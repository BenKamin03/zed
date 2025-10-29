use std::ops::Range;
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use log::{debug, error, info};
use ::edit_prediction::{Direction, EditPrediction, EditPredictionProvider};
use futures::StreamExt;
use gpui::{App, Context, Entity, Task};
use language::{Anchor, Buffer, BufferSnapshot, ToOffset, ToPoint};
use language_model::{LanguageModel, LanguageModelId, LanguageModelProviderId, LanguageModelRegistry, LanguageModelRequest, LanguageModelRequestMessage, MessageContent, Role, SelectedModel};
use language::language_settings::{all_language_settings, EditPredictionSettings};
use edit_prediction_context::{EditPredictionExcerpt, EditPredictionExcerptOptions};
use std::collections::{HashMap, VecDeque};

const DEBOUNCE_TIMEOUT: Duration = Duration::from_millis(150);
const INLINE_SYSTEM_PROMPT: &str = include_str!(
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../assets/prompts/inline_edit_prediction_system_prompt.hbs"
    )
);

#[derive(Clone)]
struct CurrentCompletion {
    snapshot: BufferSnapshot,
    edits: Arc<[(Range<Anchor>, String)]>,
    edit_preview: language::EditPreview,
}

impl CurrentCompletion {
    fn interpolate(&self, new_snapshot: &BufferSnapshot) -> Option<Vec<(Range<Anchor>, String)>> {
        edit_prediction::interpolate_edits(&self.snapshot, new_snapshot, &self.edits)
    }
}

pub struct LanguageModelEditPredictionProvider {
    pending_request: Option<Task<Result<()>>>,
    current_completion: Option<CurrentCompletion>,
    last_model_id: Option<String>,
    // Recent cross-file snippets: last N merged areas with their file path and row range
    recent_snippets: VecDeque<RecentSnippet>,
    // Track last seen full text per buffer entity to compute diffs incrementally
    last_seen_text_by_buffer: HashMap<gpui::EntityId, String>,
}

#[derive(Clone)]
struct RecentSnippet {
    file_path: String,
    start_row: u32,
    end_row: u32,
    text: String,
}

impl LanguageModelEditPredictionProvider {
    pub fn new() -> Self {
        Self {
            pending_request: None,
            current_completion: None,
            last_model_id: None,
            recent_snippets: VecDeque::new(),
            last_seen_text_by_buffer: HashMap::new(),
        }
    }

    fn resolve_model(cx: &App) -> Option<Arc<dyn LanguageModel>> {
        let registry = LanguageModelRegistry::read_global(cx);
        let settings = all_language_settings(None, cx);
        if let Some(config) = &settings.edit_predictions.language_model.model {
            // Support model ids that contain '/' (e.g. "owner/model_name:7b") by splitting only on the first '/'
            if let Some((provider_str, model_str)) = config.split_once('/') {
                let provider_id = LanguageModelProviderId::from(provider_str.to_string());
                let model_id = LanguageModelId::from(model_str.to_string());
                return registry
                    .provider(&provider_id)
                    .and_then(|provider| provider.provided_models(cx).into_iter().find(|m| m.id() == model_id));
            } else if let Ok(selected) = SelectedModel::from_str(config) {
                // Fallback to simple parser if there is exactly one '/'
                return registry
                    .provider(&selected.provider)
                    .and_then(|provider| provider.provided_models(cx).into_iter().find(|m| m.id() == selected.model));
            }
        }
        registry.default_model().map(|c| c.model)
    }

    fn build_request(prefix: &str, suffix: &str, parent_signatures: &[String], settings: &EditPredictionSettings) -> LanguageModelRequest {
        let mut stop = settings
            .language_model
            .stop
            .clone()
            .unwrap_or_default();
        // Add guard stops to discourage marker emission
        stop.push("[PREFIX]".into());
        stop.push("[/PREFIX]".into());
        stop.push("[SUFFIX]".into());
        stop.push("[/SUFFIX]".into());
        stop.sort();
        stop.dedup();
        let temperature = settings.language_model.temperature;

        let mut context = String::new();
        if !parent_signatures.is_empty() {
            context.push_str("Parent signatures:\n");
            for sig in parent_signatures.iter() {
                let trimmed = sig.trim();
                if !trimmed.is_empty() {
                    context.push_str(trimmed);
                    context.push_str("\n\n");
                }
            }
        }

        // Structure the user message with explicit prefix/suffix markers
        let user_message = if context.is_empty() {
            format!(
                "[PREFIX]\n{}\n[/PREFIX]\n[SUFFIX]\n{}\n[/SUFFIX]",
                prefix, suffix
            )
        } else {
            format!(
                "{}\n[PREFIX]\n{}\n[/PREFIX]\n[SUFFIX]\n{}\n[/SUFFIX]",
                context, prefix, suffix
            )
        };

        info!(
            "LM EditPred: user_message ({} bytes)",
            user_message.len(),
        );

        LanguageModelRequest {
            messages: vec![
                LanguageModelRequestMessage {
                    role: Role::System,
                    content: vec![MessageContent::Text(INLINE_SYSTEM_PROMPT.to_string())],
                    cache: false,
                },
                LanguageModelRequestMessage {
                    role: Role::User,
                    content: vec![MessageContent::Text(user_message)],
                    cache: false,
                },
            ],
            stop,
            temperature,
            infill: Some(!suffix.trim().is_empty()),
            thinking_allowed: false,
            ..Default::default()
        }
    }
}

impl EditPredictionProvider for LanguageModelEditPredictionProvider {
    fn name() -> &'static str { "language_model" }

    fn display_name() -> &'static str { "Language Model" }

    fn show_completions_in_menu() -> bool { true }

    fn supports_jump_to_edit() -> bool { false }

    fn is_refreshing(&self) -> bool { self.pending_request.is_some() }

    fn is_enabled(&self, _buffer: &Entity<Buffer>, _cursor_position: Anchor, cx: &App) -> bool {
        Self::resolve_model(cx).is_some()
            && LanguageModelRegistry::read_global(cx).has_authenticated_provider(cx)
    }

    fn refresh(
        &mut self,
        buffer: Entity<Buffer>,
        cursor_position: Anchor,
        debounce: bool,
        cx: &mut Context<Self>,
    ) {
        let snapshot = buffer.read(cx).snapshot();

        // Update recent edits ring buffer by diffing last seen text for this buffer
        let buffer_id = buffer.entity_id();
        let new_text = snapshot.text();
        let old_text = self
            .last_seen_text_by_buffer
            .get(&buffer_id)
            .cloned()
            .unwrap_or_default();
        if old_text != new_text {
            let file_path = buffer
                .read(cx)
                .file()
                .map(|f| f.full_path(cx).to_string_lossy().into_owned())
                .unwrap_or_else(|| format!("buffer_{}", buffer_id));

            // Use line-based diff to gather changed new-line ranges
            let line_edits = language::line_diff(&old_text, &new_text);
            
            // Indices of lines in the new text
            let mut new_lines: Vec<&str> = Vec::new();
            let mut start = 0usize;
            for (i, ch) in new_text.char_indices() {
                if ch == '\n' {
                    new_lines.push(&new_text[start..i]);
                    start = i + 1;
                }
            }
            if start <= new_text.len() {
                new_lines.push(&new_text[start..]);
            }

            // Coalesce new row ranges (merge overlapping/adjacent ranges)
            let mut merged: Vec<std::ops::Range<u32>> = Vec::new();
            for (_old_rows, new_rows) in line_edits.into_iter() {
                if new_rows.start == new_rows.end { continue; }
                if let Some(last) = merged.last_mut() {
                    if new_rows.start <= last.end + 1 {
                        last.end = last.end.max(new_rows.end);
                    } else {
                        merged.push(new_rows.clone());
                    }
                } else {
                    merged.push(new_rows.clone());
                }
            }

            // Choose the snippet covering the cursor row if possible, else the last merged range
            let cursor_row = cursor_position.to_point(&snapshot).row;
            let mut chosen = None;
            for r in merged.iter() {
                if r.start <= cursor_row && cursor_row <= r.end { chosen = Some(r.clone()); break; }
            }
            let chosen = chosen.unwrap_or_else(|| merged.last().cloned().unwrap_or(0..0));

            // Expand with surrounding context lines
            const SURROUND: u32 = 3;
            let start_row = chosen.start.saturating_sub(SURROUND);
            let end_row = (chosen.end + SURROUND).min((new_lines.len() as u32).saturating_sub(1));

            if start_row <= end_row {
                let mut text = String::new();
                for row in start_row..=end_row {
                    let idx = row as usize;
                    if idx < new_lines.len() {
                        text.push_str(new_lines[idx]);
                        text.push('\n');
                    }
                }

                // Merge with last snippet if same file and overlapping/adjacent ranges
                if let Some(last) = self.recent_snippets.back_mut() {
                    if last.file_path == file_path {
                        let overlap = !(end_row + 1 < last.start_row || start_row > last.end_row + 1);
                        if overlap {
                            let merged_start = start_row.min(last.start_row);
                            let merged_end = end_row.max(last.end_row);

                            // Rebuild merged text from current new_lines
                            let mut merged_text = String::new();
                            for row in merged_start..=merged_end {
                                let idx = row as usize;
                                if idx < new_lines.len() {
                                    merged_text.push_str(new_lines[idx]);
                                    merged_text.push('\n');
                                }
                            }
                            last.start_row = merged_start;
                            last.end_row = merged_end;
                            last.text = merged_text;
                        } else {
                            self.recent_snippets.push_back(RecentSnippet { file_path: file_path.clone(), start_row, end_row, text });
                        }
                    } else {
                        self.recent_snippets.push_back(RecentSnippet { file_path: file_path.clone(), start_row, end_row, text });
                    }
                } else {
                    self.recent_snippets.push_back(RecentSnippet { file_path: file_path.clone(), start_row, end_row, text });
                }

                let recent_limit: usize = all_language_settings(None, cx)
                    .edit_predictions
                    .language_model
                    .recent_edits_max_snippets
                    .unwrap() as usize;
                while self.recent_snippets.len() > recent_limit {
                    self.recent_snippets.pop_front();
                }
            }

            self.last_seen_text_by_buffer.insert(buffer_id, new_text);
        }

        if let Some(current) = self.current_completion.as_ref() {
            if current.interpolate(&snapshot).is_some() {
                return;
            }
        }

        let model = Self::resolve_model(cx);
        if model.is_none() {
            debug!("LM EditPred: no model available or provider not authenticated");
            return;
        }
        let model = model.unwrap();

        // If the selected model changed via settings, clear any cached completion
        let model_id_now = model.telemetry_id();
        if self.last_model_id.as_deref() != Some(&model_id_now) {
            debug!("LM EditPred: model changed to {} — clearing cached completion", model_id_now);
            self.current_completion = None;
            self.last_model_id = Some(model_id_now.clone());
        }

        // Build a structured excerpt with prefix and suffix around the cursor
        const EXCERPT_OPTIONS: EditPredictionExcerptOptions = EditPredictionExcerptOptions {
            max_bytes: 1400,
            min_bytes: 600,
            target_before_cursor_over_total_bytes: 0.66,
        };
        let cursor_offset = cursor_position.to_offset(&snapshot);
        let cursor_point = cursor_offset.to_point(&snapshot);
        let excerpt = match EditPredictionExcerpt::select_from_buffer(
            cursor_point,
            &snapshot,
            &EXCERPT_OPTIONS,
            None,
        ) {
            Some(excerpt) => excerpt,
            None => {
                // Fallback: prefix-only small window
                let start = cursor_offset.saturating_sub(1024);
                let prefix = snapshot
                    .text_for_range(start..cursor_offset)
                    .collect::<String>();
                let settings = all_language_settings(None, cx).edit_predictions.clone();
                let request = Self::build_request(&prefix, "", &[], &settings);
                let attempts = all_language_settings(None, cx)
                    .edit_predictions
                    .language_model
                    .empty_completion_total_attempts
                    .unwrap()
                    .max(1);

                self.pending_request = Some(cx.spawn(async move |this, cx| {
                    if debounce { smol::Timer::after(DEBOUNCE_TIMEOUT).await; }
                    let mut completion = String::new();
                    for _ in 0..attempts {
                        let stream = model.stream_completion_text(request.clone(), cx).await;
                        let Ok(mut stream) = stream else {
                            error!("LM EditPred: failed to start stream model={}", model.telemetry_id());
                            break;
                        };
                        let mut generated = String::new();
                        while let Some(chunk) = stream.stream.next().await {
                            match chunk { Ok(text) => generated.push_str(&text), Err(_) => break }
                        }
                        // Sanitize markers
                        for marker in ["[PREFIX]", "[/PREFIX]", "[SUFFIX]", "[/SUFFIX]"] {
                            if generated.contains(marker) {
                                generated = generated.replace(marker, "");
                            }
                        }
                        // remove internal echoes of verbatim context
                        let mut candidates: std::collections::HashSet<String> = std::collections::HashSet::new();
                        
                        // long tail of prefix
                        if !prefix.is_empty() {
                            let mut tail = String::new();
                            let mut count = 0usize;
                            for ch in prefix.chars().rev() {
                                if count >= 200 { break; }
                                tail.insert(0, ch);
                                count += 1;
                            }
                            if tail.trim().len() >= 40 { candidates.insert(tail); }
                            // long lines from prefix
                            for line in prefix.lines() {
                                let l = line.trim_end();
                                if l.len() >= 40 { candidates.insert(l.to_string()); }
                            }
                        }

                        for cand in candidates.into_iter() {
                            if !cand.is_empty() && generated.contains(&cand) {
                                generated = generated.replace(&cand, "");
                            }
                        }
                        if !generated.trim().is_empty() {
                            completion = generated;
                            break;
                        }
                    }
                    if completion.trim().is_empty() {
                        this.update(cx, |this, cx| { this.pending_request = None; cx.notify(); })?;
                        return Ok(());
                    }
                    let edits: Arc<[(Range<Anchor>, String)]> =
                        vec![(cursor_position..cursor_position, completion)].into();
                    let edit_preview = buffer
                        .read_with(cx, |buffer, cx| buffer.preview_edits(edits.clone(), cx))?
                        .await;
                    this.update(cx, |this, cx| {
                        this.current_completion = Some(CurrentCompletion { snapshot: snapshot.clone(), edits, edit_preview });
                        this.pending_request = None;
                        cx.notify();
                    })?;
                    Ok(())
                }));
                return;
            }
        };
        let excerpt_text = excerpt.text(&snapshot);
        
        // compute prefix/suffix using a line-centered window around the cursor
        let settings_window_lines = all_language_settings(None, cx)
            .edit_predictions
            .language_model
            .current_space_window_lines
            .unwrap();

        let window_lines: u32 = settings_window_lines;
        let last_row = snapshot.max_point().row;
        let total_rows = last_row.saturating_add(1);
        let cursor_row = cursor_point.row;

        let (start_row, end_row) = if total_rows > window_lines {
            let half = window_lines / 2;
            let min_start = 0u32;
            let max_start = total_rows.saturating_sub(window_lines);
            let desired_start = cursor_row.saturating_sub(half);
            let start = desired_start.clamp(min_start, max_start);
            let end = start.saturating_add(window_lines.saturating_sub(1));
            (start, end)
        } else {
            (0, last_row)
        };

        let window_start_offset = language::Point::new(start_row, 0).to_offset(&snapshot);

        let window_end_offset = if end_row < last_row {
            language::Point::new(end_row + 1, 0).to_offset(&snapshot)
        } else {
            snapshot.text().len()
        };

        let prefix = snapshot
            .text_for_range(window_start_offset.min(cursor_offset)..cursor_offset)
            .collect::<String>();
        let suffix = snapshot
            .text_for_range(cursor_offset..window_end_offset.max(cursor_offset))
            .collect::<String>();

        // current buffer path to de-duplicate same-file context
        let current_file_path = buffer
            .read(cx)
            .file()
            .map(|f| f.full_path(cx).to_string_lossy().into_owned())
            .unwrap_or_else(|| format!("buffer_{}", buffer.entity_id()));

        let parent_signatures = {
            let mut sigs = excerpt_text.parent_signatures;
            if !self.recent_snippets.is_empty() {
                let mut recent = String::from("Recent edits across files (latest first):\n");
                let mut added_any = false;
                for snippet in self.recent_snippets.iter().rev() {
                    if snippet.file_path == current_file_path { continue; }
                    recent.push_str(&snippet.file_path);
                    recent.push_str(":\n");
                    recent.push_str(&snippet.text);
                    if !recent.ends_with('\n') { recent.push('\n'); }
                    added_any = true;
                }
                if added_any {
                    sigs.insert(0, recent);
                }
            }
            sigs
        };

        // Keep a small tail of the prefix to de-duplicate from model output later
        let prefix_tail: String = {
            const TAIL_MAX_CHARS: usize = 80;
            let mut tail = String::new();
            let mut count = 0;
            for ch in prefix.chars().rev() {
                if count >= TAIL_MAX_CHARS { break; }
                tail.insert(0, ch);
                count += 1;
            }
            tail
        };

        let settings = all_language_settings(None, cx).edit_predictions.clone();
        let request = Self::build_request(&prefix, &suffix, &parent_signatures, &settings);
        let attempts = all_language_settings(None, cx)
            .edit_predictions
            .language_model
            .empty_completion_total_attempts
            .unwrap()
            .max(1);

        self.pending_request = Some(cx.spawn(async move |this, cx| {
            if debounce { smol::Timer::after(DEBOUNCE_TIMEOUT).await; }

            // Log request details
            let model_id = model.telemetry_id();
            let req_temp = request.temperature;
            let req_stop_len = request.stop.len();
            let prefix_chars = prefix.chars().count();
            debug!(
                "LM EditPred: request start model={} prefix_len={} temperature={:?} stop_count={}",
                model_id,
                prefix_chars,
                req_temp,
                req_stop_len
            );

            let started_at = Instant::now();

            // Attempts loop: stream text and collect until non-empty or attempts exhausted
            let mut completion = String::new();
            for _ in 0..attempts {
                let stream = model.stream_completion_text(request.clone(), cx).await;
                let Ok(mut stream) = stream else {
                    error!("LM EditPred: failed to start stream model={}", model_id);
                    break;
                };
                let mut generated = String::new();
                while let Some(chunk) = stream.stream.next().await {
                    match chunk {
                        Ok(text) => generated.push_str(&text),
                        Err(e) => {
                            error!("LM EditPred: stream error model={} err={}", model_id, e);
                            break;
                        }
                    }
                }

                // Sanitize: strip any prompt markers
                for marker in ["[PREFIX]", "[/PREFIX]", "[SUFFIX]", "[/SUFFIX]"] {
                    if generated.contains(marker) {
                        generated = generated.replace(marker, "");
                    }
                }

                // Sanitize: remove any duplicate text already present at the end of prefix
                let dedup_prefix_overlap = |prefix_tail: &str, generated: &str| -> usize {
                    // longest suffix of prefix_tail that is a prefix of generated
                    let pt_chars: Vec<char> = prefix_tail.chars().collect();
                    let g_chars: Vec<char> = generated.chars().collect();
                    let max_len = pt_chars.len().min(g_chars.len());
                    for len in (1..=max_len).rev() {
                        if pt_chars[pt_chars.len()-len..] == g_chars[..len] {
                            // compute byte length of that prefix in generated
                            return g_chars[..len].iter().map(|c| c.len_utf8()).sum();
                        }
                    }
                    0
                };
                let dup = dedup_prefix_overlap(&prefix_tail, &generated);
                if dup > 0 { generated = generated[dup..].to_string(); }

                // Sanitize: remove any leading overlap with provided suffix to avoid echoing
                let common_prefix_len = |a: &str, b: &str| -> usize {
                    a.chars()
                        .zip(b.chars())
                        .take_while(|(x, y)| x == y)
                        .map(|(c, _)| c.len_utf8())
                        .sum()
                };
                let overlap = common_prefix_len(&generated, &suffix);
                let generated = if overlap > 0 {
                    generated[overlap..].to_string()
                } else {
                    generated
                };

                // Additional dedupe: remove internal echoes of verbatim context
                let mut candidates: std::collections::HashSet<String> = std::collections::HashSet::new();
                // Long tail of prefix
                if !prefix.is_empty() {
                    let mut tail = String::new();
                    let mut count = 0usize;
                    for ch in prefix.chars().rev() {
                        if count >= 200 { break; }
                        tail.insert(0, ch);
                        count += 1;
                    }
                    if tail.trim().len() >= 40 { candidates.insert(tail); }
                    for line in prefix.lines() {
                        let l = line.trim_end();
                        if l.len() >= 40 { candidates.insert(l.to_string()); }
                    }
                }
                // Head of suffix
                if !suffix.is_empty() {
                    let mut head = String::new();
                    let mut count = 0usize;
                    for ch in suffix.chars() {
                        if count >= 200 { break; }
                        head.push(ch);
                        count += 1;
                    }
                    if head.trim().len() >= 40 { candidates.insert(head); }
                    for line in suffix.lines() {
                        let l = line.trim_end();
                        if l.len() >= 40 { candidates.insert(l.to_string()); }
                    }
                }
                let mut generated = generated;
                for cand in candidates.into_iter() {
                    if !cand.is_empty() && generated.contains(&cand) {
                        generated = generated.replace(&cand, "");
                    }
                }

                if !generated.trim().is_empty() {
                    completion = generated;
                    break;
                }
            }

            if completion.trim().is_empty() {
                let elapsed = started_at.elapsed();
                debug!(
                    "LM EditPred: empty completion after {} attempts model={} elapsed_ms={}",
                    attempts,
                    model_id,
                    elapsed.as_millis()
                );
                this.update(cx, |this, cx| { this.pending_request = None; cx.notify(); })?;
                return Ok(());
            }

            let elapsed = started_at.elapsed();
            debug!(
                "LM EditPred: completion received model={} chars={} elapsed_ms={}",
                model_id,
                completion.chars().count(),
                elapsed.as_millis()
            );

            // Prepare edits and preview
            let edits: Arc<[(Range<Anchor>, String)]> =
                vec![(cursor_position..cursor_position, completion)].into();
            let edit_preview = buffer
                .read_with(cx, |buffer, cx| buffer.preview_edits(edits.clone(), cx))?
                .await;

            this.update(cx, |this, cx| {
                this.current_completion = Some(CurrentCompletion {
                    snapshot: snapshot.clone(),
                    edits,
                    edit_preview,
                });
                this.pending_request = None;
                cx.notify();
            })?;

            Ok(())
        }));
    }

    fn cycle(
        &mut self,
        _buffer: Entity<Buffer>,
        _cursor_position: Anchor,
        _direction: Direction,
        _cx: &mut Context<Self>,
    ) {
        // Not supported in the initial implementation
    }

    fn accept(&mut self, _cx: &mut Context<Self>) {
        self.pending_request = None;
        self.current_completion = None;
    }

    fn discard(&mut self, _cx: &mut Context<Self>) {
        self.pending_request = None;
        self.current_completion = None;
    }

    fn suggest(
        &mut self,
        buffer: &Entity<Buffer>,
        _cursor_position: Anchor,
        cx: &mut Context<Self>,
    ) -> Option<EditPrediction> {
        let current = self.current_completion.as_ref()?;
        let buffer = buffer.read(cx);
        let edits = current.interpolate(&buffer.snapshot())?;
        if edits.is_empty() {
            return None;
        }
        Some(EditPrediction::Local { id: None, edits, edit_preview: Some(current.edit_preview.clone()) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::prelude::*;
    use language::Buffer;
    use language::language_settings::{EditPredictionSettings, LanguageModelProviderSettings};
    use language_model::{fake_provider::FakeLanguageModel, init_settings as init_lm_settings};
    use std::sync::{Mutex, OnceLock};

    fn serial_guard() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    fn set_user_language_model_settings(
        cx: &mut App,
        apply: impl FnOnce(&mut settings::SettingsContent, &App),
    ) {
        SettingsStore::global(cx).update_settings_file(<dyn fs::Fs>::global(cx), apply);
    }

    #[gpui::test]
    async fn is_enabled_true_with_default_model_and_authenticated(cx: &mut App) {
        let _g = serial_guard();
        // Initialize registry with a fake provider and default model
        init_lm_settings(cx);
        language_model::LanguageModelRegistry::test(cx);

        // Minimal buffer entity; contents are irrelevant for is_enabled()
        let buffer = cx.new(|cx| Buffer::local("", cx));

        let provider_entity = cx.new(|_| LanguageModelEditPredictionProvider::new());
        let enabled = provider_entity.update(cx, |this, cx| {
            this.is_enabled(&buffer, language::Anchor::MIN, cx)
        });

        assert!(enabled, "provider should be enabled when a default model exists and a provider is authenticated");

        // Reset registry to a clean default to avoid impacting unrelated tests
        init_lm_settings(cx);
    }

    #[gpui::test]
    async fn request_includes_guard_stops_and_temperature(cx: &mut App) {
        let _g = serial_guard();
        // Build settings locally without touching global SettingsStore
        let settings = EditPredictionSettings {
            language_model: LanguageModelProviderSettings {
                model: None,
                temperature: Some(0.7),
                max_tokens: None,
                stop: Some(vec!["X".into(), "[PREFIX]".into()]),
            },
            ..Default::default()
        };

        let req = LanguageModelEditPredictionProvider::build_request(
            "prefix text",
            "suffix text",
            &[],
            &settings,
        );

        // Temperature passes through
        assert_eq!(req.temperature, Some(0.7));

        // Stop sequences contain guards, user-provided values, and are deduped
        let stops: std::collections::HashSet<_> = req.stop.iter().cloned().collect();
        for s in ["[PREFIX]", "[/PREFIX]", "[SUFFIX]", "[/SUFFIX]", "X"] {
            assert!(stops.contains(s), "missing stop token: {s}");
        }
    }
}


