use std::ops::Range;
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use log::{debug, error};
use ::edit_prediction::{Direction, EditPrediction, EditPredictionProvider};
use futures::StreamExt;
use gpui::{App, Context, Entity, Task};
use language::{Anchor, Buffer, BufferSnapshot, ToOffset, ToPoint};
use language_model::{LanguageModel, LanguageModelId, LanguageModelProviderId, LanguageModelRegistry, LanguageModelRequest, LanguageModelRequestMessage, MessageContent, Role, SelectedModel};
use language::language_settings::{all_language_settings, EditPredictionSettings};
use edit_prediction_context::{EditPredictionExcerpt, EditPredictionExcerptOptions};

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
}

impl LanguageModelEditPredictionProvider {
    pub fn new() -> Self {
        Self { pending_request: None, current_completion: None, last_model_id: None }
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
                self.pending_request = Some(cx.spawn(async move |this, cx| {
                    if debounce { smol::Timer::after(DEBOUNCE_TIMEOUT).await; }
                    let stream = model.stream_completion_text(request, cx).await;
                    let Ok(mut stream) = stream else {
                        error!("LM EditPred: failed to start stream model={}", model.telemetry_id());
                        this.update(cx, |this, cx| { this.pending_request = None; cx.notify(); })?;
                        return Ok(());
                    };
                    let mut completion = String::new();
                    while let Some(chunk) = stream.stream.next().await {
                        match chunk { Ok(text) => completion.push_str(&text), Err(_) => break }
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
        let body = &excerpt_text.body;
        let cursor_in_excerpt = cursor_offset
            .saturating_sub(excerpt.range.start)
            .min(body.len());
        let prefix = body[..cursor_in_excerpt].to_string();
        let suffix = body[cursor_in_excerpt..].to_string();
        let parent_signatures = excerpt_text.parent_signatures;

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

            // Stream text and collect into a single completion string
            let stream = model.stream_completion_text(request, cx).await;
            let Ok(mut stream) = stream else {
                error!("LM EditPred: failed to start stream model={}", model_id);
                this.update(cx, |this, cx| { this.pending_request = None; cx.notify(); })?;
                return Ok(());
            };

            let mut completion = String::new();
            while let Some(chunk) = stream.stream.next().await {
                match chunk {
                    Ok(text) => completion.push_str(&text),
                    Err(e) => {
                        error!("LM EditPred: stream error model={} err={}", model_id, e);
                        break;
                    }
                }
            }

            // Sanitize: strip any prompt markers
            for marker in ["[PREFIX]", "[/PREFIX]", "[SUFFIX]", "[/SUFFIX]"] {
                if completion.contains(marker) {
                    completion = completion.replace(marker, "");
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
            let dup = dedup_prefix_overlap(&prefix_tail, &completion);
            if dup > 0 { completion = completion[dup..].to_string(); }

            // Sanitize: remove any leading overlap with provided suffix to avoid echoing
            let common_prefix_len = |a: &str, b: &str| -> usize {
                a.chars()
                    .zip(b.chars())
                    .take_while(|(x, y)| x == y)
                    .map(|(c, _)| c.len_utf8())
                    .sum()
            };
            let overlap = common_prefix_len(&completion, &suffix);
            if overlap > 0 {
                completion = completion[overlap..].to_string();
            }

            if completion.trim().is_empty() {
                let elapsed = started_at.elapsed();
                debug!(
                    "LM EditPred: empty completion model={} elapsed_ms={}",
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


