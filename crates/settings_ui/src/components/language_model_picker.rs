use std::sync::Arc;
use std::collections::HashSet;

use gpui::{App, Window};
use language_model::{ConfiguredModel, LanguageModel, LanguageModelRegistry};
use ui::{Button, ButtonSize, ButtonStyle, ContextMenu, ContextMenuEntry, IconName, IconPosition, PopoverMenu, prelude::*};

pub fn language_model_picker(
    active: Option<ConfiguredModel>,
    on_model_chosen: impl Fn(Arc<dyn LanguageModel>, &mut App) + 'static + Send + Sync,
    _window: &mut Window,
    _cx: &mut App,
) -> AnyElement {
    let label = match active {
        Some(ref active) => active.model.name().0,
        None => SharedString::from("Select a Model"),
    };
    let on_chosen = Arc::new(on_model_chosen);

    PopoverMenu::new("edit-prediction-model-picker")
        .trigger(
            Button::new("edit_prediction_model_picker_trigger", label)
                .tab_index(0_isize)
                .style(ButtonStyle::Outlined)
                .size(ButtonSize::Medium)
                .icon(IconName::ChevronUpDown)
                .icon_color(Color::Muted)
                .icon_size(IconSize::Small)
                .icon_position(IconPosition::End),
        )
        .menu(move |window, cx| {
            let on_chosen = on_chosen.clone();
            Some(ContextMenu::build(window, cx, move |mut menu, _window, cx| {
                let registry = LanguageModelRegistry::read_global(cx);

                // Collect recommended entries and their IDs to avoid duplication
                let mut recommended: Vec<(SharedString, IconName, Arc<dyn LanguageModel>)> = Vec::new();
                let mut recommended_ids: HashSet<language_model::LanguageModelId> = HashSet::new();
                for provider in registry.providers().iter() {
                    for model in provider.recommended_models(cx).into_iter() {
                        recommended_ids.insert(model.id());
                        recommended.push((model.name().0, provider.icon(), model.clone()));
                    }
                }

                if !recommended.is_empty() {
                    menu = menu.header("Recommended");
                    for (name, icon, model) in recommended.into_iter() {
                        let on_chosen = on_chosen.clone();
                        let model_clone = model.clone();
                        menu = menu.item(
                            ContextMenuEntry::new(name.clone())
                                .icon(icon)
                                .handler(move |_, cx| {
                                    on_chosen(model_clone.clone(), cx);
                                }),
                        );
                    }
                }

                // Collect remaining provided models not already listed as recommended
                let mut other: Vec<(SharedString, IconName, Arc<dyn LanguageModel>)> = Vec::new();
                for provider in registry.providers().iter() {
                    for model in provider.provided_models(cx).into_iter() {
                        if recommended_ids.contains(&model.id()) {
                            continue;
                        }
                        other.push((model.name().0, provider.icon(), model.clone()));
                    }
                }

                if !other.is_empty() {
                    menu = menu.header("Other");
                    for (name, icon, model) in other.into_iter() {
                        let on_chosen = on_chosen.clone();
                        let model_clone = model.clone();
                        menu = menu.item(
                            ContextMenuEntry::new(name.clone())
                                .icon(icon)
                                .handler(move |_, cx| {
                                    on_chosen(model_clone.clone(), cx);
                                }),
                        );
                    }
                }

                menu
            }))
        })
        .anchor(gpui::Corner::TopLeft)
        .offset(gpui::Point { x: px(0.0), y: px(2.0) })
        .with_handle(ui::PopoverMenuHandle::default())
        .into_any_element()
}


