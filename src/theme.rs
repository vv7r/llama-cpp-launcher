//! Thèmes de l'interface : sombre et clair inspirés de VS Code (Dark+ et
//! Light+), puis Nord, Dracula, Gruvbox et Solarized clair.
//!
//! Les couleurs sont lues à travers des fonctions (`theme::text()`…) et non
//! des constantes : le thème se change à chaud, depuis la barre du haut.

use crate::config::ThemeChoice;
use egui::{Color32, CornerRadius, Stroke, Visuals};
use std::sync::atomic::{AtomicU8, Ordering};

/// Toutes les couleurs de l'interface pour un thème.
struct Palette {
    /// Fond clair : widgets d'egui en variante claire, boutons d'action
    /// teintés plutôt que pleins.
    light: bool,
    bg: Color32,
    card: Color32,
    card_hi: Color32,
    hover: Color32,
    border: Color32,
    hover_border: Color32,
    accent: Color32,
    text: Color32,
    /// Texte le plus contrasté (titre d'une carte active, survol).
    strong: Color32,
    muted: Color32,
    green: Color32,
    red: Color32,
    orange: Color32,
    yellow: Color32,
    /// Fond de la console et de la commande générée.
    code_bg: Color32,
    active_card: Color32,
    danger_card: Color32,
    chip: Color32,
    chip_text: Color32,
    chip_primary: Color32,
    chip_primary_text: Color32,
}

const fn rgb(r: u8, g: u8, b: u8) -> Color32 {
    Color32::from_rgb(r, g, b)
}

const DARK: Palette = Palette {
    light: false,
    bg: rgb(0x1E, 0x1E, 0x1E),
    card: rgb(0x25, 0x25, 0x26),
    card_hi: rgb(0x2D, 0x2D, 0x30),
    hover: rgb(0x3A, 0x3A, 0x3D),
    border: rgb(0x3E, 0x3E, 0x42),
    hover_border: rgb(0x55, 0x55, 0x5A),
    accent: rgb(0x00, 0x7A, 0xCC),
    text: rgb(0xE0, 0xE0, 0xE0),
    strong: Color32::WHITE,
    muted: rgb(0x9A, 0x9A, 0x9A),
    green: rgb(0x2E, 0xA0, 0x43),
    red: rgb(0xC4, 0x3B, 0x3B),
    orange: rgb(0xCC, 0x7A, 0x00),
    yellow: rgb(0xD7, 0xBA, 0x7D),
    code_bg: rgb(0x18, 0x18, 0x18),
    active_card: rgb(0x1B, 0x33, 0x47),
    danger_card: rgb(0x3A, 0x22, 0x22),
    chip: rgb(0x36, 0x36, 0x3A),
    chip_text: rgb(0xE0, 0xE0, 0xE0),
    chip_primary: rgb(0x0E, 0x4A, 0x75),
    chip_primary_text: Color32::WHITE,
};

/// Couleurs d'état plus foncées qu'en sombre : lisibles sur fond blanc.
const LIGHT: Palette = Palette {
    light: true,
    bg: rgb(0xF3, 0xF3, 0xF3),
    card: rgb(0xFF, 0xFF, 0xFF),
    card_hi: rgb(0xE8, 0xE8, 0xE8),
    hover: rgb(0xDC, 0xDC, 0xDC),
    border: rgb(0xCE, 0xCE, 0xCE),
    hover_border: rgb(0x9E, 0x9E, 0x9E),
    accent: rgb(0x00, 0x5F, 0xB8),
    text: rgb(0x1F, 0x1F, 0x1F),
    strong: rgb(0x00, 0x00, 0x00),
    muted: rgb(0x61, 0x61, 0x61),
    green: rgb(0x1A, 0x7F, 0x37),
    red: rgb(0xC4, 0x2B, 0x1C),
    orange: rgb(0xA8, 0x55, 0x00),
    yellow: rgb(0x8A, 0x63, 0x00),
    code_bg: rgb(0xFF, 0xFF, 0xFF),
    active_card: rgb(0xDC, 0xEB, 0xFA),
    danger_card: rgb(0xFB, 0xE4, 0xE2),
    chip: rgb(0xE4, 0xE4, 0xE4),
    chip_text: rgb(0x1F, 0x1F, 0x1F),
    chip_primary: rgb(0xCF, 0xE3, 0xF7),
    chip_primary_text: rgb(0x0B, 0x3D, 0x66),
};

const NORD: Palette = Palette {
    light: false,
    bg: rgb(0x2E, 0x34, 0x40),
    card: rgb(0x3B, 0x42, 0x52),
    card_hi: rgb(0x43, 0x4C, 0x5E),
    hover: rgb(0x4C, 0x56, 0x6A),
    border: rgb(0x4C, 0x56, 0x6A),
    hover_border: rgb(0x61, 0x6E, 0x88),
    accent: rgb(0x5E, 0x81, 0xAC),
    text: rgb(0xD8, 0xDE, 0xE9),
    strong: rgb(0xEC, 0xEF, 0xF4),
    muted: rgb(0x9A, 0xA3, 0xB5),
    green: rgb(0x6A, 0x9A, 0x5B),
    red: rgb(0xBF, 0x61, 0x6A),
    orange: rgb(0xC0, 0x7A, 0x4E),
    yellow: rgb(0xEB, 0xCB, 0x8B),
    code_bg: rgb(0x27, 0x2C, 0x36),
    active_card: rgb(0x3A, 0x4D, 0x66),
    danger_card: rgb(0x4A, 0x30, 0x36),
    chip: rgb(0x43, 0x4C, 0x5E),
    chip_text: rgb(0xE5, 0xE9, 0xF0),
    chip_primary: rgb(0x4C, 0x6A, 0x8C),
    chip_primary_text: rgb(0xEC, 0xEF, 0xF4),
};

const DRACULA: Palette = Palette {
    light: false,
    bg: rgb(0x21, 0x22, 0x2C),
    card: rgb(0x28, 0x2A, 0x36),
    card_hi: rgb(0x34, 0x37, 0x46),
    hover: rgb(0x44, 0x47, 0x5A),
    border: rgb(0x44, 0x47, 0x5A),
    hover_border: rgb(0x62, 0x72, 0xA4),
    accent: rgb(0x7C, 0x5C, 0xC4),
    text: rgb(0xF8, 0xF8, 0xF2),
    strong: Color32::WHITE,
    muted: rgb(0x8E, 0x95, 0xB8),
    green: rgb(0x2F, 0xA8, 0x5A),
    red: rgb(0xD9, 0x44, 0x44),
    orange: rgb(0xD9, 0x8A, 0x3D),
    yellow: rgb(0xF1, 0xFA, 0x8C),
    code_bg: rgb(0x1E, 0x1F, 0x29),
    active_card: rgb(0x3C, 0x35, 0x60),
    danger_card: rgb(0x4A, 0x25, 0x30),
    chip: rgb(0x34, 0x37, 0x46),
    chip_text: rgb(0xF8, 0xF8, 0xF2),
    chip_primary: rgb(0x5A, 0x4A, 0x8A),
    chip_primary_text: Color32::WHITE,
};

const GRUVBOX: Palette = Palette {
    light: false,
    bg: rgb(0x28, 0x28, 0x28),
    card: rgb(0x32, 0x30, 0x2F),
    card_hi: rgb(0x3C, 0x38, 0x36),
    hover: rgb(0x50, 0x49, 0x45),
    border: rgb(0x50, 0x49, 0x45),
    hover_border: rgb(0x66, 0x5C, 0x54),
    accent: rgb(0x45, 0x85, 0x88),
    text: rgb(0xEB, 0xDB, 0xB2),
    strong: rgb(0xFB, 0xF1, 0xC7),
    muted: rgb(0xA8, 0x99, 0x84),
    green: rgb(0x79, 0x86, 0x0E),
    red: rgb(0xCC, 0x24, 0x1D),
    orange: rgb(0xD6, 0x5D, 0x0E),
    yellow: rgb(0xFA, 0xBD, 0x2F),
    code_bg: rgb(0x1D, 0x20, 0x21),
    active_card: rgb(0x2F, 0x45, 0x46),
    danger_card: rgb(0x4A, 0x2A, 0x26),
    chip: rgb(0x3C, 0x38, 0x36),
    chip_text: rgb(0xEB, 0xDB, 0xB2),
    chip_primary: rgb(0x3E, 0x66, 0x68),
    chip_primary_text: rgb(0xFB, 0xF1, 0xC7),
};

const SOLARIZED: Palette = Palette {
    light: true,
    bg: rgb(0xEE, 0xE8, 0xD5),
    card: rgb(0xFD, 0xF6, 0xE3),
    card_hi: rgb(0xE8, 0xE1, 0xCC),
    hover: rgb(0xDD, 0xD6, 0xC1),
    border: rgb(0xD3, 0xCB, 0xB4),
    hover_border: rgb(0x93, 0xA1, 0xA1),
    accent: rgb(0x26, 0x8B, 0xD2),
    text: rgb(0x3B, 0x4E, 0x55),
    strong: rgb(0x00, 0x2B, 0x36),
    muted: rgb(0x6C, 0x7F, 0x84),
    green: rgb(0x6B, 0x7A, 0x00),
    red: rgb(0xC8, 0x2A, 0x27),
    orange: rgb(0xB8, 0x44, 0x14),
    yellow: rgb(0x8A, 0x6A, 0x00),
    code_bg: rgb(0xFD, 0xF6, 0xE3),
    active_card: rgb(0xDC, 0xE8, 0xEE),
    danger_card: rgb(0xF5, 0xDD, 0xD3),
    chip: rgb(0xE6, 0xDF, 0xCA),
    chip_text: rgb(0x3B, 0x4E, 0x55),
    chip_primary: rgb(0xCF, 0xE2, 0xEE),
    chip_primary_text: rgb(0x0B, 0x4A, 0x70),
};

/// Dans l'ordre de `ThemeChoice`.
const PALETTES: [&Palette; 6] = [&DARK, &LIGHT, &NORD, &DRACULA, &GRUVBOX, &SOLARIZED];

static CURRENT: AtomicU8 = AtomicU8::new(0);

fn p() -> &'static Palette {
    PALETTES[CURRENT.load(Ordering::Relaxed) as usize % PALETTES.len()]
}

pub fn bg() -> Color32 { p().bg }
pub fn card() -> Color32 { p().card }
pub fn card_hi() -> Color32 { p().card_hi }
pub fn border() -> Color32 { p().border }
pub fn hover_border() -> Color32 { p().hover_border }
pub fn accent() -> Color32 { p().accent }
pub fn text() -> Color32 { p().text }
pub fn strong() -> Color32 { p().strong }
pub fn muted() -> Color32 { p().muted }
pub fn green() -> Color32 { p().green }
pub fn red() -> Color32 { p().red }
pub fn orange() -> Color32 { p().orange }
pub fn yellow() -> Color32 { p().yellow }
pub fn code_bg() -> Color32 { p().code_bg }
pub fn active_card() -> Color32 { p().active_card }
pub fn danger_card() -> Color32 { p().danger_card }
pub fn chip() -> (Color32, Color32) { (p().chip, p().chip_text) }
pub fn chip_primary() -> (Color32, Color32) { (p().chip_primary, p().chip_primary_text) }

/// Applique le thème choisi à egui.
pub fn apply(ctx: &egui::Context, choice: ThemeChoice) {
    let index = ThemeChoice::ALL.iter().position(|&t| t == choice).unwrap_or(0);
    CURRENT.store(index as u8, Ordering::Relaxed);
    let c = p();
    let light = c.light;
    let mut visuals = if light { Visuals::light() } else { Visuals::dark() };
    visuals.panel_fill = c.bg;
    visuals.window_fill = c.card;
    visuals.extreme_bg_color = c.code_bg;
    visuals.faint_bg_color = c.card;
    visuals.override_text_color = Some(c.text);
    visuals.hyperlink_color = c.accent;
    visuals.selection.bg_fill = c.accent.gamma_multiply(if light { 0.35 } else { 0.55 });
    visuals.window_stroke = Stroke::new(1.0, c.border);

    let r = CornerRadius::same(4);
    visuals.widgets.noninteractive.bg_fill = c.card;
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, c.border);
    visuals.widgets.noninteractive.corner_radius = r;
    visuals.widgets.inactive.bg_fill = c.card_hi;
    visuals.widgets.inactive.weak_bg_fill = c.card_hi;
    visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, c.border);
    visuals.widgets.inactive.corner_radius = r;
    visuals.widgets.hovered.bg_fill = c.hover;
    visuals.widgets.hovered.weak_bg_fill = c.hover;
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, c.accent);
    visuals.widgets.hovered.corner_radius = r;
    visuals.widgets.active.bg_fill = c.accent;
    visuals.widgets.active.weak_bg_fill = c.accent;
    visuals.widgets.active.bg_stroke = Stroke::new(1.0, c.accent);
    visuals.widgets.active.corner_radius = r;
    visuals.widgets.open.bg_fill = c.card_hi;
    visuals.widgets.open.weak_bg_fill = c.card_hi;
    visuals.widgets.open.bg_stroke = Stroke::new(1.0, c.accent);
    visuals.widgets.open.corner_radius = r;

    // Couleur du texte dans chaque état : explicite plutôt qu'héritée, pour
    // qu'aucun libellé ne se fonde dans son fond.
    //
    // egui colore aussi les textes « forts » (`RichText::strong`, titres des
    // groupes) avec la couleur des widgets actifs : elle doit donc être la
    // plus contrastée du thème — blanc en sombre, noir en clair. Les boutons
    // d'action (Démarrer, Supprimer…) fixent eux-mêmes leur texte :
    // voir `action_button`.
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, c.text);
    visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, c.text);
    visuals.widgets.hovered.fg_stroke = Stroke::new(1.5, c.strong);
    visuals.widgets.active.fg_stroke = Stroke::new(2.0, c.strong);
    visuals.widgets.open.fg_stroke = Stroke::new(1.0, c.strong);

    // `set_visuals` ne touche que le thème actif au moment de l'appel ; eframe
    // bascule ensuite sur le thème du système. On impose donc le thème voulu
    // et on écrit la palette dans les deux variantes : le thème de Windows ne
    // peut plus s'y substituer.
    ctx.set_theme(if light { egui::Theme::Light } else { egui::Theme::Dark });
    ctx.all_styles_mut(|style| style.visuals = visuals.clone());

    ctx.all_styles_mut(|style| {
        style.spacing.item_spacing = egui::vec2(8.0, 6.0);
        style.spacing.button_padding = egui::vec2(10.0, 5.0);
        style.spacing.interact_size.y = 24.0;
    });
}

/// Bouton d'action coloré (Démarrer, Arrêter, Supprimer…). Plein avec texte
/// blanc sur fond sombre ; sur fond clair, un aplat plein criait au milieu
/// d'une interface sobre : fond légèrement teinté, texte et bord colorés.
pub fn action_button<'a>(text: egui::RichText, color: Color32) -> egui::Button<'a> {
    if p().light {
        egui::Button::new(text.color(color))
            .fill(mix(p().card, color, 0.10))
            .stroke(Stroke::new(1.0, mix(p().card, color, 0.55)))
    } else {
        egui::Button::new(text.color(Color32::WHITE)).fill(color)
    }
}

/// Mélange de `a` vers `b` (t = 0 : a, t = 1 : b).
fn mix(a: Color32, b: Color32, t: f32) -> Color32 {
    let l = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    rgb(l(a.r(), b.r()), l(a.g(), b.g()), l(a.b(), b.b()))
}

/// Pastille de couleur (état du serveur, profil modifié). Dessinée plutôt
/// qu'écrite : le caractère « ● » n'existe pas dans les polices d'egui.
pub fn dot(ui: &mut egui::Ui, color: Color32) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
    ui.painter().circle_filled(rect.center(), 4.0, color);
    response
}

/// Icône de la fenêtre, dessinée par `icon.rs` comme celle de l'exécutable.
pub fn icon() -> egui::IconData {
    const S: u32 = 256;
    egui::IconData {
        rgba: crate::icon::rgba(S),
        width: S,
        height: S,
    }
}

#[cfg(test)]
mod tests {
    /// Tout caractère non latin présent dans le code de l'interface doit
    /// exister dans les polices embarquées par egui, sinon il s'affiche en
    /// carré « □ ». Les commentaires sont ignorés (retirés à partir de `//`).
    #[test]
    fn every_ui_symbol_exists_in_egui_fonts() {
        let sources = [
            ("app.rs", include_str!("app.rs")),
            ("params.rs", include_str!("params.rs")),
            ("bench.rs", include_str!("bench.rs")),
            ("profiles.rs", include_str!("profiles.rs")),
            ("process.rs", include_str!("process.rs")),
            ("main.rs", include_str!("main.rs")),
        ];

        // Seule la chaîne de repli proportionnelle compte : elle sert au texte
        // d'interface et elle est incluse dans la chaîne monospace. La police
        // Hack couvre davantage de symboles, mais uniquement en monospace —
        // un « ● » y existe et s'affiche pourtant en carré dans un bouton.
        let defs = egui::FontDefinitions::default();
        let faces: Vec<ttf_parser::Face<'_>> = defs.families[&egui::FontFamily::Proportional]
            .iter()
            .map(|name| {
                let d = &defs.font_data[name];
                ttf_parser::Face::parse(&d.font, d.index).expect("police illisible")
            })
            .collect();
        let covered = |c: char| faces.iter().any(|f| f.glyph_index(c).is_some());

        let mut missing = Vec::new();
        for (file, text) in sources {
            // Les tests eux-mêmes citent des caractères absents : on s'arrête
            // au module de tests.
            let code = text.split("#[cfg(test)]").next().unwrap_or(text);
            for (n, line) in code.lines().enumerate() {
                let line = line.split("//").next().unwrap_or("");
                for c in line.chars().filter(|c| (*c as u32) > 0xFF) {
                    if !covered(c) {
                        missing.push(format!("{file}:{} U+{:04X} « {c} »", n + 1, c as u32));
                    }
                }
            }
        }
        assert!(
            missing.is_empty(),
            "caractères sans glyphe :\n{}",
            missing.join("\n")
        );
    }
}
