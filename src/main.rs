// Pas de console derrière la fenêtre en release.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod bench;
mod cmdparse;
mod config;
mod discovery;
mod fattn;
mod gguf;
mod gpu;
mod hub;
mod icon;
mod params;
mod power;
mod process;
mod profiles;
mod stats;
mod theme;
mod updater;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("llama.cpp launcher")
            .with_inner_size([1440.0, 880.0])
            .with_min_inner_size([1020.0, 620.0])
            .with_icon(theme::icon()),
        wgpu_options: eframe::WgpuConfiguration {
            wgpu_setup: eframe::egui_wgpu::WgpuSetup::CreateNew(render_setup()),
            ..Default::default()
        },
        ..Default::default()
    };

    eframe::run_native(
        "llama.cpp launcher",
        options,
        Box::new(|cc| Ok(Box::new(app::App::new(cc)))),
    )
}

/// Rendu sur le GPU économe (la puce intégrée, s'il y en a une).
///
/// Par défaut wgpu choisit le GPU le plus puissant : l'interface dessinait
/// alors sur la carte dédiée et lui prenait ~260 Mo de VRAM, retirés aux
/// modèles. Sur la puce intégrée il n'en reste que ~50 Mo, pour l'affichage.
/// Sans puce intégrée, `LowPower` retombe sur le seul GPU présent.
/// `WGPU_POWER_PREF=high` rétablit l'ancien comportement.
///
/// Cela n'empêche PAS la notification « Alt+Z » de l'overlay NVIDIA : NVIDIA
/// détecte les applications graphiques quel que soit le GPU de rendu, et
/// affirme qu'aucun moyen programmatique ne l'évite.
fn render_setup() -> eframe::egui_wgpu::WgpuSetupCreateNew {
    use eframe::wgpu::PowerPreference;
    let mut setup = eframe::egui_wgpu::WgpuSetupCreateNew::without_display_handle();
    if PowerPreference::from_env().is_none() {
        setup.power_preference = PowerPreference::LowPower;
    }
    setup
}
