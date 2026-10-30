// Pas de console derrière la fenêtre en release.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod bench;
mod cmdparse;
mod config;
#[cfg(not(windows))]
mod desktop;
mod discovery;
mod fattn;
mod gguf;
mod gpu;
mod hub;
mod icon;
mod params;
#[cfg(windows)]
mod pdh;
mod power;
mod process;
mod profiles;
mod stats;
mod sysmon;
mod theme;
mod updater;

/// Nom de l'application : identité de la fenêtre (`app_id`, groupe de la
/// barre des tâches), sans le numéro de version.
const APP_NAME: &str = "llama.cpp launcher";

/// Titre de la fenêtre : le nom suivi de la version du binaire, pour voir
/// d'un coup d'œil quelle version tourne.
fn title() -> String {
    format!("{APP_NAME} {}", env!("CARGO_PKG_VERSION"))
}

fn main() -> eframe::Result<()> {
    let viewport = egui::ViewportBuilder::default()
        .with_title(title())
        .with_inner_size([1440.0, 880.0])
        .with_min_inner_size([1020.0, 620.0])
        .with_icon(theme::icon());
    // Linux : l'icône vient du fichier .desktop qui porte l'app_id de la
    // fenêtre (Wayland ignore l'icône de la fenêtre).
    #[cfg(not(windows))]
    let viewport = {
        std::thread::spawn(desktop::install);
        viewport.with_app_id(desktop::APP_ID)
    };
    let options = eframe::NativeOptions {
        viewport,
        wgpu_options: eframe::WgpuConfiguration {
            wgpu_setup: eframe::egui_wgpu::WgpuSetup::CreateNew(render_setup()),
            ..Default::default()
        },
        ..Default::default()
    };

    eframe::run_native(
        APP_NAME,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Le titre affiché porte la version du binaire, pour savoir d'un coup
    /// d'œil quelle version tourne.
    #[test]
    fn the_title_carries_the_version() {
        let title = title();
        assert!(
            title.ends_with(env!("CARGO_PKG_VERSION")),
            "titre sans version : {title}"
        );
        assert!(title.starts_with(APP_NAME), "titre sans nom : {title}");
        // L'identité de la fenêtre, elle, ne bouge pas avec la version :
        // l'entrée `.desktop` de Linux et le groupe de la barre des tâches
        // s'appuient dessus.
        assert_eq!(APP_NAME, "llama.cpp launcher");
    }
}
