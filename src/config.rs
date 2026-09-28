//! Configuration persistante (`config.json`, à côté de l'exécutable).

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
pub enum Lang {
    Fr,
    En,
}

impl Lang {
    pub fn is_fr(self) -> bool {
        self == Lang::Fr
    }

    /// Langue de l'interface du système : français s'il est en français,
    /// anglais sinon. Sert de défaut quand `config.json` n'en fixe pas.
    pub fn system() -> Lang {
        if system_is_french() { Lang::Fr } else { Lang::En }
    }
}

#[cfg(windows)]
fn system_is_french() -> bool {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetUserDefaultUILanguage() -> u16;
    }
    // LANGID : les 10 bits de poids faible donnent la langue principale.
    const LANG_FRENCH: u16 = 0x0C;
    (unsafe { GetUserDefaultUILanguage() } & 0x3FF) == LANG_FRENCH
}

#[cfg(not(windows))]
fn system_is_french() -> bool {
    ["LC_ALL", "LC_MESSAGES", "LANG"]
        .iter()
        .filter_map(|v| std::env::var(v).ok())
        .find(|v| !v.is_empty())
        .is_some_and(|v| v.to_ascii_lowercase().starts_with("fr"))
}

/// Thème de l'interface. L'ordre des variantes est celui des palettes de
/// `theme.rs` et du menu de la barre du haut.
#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Debug, Default)]
#[serde(rename_all = "lowercase")]
pub enum ThemeChoice {
    #[default]
    Dark,
    Light,
    Nord,
    Dracula,
    Gruvbox,
    Solarized,
}

impl ThemeChoice {
    pub const ALL: [ThemeChoice; 6] = [
        ThemeChoice::Dark,
        ThemeChoice::Light,
        ThemeChoice::Nord,
        ThemeChoice::Dracula,
        ThemeChoice::Gruvbox,
        ThemeChoice::Solarized,
    ];

    pub fn label(self, fr: bool) -> &'static str {
        match (self, fr) {
            (ThemeChoice::Dark, true) => "Sombre",
            (ThemeChoice::Dark, false) => "Dark",
            (ThemeChoice::Light, true) => "Clair",
            (ThemeChoice::Light, false) => "Light",
            (ThemeChoice::Nord, _) => "Nord",
            (ThemeChoice::Dracula, _) => "Dracula",
            (ThemeChoice::Gruvbox, _) => "Gruvbox",
            (ThemeChoice::Solarized, true) => "Solarized clair",
            (ThemeChoice::Solarized, false) => "Solarized light",
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct AppConfig {
    pub llama_cpp_directory: String,
    pub models_directory: String,
    /// Dépôt git de llama.cpp, pour les mises à jour depuis les sources.
    pub llama_source_directory: String,
    /// Dossier où chaque mise à jour publie sa release.
    pub llama_releases_directory: String,
    pub last_selected_version: String,
    pub last_selected_model: String,
    pub host: String,
    pub port: String,
    pub language: Lang,
    pub theme: ThemeChoice,
    /// Dernières valeurs choisies pour chaque paramètre.
    pub parameters: BTreeMap<String, String>,
    /// `parameters` a déjà été enregistré : vide, il signifie alors « tout
    /// désactivé », et non « jamais configuré » (valeurs par défaut).
    pub parameters_saved: bool,
    /// Valeurs personnalisées ajoutées aux listes via l'import de commande.
    pub extra_options: BTreeMap<String, Vec<String>>,
    /// Options importées absentes du catalogue, transmises telles quelles
    /// (une entrée = l'option et sa valeur éventuelle).
    pub extra_args: Vec<Vec<String>>,
    /// Réglages de llama-bench.
    pub bench: crate::bench::BenchSettings,
    /// Modèles (chemins complets) et versions (noms) cochés pour le
    /// benchmark.
    pub bench_models: Vec<String>,
    pub bench_versions: Vec<String>,
    /// Prix du kWh et devise, pour chiffrer la consommation du GPU.
    pub kwh_price: f64,
    pub currency: String,
    /// Énergie consommée par le GPU, cumulée d'une session à l'autre (Wh),
    /// et date du début du cumul.
    pub energy_total_wh: f64,
    pub energy_since: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            llama_cpp_directory: String::new(),
            models_directory: String::new(),
            llama_source_directory: String::new(),
            llama_releases_directory: String::new(),
            last_selected_version: String::new(),
            last_selected_model: String::new(),
            host: "127.0.0.1".into(),
            port: "8080".into(),
            language: Lang::system(),
            theme: ThemeChoice::Dark,
            parameters: BTreeMap::new(),
            parameters_saved: false,
            extra_options: BTreeMap::new(),
            extra_args: Vec::new(),
            bench: Default::default(),
            bench_models: Vec::new(),
            bench_versions: Vec::new(),
            kwh_price: 0.25,
            currency: "€".into(),
            energy_total_wh: 0.0,
            energy_since: String::new(),
        }
    }
}

/// Répertoire de travail de l'application : celui de l'exécutable, avec
/// repli sur le répertoire courant si l'emplacement n'est pas inscriptible.
pub fn app_dir() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            if is_writable(dir) {
                return dir.to_path_buf();
            }
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn is_writable(dir: &Path) -> bool {
    let probe = dir.join(".uillamacpp-write-test");
    match std::fs::write(&probe, b"") {
        Ok(()) => {
            let _ = std::fs::remove_file(&probe);
            true
        }
        Err(_) => false,
    }
}

pub fn config_path() -> PathBuf {
    app_dir().join("config.json")
}

/// Charge `config.json`. Un fichier illisible (corrompu, ou écrit par une
/// version incompatible) est copié en `config.json.bak` avant que la
/// sauvegarde automatique ne le remplace par les valeurs par défaut : il
/// n'est jamais perdu. Le message d'erreur est rendu pour la console.
pub fn load() -> (AppConfig, Option<String>) {
    let path = config_path();
    let Ok(text) = std::fs::read_to_string(&path) else {
        return (AppConfig::default(), None);
    };
    match serde_json::from_str(&text) {
        Ok(cfg) => (cfg, None),
        Err(e) => {
            let backup = path.with_extension("json.bak");
            let saved = std::fs::write(&backup, &text).is_ok();
            let msg = if saved {
                format!("config.json illisible ({e}) : copié dans {}, réglages par défaut", backup.display())
            } else {
                format!("config.json illisible ({e}) : réglages par défaut")
            };
            (AppConfig::default(), Some(msg))
        }
    }
}

pub fn save(cfg: &AppConfig) -> anyhow::Result<()> {
    let json = serde_json::to_string_pretty(cfg)?;
    std::fs::write(config_path(), json)?;
    Ok(())
}
