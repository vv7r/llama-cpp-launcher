//! Interface egui : configuration, paramètres, console, benchmark, modèles.

use crate::bench::{self, BenchEvent, BenchHandle, BenchRow};
use crate::cmdparse;
use crate::config::{self, AppConfig, Lang, ThemeChoice};
use crate::discovery::{self, ModelInfo, VersionInfo};
use crate::fattn::FaSupport;
use crate::gguf;
use crate::gpu::{self, GpuMemory};
use crate::hub;
use crate::params::{self, Group, Kind, ParamState};
use crate::process::{self, ModelSource, ProcEvent, ServerProcess, Speeds};
use crate::profiles::{self, LaunchProfile, ProfileEntry};
use crate::power::{self, PowerLog};
use crate::stats::{Spread, TokenStats};
use crate::theme;
use crate::updater;

use egui::{Color32, ComboBox, RichText, Ui};
use std::collections::HashMap;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver};
use std::time::{Duration, Instant};

const MAX_CONSOLE_LINES: usize = 4000;

/// Fichiers d'un dépôt Hugging Face, lus en tâche de fond.
type HubFilesResult = (String, Result<Vec<hub::HubFile>, String>);
/// Numéro de build et commit d'une version (`llama-server --version`).
type ProbeResult = (PathBuf, Option<(u32, String)>);

/// Sous-onglets de Statistiques.
#[derive(PartialEq, Clone, Copy)]
enum StatsView {
    Throughput,
    Power,
}

/// Devises proposées pour le prix du kWh.
const CURRENCIES: &[&str] = &["€", "$", "£", "CHF"];

#[derive(PartialEq, Clone, Copy)]
enum Tab {
    Console,
    Stats,
    Benchmark,
    Models,
}

enum Action {
    Start,
    Stop,
    Restart,
    BrowseLlamaDir,
    BrowseModelsDir,
    Refresh,
    ResetDefaults,
    DisableAll,
    OpenImport,
    ApplyImport,
    SaveProfile,
    LoadProfile(String),
    DeleteProfile(String),
    RenameProfile { stem: String, name: String },
    UpdateProfile(String),
    RemoveExtraArg(usize),
    OpenUpdate,
    BrowseSource,
    BrowseReleases,
    CheckUpdates,
    StartUpdate,
    CancelUpdate,
    BrowseInstallDir,
    StartInstall,
    RecheckPrereqs,
    CopyText(String),
    AskDeleteVersion(PathBuf),
    SetTheme(ThemeChoice),
    HubLoad,
    HubSort(hub::Sort),
    HubPickRepo(String),
    DeleteVersion(PathBuf),
    CopyConsole,
    ClearConsole,
    CopyCommand,
    CopyCommandOneLine,
    /// Mesure la sélection ; `true` : seulement ce qui manque.
    BenchRun(bool),
    SetBenchSettings(bench::BenchSettings),
    SetBenchSelection {
        models: Vec<String>,
        versions: Vec<String>,
    },
    BenchStop,
    /// Supprime une mesure (clé de `BenchRow::key`), ou toutes avec `None`.
    BenchDelete(Option<(String, String, String, String)>),
    BenchConfirmClear(bool),
    ResetStats,
    SetEnergyPrice(f64, String),
    ResetPowerCurve,
    ResetEnergyTotal,
    SaveBench,
    DeleteModel(PathBuf),
    RevealModel(PathBuf),
    HubDownload,
    HubCancelDownload,
}

pub struct App {
    /// Pour réveiller l'interface depuis les fils de lecture.
    ctx: egui::Context,
    cfg: AppConfig,
    states: Vec<ParamState>,
    versions: Vec<VersionInfo>,
    models: Vec<ModelInfo>,
    selected_version: String,
    selected_model: String,

    console: VecDeque<String>,
    autoscroll: bool,
    server: Option<ServerProcess>,
    speeds: Speeds,
    /// Débits par requête depuis le dernier démarrage du serveur.
    stats: TokenStats,
    stats_view: StatsView,
    /// Consommation du GPU : courbe et énergie.
    power: PowerLog,
    restart_pending: Option<Instant>,

    tab: Tab,
    profiles: Vec<ProfileEntry>,
    profile_name: String,
    /// Profil chargé ou enregistré en dernier (nom de fichier).
    active_profile: String,
    /// Carte en cours d'édition.
    profile_edit: Option<ProfileEdit>,
    /// Carte dont la suppression attend confirmation.
    profile_delete: Option<String>,

    import_open: bool,
    import_text: String,

    bench_handle: Option<BenchHandle>,
    bench_rows: Vec<BenchRow>,
    bench_progress: (usize, usize),
    bench_label: String,
    /// Résultats : seulement ceux mesurés avec les réglages actuels.
    bench_current_only: bool,
    /// « Tout supprimer » attend sa confirmation.
    bench_confirm_clear: bool,

    /// En-têtes GGUF, mémorisés dans `gguf-cache.json` et lus en tâche de
    /// fond pour chaque nouveau modèle (téléchargé ou copié).
    meta: HashMap<PathBuf, gguf::Cached>,
    meta_rx: Option<Receiver<(PathBuf, gguf::Cached)>>,
    /// Une lecture est demandée pendant qu'une autre tourne.
    meta_rescan: bool,
    /// En-têtes lus / à lire par la lecture en cours.
    meta_progress: (usize, usize),
    /// Dernier enregistrement du cache pendant une lecture.
    meta_saved: Instant,
    /// Paires K/V FlashAttention compilées par chaque release CUDA (lues
    /// dans sa ggml-cuda.dll, en tâche de fond) ; `None` : inconnu.
    fa_support: HashMap<PathBuf, Option<FaSupport>>,
    fa_rx: Option<Receiver<(PathBuf, Option<FaSupport>)>>,
    /// Hugging Face : dépôts et fichiers lus sur l'API, chargés à la demande.
    hub_sort: hub::Sort,
    hub_repos: Option<Result<Vec<hub::Repo>, String>>,
    hub_repos_rx: Option<Receiver<Result<Vec<hub::Repo>, String>>>,
    hub_repo: String,
    hub_files: Option<Result<Vec<hub::HubFile>, String>>,
    hub_files_rx: Option<Receiver<HubFilesResult>>,
    hub_file: String,
    hub_download: Option<hub::Download>,
    hub_progress: (u64, u64),
    /// Mémoire vidéo, rafraîchie toutes les deux secondes.
    gpus: Vec<GpuMemory>,
    gpu_rx: Receiver<gpu::Sample>,
    confirm_delete: Option<PathBuf>,
    toast: Option<(String, Color32, Instant)>,
    last_save: Instant,
    saved_json: String,
    /// Dernière vérification des changements à enregistrer.
    last_save_check: Instant,
    /// Énergie cumulée, à jour à chaque relevé. Elle n'est recopiée dans
    /// `cfg` qu'avec une autre modification, toutes les cinq minutes ou à la
    /// fermeture : sinon `config.json` était réécrit toutes les deux secondes.
    energy_total_wh: f64,
    energy_saved: Instant,

    // ---- Mise à jour de llama.cpp ----
    update_open: bool,
    /// État git du dépôt source et options de compilation, relus à
    /// l'ouverture de la fenêtre (pas à chaque image : git est lent).
    update_git: Option<Result<updater::GitState, String>>,
    update_config: Option<(String, updater::BuildConfig)>,
    update_check_task: Option<updater::Task>,
    update_check: Option<Result<updater::Check, String>>,
    update_task: Option<updater::Task>,
    update_stage: Option<(usize, usize, &'static str)>,
    update_progress: Option<f32>,
    update_started: Option<Instant>,
    update_last_line: String,
    update_result: Option<Result<updater::Release, String>>,
    /// Fichiers mmproj trouvés dans le répertoire des modèles.
    mmproj_count: usize,
    /// Supprimer la version sélectionnée une fois la mise à jour vérifiée.
    /// Toujours décoché à l'ouverture : par défaut, l'ancienne version reste.
    update_delete_previous: bool,
    /// FlashAttention CUDA pour toutes les paires de types K/V.
    update_fa_all: bool,
    /// Version depuis laquelle le serveur en cours a été lancé.
    server_dir: Option<PathBuf>,
    /// Version dont la suppression attend confirmation.
    confirm_version_delete: Option<PathBuf>,
    /// Installation depuis zéro : outils détectés, dossier et backend choisis.
    prereqs: Option<updater::Prereqs>,
    prereq_rx: Option<Receiver<updater::Prereqs>>,
    install_dir: String,
    install_backend: updater::Backend,
    /// L'utilisateur a choisi le backend : ne plus appliquer la suggestion.
    install_backend_chosen: bool,
    /// Résultats de `llama-server --version`, par dossier (None = échec).
    probed: HashMap<PathBuf, Option<(u32, String)>>,
    probe_rx: Option<Receiver<ProbeResult>>,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let (cfg, cfg_error) = config::load();
        theme::apply(&cc.egui_ctx, cfg.theme);

        let mut states = params::default_states();
        // Restaure les listes enrichies puis les valeurs choisies.
        for (name, extra) in &cfg.extra_options {
            if let Some(i) = params::index_of(name) {
                for value in extra {
                    if !states[i].options.iter().any(|o| o == value) {
                        states[i].options.push(value.clone());
                    }
                }
            }
        }
        if cfg.parameters_saved || !cfg.parameters.is_empty() {
            for st in states.iter_mut() {
                st.value.clear();
            }
            for (name, value) in &cfg.parameters {
                if let Some(i) = params::index_of(name) {
                    states[i].adopt(value);
                }
            }
        }

        let mut app = Self {
            ctx: cc.egui_ctx.clone(),
            selected_version: cfg.last_selected_version.clone(),
            selected_model: cfg.last_selected_model.clone(),
            states,
            versions: Vec::new(),
            models: Vec::new(),
            console: VecDeque::new(),
            autoscroll: true,
            server: None,
            speeds: Speeds::default(),
            stats: TokenStats::default(),
            stats_view: StatsView::Throughput,
            power: PowerLog::new(),
            restart_pending: None,
            tab: Tab::Console,
            profiles: profiles::entries(),
            profile_name: String::new(),
            active_profile: String::new(),
            profile_edit: None,
            profile_delete: None,
            import_open: false,
            import_text: String::new(),
            bench_handle: None,
            bench_rows: bench::load_results(),
            bench_progress: (0, 0),
            bench_label: String::new(),
            bench_current_only: false,
            bench_confirm_clear: false,
            meta: gguf::load_cache(),
            meta_rx: None,
            meta_rescan: false,
            meta_progress: (0, 0),
            meta_saved: Instant::now(),
            fa_support: HashMap::new(),
            fa_rx: None,
            hub_sort: hub::Sort::Trending,
            hub_repos: None,
            hub_repos_rx: None,
            hub_repo: String::new(),
            hub_files: None,
            hub_files_rx: None,
            hub_file: String::new(),
            hub_download: None,
            hub_progress: (0, 0),
            gpus: Vec::new(),
            gpu_rx: gpu::monitor(),
            confirm_delete: None,
            toast: None,
            last_save: Instant::now(),
            last_save_check: Instant::now(),
            energy_total_wh: cfg.energy_total_wh,
            energy_saved: Instant::now(),
            saved_json: serde_json::to_string_pretty(&cfg).unwrap_or_default(),
            update_open: false,
            update_git: None,
            update_config: None,
            update_check_task: None,
            update_check: None,
            update_task: None,
            update_stage: None,
            update_progress: None,
            update_started: None,
            update_last_line: String::new(),
            update_result: None,
            probed: HashMap::new(),
            probe_rx: None,
            mmproj_count: 0,
            update_delete_previous: false,
            update_fa_all: false,
            server_dir: None,
            confirm_version_delete: None,
            prereqs: None,
            prereq_rx: None,
            install_dir: std::env::var("USERPROFILE")
                .map(|home| PathBuf::from(home).join("llama").to_string_lossy().into_owned())
                .unwrap_or_default(),
            install_backend: updater::Backend::Cpu,
            install_backend_chosen: false,
            cfg,
        };
        app.refresh();
        // Premier lancement, rien de configuré : proposer d'installer.
        if app.versions.is_empty()
            && app.cfg.llama_cpp_directory.is_empty()
            && app.cfg.llama_releases_directory.is_empty()
        {
            app.update_open = true;
            app.inspect_source();
            app.start_prereqs();
        }
        app.log(format!(
            "llama.cpp launcher {} — {}",
            env!("CARGO_PKG_VERSION"),
            app.t("prêt.", "ready.")
        ));
        if let Some(e) = cfg_error {
            app.log(format!("! {e}"));
        }
        app
    }

    fn fr(&self) -> bool {
        self.cfg.language.is_fr()
    }

    fn t(&self, fr: &'static str, en: &'static str) -> &'static str {
        if self.fr() {
            fr
        } else {
            en
        }
    }

    fn log(&mut self, line: impl Into<String>) {
        if self.console.len() >= MAX_CONSOLE_LINES {
            self.console.pop_front();
        }
        self.console.push_back(line.into());
    }

    fn notify(&mut self, text: impl Into<String>, color: Color32) {
        self.toast = Some((text.into(), color, Instant::now()));
    }

    fn refresh(&mut self) {
        // Versions du répertoire llama.cpp, puis releases publiées par les
        // mises à jour ; la plus récente en tête.
        let mut versions = discovery::find_versions(&self.cfg.llama_cpp_directory);
        for v in discovery::find_versions(&self.cfg.llama_releases_directory) {
            if !versions.iter().any(|x| x.name == v.name || x.dir == v.dir) {
                versions.push(v);
            }
        }
        for v in versions.iter_mut() {
            if let Some(Some((build, commit))) = self.probed.get(&v.dir) {
                v.build = Some(*build);
                v.commit = Some(commit.clone());
            }
        }
        versions.sort_by_key(|v| std::cmp::Reverse(v.build));
        self.versions = versions;
        self.start_probe();
        let (models, mmproj) = discovery::scan_models(&self.cfg.models_directory);
        self.models = models;
        self.mmproj_count = mmproj.len();
        if let Some(i) = params::index_of("--mmproj") {
            for path in mmproj {
                self.states[i].offer(&path.to_string_lossy());
            }
        }
        if !self.versions.iter().any(|v| v.name == self.selected_version) {
            self.selected_version = self.versions.first().map(|v| v.name.clone()).unwrap_or_default();
        }
        if !self
            .models
            .iter()
            .any(|m| m.full_path.to_string_lossy() == self.selected_model)
        {
            self.selected_model = self
                .models
                .first()
                .map(|m| m.full_path.to_string_lossy().into_owned())
                .unwrap_or_default();
        }
        self.profiles = profiles::entries();
        self.scan_metadata();
        self.scan_fa_support();
    }

    /// Lit en tâche de fond les paires FlashAttention des releases pas
    /// encore examinées.
    fn scan_fa_support(&mut self) {
        let todo: Vec<PathBuf> = self
            .versions
            .iter()
            .map(|v| v.dir.clone())
            .filter(|d| !self.fa_support.contains_key(d))
            .collect();
        if todo.is_empty() {
            return;
        }
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            for dir in todo {
                let found = crate::fattn::read(&dir);
                if tx.send((dir, found)).is_err() {
                    break;
                }
            }
        });
        self.fa_rx = Some(rx);
    }

    /// Paires FlashAttention de la release sélectionnée, si elle les indique.
    fn current_fa(&self) -> Option<FaSupport> {
        self.version()
            .and_then(|v| self.fa_support.get(&v.dir))
            .cloned()
            .flatten()
    }

    /// Interroge `llama-server --version` des versions pas encore connues,
    /// en tâche de fond (≈ 120 ms par exe, à ne pas faire pendant le rendu).
    fn start_probe(&mut self) {
        if self.probe_rx.is_some() {
            return;
        }
        let todo: Vec<(PathBuf, PathBuf)> = self
            .versions
            .iter()
            .filter(|v| !self.probed.contains_key(&v.dir))
            .map(|v| (v.dir.clone(), v.server()))
            .collect();
        if todo.is_empty() {
            return;
        }
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            for (dir, exe) in todo {
                if tx.send((dir, discovery::probe_version(&exe))).is_err() {
                    break;
                }
            }
        });
        self.probe_rx = Some(rx);
    }

    fn pump_probe(&mut self, ctx: &egui::Context) {
        let Some(rx) = &self.probe_rx else {
            return;
        };
        let mut changed = false;
        loop {
            match rx.try_recv() {
                Ok((dir, result)) => {
                    if let Some((build, commit)) = &result {
                        if let Some(v) = self.versions.iter_mut().find(|v| v.dir == dir) {
                            v.build = Some(*build);
                            v.commit = Some(commit.clone());
                        }
                    }
                    self.probed.insert(dir, result);
                    changed = true;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    ctx.request_repaint_after(Duration::from_millis(100));
                    break;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.probe_rx = None;
                    // Des versions ont pu apparaître pendant l'interrogation.
                    self.start_probe();
                    break;
                }
            }
        }
        if changed {
            self.versions.sort_by_key(|v| std::cmp::Reverse(v.build));
        }
    }

    /// Le serveur en cours a-t-il été lancé depuis cette version ?
    fn version_in_use(&self, dir: &Path) -> bool {
        self.server.is_some() && self.server_dir.as_deref() == Some(dir)
    }

    /// La version sélectionnée, si elle peut être supprimée : une release du
    /// dossier des releases, pas un build de l'utilisateur ailleurs.
    fn deletable_version(&self) -> Option<&VersionInfo> {
        let releases = Path::new(&self.cfg.llama_releases_directory);
        self.version()
            .filter(|v| updater::is_release(&v.dir, releases))
    }

    fn version(&self) -> Option<&VersionInfo> {
        self.versions.iter().find(|v| v.name == self.selected_version)
    }

    fn model(&self) -> Option<&ModelInfo> {
        self.models
            .iter()
            .find(|m| m.full_path.to_string_lossy() == self.selected_model)
    }

    fn running(&self) -> bool {
        self.server.is_some()
    }

    /// Écrit `config.json` si l'état a changé, au plus une fois toutes les
    /// deux secondes : la configuration survit à un arrêt brutal.
    ///
    /// egui n'appelle `ui()` que sur événement ; tant qu'il reste quelque
    /// chose à écrire on programme donc un réveil, sans quoi une fenêtre
    /// laissée au repos ne sauvegarderait jamais.
    ///
    /// La comparaison (toute la configuration sérialisée) n'est faite
    /// qu'une fois par demi-seconde, pas à chaque image.
    fn autosave(&mut self, ctx: &egui::Context) {
        let check = Duration::from_millis(500);
        let since_check = self.last_save_check.elapsed();
        if since_check < check {
            ctx.request_repaint_after(check - since_check);
            return;
        }
        self.last_save_check = Instant::now();
        self.sync_cfg();
        let energy_due = self.energy_total_wh != self.cfg.energy_total_wh
            && self.energy_saved.elapsed() >= Duration::from_secs(300);
        let json = serde_json::to_string_pretty(&self.cfg).unwrap_or_default();
        if json == self.saved_json && !energy_due {
            return;
        }
        let elapsed = self.last_save.elapsed();
        let period = Duration::from_secs(2);
        if elapsed < period {
            ctx.request_repaint_after(period - elapsed);
            return;
        }
        self.last_save = Instant::now();
        self.cfg.energy_total_wh = self.energy_total_wh;
        self.energy_saved = Instant::now();
        let json = serde_json::to_string_pretty(&self.cfg).unwrap_or_default();
        self.write_config(json);
    }

    fn persist(&mut self) {
        self.sync_cfg();
        self.cfg.energy_total_wh = self.energy_total_wh;
        let json = serde_json::to_string_pretty(&self.cfg).unwrap_or_default();
        if json != self.saved_json {
            self.write_config(json);
        }
    }

    fn write_config(&mut self, json: String) {
        match config::save(&self.cfg) {
            Ok(()) => self.saved_json = json,
            Err(e) => self.log(format!("! config.json : {e}")),
        }
    }

    /// Reporte l'état de l'interface dans la configuration sérialisable.
    fn sync_cfg(&mut self) {
        self.cfg.last_selected_version = self.selected_version.clone();
        self.cfg.last_selected_model = self.selected_model.clone();
        self.cfg.parameters = params::to_map(&self.states).into_iter().collect();
        self.cfg.parameters_saved = true;
        self.cfg.extra_options.clear();
        for (i, st) in self.states.iter().enumerate() {
            let def = params::def(i);
            let extra: Vec<String> = st
                .options
                .iter()
                .filter(|o| !def.presets.contains(&o.as_str()))
                // Les mmproj découverts sont reproposés à chaque scan : on ne
                // retient que celui choisi, sinon un fichier supprimé
                // resterait dans la liste pour toujours.
                .filter(|o| def.name != "--mmproj" || **o == st.value)
                // Un secret se saisit : inutile d'en garder un historique.
                .filter(|_| !def.secret)
                .cloned()
                .collect();
            if !extra.is_empty() {
                self.cfg.extra_options.insert(def.name.to_string(), extra);
            }
        }
    }

    // ---------------------------------------------------------------- actions

    fn start_server(&mut self) {
        if self.running() {
            return;
        }
        let Some(version) = self.version().cloned() else {
            self.notify(
                self.t("Aucune version sélectionnée.", "No version selected."),
                theme::red(),
            );
            return;
        };
        let Some(model) = self.model().cloned() else {
            self.notify(
                self.t("Aucun modèle sélectionné.", "No model selected."),
                theme::red(),
            );
            return;
        };

        let args = params::build_args(&self.states, &self.cfg.extra_args);
        let host = self.cfg.host.clone();
        let port = self.cfg.port.clone();
        self.speeds = Speeds::default();
        self.stats.clear();
        self.power.reset_server();

        match process::start(
            &version.server(),
            &ModelSource::File(model.full_path.clone()),
            &host,
            &port,
            &args,
            {
                let ctx = self.ctx.clone();
                std::sync::Arc::new(move || ctx.request_repaint())
            },
        ) {
            Ok(proc) => {
                let line = self.mask_secrets(&proc.command_line);
                self.log(format!("$ {line}"));
                self.log(format!(
                    "▶ {} (PID {}) — http://{host}:{port}",
                    version.name,
                    proc.pid()
                ));
                self.server_dir = Some(version.dir.clone());
                self.server = Some(proc);
                self.tab = Tab::Console;
                self.notify(self.t("Serveur démarré.", "Server started."), theme::green());
            }
            Err(e) => {
                self.log(format!("! {e}"));
                self.notify(format!("{e}"), theme::red());
            }
        }
    }

    fn stop_server(&mut self) {
        if let Some(mut proc) = self.server.take() {
            proc.stop();
            self.log(self.t("■ Serveur arrêté.", "■ Server stopped.").to_string());
        }
        self.speeds = Speeds::default();
    }

    fn apply_import(&mut self) {
        let parsed = cmdparse::parse(&self.import_text);

        for st in self.states.iter_mut() {
            st.value.clear();
        }
        let mut applied = 0;
        for (name, value) in &parsed.params {
            if let Some(i) = params::index_of(name) {
                self.states[i].adopt(value);
                applied += 1;
            }
        }
        if let Some(host) = parsed.host {
            if !params::HOSTS.contains(&host.as_str()) {
                self.log(format!("  host importé : {host}"));
            }
            self.cfg.host = host;
        }
        if let Some(port) = parsed.port {
            self.cfg.port = port;
        }
        if let Some(model) = &parsed.model {
            let path = PathBuf::from(model);
            if let Some(found) = self.models.iter().find(|m| {
                m.full_path == path
                    || m.full_path.file_name() == path.file_name()
            }) {
                self.selected_model = found.full_path.to_string_lossy().into_owned();
            } else {
                self.log(format!(
                    "  modèle « {model} » absent du répertoire configuré, sélection inchangée"
                ));
            }
        }
        // Hors catalogue : conservées et transmises telles quelles, pour que
        // la commande lancée soit bien celle qui a été collée.
        for tokens in &parsed.unknown {
            self.log(format!("  option hors catalogue conservée : {}", tokens.join(" ")));
        }
        let kept = parsed.unknown.len();
        self.cfg.extra_args = parsed.unknown;

        self.import_open = false;
        self.import_text.clear();
        let msg = match (self.fr(), kept) {
            (true, 0) => format!("{applied} paramètre(s) importé(s)."),
            (true, k) => format!(
                "{applied} paramètre(s) importé(s), {k} option(s) hors catalogue conservée(s)."
            ),
            (false, 0) => format!("{applied} parameter(s) imported."),
            (false, k) => {
                format!("{applied} parameter(s) imported, {k} uncatalogued option(s) kept.")
            }
        };
        self.log(msg.clone());
        self.notify(msg, theme::accent());
    }

    /// Profil décrivant l'état courant : paramètres, hôte, port, modèle et
    /// version sélectionnés.
    fn current_profile(&self, name: String) -> LaunchProfile {
        LaunchProfile {
            name,
            parameters: params::to_map(&self.states),
            host: self.cfg.host.clone(),
            port: self.cfg.port.clone(),
            model: self.selected_model.clone(),
            version: self.selected_version.clone(),
            updated: String::new(),
            extra_args: self.cfg.extra_args.clone(),
        }
    }

    /// Crée un nouveau profil à partir du champ « nom ». Refuse d'écraser un
    /// profil existant : la mise à jour passe par le bouton de la carte.
    fn save_profile(&mut self) {
        let name = self.profile_name.trim().to_string();
        if name.is_empty() {
            self.notify(
                self.t("Donnez un nom au profil.", "Give the profile a name."),
                theme::orange(),
            );
            return;
        }
        if profiles::exists(&name) {
            self.notify(
                self.t(
                    "Ce nom existe déjà — utilisez ✏ sur sa carte pour le mettre à jour.",
                    "That name exists — use ✏ on its card to update it.",
                ),
                theme::orange(),
            );
            return;
        }
        match profiles::save(&self.current_profile(name.clone())) {
            Ok(()) => {
                self.profiles = profiles::entries();
                self.active_profile = profiles::sanitize(&name);
                self.profile_name.clear();
                self.notify(
                    format!("{} « {name} »", self.t("Profil créé", "Profile created")),
                    theme::green(),
                );
            }
            Err(e) => self.notify(format!("{e}"), theme::red()),
        }
    }

    /// Écrase les réglages d'un profil existant avec l'état courant, en
    /// conservant son nom.
    fn update_profile(&mut self, stem: String) {
        let name = match profiles::load(&stem) {
            Ok(p) if !p.name.trim().is_empty() => p.name,
            _ => stem.clone(),
        };
        match profiles::save(&self.current_profile(name.clone())) {
            Ok(()) => {
                self.profiles = profiles::entries();
                self.active_profile = stem;
                self.profile_edit = None;
                self.notify(
                    format!("{} « {name} »", self.t("Profil mis à jour", "Profile updated")),
                    theme::green(),
                );
            }
            Err(e) => self.notify(format!("{e}"), theme::red()),
        }
    }

    fn rename_profile(&mut self, stem: String, name: String) {
        match profiles::rename(&stem, &name) {
            Ok(new_stem) => {
                if self.active_profile == stem {
                    self.active_profile = new_stem;
                }
                self.profile_edit = None;
                self.profiles = profiles::entries();
            }
            Err(e) => self.notify(format!("{e}"), theme::red()),
        }
    }

    fn load_profile(&mut self, stem: String) {
        match profiles::load(&stem) {
            Ok(profile) => {
                // Les valeurs ajoutées aux listes par un import restent
                // disponibles après le chargement.
                let options: Vec<Vec<String>> =
                    self.states.iter().map(|s| s.options.clone()).collect();
                self.states = params::from_map(&profile.parameters);
                for (st, opts) in self.states.iter_mut().zip(options) {
                    for o in opts {
                        if !st.options.contains(&o) {
                            st.options.push(o);
                        }
                    }
                }
                self.cfg.extra_args = profile.extra_args.clone();
                if !profile.host.is_empty() {
                    self.cfg.host = profile.host.clone();
                }
                if !profile.port.is_empty() {
                    self.cfg.port = profile.port.clone();
                }
                // Modèle et version ne sont restaurés que s'ils existent
                // encore sur le disque.
                if !profile.model.is_empty() {
                    if self
                        .models
                        .iter()
                        .any(|m| m.full_path.to_string_lossy() == profile.model)
                    {
                        self.selected_model = profile.model.clone();
                    } else {
                        self.log(format!("  modèle du profil introuvable : {}", profile.model));
                    }
                }
                if !profile.version.is_empty() {
                    if self.versions.iter().any(|v| v.name == profile.version) {
                        self.selected_version = profile.version.clone();
                    } else {
                        self.log(format!("  version du profil introuvable : {}", profile.version));
                    }
                }
                self.active_profile = stem.clone();
                let name = if profile.name.is_empty() { stem } else { profile.name };
                self.notify(
                    format!("{} « {name} »", self.t("Profil chargé", "Profile loaded")),
                    theme::accent(),
                );
            }
            Err(e) => self.notify(format!("{e}"), theme::red()),
        }
    }

    /// Combinaisons version × modèle cochées, en indices dans `versions` et
    /// `models` ; `only_missing` écarte celles déjà mesurées avec les
    /// réglages actuels. Appelé à chaque image pour les compteurs : sans
    /// copie, et avec des ensembles plutôt que des recherches linéaires.
    fn bench_pairs(&self, only_missing: bool) -> Vec<(usize, usize)> {
        use std::collections::HashSet;
        let versions: HashSet<&str> = self.cfg.bench_versions.iter().map(String::as_str).collect();
        let models: HashSet<&str> = self.cfg.bench_models.iter().map(String::as_str).collect();
        let done: HashSet<(String, String, String, String)> = if only_missing {
            self.bench_rows.iter().map(BenchRow::key).collect()
        } else {
            HashSet::new()
        };
        let config = self.cfg.bench.label().to_lowercase();
        let chosen: Vec<(usize, &ModelInfo)> = self
            .models
            .iter()
            .enumerate()
            .filter(|(_, m)| m.full_path.to_str().is_some_and(|p| models.contains(p)))
            .collect();
        let mut pairs = Vec::new();
        for (vi, version) in self.versions.iter().enumerate() {
            if !version.has_bench() || !versions.contains(version.name.as_str()) {
                continue;
            }
            for &(mi, model) in &chosen {
                if only_missing {
                    let key = (
                        discovery::base_name(&model.name).to_lowercase(),
                        model.quant.to_lowercase(),
                        version.name.to_lowercase(),
                        config.clone(),
                    );
                    if done.contains(&key) {
                        continue;
                    }
                }
                pairs.push((vi, mi));
            }
        }
        pairs
    }

    fn bench_jobs(&self, only_missing: bool) -> Vec<bench::Job> {
        self.bench_pairs(only_missing)
            .into_iter()
            .map(|(v, m)| (self.versions[v].clone(), self.models[m].clone()))
            .collect()
    }

    fn start_bench(&mut self, only_missing: bool) {
        if self.bench_handle.is_some() {
            return;
        }
        let jobs = self.bench_jobs(only_missing);
        if jobs.is_empty() {
            let nothing_selected = self.bench_pairs(false).is_empty();
            if nothing_selected {
                self.notify(
                    self.t(
                        "Cochez au moins une version et un modèle.",
                        "Tick at least one version and one model.",
                    ),
                    theme::red(),
                );
            } else {
                self.notify(
                    self.t(
                        "La sélection est déjà mesurée avec ces réglages.",
                        "The selection is already benchmarked with these settings.",
                    ),
                    theme::green(),
                );
            }
            return;
        }

        self.bench_progress = (0, jobs.len());
        self.tab = Tab::Benchmark;
        self.log(format!(
            "—— Benchmark : {} combinaison(s) · {} ——",
            jobs.len(),
            self.cfg.bench.label()
        ));
        self.bench_handle = Some(bench::run(jobs, self.cfg.bench.clone()));
    }

    fn save_bench(&mut self) {
        match bench::save_results(&self.bench_rows) {
            Ok(path) => {
                self.notify(format!("{}", path.display()), theme::green());
                self.log(format!("✔ {}", path.display()));
            }
            Err(e) => self.notify(format!("{e}"), theme::red()),
        }
    }

    /// Lit en tâche de fond l'en-tête des modèles absents du cache ou
    /// modifiés depuis. Au premier lancement sur un gros dossier de modèles,
    /// l'interface reste fluide : les colonnes se remplissent au fil de la
    /// lecture, avec sa progression dans l'onglet Modèles.
    fn scan_metadata(&mut self) {
        if self.meta_rx.is_some() {
            self.meta_rescan = true;
            return;
        }
        let todo: Vec<PathBuf> = self
            .models
            .iter()
            .map(|m| m.full_path.clone())
            .filter(|p| self.meta.get(p).is_none_or(|c| !gguf::is_fresh(c, p)))
            .collect();
        if todo.is_empty() {
            return;
        }
        self.meta_progress = (0, todo.len());
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            for path in todo {
                let entry = gguf::read_cached(&path);
                if tx.send((path, entry)).is_err() {
                    break;
                }
            }
        });
        self.meta_rx = Some(rx);
    }

    fn pump_metadata(&mut self, ctx: &egui::Context) {
        let Some(rx) = &self.meta_rx else {
            return;
        };
        loop {
            match rx.try_recv() {
                Ok((path, entry)) => {
                    self.meta.insert(path, entry);
                    self.meta_progress.0 += 1;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    // Enregistré en cours de route : une première lecture
                    // interrompue (application fermée) n'est pas à refaire.
                    if self.meta_saved.elapsed() >= Duration::from_secs(2) {
                        gguf::save_cache(&self.meta);
                        self.meta_saved = Instant::now();
                    }
                    ctx.request_repaint_after(Duration::from_millis(100));
                    return;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
            }
        }
        self.meta_rx = None;
        gguf::save_cache(&self.meta);
        if std::mem::take(&mut self.meta_rescan) {
            self.scan_metadata();
        }
    }

    fn handle(&mut self, action: Action, ctx: &egui::Context) {
        match action {
            Action::Start => self.start_server(),
            Action::Stop => self.stop_server(),
            Action::Restart => {
                self.stop_server();
                // Laisse le port se libérer avant de relancer.
                self.restart_pending = Some(Instant::now() + Duration::from_millis(1000));
                self.log(self.t("↻ Redémarrage…", "↻ Restarting…").to_string());
            }
            Action::BrowseLlamaDir => {
                if let Some(dir) = pick_folder(&self.cfg.llama_cpp_directory) {
                    self.cfg.llama_cpp_directory = dir.to_string_lossy().into_owned();
                    self.sync_install_dir();
                    self.refresh();
                }
            }
            Action::BrowseModelsDir => {
                if let Some(dir) = pick_folder(&self.cfg.models_directory) {
                    self.cfg.models_directory = dir.to_string_lossy().into_owned();
                    self.refresh();
                }
            }
            Action::Refresh => {
                self.refresh();
                self.notify(
                    if self.fr() {
                        format!(
                            "{} version(s), {} modèle(s).",
                            self.versions.len(),
                            self.models.len()
                        )
                    } else {
                        format!(
                            "{} version(s), {} model(s).",
                            self.versions.len(),
                            self.models.len()
                        )
                    },
                    theme::accent(),
                );
            }
            Action::ResetDefaults => {
                let options: Vec<Vec<String>> =
                    self.states.iter().map(|s| s.options.clone()).collect();
                self.states = params::default_states();
                for (st, opts) in self.states.iter_mut().zip(options) {
                    st.options = opts;
                }
            }
            Action::DisableAll => {
                for st in self.states.iter_mut() {
                    st.value.clear();
                }
            }
            Action::OpenImport => {
                self.import_open = true;
            }
            Action::ApplyImport => self.apply_import(),
            Action::SaveProfile => self.save_profile(),
            Action::LoadProfile(name) => self.load_profile(name),
            Action::DeleteProfile(stem) => {
                self.profile_delete = None;
                if let Err(e) = profiles::delete(&stem) {
                    self.notify(format!("{e}"), theme::red());
                } else {
                    self.profiles = profiles::entries();
                    if self.active_profile == stem {
                        self.active_profile.clear();
                    }
                }
            }
            Action::RenameProfile { stem, name } => self.rename_profile(stem, name),
            Action::OpenUpdate => {
                self.update_open = true;
                self.update_delete_previous = false;
                self.inspect_source();
                // Coché si le build actuel compile déjà toutes les paires.
                self.update_fa_all =
                    self.update_config.as_ref().is_some_and(|(_, c)| c.fa_all());
                self.sync_install_dir();
                if !updater::is_source(Path::new(&self.cfg.llama_source_directory)) {
                    self.start_prereqs();
                }
            }
            Action::BrowseInstallDir => {
                if let Some(dir) = pick_folder(&self.install_dir) {
                    // Même dossier que le « Répertoire llama.cpp » : l'un suit
                    // l'autre, pour qu'il n'y ait qu'un seul choix à faire.
                    self.install_dir = dir.to_string_lossy().into_owned();
                    self.cfg.llama_cpp_directory = self.install_dir.clone();
                    self.refresh();
                }
            }
            Action::StartInstall => self.start_install(),
            Action::AskDeleteVersion(dir) => self.confirm_version_delete = Some(dir),
            Action::HubLoad => self.hub_load(),
            Action::HubSort(sort) => {
                self.hub_sort = sort;
                self.hub_load();
            }
            Action::HubPickRepo(repo) => self.hub_pick_repo(repo),
            Action::SetTheme(choice) => {
                self.cfg.theme = choice;
                theme::apply(ctx, choice);
            }
            Action::DeleteVersion(dir) => {
                self.confirm_version_delete = None;
                let name = dir
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                if self.version_in_use(&dir) {
                    self.notify(
                        self.t(
                            "Arrêtez d'abord le serveur lancé depuis cette version.",
                            "Stop the server running from this version first.",
                        ),
                        theme::orange(),
                    );
                } else {
                    let releases = PathBuf::from(&self.cfg.llama_releases_directory);
                    match updater::delete_release(&dir, &releases) {
                        Ok(()) => {
                            self.log(format!("🗑 {name}"));
                            self.probed.remove(&dir);
                            self.refresh();
                            self.notify(
                                format!("{} {name}", self.t("Version supprimée :", "Version deleted:")),
                                theme::accent(),
                            );
                        }
                        Err(e) => {
                            self.log(format!("! {e}"));
                            self.notify(e.to_string(), theme::red());
                        }
                    }
                }
            }
            Action::RecheckPrereqs => {
                self.prereqs = None;
                self.start_prereqs();
            }
            Action::CopyText(text) => {
                ctx.copy_text(text);
                self.notify(self.t("Commande copiée.", "Command copied."), theme::accent());
            }
            Action::BrowseSource => {
                if let Some(dir) = pick_folder(&self.cfg.llama_source_directory) {
                    if updater::is_source(&dir) {
                        self.cfg.llama_source_directory = dir.to_string_lossy().into_owned();
                        self.update_check = None;
                        self.inspect_source();
                    } else {
                        self.notify(
                            self.t(
                                "Ce dossier n'est pas un dépôt llama.cpp (git + CMakeLists.txt).",
                                "This folder is not a llama.cpp repository (git + CMakeLists.txt).",
                            ),
                            theme::red(),
                        );
                    }
                }
            }
            Action::BrowseReleases => {
                if let Some(dir) = pick_folder(&self.cfg.llama_releases_directory) {
                    self.cfg.llama_releases_directory = dir.to_string_lossy().into_owned();
                    self.refresh();
                }
            }
            Action::CheckUpdates => {
                if self.update_check_task.is_none() {
                    self.update_check = None;
                    self.update_check_task = Some(updater::check(PathBuf::from(
                        &self.cfg.llama_source_directory,
                    )));
                }
            }
            Action::StartUpdate => self.start_update(),
            Action::CancelUpdate => {
                if let Some(task) = &self.update_task {
                    task.cancel();
                }
            }
            Action::RemoveExtraArg(i) => {
                if i < self.cfg.extra_args.len() {
                    self.cfg.extra_args.remove(i);
                }
            }
            Action::UpdateProfile(stem) => self.update_profile(stem),
            Action::CopyConsole => {
                let text: Vec<&str> = self.console.iter().map(String::as_str).collect();
                ctx.copy_text(text.join("\n"));
                self.notify(self.t("Console copiée.", "Console copied."), theme::accent());
            }
            Action::ClearConsole => self.console.clear(),
            // La copie contient la vraie clé API : elle doit fonctionner une
            // fois collée. Seul l'affichage la masque.
            Action::CopyCommand => {
                ctx.copy_text(self.command_text(false, false));
                self.notify(self.t("Commande copiée.", "Command copied."), theme::accent());
            }
            Action::CopyCommandOneLine => {
                ctx.copy_text(self.command_text(true, false));
                self.notify(self.t("Commande copiée.", "Command copied."), theme::accent());
            }
            Action::BenchRun(only_missing) => self.start_bench(only_missing),
            Action::SetBenchSettings(settings) => self.cfg.bench = settings,
            Action::SetBenchSelection { models, versions } => {
                self.cfg.bench_models = models;
                self.cfg.bench_versions = versions;
            }
            Action::BenchStop => {
                if let Some(handle) = &self.bench_handle {
                    handle.cancel();
                }
            }
            Action::SaveBench => self.save_bench(),
            Action::BenchDelete(key) => {
                self.bench_confirm_clear = false;
                let before = self.bench_rows.len();
                match &key {
                    Some(k) => self.bench_rows.retain(|r| &r.key() != k),
                    None => self.bench_rows.clear(),
                }
                let removed = before - self.bench_rows.len();
                if removed > 0 {
                    if let Err(e) = bench::save_results(&self.bench_rows) {
                        self.notify(format!("{e}"), theme::red());
                    } else {
                        self.log(format!(
                            "{} {removed} {}",
                            self.t("Benchmark :", "Benchmark:"),
                            self.t("résultat(s) supprimé(s)", "result(s) deleted")
                        ));
                    }
                }
            }
            Action::BenchConfirmClear(on) => self.bench_confirm_clear = on,
            Action::ResetStats => {
                self.stats.clear();
                self.power.reset_server();
            }
            Action::SetEnergyPrice(price, currency) => {
                self.cfg.kwh_price = price;
                self.cfg.currency = currency;
            }
            Action::ResetPowerCurve => self.power.reset(),
            Action::ResetEnergyTotal => {
                self.energy_total_wh = 0.0;
                self.cfg.energy_since = chrono::Local::now().format("%d/%m/%Y").to_string();
            }
            Action::RevealModel(path) => reveal(&path),
            Action::DeleteModel(path) => {
                match std::fs::remove_file(&path) {
                    Ok(()) => {
                        self.log(format!("🗑 {}", path.display()));
                        self.meta.remove(&path);
                        self.refresh();
                    }
                    Err(e) => self.notify(format!("{e}"), theme::red()),
                }
                self.confirm_delete = None;
            }
            Action::HubDownload => self.hub_start_download(),
            Action::HubCancelDownload => {
                if let Some(d) = &self.hub_download {
                    d.cancel();
                }
            }
        }
    }

    /// Exécutable et modèle à afficher dans la commande, avec des marqueurs
    /// lisibles tant que la sélection est incomplète.
    fn command_targets(&self) -> (String, String) {
        let exe = self
            .version()
            .map(|v| v.server().to_string_lossy().into_owned())
            .unwrap_or_else(|| discovery::SERVER_EXE.to_string());
        let model = self
            .model()
            .map(|m| m.full_path.to_string_lossy().into_owned())
            .unwrap_or_else(|| self.t("<aucun modèle>", "<no model>").to_string());
        (exe, model)
    }

    /// La commande sur plusieurs lignes ou une seule, secrets masqués ou non.
    fn command_text(&self, one_line: bool, masked: bool) -> String {
        let (exe, model) = self.command_targets();
        let cmd = params::Command {
            exe: &exe,
            model: &model,
            host: &self.cfg.host,
            port: &self.cfg.port,
            states: &self.states,
            extra: &self.cfg.extra_args,
        };
        if one_line {
            cmd.one_line(masked)
        } else {
            cmd.multiline(masked)
        }
    }

    /// Masque dans un texte les valeurs des paramètres secrets actifs, avant
    /// de l'écrire dans la console.
    fn mask_secrets(&self, text: &str) -> String {
        let mut out = text.to_string();
        for (i, st) in self.states.iter().enumerate() {
            if params::def(i).secret && st.enabled() {
                out = out.replace(&st.value, &params::mask(&st.value));
            }
        }
        out
    }

    // ------------------------------------------------------------------ pumps

    fn pump_server(&mut self, ctx: &egui::Context) {
        let mut lines = Vec::new();
        let mut finished = None;
        if let Some(proc) = &mut self.server {
            while let Ok(ProcEvent::Line(line)) = proc.rx.try_recv() {
                lines.push(line);
            }
            if !proc.streams_open() {
                if let Some(code) = proc.try_wait() {
                    finished = Some(code);
                }
            }
        }

        let had_output = !lines.is_empty();
        for line in lines {
            self.speeds.observe(&line);
            self.stats.observe(&line);
            self.log(line);
        }

        if let Some(code) = finished {
            self.server = None;
            self.log(match code {
                Some(0) => self.t("■ Processus terminé.", "■ Process finished.").to_string(),
                Some(c) => format!("■ Code de sortie : {c}"),
                None => self.t("■ Processus arrêté.", "■ Process stopped.").to_string(),
            });
        }

        // Les lignes réveillent l'interface d'elles-mêmes (`process::Wake`) :
        // reste à surveiller la fin du processus une fois ses flux fermés.
        let exiting = self.server.as_ref().is_some_and(|p| !p.streams_open());
        if exiting || had_output {
            ctx.request_repaint_after(Duration::from_millis(100));
        }

        if let Some(at) = self.restart_pending {
            if Instant::now() >= at {
                self.restart_pending = None;
                self.start_server();
            } else {
                ctx.request_repaint_after(Duration::from_millis(120));
            }
        }
    }

    /// Relit l'état du dépôt source et les options de compilation, en
    /// complétant les chemins par détection s'ils sont vides.
    fn inspect_source(&mut self) {
        let versions_dir = PathBuf::from(&self.cfg.llama_cpp_directory);
        if self.cfg.llama_source_directory.is_empty() {
            if let Some(src) = updater::detect_source(&versions_dir) {
                self.cfg.llama_source_directory = src.to_string_lossy().into_owned();
            }
        }
        let source = PathBuf::from(&self.cfg.llama_source_directory);
        if self.cfg.llama_releases_directory.is_empty() && updater::is_source(&source) {
            self.cfg.llama_releases_directory = updater::default_releases_dir(&source)
                .to_string_lossy()
                .into_owned();
        }
        if !updater::is_source(&source) {
            self.update_git = Some(Err(self
                .t(
                    "Aucun dépôt llama.cpp trouvé : choisissez le dossier cloné avec git.",
                    "No llama.cpp repository found: pick the folder cloned with git.",
                )
                .to_string()));
            self.update_config = None;
            return;
        }
        self.update_git = Some(updater::git_state(&source).map_err(|e| e.to_string()));

        // Options : celles du dossier de compilation dédié s'il existe déjà,
        // sinon celles du build de référence qui seront reprises.
        let own = source.join(updater::BUILD_DIR_NAME).join("CMakeCache.txt");
        self.update_config = if own.is_file() {
            updater::BuildConfig::read(&own)
                .ok()
                .map(|c| (updater::BUILD_DIR_NAME.to_string(), c))
        } else {
            updater::detect_reference_cache(&versions_dir, &source).and_then(|cache| {
                updater::BuildConfig::read(&cache)
                    .ok()
                    .map(|c| (cache.to_string_lossy().into_owned(), c))
            })
        };
    }

    fn start_update(&mut self) {
        if self.update_task.is_some() {
            return;
        }
        let source = PathBuf::from(&self.cfg.llama_source_directory);
        if !updater::is_source(&source) || self.cfg.llama_releases_directory.is_empty() {
            self.notify(
                self.t(
                    "Choisissez d'abord le dépôt source et le dossier des releases.",
                    "Pick the source repository and the releases folder first.",
                ),
                theme::red(),
            );
            return;
        }
        let reference =
            updater::detect_reference_cache(Path::new(&self.cfg.llama_cpp_directory), &source);
        self.update_result = None;
        self.update_stage = None;
        self.update_progress = None;
        self.update_last_line.clear();
        self.update_started = Some(Instant::now());
        self.log(self.t("—— Mise à jour de llama.cpp ——", "—— Updating llama.cpp ——"));
        self.update_task = Some(updater::update(updater::Plan {
            source,
            reference,
            // Sert seulement si aucun build n'a jamais été configuré (une
            // installation interrompue avant la configuration).
            backend: self.install_backend,
            releases_dir: PathBuf::from(&self.cfg.llama_releases_directory),
            clone: false,
            // Seulement si demandé, et seulement une release de l'app.
            delete_previous: if self.update_delete_previous {
                self.deletable_version().map(|v| v.dir.clone())
            } else {
                None
            },
            fa_all: Some(self.update_fa_all),
        }));
    }

    /// Tant que llama.cpp n'est pas installé, le dossier d'installation est
    /// celui choisi comme « Répertoire llama.cpp ». Deux champs distincts
    /// faisaient installer ailleurs que dans le dossier choisi à gauche.
    fn sync_install_dir(&mut self) {
        let installed = updater::is_source(Path::new(&self.cfg.llama_source_directory));
        if !installed && !self.cfg.llama_cpp_directory.trim().is_empty() {
            self.install_dir = self.cfg.llama_cpp_directory.trim().to_string();
        }
    }

    /// Charge la liste des dépôts (réseau, en tâche de fond).
    fn hub_load(&mut self) {
        let (tx, rx) = channel();
        let sort = self.hub_sort;
        std::thread::spawn(move || {
            let _ = tx.send(hub::list_repos(sort).map_err(|e| e.to_string()));
        });
        self.hub_repos = None;
        self.hub_repos_rx = Some(rx);
    }

    /// Choisit un dépôt et charge ses fichiers.
    fn hub_pick_repo(&mut self, repo: String) {
        if repo == self.hub_repo && self.hub_files.is_some() {
            return;
        }
        self.hub_repo = repo.clone();
        self.hub_file.clear();
        self.hub_files = None;
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let files = hub::list_files(&repo).map_err(|e| e.to_string());
            let _ = tx.send((repo, files));
        });
        self.hub_files_rx = Some(rx);
    }

    fn hub_start_download(&mut self) {
        if self.hub_download.is_some() {
            return;
        }
        let file = self
            .hub_files
            .as_ref()
            .and_then(|r| r.as_ref().ok())
            .and_then(|files| files.iter().find(|f| f.file == self.hub_file))
            .cloned();
        let Some(file) = file else {
            return;
        };
        let models = PathBuf::from(&self.cfg.models_directory);
        if !models.is_dir() {
            self.notify(
                self.t(
                    "Choisissez d'abord le répertoire des modèles.",
                    "Pick the models directory first.",
                ),
                theme::red(),
            );
            return;
        }
        self.hub_progress = (0, file.size);
        self.log(format!(
            "⬇ {} · {} ({})",
            self.hub_repo,
            file.file,
            discovery::human_size(file.size)
        ));
        self.hub_download = Some(hub::download(self.hub_repo.clone(), file, models));
    }

    fn pump_hub(&mut self, ctx: &egui::Context) {
        let mut finished = None;
        let mut notices = Vec::new();
        if let Some(d) = &self.hub_download {
            while let Ok(event) = d.rx.try_recv() {
                match event {
                    hub::DownloadEvent::Progress { done, total } => self.hub_progress = (done, total),
                    hub::DownloadEvent::Notice(msg) => notices.push(msg),
                    hub::DownloadEvent::Done(result) => finished = Some(result),
                }
            }
            ctx.request_repaint_after(Duration::from_millis(250));
        }
        for msg in notices {
            self.log(format!("⬇ {msg}"));
        }
        if let Some(result) = finished {
            self.hub_download = None;
            match result {
                Ok(path) => {
                    // Le modèle rejoint la liste et devient le modèle choisi.
                    self.refresh();
                    self.selected_model = path.to_string_lossy().into_owned();
                    let msg = format!(
                        "{} {}",
                        self.t("Modèle téléchargé :", "Model downloaded:"),
                        path.display()
                    );
                    self.log(format!("✔ {msg}"));
                    self.notify(msg, theme::green());
                }
                Err(e) => {
                    self.log(format!("! {e}"));
                    self.notify(e, theme::red());
                }
            }
        }

        if let Some(rx) = &self.hub_repos_rx {
            match rx.try_recv() {
                Ok(result) => {
                    self.hub_repos_rx = None;
                    // Premier dépôt de la liste choisi d'office.
                    let first = result
                        .as_ref()
                        .ok()
                        .and_then(|r| r.first())
                        .map(|r| r.id.clone());
                    self.hub_repos = Some(result);
                    if let Some(first) = first {
                        let keep = self
                            .hub_repos
                            .as_ref()
                            .and_then(|r| r.as_ref().ok())
                            .is_some_and(|list| list.iter().any(|r| r.id == self.hub_repo));
                        if !keep {
                            self.hub_pick_repo(first);
                        }
                    }
                }
                Err(_) => ctx.request_repaint_after(Duration::from_millis(150)),
            }
        }
        if let Some(rx) = &self.hub_files_rx {
            match rx.try_recv() {
                Ok((repo, result)) => {
                    self.hub_files_rx = None;
                    // Réponse d'un dépôt qu'on a quitté entre-temps : ignorée.
                    if repo == self.hub_repo {
                        // Q4_K_M par défaut, comme llama-server ; sinon le
                        // plus léger.
                        if let Ok(files) = &result {
                            self.hub_file = files
                                .iter()
                                .find(|f| f.quant == "Q4_K_M")
                                .or_else(|| files.first())
                                .map(|f| f.file.clone())
                                .unwrap_or_default();
                        }
                        self.hub_files = Some(result);
                    }
                }
                Err(_) => ctx.request_repaint_after(Duration::from_millis(150)),
            }
        }
    }

    /// Détecte les outils de compilation en tâche de fond.
    fn start_prereqs(&mut self) {
        if self.prereq_rx.is_some() {
            return;
        }
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let _ = tx.send(updater::detect_prereqs());
        });
        self.prereq_rx = Some(rx);
    }

    fn start_install(&mut self) {
        if self.update_task.is_some() || self.install_dir.trim().is_empty() {
            return;
        }
        let source = PathBuf::from(self.install_dir.trim()).join("llama.cpp");
        // Un clone existe déjà là : l'adopter plutôt que d'échouer.
        if updater::is_source(&source) {
            self.cfg.llama_source_directory = source.to_string_lossy().into_owned();
            self.inspect_source();
            self.notify(
                self.t(
                    "Un dépôt llama.cpp existe déjà dans ce dossier : il sera mis à jour.",
                    "A llama.cpp repository already exists there: it will be updated.",
                ),
                theme::accent(),
            );
            return;
        }
        let releases = updater::default_releases_dir(&source);
        // La colonne de gauche montre ensuite où tout se trouve.
        self.cfg.llama_cpp_directory = self.install_dir.trim().to_string();
        self.cfg.llama_source_directory = source.to_string_lossy().into_owned();
        self.cfg.llama_releases_directory = releases.to_string_lossy().into_owned();
        self.update_result = None;
        self.update_stage = None;
        self.update_progress = None;
        self.update_last_line.clear();
        self.update_started = Some(Instant::now());
        self.log(self.t("—— Installation de llama.cpp ——", "—— Installing llama.cpp ——"));
        self.update_task = Some(updater::install(updater::Plan {
            source,
            reference: None,
            backend: self.install_backend,
            releases_dir: releases,
            clone: true,
            delete_previous: None,
            fa_all: Some(self.update_fa_all),
        }));
    }

    fn pump_update(&mut self, ctx: &egui::Context) {
        if let Some(rx) = &self.prereq_rx {
            match rx.try_recv() {
                Ok(found) => {
                    if !self.install_backend_chosen {
                        self.install_backend = found.suggested();
                    }
                    self.prereqs = Some(found);
                    self.prereq_rx = None;
                }
                Err(_) => ctx.request_repaint_after(Duration::from_millis(150)),
            }
        }

        if let Some(task) = &self.update_check_task {
            if let Ok(updater::Event::Checked(result)) = task.rx.try_recv() {
                self.update_check = Some(result);
                self.update_check_task = None;
            } else {
                ctx.request_repaint_after(Duration::from_millis(200));
            }
        }

        let mut events = Vec::new();
        if let Some(task) = &self.update_task {
            while let Ok(event) = task.rx.try_recv() {
                events.push(event);
            }
            ctx.request_repaint_after(Duration::from_millis(250));
        }
        for event in events {
            match event {
                updater::Event::Line(line) => {
                    self.update_last_line = line.clone();
                    self.log(line);
                }
                updater::Event::Stage {
                    index,
                    total,
                    label,
                } => {
                    self.update_stage = Some((index, total, label));
                    self.update_progress = None;
                }
                updater::Event::Progress(p) => self.update_progress = Some(p),
                updater::Event::Checked(_) => {}
                updater::Event::Done(result) => {
                    self.update_task = None;
                    match &result {
                        Ok(release) => {
                            if let Some(removed) = &release.removed {
                                self.probed.retain(|dir, _| {
                                    dir.file_name().is_none_or(|n| n.to_string_lossy() != *removed)
                                });
                            }
                            self.refresh();
                            self.selected_version = release.name.clone();
                            self.update_check = None;
                            self.inspect_source();
                            let msg = format!(
                                "{} {} (b{})",
                                self.t(
                                    "Release créée et sélectionnée :",
                                    "Release created and selected:"
                                ),
                                release.name,
                                release.build
                            );
                            self.log(format!("✔ {msg}"));
                            self.notify(msg, theme::green());
                        }
                        Err(e) => {
                            self.log(format!("! {e}"));
                            self.notify(e.clone(), theme::red());
                        }
                    }
                    self.update_result = Some(result);
                }
            }
        }
    }

    fn pump_bench(&mut self, ctx: &egui::Context) {
        let mut logs = Vec::new();
        let mut rows = Vec::new();
        let mut done = None;
        let mut progress = None;

        if let Some(handle) = &self.bench_handle {
            while let Ok(event) = handle.rx.try_recv() {
                match event {
                    BenchEvent::Log(line) => logs.push(line),
                    BenchEvent::Row(row) => rows.push(row),
                    BenchEvent::Progress { done, total, label } => {
                        progress = Some((done, total, label))
                    }
                    BenchEvent::Done { completed, failed } => done = Some((completed, failed)),
                }
            }
        }

        for line in logs {
            self.log(line);
        }
        for row in rows {
            self.log(format!(
                "  ✔ {} {} — PP {} · TG {}",
                row.model, row.quant, row.pp, row.tg
            ));
            bench::upsert(&mut self.bench_rows, row);
        }
        if let Some((done, total, label)) = progress {
            self.bench_progress = (done, total);
            self.bench_label = label;
        }
        if let Some((completed, failed)) = done {
            self.bench_handle = None;
            self.bench_label.clear();
            self.log(format!(
                "—— Benchmark terminé : {completed} réussi(s), {failed} échec(s) ——"
            ));
            self.save_bench();
        }

        if self.bench_handle.is_some() {
            ctx.request_repaint_after(Duration::from_millis(200));
        }
    }
}

// ---------------------------------------------------------------------- eframe

impl eframe::App for App {
    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.pump_server(&ctx);
        self.pump_bench(&ctx);
        self.pump_probe(&ctx);
        self.pump_update(&ctx);
        self.pump_hub(&ctx);
        self.pump_metadata(&ctx);
        if let Some(rx) = &self.fa_rx {
            while let Ok((dir, found)) = rx.try_recv() {
                self.fa_support.insert(dir, found);
            }
        }
        while let Ok((at, gpus)) = self.gpu_rx.try_recv() {
            let watts: Vec<f64> = gpus.iter().filter_map(|g| g.power_w).map(f64::from).collect();
            if !watts.is_empty() {
                let running = self.running();
                let wh = self.power.record(watts.iter().sum(), running, at);
                self.energy_total_wh += wh;
                if self.cfg.energy_since.is_empty() {
                    self.cfg.energy_since = chrono::Local::now().format("%d/%m/%Y").to_string();
                }
            }
            self.gpus = gpus;
        }
        // Le suivi réveille l'interface toutes les deux secondes pour
        // rafraîchir la jauge, seulement s'il y a un GPU à afficher.
        if !self.gpus.is_empty() {
            ctx.request_repaint_after(Duration::from_secs(2));
        }

        let mut actions: Vec<Action> = Vec::new();

        self.top_bar(ui, &mut actions);
        self.config_panel(ui, &mut actions);
        self.params_panel(ui, &mut actions);
        self.bottom_bar(ui, &mut actions);
        self.central(ui, &mut actions);
        self.import_window(&ctx, &mut actions);
        self.confirm_window(&ctx, &mut actions);
        self.confirm_version_window(&ctx, &mut actions);
        self.update_window(&ctx, &mut actions);

        for action in actions {
            self.handle(action, &ctx);
        }
        self.autosave(&ctx);

        if let Some((_, _, at)) = self.toast {
            if at.elapsed() > Duration::from_secs(4) {
                self.toast = None;
            } else {
                ctx.request_repaint_after(Duration::from_millis(300));
            }
        }
    }

    fn on_exit(&mut self) {
        // Une compilation ou un git en cours ne doivent pas survivre à l'app.
        if let Some(task) = &self.update_task {
            task.cancel();
        }
        if let Some(task) = &self.update_check_task {
            task.cancel();
        }
        if let Some(d) = &self.hub_download {
            d.cancel();
        }
        if let Some(handle) = &self.bench_handle {
            handle.cancel();
        }
        self.stop_server();
        self.persist();
    }
}

// -------------------------------------------------------------------- panneaux

impl App {
    fn top_bar(&mut self, root: &mut Ui, actions: &mut Vec<Action>) {
        egui::Panel::top("top")
            .exact_size(52.0)
            .frame(egui::Frame::new().fill(theme::card()).inner_margin(egui::Margin::symmetric(12, 8)))
            .show(root, |ui| {
                ui.horizontal_centered(|ui| {
                    let running = self.running();
                    let can_start = !running
                        && self.version().is_some()
                        && self.model().is_some()
                        && self.restart_pending.is_none();

                    if ui
                        .add_enabled(
                            can_start,
                            theme::action_button(
                                RichText::new(format!("▶  {}", self.t("Démarrer", "Start")))
                                    .strong(),
                                theme::green(),
                            ),
                        )
                        .on_hover_text(self.t(
                            "Lance llama-server avec la commande affichée en bas de la fenêtre.",
                            "Starts llama-server with the command shown at the bottom of the window.",
                        ))
                        .on_disabled_hover_text(if running {
                            self.t(
                                "Un serveur tourne déjà : arrêtez-le d'abord.",
                                "A server is already running: stop it first.",
                            )
                        } else {
                            self.t(
                                "Sélectionnez une version de llama.cpp et un modèle à gauche.",
                                "Pick a llama.cpp version and a model on the left.",
                            )
                        })
                        .clicked()
                    {
                        actions.push(Action::Start);
                    }
                    if ui
                        .add_enabled(
                            running,
                            theme::action_button(
                                RichText::new(format!("■  {}", self.t("Arrêter", "Stop")))
                                    .strong(),
                                theme::red(),
                            ),
                        )
                        .on_hover_text(self.t(
                            "Arrête le serveur et toute son arborescence de processus.",
                            "Stops the server and its whole process tree.",
                        ))
                        .on_disabled_hover_text(self.t(
                            "Aucun serveur en cours.",
                            "No server running.",
                        ))
                        .clicked()
                    {
                        actions.push(Action::Stop);
                    }
                    if ui
                        .add_enabled(
                            running,
                            theme::action_button(
                                RichText::new(format!("↻  {}", self.t("Redémarrer", "Restart")))
                                    .strong(),
                                theme::orange(),
                            ),
                        )
                        .on_hover_text(self.t(
                            "Arrête puis relance après une seconde, le temps que le port se libère. \
                             À utiliser après avoir changé un paramètre.",
                            "Stops, then restarts after a second so the port is released. \
                             Use it after changing a parameter.",
                        ))
                        .on_disabled_hover_text(self.t(
                            "Aucun serveur en cours.",
                            "No server running.",
                        ))
                        .clicked()
                    {
                        actions.push(Action::Restart);
                    }

                    ui.add_space(12.0);
                    let (dot, label) = if running {
                        (theme::green(), format!("http://{}:{}", self.cfg.host, self.cfg.port))
                    } else {
                        (theme::muted(), self.t("arrêté", "stopped").to_string())
                    };
                    let state_tip = if running {
                        self.t(
                            "Serveur en écoute. Ouvrez cette adresse dans un navigateur pour \
                             l'interface web de llama.cpp, ou pointez-y un client compatible \
                             OpenAI.",
                            "Server listening. Open this address in a browser for the llama.cpp \
                             web UI, or point an OpenAI-compatible client at it.",
                        )
                    } else {
                        self.t("Aucun serveur en cours.", "No server running.")
                    };
                    status_chip(ui, dot, &label).on_hover_text(state_tip);

                    if self.update_task.is_some() {
                        ui.add_space(8.0);
                        ui.add(egui::Spinner::new().size(14.0));
                        let elapsed = self
                            .update_started
                            .map(|t| format_elapsed(t.elapsed()))
                            .unwrap_or_default();
                        let stage = self
                            .update_stage
                            .map(|(i, n, _)| format!("{i}/{n}"))
                            .unwrap_or_default();
                        if ui
                            .add(
                                egui::Label::new(
                                    RichText::new(format!(
                                        "llama.cpp {stage} · {elapsed}"
                                    ))
                                    .color(theme::yellow())
                                    .size(12.0),
                                )
                                .sense(egui::Sense::click()),
                            )
                            .on_hover_text(self.t(
                                "Mise à jour de llama.cpp en cours. Cliquer pour suivre.",
                                "llama.cpp update in progress. Click to follow it.",
                            ))
                            .clicked()
                        {
                            actions.push(Action::OpenUpdate);
                        }
                    }

                    for g in &self.gpus {
                        ui.add_space(10.0);
                        let fraction = g.fraction();
                        let color = match fraction {
                            f if f >= 0.9 => theme::red(),
                            f if f >= 0.7 => theme::orange(),
                            _ => theme::green(),
                        };
                        ui.label(RichText::new("VRAM").color(theme::muted()).size(11.0));
                        gauge(
                            ui,
                            fraction,
                            color,
                            &format!(
                                "{:.1} / {:.1} {}",
                                g.used_mib as f64 / 1024.0,
                                g.total_mib as f64 / 1024.0,
                                self.t("Go", "GB")
                            ),
                        )
                        .on_hover_text(format!(
                            "{}\n{} {} Mio / {} Mio ({:.0} %)\n{} {} Mio\n{} {} %\n\n{}",
                            g.name,
                            self.t("Utilisée :", "Used:"),
                            g.used_mib,
                            g.total_mib,
                            fraction * 100.0,
                            self.t("Libre :", "Free:"),
                            g.total_mib.saturating_sub(g.used_mib),
                            self.t("Charge GPU :", "GPU load:"),
                            g.load,
                            self.t(
                                "Toute la carte : llama-server et les autres applications.",
                                "The whole card: llama-server and every other application.",
                            ),
                        ));
                        if let Some(w) = g.power_w {
                            let limit = g
                                .power_limit_w
                                .map(|l| format!(" / {l:.0} W"))
                                .unwrap_or_default();
                            ui.label(
                                RichText::new(format!("{w:.0} W"))
                                    .color(theme::text())
                                    .size(12.0),
                            )
                            .on_hover_text(format!(
                                "{}\n{} {w:.1} W{limit}\n\n{}",
                                g.name,
                                self.t("Consommation :", "Power draw:"),
                                self.t(
                                    "Puissance électrique tirée par la carte graphique, relevée \
                                     toutes les deux secondes. La limite est celle du pilote.",
                                    "Electrical power drawn by the graphics card, read every two \
                                     seconds. The limit is the driver's.",
                                ),
                            ));
                        }
                    }

                    if let Some(pp) = self.speeds.prompt {
                        ui.label(
                            RichText::new(format!("PP {pp:.1} t/s"))
                                .color(theme::yellow())
                                .size(12.0),
                        )
                        .on_hover_text(self.t(
                            "Prompt processing : vitesse de lecture du prompt, en tokens par \
                             seconde. Relevée dans les logs de la dernière requête.",
                            "Prompt processing: how fast the prompt is read, in tokens per second. \
                             Taken from the last request's logs.",
                        ));
                    }
                    if let Some(tg) = self.speeds.generation {
                        ui.label(
                            RichText::new(format!("TG {tg:.1} t/s"))
                                .color(theme::yellow())
                                .size(12.0),
                        )
                        .on_hover_text(self.t(
                            "Token generation : vitesse d'écriture de la réponse, en tokens par \
                             seconde. C'est ce que perçoit l'utilisateur.",
                            "Token generation: how fast the answer is written, in tokens per \
                             second. This is what the user perceives.",
                        ));
                    }

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let lang_tip = self.t(
                            "Langue de l'interface et des infobulles.",
                            "Language of the interface and tooltips.",
                        );
                        let mut lang = self.cfg.language;
                        ComboBox::from_id_salt("lang")
                            .width(64.0)
                            .selected_text(if lang.is_fr() { "FR" } else { "EN" })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut lang, Lang::Fr, "FR");
                                ui.selectable_value(&mut lang, Lang::En, "EN");
                            })
                            .response
                            .on_hover_text(lang_tip);
                        self.cfg.language = lang;

                        let mut choice = self.cfg.theme;
                        let fr = self.fr();
                        ComboBox::from_id_salt("theme")
                            .width(118.0)
                            .selected_text(choice.label(fr))
                            .show_ui(ui, |ui| {
                                for t in ThemeChoice::ALL {
                                    ui.selectable_value(&mut choice, t, t.label(fr));
                                }
                            })
                            .response
                            .on_hover_text(self.t("Thème de l'interface.", "Interface theme."));
                        if choice != self.cfg.theme {
                            actions.push(Action::SetTheme(choice));
                        }

                        if let Some((text, color, _)) = &self.toast {
                            ui.label(RichText::new(text.clone()).color(*color).size(12.0));
                        }
                    });
                });
            });
    }

    fn config_panel(&mut self, root: &mut Ui, actions: &mut Vec<Action>) {
        egui::Panel::left("config")
            .default_size(330.0)
            .size_range(280.0..=460.0)
            .frame(panel_frame())
            .show(root, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("config_scroll")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        let fr = self.fr();
                        if fold_header(
                            ui,
                            "fold_config",
                            tr(fr, "Configuration", "Configuration"),
                            tr(
                                fr,
                                "Dossiers, version de llama.cpp, modèle et modèle vision.",
                                "Folders, llama.cpp version, model and vision model.",
                            ),
                            None,
                        ) {
                            dir_row(
                                ui,
                                self.t("Répertoire llama.cpp", "llama.cpp directory"),
                                &self.cfg.llama_cpp_directory,
                                self.t("Parcourir…", "Browse…"),
                                self.t(
                                    "Dossier qui contient vos versions de llama.cpp. Chaque \
                                     sous-dossier doit contenir llama-server.exe ; les dispositions \
                                     bin/ et build/bin/ sont reconnues.",
                                    "Folder holding your llama.cpp versions. Each subfolder must \
                                     contain llama-server.exe; the bin/ and build/bin/ layouts are \
                                     recognized.",
                                ),
                                || actions.push(Action::BrowseLlamaDir),
                            );
                            dir_row(
                                ui,
                                self.t("Répertoire modèles", "Models directory"),
                                &self.cfg.models_directory,
                                self.t("Parcourir…", "Browse…"),
                                self.t(
                                    "Dossier de vos fichiers .gguf, parcouru récursivement \
                                     (4 niveaux).",
                                    "Folder holding your .gguf files, scanned recursively (4 levels).",
                                ),
                                || actions.push(Action::BrowseModelsDir),
                            );

                            ui.add_space(6.0);
                            if ui
                                .button(format!("⟳  {}", self.t("Rafraîchir", "Refresh")))
                                .on_hover_text(self.t(
                                    "Relit les deux répertoires. À utiliser après avoir ajouté une \
                                     version ou téléchargé un modèle.",
                                    "Re-scans both directories. Use it after adding a version or \
                                     downloading a model.",
                                ))
                                .clicked()
                            {
                                actions.push(Action::Refresh);
                            }
                            ui.add_space(10.0);

                            // ---- Version ----
                            let version_tip = self.t(
                                "Build de llama.cpp utilisé pour lancer le serveur et les benchmarks. \
                                 Le backend (Vulkan, CUDA, CPU…) dépend de ce choix. Une version \
                                 marquée « sans llama-bench » ne pourra pas être mesurée.",
                                "The llama.cpp build used to run the server and the benchmarks. The \
                                 backend (Vulkan, CUDA, CPU…) follows from this choice. A version \
                                 marked “sans llama-bench” cannot be benchmarked.",
                            );
                            field_label(
                                ui,
                                self.t("Version llama.cpp", "llama.cpp version"),
                                version_tip,
                            );
                            let none_label = self.t("— aucune —", "— none —");
                            {
                                let versions = &self.versions;
                                let selected = &mut self.selected_version;
                                let text = match versions.iter().find(|v| v.name == *selected) {
                                    Some(v) => format!("{}  ·  {}", v.tag(), v.name),
                                    None if selected.is_empty() => none_label.to_string(),
                                    None => selected.clone(),
                                };
                                ComboBox::from_id_salt("version")
                                    .width(ui.available_width())
                                    .selected_text(text)
                                    .show_ui(ui, |ui| {
                                        ui.set_min_width(360.0);
                                        for v in versions {
                                            let mut label = format!("{}  ·  {}", v.tag(), v.name);
                                            if !v.has_bench() {
                                                label.push_str("  (sans llama-bench)");
                                            }
                                            let mut tip = v.dir.to_string_lossy().into_owned();
                                            if let Some(commit) = &v.commit {
                                                tip.push_str(&format!("\ncommit {commit}"));
                                            }
                                            ui.selectable_value(selected, v.name.clone(), label)
                                                .on_hover_text(tip);
                                        }
                                    })
                                    .response
                                    .on_hover_text(version_tip);
                            }
                            ui.add_space(4.0);
                            let updating = self.update_task.is_some();
                            let installed = !self.versions.is_empty();
                            if !installed {
                                ui.label(
                                    RichText::new(self.t(
                                        "Aucun llama.cpp : Options › Installer llama.cpp.",
                                        "No llama.cpp: Options › Install llama.cpp.",
                                    ))
                                    .color(theme::orange())
                                    .size(11.0),
                                );
                            }

                            // Un seul menu pour tout ce qui touche aux versions.
                            let update_label = match (updating, installed) {
                                (true, _) => self.t("⟳ Suivre la compilation…", "⟳ Follow the build…"),
                                (false, true) => {
                                    self.t("⟳ Mettre à jour llama.cpp…", "⟳ Update llama.cpp…")
                                }
                                (false, false) => self.t("📥 Installer llama.cpp…", "📥 Install llama.cpp…"),
                            };
                            let update_tip = if installed {
                                self.t(
                                    "Récupère les dernières sources de llama.cpp, les compile avec \
                                     les mêmes options que votre build actuel, et publie le résultat \
                                     dans une nouvelle release. L'ancienne version est conservée.",
                                    "Fetches the latest llama.cpp sources, builds them with the same \
                                     options as your current build, and publishes the result as a new \
                                     release. The old version is kept.",
                                )
                            } else {
                                self.t(
                                    "Télécharge les sources de llama.cpp et les compile pour votre \
                                     machine.",
                                    "Downloads the llama.cpp sources and builds them for your machine.",
                                )
                            };
                            let delete_label = format!(
                                "🗑 {}",
                                self.t("Supprimer cette version…", "Delete this version…")
                            );
                            // (dossier, supprimable, raison si non supprimable)
                            let delete = match (self.version(), self.deletable_version()) {
                                (None, _) => (None, self.t("Aucune version sélectionnée.", "No version selected.")),
                                (Some(_), None) => (
                                    None,
                                    self.t(
                                        "Seules les releases du dossier des releases peuvent être \
                                         supprimées ; celle-ci est ailleurs sur le disque.",
                                        "Only releases in the releases folder can be deleted; this \
                                         one lives elsewhere on disk.",
                                    ),
                                ),
                                (Some(_), Some(v)) if self.version_in_use(&v.dir) => (
                                    None,
                                    self.t(
                                        "Le serveur tourne depuis cette version : arrêtez-le d'abord.",
                                        "The server runs from this version: stop it first.",
                                    ),
                                ),
                                (Some(_), Some(v)) => (Some(v.dir.clone()), ""),
                            };
                            // `menu_button` ne prend que du texte : sans installation,
                            // c'est lui qui prend la couleur d'accent pour guider.
                            let mut menu_text =
                                RichText::new(format!("⚙ {}", self.t("Options", "Options")));
                            if !installed && !updating {
                                menu_text = menu_text.color(theme::accent()).strong();
                            }
                            ui.menu_button(menu_text, |ui| {
                                ui.set_min_width(240.0);
                                if ui.button(update_label).on_hover_text(update_tip).clicked() {
                                    actions.push(Action::OpenUpdate);
                                    ui.close();
                                }
                                ui.separator();
                                let (dir, reason) = &delete;
                                let response = ui.add_enabled(
                                    dir.is_some(),
                                    egui::Button::new(RichText::new(&delete_label).color(theme::red())),
                                );
                                let response = if dir.is_some() {
                                    response.on_hover_text(self.t(
                                        "Supprime du disque la version sélectionnée. Une \
                                         confirmation est demandée.",
                                        "Deletes the selected version from disk. You will be asked \
                                         to confirm.",
                                    ))
                                } else {
                                    response.on_disabled_hover_text(*reason)
                                };
                                if response.clicked() {
                                    if let Some(dir) = dir {
                                        actions.push(Action::AskDeleteVersion(dir.clone()));
                                    }
                                    ui.close();
                                }
                            })
                            .response
                            .on_hover_text(self.t(
                                "Mettre à jour, installer ou supprimer une version de llama.cpp.",
                                "Update, install or delete a llama.cpp version.",
                            ));
                            if self.versions.is_empty() && !self.cfg.llama_cpp_directory.is_empty() {
                                hint(
                                    ui,
                                    self.t(
                                        "Aucun llama-server.exe trouvé dans ce répertoire.",
                                        "No llama-server.exe found in this directory.",
                                    ),
                                );
                            }

                            ui.add_space(8.0);

                            // ---- Modèle ----
                            let model_tip = self.t(
                                "Modèle à charger. Les projecteurs multimodaux (mmproj) et les parties \
                                 2 et suivantes des modèles scindés sont masqués : sélectionnez la \
                                 première partie, llama.cpp charge les autres tout seul.",
                                "Model to load. Multimodal projectors (mmproj) and parts 2+ of split \
                                 models are hidden: pick the first part, llama.cpp loads the rest by \
                                 itself.",
                            );
                            field_label(ui, self.t("Modèle GGUF", "GGUF model"), model_tip);
                            let none_model = self.t("— aucun —", "— none —");
                            {
                                let models = &self.models;
                                let selected = &mut self.selected_model;
                                let text = models
                                    .iter()
                                    .find(|m| m.full_path.to_string_lossy() == *selected)
                                    .map(|m| format!("{}  ·  {}", m.name, m.size_human()))
                                    .unwrap_or_else(|| none_model.to_string());
                                ComboBox::from_id_salt("model")
                                    .width(ui.available_width())
                                    .selected_text(text)
                                    .show_ui(ui, |ui| {
                                        ui.set_min_width(420.0);
                                        for m in models {
                                            let value = m.full_path.to_string_lossy().into_owned();
                                            ui.selectable_value(
                                                selected,
                                                value,
                                                format!("{}  ·  {}  ·  {}", m.name, m.quant, m.size_human()),
                                            )
                                            .on_hover_text(m.full_path.to_string_lossy());
                                        }
                                    })
                                    .response
                                    .on_hover_text(model_tip);
                            }
                            if !self.models.is_empty() {
                                hint(
                                    ui,
                                    &format!(
                                        "{} {}",
                                        self.models.len(),
                                        self.t("modèle(s) détecté(s)", "model(s) found")
                                    ),
                                );
                            }

                            // ---- Modèle vision : présenté comme le modèle GGUF, qu'il
                            //      complète (le projecteur doit lui correspondre) ----
                            ui.add_space(8.0);
                            let fr = self.fr();
                            if let (Some(mm), Some(cpu)) = (
                                params::index_of("--mmproj"),
                                params::index_of("--no-mmproj-offload"),
                            ) {
                                let def = params::def(mm);
                                let tip = param_tip(def, fr);
                                field_label(ui, tr(fr, "Modèle vision", "Vision model"), &tip);
                                choice_field(ui, def, &mut self.states[mm], &tip, fr, &KvCheck::default());
                                hint(
                                    ui,
                                    &format!(
                                        "{} {}",
                                        self.mmproj_count,
                                        tr(fr, "modèle(s) détecté(s)", "model(s) found")
                                    ),
                                );
                                let cpu_tip = param_tip(params::def(cpu), fr);
                                flag_field_labeled(
                                    ui,
                                    &mut self.states[cpu].value,
                                    tr(fr, "Modèle vision sur CPU", "Vision model on CPU"),
                                    &cpu_tip,
                                );
                            }

                        }
                        ui.add_space(10.0);

                        // ---- Serveur : adresse d'écoute puis paramètres ----
                        ui.add_space(14.0);
                        if fold_header(
                            ui,
                            "fold_server",
                            Group::Server.title(fr),
                            Group::Server.help(fr),
                            None,
                        ) {
                            ui.add_space(4.0);
                            let fa = self.current_fa();
                            let cfg = &mut self.cfg;
                            param_grid_with(ui, Group::Server, &mut self.states, fr, fa.as_ref(), |ui| {
                                address_rows(ui, &mut cfg.host, &mut cfg.port, fr);
                            });
                            // Joignable depuis le réseau sans authentification.
                            let exposed =
                                !matches!(self.cfg.host.as_str(), "127.0.0.1" | "localhost" | "::1");
                            let has_key = params::index_of("--api-key")
                                .is_some_and(|i| self.states[i].enabled());
                            if exposed && !has_key {
                                ui.horizontal_wrapped(|ui| {
                                    theme::dot(ui, theme::orange());
                                    ui.label(
                                        RichText::new(self.t(
                                            "Serveur ouvert au réseau sans clé API.",
                                            "Server open to the network without an API key.",
                                        ))
                                        .color(theme::orange())
                                        .size(11.0),
                                    )
                                    .on_hover_text(self.t(
                                        "Avec cet hôte, toute machine du réseau local peut utiliser \
                                         le serveur. Renseignez une clé API, ou choisissez 127.0.0.1 \
                                         pour le réserver à cette machine.",
                                        "With this host, any machine on the local network can use \
                                         the server. Set an API key, or pick 127.0.0.1 to keep it to \
                                         this machine.",
                                    ));
                                });
                            }
                        }
                        ui.add_space(16.0);
                        self.profiles_section(ui, actions);
                    });
            });
    }

    /// Profils affichés comme une liste de conversations : une carte par
    /// profil, la plus récente en haut. Cliquer la carte charge le profil.
    fn profiles_section(&mut self, ui: &mut Ui, actions: &mut Vec<Action>) {
        let fr = self.fr();

        let count = (!self.profiles.is_empty()).then(|| self.profiles.len().to_string());
        if !fold_header(
            ui,
            "fold_profiles",
            tr(fr, "Profils", "Profiles"),
            tr(
                fr,
                "Jeux de réglages enregistrés dans profiles/, à côté de l'exécutable : paramètres, \
                 hôte, port, modèle et version. Le plus récent en haut.",
                "Settings saved in profiles/ next to the executable: parameters, host, port, model \
                 and version. Most recent first.",
            ),
            count,
        ) {
            return;
        }
        ui.add_space(6.0);

        // Création, comme « Nouvelle conversation ».
        let name_empty = self.profile_name.trim().is_empty();
        ui.horizontal(|ui| {
            let width = (ui.available_width() - 84.0).max(80.0);
            let field = ui
                .add(
                    egui::TextEdit::singleline(&mut self.profile_name)
                        .desired_width(width)
                        .hint_text(tr(fr, "nom du nouveau profil…", "new profile name…")),
                )
                .on_hover_text(tr(
                    fr,
                    "Nom du profil à créer à partir des réglages actuels. Entrée pour valider.",
                    "Name of the profile to create from the current settings. Enter to confirm.",
                ));
            if field.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) && !name_empty {
                actions.push(Action::SaveProfile);
            }
            if ui
                .add_enabled(
                    !name_empty,
                    theme::action_button(
                        RichText::new(tr(fr, "+ Créer", "+ Create")),
                        theme::accent(),
                    ),
                )
                .on_hover_text(tr(
                    fr,
                    "Crée un profil avec les paramètres, l'hôte, le port, le modèle et la version \
                     actuellement sélectionnés.",
                    "Creates a profile with the currently selected parameters, host, port, model \
                     and version.",
                ))
                .on_disabled_hover_text(tr(fr, "Saisissez d'abord un nom.", "Type a name first."))
                .clicked()
            {
                actions.push(Action::SaveProfile);
            }
        });
        ui.add_space(8.0);

        if self.profiles.is_empty() {
            hint(
                ui,
                tr(
                    fr,
                    "Aucun profil. Réglez les paramètres puis créez-en un ci-dessus.",
                    "No profiles yet. Tune the parameters, then create one above.",
                ),
            );
            return;
        }

        let current = params::to_map(&self.states);
        let now = chrono::Local::now();
        let entries = self.profiles.clone();
        for entry in &entries {
            let active = self.active_profile == entry.stem;
            let p = &entry.profile;
            // Écart entre le profil actif et ce qui est affiché à l'écran.
            let dirty = active
                && (p.parameters != current
                    || p.extra_args != self.cfg.extra_args
                    || (!p.host.is_empty() && p.host != self.cfg.host)
                    || (!p.port.is_empty() && p.port != self.cfg.port)
                    || (!p.model.is_empty() && p.model != self.selected_model)
                    || (!p.version.is_empty() && p.version != self.selected_version));
            profile_card(
                ui,
                entry,
                CardState {
                    active,
                    dirty,
                    now,
                    fr,
                },
                &mut self.profile_edit,
                &mut self.profile_delete,
                actions,
            );
            ui.add_space(6.0);
        }
    }

    fn params_panel(&mut self, root: &mut Ui, actions: &mut Vec<Action>) {
        egui::Panel::left("params")
            .default_size(360.0)
            .size_range(300.0..=520.0)
            .frame(panel_frame())
            .show(root, |ui| {
                // Titre et boutons sur deux lignes : le panneau est étroit et
                // les deux se chevauchaient.
                ui.vertical(|ui| {
                    ui.label(
                        RichText::new(self.t("Paramètres de lancement", "Launch parameters"))
                            .color(theme::text())
                            .strong()
                            .size(14.0),
                    )
                    .on_hover_text(self.t(
                        "Options passées à llama-server. Survolez un libellé pour savoir à quoi \
                         il sert ; le nombre entre parenthèses après chaque groupe indique combien \
                         de ses options sont actives.",
                        "Options handed to llama-server. Hover a label to see what it does; the \
                         number in parentheses after each group says how many of its options are \
                         active.",
                    ));
                    // Passe à la ligne si le panneau est trop étroit.
                    ui.horizontal_wrapped(|ui| {
                        if ui
                            .small_button(self.t("Défauts", "Defaults"))
                            .on_hover_text(self.t(
                                "Rétablit la valeur par défaut de chaque paramètre. Les valeurs \
                                 ajoutées aux listes par un import sont conservées.",
                                "Restores every parameter's default value. Values added to the \
                                 lists by an import are kept.",
                            ))
                            .clicked()
                        {
                            actions.push(Action::ResetDefaults);
                        }
                        if ui
                            .small_button(self.t("Tout désactiver", "Disable all"))
                            .on_hover_text(self.t(
                                "Désactive tous les paramètres d'un coup : la commande se réduit à \
                                 l'hôte, le port et le modèle. Utile pour repartir de zéro ou \
                                 isoler un paramètre fautif.",
                                "Disables every parameter at once: the command shrinks to host, \
                                 port and model. Useful to start over or isolate a faulty setting.",
                            ))
                            .clicked()
                        {
                            actions.push(Action::DisableAll);
                        }
                        if ui
                            .small_button(format!(
                                "📥  {}",
                                self.t("Importer une commande", "Import a command")
                            ))
                            .on_hover_text(self.t(
                                "Collez une commande llama-server existante pour en extraire les \
                                 paramètres. Une valeur absente d'une liste y est ajoutée plutôt \
                                 que perdue.",
                                "Paste an existing llama-server command to extract its parameters. \
                                 A value missing from a list is added to it rather than lost.",
                            ))
                            .clicked()
                        {
                            actions.push(Action::OpenImport);
                        }
                    });
                });
                hint_tip(
                    ui,
                    self.t(
                        "Listes préenregistrées ; « désactivé » retire l'option.",
                        "Preset lists; “disabled” drops the option.",
                    ),
                    self.t(
                        "Aucune valeur ne se tape : chaque paramètre n'accepte que les valeurs de \
                         sa liste. Une commande importée dont une valeur manque à la liste l'y \
                         ajoute, elle n'est donc jamais perdue.",
                        "Nothing is typed: each parameter only accepts values from its list. An \
                         imported command whose value is missing from the list adds it there, so \
                         it is never lost.",
                    ),
                );
                ui.add_space(6.0);

                let fr = self.fr();
                let fa = self.current_fa();
                let states = &mut self.states;
                let extra = &self.cfg.extra_args;

                egui::ScrollArea::vertical()
                    .id_salt("params_scroll")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        for group in Group::CENTER {
                            let count = params::PARAMS
                                .iter()
                                .enumerate()
                                .filter(|(i, d)| d.group == group && states[*i].enabled())
                                .count();
                            let title = format!("{}  ({count})", group.title(fr));
                            let group_tip = format!(
                                "{}\n\n{count} {}",
                                group.help(fr),
                                if fr {
                                    "option(s) active(s) dans ce groupe."
                                } else {
                                    "active option(s) in this group."
                                }
                            );

                            egui::CollapsingHeader::new(RichText::new(title).strong())
                                .id_salt(format!("group_{group:?}"))
                                .default_open(true)
                                .show_background(false)
                                .show(ui, |ui| {
                                    param_grid(ui, group, states, fr, fa.as_ref());
                                })
                                .header_response
                                .on_hover_text(group_tip);
                            ui.add_space(4.0);
                        }
                        if !extra.is_empty() {
                            extra_args_section(ui, extra, fr, actions);
                        }
                        ui.add_space(12.0);
                    });
            });
    }

    fn bottom_bar(&mut self, root: &mut Ui, actions: &mut Vec<Action>) {
        egui::Panel::bottom("cmd")
            .resizable(true)
            .default_size(236.0)
            .size_range(72.0..=620.0)
            .frame(
                egui::Frame::new()
                    .fill(theme::card())
                    .inner_margin(egui::Margin::symmetric(12, 8)),
            )
            .show(root, |ui| {
                let command = self.command_text(false, true);
                let lines = command.lines().count();

                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(self.t("Commande générée", "Generated command"))
                            .color(theme::text())
                            .strong()
                            .size(13.0),
                    )
                    .on_hover_text(self.t(
                        "Exactement ce qui sera exécuté au démarrage. Une option par ligne, \
                         avec les continuations ^ de cmd.exe : la commande est collable telle \
                         quelle dans un terminal. Le bord supérieur du panneau se redimensionne.",
                        "Exactly what will be executed on start. One option per line, with \
                         cmd.exe ^ continuations: the command can be pasted straight into a \
                         terminal. Drag the panel's top edge to resize it.",
                    ));
                    ui.label(
                        RichText::new(format!(
                            "{lines} {}",
                            self.t("lignes", "lines")
                        ))
                        .color(theme::muted())
                        .size(11.0),
                    );

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .small_button(self.t("Copier", "Copy"))
                            .on_hover_text(self.t(
                                "Copie la commande multi-lignes telle qu'affichée.",
                                "Copies the multi-line command as shown.",
                            ))
                            .clicked()
                        {
                            actions.push(Action::CopyCommand);
                        }
                        if ui
                            .small_button(self.t("Copier sur une ligne", "Copy as one line"))
                            .on_hover_text(self.t(
                                "Même commande sans continuation, pour PowerShell ou un script.",
                                "Same command without continuations, for PowerShell or a script.",
                            ))
                            .clicked()
                        {
                            actions.push(Action::CopyCommandOneLine);
                        }
                    });
                });

                ui.add_space(4.0);
                egui::ScrollArea::vertical()
                    .id_salt("cmd_scroll")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        // `&str` implémente `TextBuffer` en lecture seule : le
                        // texte reste sélectionnable sans être modifiable.
                        ui.add(
                            egui::TextEdit::multiline(&mut command.as_str())
                                .desired_width(f32::INFINITY)
                                .desired_rows(lines.clamp(4, 24))
                                .code_editor()
                                .text_color(theme::text())
                                .background_color(theme::code_bg()),
                        )
                        .on_hover_text(self.t(
                            "Lecture seule : les valeurs se changent dans le panneau « Paramètres \
                             de lancement ». Sélectionnez pour copier une partie.",
                            "Read-only: values are changed in the “Launch parameters” panel. \
                             Select to copy a portion.",
                        ));
                    });
            });
    }

    fn central(&mut self, root: &mut Ui, actions: &mut Vec<Action>) {
        egui::CentralPanel::default()
            .frame(panel_frame())
            .show(root, |ui| {
                ui.horizontal(|ui| {
                    for (tab, label, tip) in [
                        (
                            Tab::Console,
                            self.t("Console", "Console"),
                            self.t(
                                "Sortie en direct de llama-server : chargement du modèle, \
                                 répartition sur le GPU, requêtes servies, erreurs.",
                                "Live output from llama-server: model loading, GPU offload, \
                                 served requests, errors.",
                            ),
                        ),
                        (
                            Tab::Stats,
                            self.t("Statistiques", "Statistics"),
                            self.t(
                                "Débits de chaque requête servie depuis le démarrage du \
                                 serveur : minimum, moyenne, maximum, et leur évolution selon \
                                 le contexte.",
                                "Throughput of each request served since the server started: \
                                 minimum, average, maximum, and how they evolve with context.",
                            ),
                        ),
                        (
                            Tab::Benchmark,
                            self.t("Benchmark", "Benchmark"),
                            self.t(
                                "Mesures llama-bench pour chaque combinaison modèle × version, \
                                 afin de comparer les backends et les quantisations.",
                                "llama-bench measurements for each model × version combination, \
                                 to compare backends and quantizations.",
                            ),
                        ),
                        (
                            Tab::Models,
                            self.t("Modèles", "Models"),
                            self.t(
                                "Inventaire des modèles GGUF détectés, avec leurs métadonnées \
                                 lues dans l'en-tête du fichier.",
                                "Inventory of the detected GGUF models, with the metadata read \
                                 from each file's header.",
                            ),
                        ),
                    ] {
                        let selected = self.tab == tab;
                        let text = RichText::new(label)
                            .color(if selected { Color32::WHITE } else { theme::muted() })
                            .strong();
                        let button = egui::Button::new(text)
                            .fill(if selected { theme::accent() } else { theme::card() });
                        if ui.add(button).on_hover_text(tip).clicked() {
                            self.tab = tab;
                        }
                    }
                });
                ui.separator();

                match self.tab {
                    Tab::Console => self.console_tab(ui, actions),
                    Tab::Benchmark => self.benchmark_tab(ui, actions),
                    Tab::Models => self.models_tab(ui, actions),
                    Tab::Stats => self.stats_tab(ui, actions),
                }
            });
    }

    fn console_tab(&mut self, ui: &mut Ui, actions: &mut Vec<Action>) {
        let autoscroll_label = self.t("Défilement auto", "Auto-scroll");
        let autoscroll_tip = self.t(
            "Suit les dernières lignes. Décochez pour remonter dans l'historique sans être \
             ramené en bas à chaque nouvelle ligne.",
            "Follows the latest lines. Uncheck to scroll back through the history without being \
             dragged to the bottom on every new line.",
        );
        let count_tip = self.t(
            "Lignes conservées. Au-delà de 4000, les plus anciennes sont oubliées.",
            "Lines kept. Past 4000, the oldest ones are dropped.",
        );
        ui.horizontal(|ui| {
            if ui
                .button(self.t("Copier", "Copy"))
                .on_hover_text(self.t(
                    "Copie toute la console dans le presse-papier — pratique pour joindre un log \
                     à un rapport de bug.",
                    "Copies the whole console to the clipboard — handy to attach a log to a bug \
                     report.",
                ))
                .clicked()
            {
                actions.push(Action::CopyConsole);
            }
            if ui
                .button(self.t("Effacer", "Clear"))
                .on_hover_text(self.t(
                    "Vide l'affichage. Le serveur en cours n'est pas affecté.",
                    "Empties the display. A running server is not affected.",
                ))
                .clicked()
            {
                actions.push(Action::ClearConsole);
            }
            ui.checkbox(&mut self.autoscroll, autoscroll_label)
                .on_hover_text(autoscroll_tip);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(
                    RichText::new(format!("{} ", self.console.len()))
                        .color(theme::muted())
                        .size(11.0),
                )
                .on_hover_text(count_tip);
            });
        });
        ui.add_space(4.0);

        egui::Frame::new()
            .fill(theme::code_bg())
            .inner_margin(egui::Margin::same(8))
            .corner_radius(4)
            .show(ui, |ui| {
                // Seules les lignes visibles sont mises en page : dessiner les
                // 4000 lignes à chaque image occupait le processeur tant que le
                // serveur écrivait.
                let font = egui::FontId::monospace(12.0);
                let row_height = ui.fonts_mut(|f| f.row_height(&font));
                egui::ScrollArea::both()
                    .id_salt("console")
                    .auto_shrink([false, false])
                    .stick_to_bottom(self.autoscroll)
                    .show_rows(ui, row_height, self.console.len(), |ui, rows| {
                        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                        ui.spacing_mut().item_spacing.y = 0.0;
                        for line in self.console.range(rows) {
                            ui.label(
                                RichText::new(line)
                                    .font(font.clone())
                                    .color(line_color(line)),
                            );
                        }
                    });
            });
    }

    fn benchmark_tab(&mut self, ui: &mut Ui, actions: &mut Vec<Action>) {
        let busy = self.bench_handle.is_some();
        let fr = self.fr();
        let busy_tip = tr(fr, "Un benchmark est déjà en cours.", "A benchmark is already running.");

        // Réglages : listes fermées, comme les paramètres de lancement.
        let mut settings = self.cfg.bench.clone();
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(tr(fr, "Réglages de llama-bench", "llama-bench settings"))
                    .strong()
                    .size(12.0),
            )
            .on_hover_text(tr(
                fr,
                "Options passées à llama-bench pour chaque mesure. Chaque résultat garde les \
                 siens (colonne Config).",
                "Options handed to llama-bench for each measurement. Each result keeps its own \
                 (Config column).",
            ));
            if ui
                .add_enabled(
                    !busy && settings != bench::BenchSettings::default(),
                    egui::Button::new(tr(fr, "Défauts", "Defaults")).small(),
                )
                .on_hover_text(tr(
                    fr,
                    "NGL 999, cache f16, flash attention auto, pp512, tg128, sans profondeur, \
                     3 répétitions : les réglages des mesures d'avant leur introduction.",
                    "NGL 999, f16 cache, flash attention auto, pp512, tg128, no depth, 3 \
                     repetitions: the settings of measurements taken before they existed.",
                ))
                .on_disabled_hover_text(tr(
                    fr,
                    "Les réglages sont déjà ceux par défaut.",
                    "The settings are already the defaults.",
                ))
                .clicked()
            {
                settings = bench::BenchSettings::default();
            }
        });
        ui.add_space(2.0);
        egui::Grid::new("bench_settings")
            .num_columns(8)
            .spacing([8.0, 6.0])
            .show(ui, |ui| {
                bench_combo(
                    ui,
                    "bench_ngl",
                    "NGL",
                    tr(
                        fr,
                        "Couches du modèle placées sur le GPU (-ngl). 999 : toutes.",
                        "Model layers placed on the GPU (-ngl). 999: all of them.",
                    ),
                    &mut settings.ngl,
                    bench::NGL_CHOICES,
                    !busy,
                );
                bench_combo(
                    ui,
                    "bench_cache",
                    "Cache K/V",
                    tr(
                        fr,
                        "Type du cache KV (-ctk et -ctv). f16 : pleine précision ; q8_0 et q4_0 \
                         réduisent la mémoire, au prix d'un peu de qualité.",
                        "KV cache type (-ctk and -ctv). f16: full precision; q8_0 and q4_0 save \
                         memory at a small quality cost.",
                    ),
                    &mut settings.cache_type,
                    params::CACHE_TYPES,
                    !busy,
                );
                bench_combo(
                    ui,
                    "bench_fa",
                    "Flash attn",
                    tr(
                        fr,
                        "Flash attention (-fa). auto : llama-bench décide selon le GPU.",
                        "Flash attention (-fa). auto: llama-bench decides from the GPU.",
                    ),
                    &mut settings.flash_attn,
                    bench::FLASH_ATTN_CHOICES,
                    !busy,
                );
                bench_combo(
                    ui,
                    "bench_p",
                    "Prompt",
                    tr(
                        fr,
                        "Tokens de prompt lus pour mesurer PP (-p).",
                        "Prompt tokens read to measure PP (-p).",
                    ),
                    &mut settings.prompt,
                    bench::PROMPT_CHOICES,
                    !busy,
                );
                ui.end_row();
                bench_combo(
                    ui,
                    "bench_n",
                    tr(fr, "Génération", "Generation"),
                    tr(
                        fr,
                        "Tokens générés pour mesurer TG (-n).",
                        "Tokens generated to measure TG (-n).",
                    ),
                    &mut settings.gen,
                    bench::GEN_CHOICES,
                    !busy,
                );
                bench_combo(
                    ui,
                    "bench_d",
                    tr(fr, "Profondeur", "Depth"),
                    tr(
                        fr,
                        "Contexte déjà rempli avant la mesure (-d). Les débits baissent quand le \
                         contexte grandit : 0 mesure une conversation qui commence, 16384 une \
                         longue conversation.",
                        "Context already filled before measuring (-d). Throughput drops as the \
                         context grows: 0 measures a conversation that starts, 16384 a long one.",
                    ),
                    &mut settings.depth,
                    bench::DEPTH_CHOICES,
                    !busy,
                );
                bench_combo(
                    ui,
                    "bench_r",
                    tr(fr, "Répétitions", "Repetitions"),
                    tr(
                        fr,
                        "Mesures moyennées (-r). Le ± des résultats est leur écart-type.",
                        "Averaged runs (-r). The ± in the results is their standard deviation.",
                    ),
                    &mut settings.repetitions,
                    bench::REPETITION_CHOICES,
                    !busy,
                );
            });
        if settings != self.cfg.bench {
            actions.push(Action::SetBenchSettings(settings));
        }

        // Sélection : versions et modèles à mesurer.
        ui.add_space(6.0);
        let mut versions = self.cfg.bench_versions.clone();
        let mut models = self.cfg.bench_models.clone();
        let version_items: Vec<(String, String)> = self
            .versions
            .iter()
            .filter(|v| v.has_bench())
            .map(|v| (v.name.clone(), format!("{} · {}", v.tag(), v.name)))
            .collect();
        let model_items: Vec<(String, String)> = self
            .models
            .iter()
            .map(|m| {
                (
                    m.full_path.to_string_lossy().into_owned(),
                    format!("{} · {}", m.name, m.size_human()),
                )
            })
            .collect();
        ui.columns(2, |cols| {
            selection_list(
                &mut cols[0],
                "bench_versions",
                "Versions",
                tr(
                    fr,
                    "Versions de llama.cpp à mesurer (seules celles qui contiennent llama-bench \
                     sont listées).",
                    "llama.cpp versions to benchmark (only those containing llama-bench are \
                     listed).",
                ),
                tr(fr, "Aucune version avec llama-bench.", "No version with llama-bench."),
                &version_items,
                &mut versions,
                !busy,
                fr,
            );
            selection_list(
                &mut cols[1],
                "bench_models",
                tr(fr, "Modèles", "Models"),
                tr(
                    fr,
                    "Modèles GGUF à mesurer. Chaque modèle coché est mesuré avec chaque version \
                     cochée.",
                    "GGUF models to benchmark. Each ticked model is measured with each ticked \
                     version.",
                ),
                tr(fr, "Aucun modèle GGUF.", "No GGUF model."),
                &model_items,
                &mut models,
                !busy,
                fr,
            );
        });
        if versions != self.cfg.bench_versions || models != self.cfg.bench_models {
            actions.push(Action::SetBenchSelection { models, versions });
        }

        ui.add_space(6.0);
        let selected = self.bench_pairs(false).len();
        let missing = self.bench_pairs(true).len();
        let empty_tip = tr(
            fr,
            "Cochez au moins une version et un modèle.",
            "Tick at least one version and one model.",
        );
        let mut current_only = self.bench_current_only;
        ui.horizontal_wrapped(|ui| {
            if ui
                .add_enabled(
                    !busy && selected > 0,
                    theme::action_button(
                        RichText::new(format!(
                            "▶  {} ({selected})",
                            tr(fr, "Mesurer la sélection", "Benchmark selection")
                        ))
                        .strong(),
                        theme::green(),
                    ),
                )
                .on_hover_text(tr(
                    fr,
                    "Mesure chaque version cochée avec chaque modèle coché, avec les réglages \
                     ci-dessus. Une mesure déjà présente aux mêmes réglages est remplacée. \
                     Comptez 1 à 5 minutes par combinaison.",
                    "Benchmarks each ticked version with each ticked model, using the settings \
                     above. A measurement already present with the same settings is replaced. \
                     Expect 1 to 5 minutes per combination.",
                ))
                .on_disabled_hover_text(if busy { busy_tip } else { empty_tip })
                .clicked()
            {
                actions.push(Action::BenchRun(false));
            }
            if ui
                .add_enabled(
                    !busy && missing > 0,
                    egui::Button::new(format!(
                        "{} ({missing})",
                        tr(fr, "Seulement les manquants", "Missing only")
                    )),
                )
                .on_hover_text(tr(
                    fr,
                    "Parmi la sélection, ne mesure que les combinaisons absentes de \
                     benchmark.md avec ces réglages. Le choix habituel après l'ajout d'un \
                     modèle ou d'une version.",
                    "Within the selection, only benchmarks combinations missing from \
                     benchmark.md with these settings. The usual choice after adding a model \
                     or a version.",
                ))
                .on_disabled_hover_text(if busy {
                    busy_tip
                } else if selected == 0 {
                    empty_tip
                } else {
                    tr(
                        fr,
                        "Toute la sélection est déjà mesurée avec ces réglages.",
                        "The whole selection is already measured with these settings.",
                    )
                })
                .clicked()
            {
                actions.push(Action::BenchRun(true));
            }
            if ui
                .add_enabled(
                    busy,
                    theme::action_button(RichText::new(tr(fr, "Interrompre", "Cancel")), theme::red()),
                )
                .on_hover_text(tr(
                    fr,
                    "Tue llama-bench et abandonne les combinaisons restantes. Les mesures déjà \
                     obtenues sont conservées.",
                    "Kills llama-bench and drops the remaining combinations. Measurements already \
                     obtained are kept.",
                ))
                .on_disabled_hover_text(tr(fr, "Aucun benchmark en cours.", "No benchmark running."))
                .clicked()
            {
                actions.push(Action::BenchStop);
            }
            if ui
                .add_enabled(!self.bench_rows.is_empty(), egui::Button::new("benchmark.md"))
                .on_hover_text(tr(
                    fr,
                    "Réécrit benchmark.md à côté de l'exécutable, trié par débit de prompt \
                     décroissant. L'écriture est déjà automatique en fin de benchmark.",
                    "Rewrites benchmark.md next to the executable, sorted by descending prompt \
                     throughput. This is already done automatically when a benchmark ends.",
                ))
                .on_disabled_hover_text(tr(fr, "Aucun résultat à écrire.", "No results to write."))
                .clicked()
            {
                actions.push(Action::SaveBench);
            }
            if self.bench_confirm_clear {
                ui.label(
                    RichText::new(tr(fr, "Supprimer tous les résultats ?", "Delete every result?"))
                        .color(theme::red())
                        .size(12.0),
                );
                if ui
                    .add(theme::action_button(
                        RichText::new(tr(fr, "Supprimer", "Delete")),
                        theme::red(),
                    ))
                    .on_hover_text(tr(
                        fr,
                        "Vide la liste et benchmark.md. Irréversible.",
                        "Empties the list and benchmark.md. Cannot be undone.",
                    ))
                    .clicked()
                {
                    actions.push(Action::BenchDelete(None));
                }
                if ui.button(tr(fr, "Annuler", "Cancel")).clicked() {
                    actions.push(Action::BenchConfirmClear(false));
                }
            } else if ui
                .add_enabled(
                    !busy && !self.bench_rows.is_empty(),
                    egui::Button::new(format!("🗑  {}", tr(fr, "Tout supprimer", "Delete all"))),
                )
                .on_hover_text(tr(
                    fr,
                    "Supprime tous les résultats, après confirmation. Pour une seule ligne, \
                     utilisez la corbeille au bout de la ligne.",
                    "Deletes every result, after confirmation. For a single row, use the bin \
                     at the end of the row.",
                ))
                .on_disabled_hover_text(if busy {
                    busy_tip
                } else {
                    tr(fr, "Aucun résultat.", "No results.")
                })
                .clicked()
            {
                actions.push(Action::BenchConfirmClear(true));
            }
            ui.checkbox(
                &mut current_only,
                tr(fr, "Réglages actuels seulement", "Current settings only"),
            )
            .on_hover_text(tr(
                fr,
                "N'affiche que les résultats mesurés avec les réglages ci-dessus : les seuls \
                 directement comparables entre eux.",
                "Only shows results measured with the settings above: the only ones directly \
                 comparable with each other.",
            ));
        });
        self.bench_current_only = current_only;

        if busy {
            let (done, total) = self.bench_progress;
            let fraction = if total == 0 { 0.0 } else { done as f32 / total as f32 };
            ui.add_space(6.0);
            let label = format!("{done}/{total}  {}", self.bench_label);
            bar(ui, ui.available_width(), 20.0, fraction, theme::accent(), &label).on_hover_text(tr(
                fr,
                "Combinaisons terminées sur le total. Le détail de chaque exécution défile dans \
                 l'onglet Console.",
                "Completed combinations out of the total. Each run's detail scrolls in the \
                 Console tab.",
            ));
        }
        ui.add_space(6.0);

        let current = self.cfg.bench.label();
        let mut rows: Vec<BenchRow> = self
            .bench_rows
            .iter()
            .filter(|r| !self.bench_current_only || r.config == current)
            .cloned()
            .collect();
        rows.sort_by(|a, b| {
            b.pp_value()
                .partial_cmp(&a.pp_value())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        egui::ScrollArea::both()
            .id_salt("bench_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                egui::Grid::new("bench_grid")
                    .num_columns(10)
                    .striped(true)
                    .spacing([14.0, 5.0])
                    .show(ui, |ui| {
                        for (header, tip) in [
                            (
                                self.t("Modèle", "Model"),
                                self.t(
                                    "Nom du modèle, privé de sa quantisation.",
                                    "Model name, with its quantization stripped off.",
                                ),
                            ),
                            (
                                "Quant",
                                self.t(
                                    "Quantisation, déduite du nom de fichier. Plus les bits sont \
                                     nombreux, meilleure est la qualité et plus le modèle est lent \
                                     et gros.",
                                    "Quantization, inferred from the file name. More bits means \
                                     better quality, and a slower, bigger model.",
                                ),
                            ),
                            (
                                "Version",
                                self.t(
                                    "Build de llama.cpp qui a produit la mesure.",
                                    "The llama.cpp build that produced the measurement.",
                                ),
                            ),
                            (
                                "Backend",
                                self.t(
                                    "Moteur de calcul utilisé (Vulkan, CUDA, CPU…), rapporté par \
                                     llama-bench.",
                                    "Compute backend used (Vulkan, CUDA, CPU…), as reported by \
                                     llama-bench.",
                                ),
                            ),
                            (
                                "Size",
                                self.t(
                                    "Taille du modèle en mémoire.",
                                    "Model size in memory.",
                                ),
                            ),
                            (
                                "Params",
                                self.t(
                                    "Nombre de paramètres du modèle.",
                                    "Number of model parameters.",
                                ),
                            ),
                            (
                                "PP (t/s)",
                                self.t(
                                    "Prompt processing : tokens lus par seconde. Détermine le délai \
                                     avant le premier mot sur un long prompt. Le ± est l'écart-type \
                                     sur les 3 répétitions. Le tableau est trié sur cette colonne.",
                                    "Prompt processing: tokens read per second. Drives the delay \
                                     before the first word on a long prompt. The ± is the standard \
                                     deviation over the 3 repetitions. The table is sorted on this \
                                     column.",
                                ),
                            ),
                            (
                                "TG (t/s)",
                                self.t(
                                    "Token generation : tokens écrits par seconde. C'est la vitesse \
                                     que ressent l'utilisateur pendant la réponse.",
                                    "Token generation: tokens written per second. This is the speed \
                                     the user feels while the answer streams.",
                                ),
                            ),
                            (
                                "Config",
                                self.t(
                                    "Réglages de la mesure : couches GPU, cache K/V, flash \
                                     attention, tokens de prompt et générés, répétitions, et \
                                     profondeur de contexte (d…) s'il y en a une. Seules les \
                                     lignes aux mêmes réglages se comparent.",
                                    "Measurement settings: GPU layers, K/V cache, flash \
                                     attention, prompt and generated tokens, repetitions, and \
                                     context depth (d…) if any. Only rows with the same settings \
                                     compare.",
                                ),
                            ),
                        ] {
                            ui.label(RichText::new(header).strong().color(theme::accent()))
                                .on_hover_text(tip);
                        }
                        ui.label("");
                        ui.end_row();

                        for r in &rows {
                            ui.label(RichText::new(&r.model).monospace().size(12.0));
                            ui.label(RichText::new(&r.quant).monospace().size(12.0));
                            ui.label(RichText::new(&r.version).monospace().size(12.0));
                            ui.label(RichText::new(&r.backend).monospace().size(12.0));
                            ui.label(RichText::new(&r.size).monospace().size(12.0));
                            ui.label(RichText::new(&r.params).monospace().size(12.0));
                            ui.label(
                                RichText::new(&r.pp)
                                    .monospace()
                                    .size(12.0)
                                    .color(theme::green()),
                            );
                            ui.label(
                                RichText::new(&r.tg)
                                    .monospace()
                                    .size(12.0)
                                    .color(theme::yellow()),
                            );
                            let same = r.config == current;
                            ui.label(RichText::new(&r.config).size(11.0).color(if same {
                                theme::text()
                            } else {
                                theme::muted()
                            }));
                            if ui
                                .add_enabled(!busy, egui::Button::new("🗑").small())
                                .on_hover_text(tr(
                                    fr,
                                    "Supprime cette mesure de la liste et de benchmark.md.",
                                    "Deletes this measurement from the list and benchmark.md.",
                                ))
                                .clicked()
                            {
                                actions.push(Action::BenchDelete(Some(r.key())));
                            }
                            ui.end_row();
                        }
                    });

                if rows.is_empty() {
                    ui.add_space(12.0);
                    hint(
                        ui,
                        self.t(
                            "Aucun résultat. Cochez des versions et des modèles, puis lancez la \
                             mesure.",
                            "No results yet. Tick versions and models, then run the benchmark.",
                        ),
                    );
                }
            });
    }

    fn stats_tab(&mut self, ui: &mut Ui, actions: &mut Vec<Action>) {
        let fr = self.fr();
        ui.horizontal(|ui| {
            for (view, label, tip) in [
                (
                    StatsView::Throughput,
                    tr(fr, "Tokens", "Tokens"),
                    tr(
                        fr,
                        "Vitesse de lecture du prompt et de génération, requête par requête.",
                        "Prompt reading and generation speed, request by request.",
                    ),
                ),
                (
                    StatsView::Power,
                    tr(fr, "Consommation GPU", "GPU power"),
                    tr(
                        fr,
                        "Puissance tirée par la carte graphique, énergie consommée et son coût.",
                        "Power drawn by the graphics card, energy used and its cost.",
                    ),
                ),
            ] {
                if ui
                    .selectable_label(self.stats_view == view, label)
                    .on_hover_text(tip)
                    .clicked()
                {
                    self.stats_view = view;
                }
            }
        });
        ui.add_space(4.0);
        match self.stats_view {
            StatsView::Throughput => self.throughput_view(ui, actions, fr),
            StatsView::Power => self.power_view(ui, actions, fr),
        }
    }

    fn throughput_view(&mut self, ui: &mut Ui, actions: &mut Vec<Action>, fr: bool) {
        let n = self.stats.requests.len();
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(format!("{} {}", spaced(n as f64, 0, fr), tr(fr, "requête(s)", "request(s)")))
                    .strong()
                    .color(theme::text()),
            );
            ui.label(
                RichText::new(format!(
                    "· {} {} · {} {}",
                    spaced(self.stats.prompt_tokens() as f64, 0, fr),
                    tr(fr, "tokens de prompt", "prompt tokens"),
                    spaced(self.stats.gen_tokens() as f64, 0, fr),
                    tr(fr, "tokens générés", "generated tokens"),
                ))
                .color(theme::muted())
                .size(12.0),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .add_enabled(n > 0, egui::Button::new(tr(fr, "Réinitialiser", "Reset")))
                    .on_hover_text(tr(
                        fr,
                        "Efface les statistiques, par exemple après avoir changé un réglage \
                         sans redémarrer. Elles repartent aussi de zéro à chaque démarrage du \
                         serveur.",
                        "Clears the statistics, for instance after changing a setting without \
                         restarting. They also start over each time the server starts.",
                    ))
                    .clicked()
                {
                    actions.push(Action::ResetStats);
                }
            });
        });

        if n == 0 {
            ui.add_space(8.0);
            hint(
                ui,
                tr(
                    fr,
                    "Aucune requête depuis le démarrage du serveur. Les débits de chaque \
                     réponse (lecture du prompt et génération) apparaîtront ici.",
                    "No request since the server started. The throughput of each answer \
                     (prompt reading and generation) will show up here.",
                ),
            );
            return;
        }

        ui.add_space(6.0);
        let weighted_tip = tr(
            fr,
            "Total des tokens divisé par le temps total. Une petite requête (quelques tokens \
             de prompt déjà en cache) pèse selon ses tokens : elle ne fait plus chuter la \
             moyenne comme dans une moyenne simple des débits.",
            "Total tokens divided by total time. A small request (a few prompt tokens already \
             cached) weighs by its tokens: it no longer drags the average down as in a plain \
             average of the rates.",
        );
        let spread_tip = tr(
            fr,
            "Sur toutes les requêtes depuis le démarrage (ou la dernière réinitialisation). \
             Minimum, médiane et maximum portent sur les débits de chaque requête, quelle que \
             soit sa taille.",
            "Over every request since the start (or the last reset). Minimum, median and \
             maximum are over each request's rate, whatever its size.",
        );
        egui::Grid::new("stats_grid")
            .num_columns(6)
            .striped(true)
            .spacing([22.0, 5.0])
            .show(ui, |ui| {
                ui.label("");
                for (h, tip) in [
                    (tr(fr, "Min", "Min"), spread_tip),
                    (tr(fr, "Moyenne pondérée", "Weighted average"), weighted_tip),
                    (tr(fr, "Médiane", "Median"), spread_tip),
                    (tr(fr, "Max", "Max"), spread_tip),
                    (tr(fr, "Dernière", "Last"), spread_tip),
                ] {
                    ui.label(RichText::new(h).strong().color(theme::accent()))
                        .on_hover_text(tip);
                }
                ui.end_row();
                let rows: [(&str, &str, Option<Spread>, Color32); 2] = [
                    (
                        "PP (t/s)",
                        tr(
                            fr,
                            "Prompt processing : tokens du prompt lus par seconde.",
                            "Prompt processing: prompt tokens read per second.",
                        ),
                        self.stats.prompt(),
                        theme::green(),
                    ),
                    (
                        "TG (t/s)",
                        tr(
                            fr,
                            "Token generation : tokens de la réponse écrits par seconde.",
                            "Token generation: answer tokens written per second.",
                        ),
                        self.stats.generation(),
                        theme::yellow(),
                    ),
                ];
                for (name, tip, spread, color) in rows {
                    ui.label(RichText::new(name).strong()).on_hover_text(tip);
                    match spread {
                        Some(s) => {
                            for v in [s.min, s.weighted, s.median, s.max, s.last] {
                                ui.label(
                                    RichText::new(spaced(v, 1, fr))
                                        .monospace()
                                        .size(12.0)
                                        .color(color),
                                );
                            }
                        }
                        None => {
                            ui.label(RichText::new("-").color(theme::muted()));
                        }
                    }
                    ui.end_row();
                }
            });

        if let Some((accepted, generated)) = self.stats.draft() {
            ui.add_space(4.0);
            ui.label(
                RichText::new(format!(
                    "{} {} % ({} / {})",
                    tr(fr, "Brouillon accepté :", "Draft accepted:"),
                    spaced(accepted as f64 * 100.0 / generated as f64, 1, fr),
                    spaced(accepted as f64, 0, fr),
                    spaced(generated as f64, 0, fr),
                ))
                .size(12.0),
            )
            .on_hover_text(tr(
                fr,
                "Décodage spéculatif : part des tokens proposés par le brouillon que le modèle \
                 principal a gardés. Plus elle est haute, plus la génération est accélérée.",
                "Speculative decoding: share of the draft's proposed tokens that the main model \
                 kept. The higher it is, the more generation is sped up.",
            ));
        }

        ui.add_space(8.0);
        let detail = |r: &crate::stats::Request| {
            let mut s = format!(
                "{} {} tokens\n{} {} tokens ({} ms)",
                tr(fr, "Contexte :", "Context:"),
                spaced(r.context.unwrap_or(0) as f64, 0, fr),
                tr(fr, "Prompt :", "Prompt:"),
                spaced(r.prompt_tokens as f64, 0, fr),
                spaced(r.prompt_ms, 0, fr),
            );
            s.push_str(&format!(
                "\n{} {} tokens ({} ms)",
                tr(fr, "Générés :", "Generated:"),
                spaced(r.gen_tokens as f64, 0, fr),
                spaced(r.gen_ms, 0, fr),
            ));
            if let Some(v) = r.prompt_tps {
                s.push_str(&format!("\nPP {} t/s", spaced(v, 1, fr)));
            }
            if let Some(v) = r.gen_tps {
                s.push_str(&format!("\nTG {} t/s", spaced(v, 1, fr)));
            }
            s
        };
        let tg: Vec<([f64; 2], String)> = self
            .stats
            .by_context(|r| r.gen_tps)
            .into_iter()
            .map(|(p, r)| (p, detail(r)))
            .collect();
        let pp: Vec<([f64; 2], String)> = self
            .stats
            .by_context(|r| r.prompt_tps)
            .into_iter()
            .map(|(p, r)| (p, detail(r)))
            .collect();
        if tg.is_empty() && pp.is_empty() {
            hint(
                ui,
                tr(
                    fr,
                    "Graphiques indisponibles : cette version de llama-server n'indique pas la \
                     taille du contexte des requêtes.",
                    "Charts unavailable: this llama-server version does not report the requests' \
                     context size.",
                ),
            );
            return;
        }
        let height = ((ui.available_height() - 60.0) / 2.0).clamp(120.0, 280.0);
        context_plot(ui, "stats_tg", "TG (t/s)", tg, theme::yellow(), height, fr);
        ui.add_space(6.0);
        context_plot(ui, "stats_pp", "PP (t/s)", pp, theme::green(), height, fr);
    }

    fn power_view(&mut self, ui: &mut Ui, actions: &mut Vec<Action>, fr: bool) {
        if self.power.samples.is_empty() {
            hint(
                ui,
                tr(
                    fr,
                    "Aucun relevé : la consommation se lit avec nvidia-smi, sur les cartes \
                     NVIDIA uniquement.",
                    "No reading: power is read through nvidia-smi, on NVIDIA cards only.",
                ),
            );
            return;
        }

        // Prix du kWh : un nombre borné, pas un texte libre.
        let mut price = self.cfg.kwh_price;
        let mut currency = self.cfg.currency.clone();
        ui.horizontal(|ui| {
            field_label(
                ui,
                tr(fr, "Prix du kWh", "Price per kWh"),
                tr(
                    fr,
                    "Tarif de votre électricité, pour chiffrer l'énergie consommée. Glissez ou \
                     double-cliquez pour saisir la valeur.",
                    "Your electricity rate, to price the energy used. Drag or double-click to \
                     type the value.",
                ),
            );
            ui.add(
                egui::DragValue::new(&mut price)
                    .range(0.0..=10.0)
                    .speed(0.001)
                    .fixed_decimals(4),
            );
            ComboBox::from_id_salt("currency")
                .width(56.0)
                .selected_text(currency.as_str())
                .show_ui(ui, |ui| {
                    for c in CURRENCIES {
                        ui.selectable_value(&mut currency, c.to_string(), *c);
                    }
                });
        });
        if price != self.cfg.kwh_price || currency != self.cfg.currency {
            actions.push(Action::SetEnergyPrice(price, currency.clone()));
        }

        ui.add_space(6.0);
        let tokens = self.stats.gen_tokens();
        let since = if self.cfg.energy_since.is_empty() {
            "-".to_string()
        } else {
            self.cfg.energy_since.clone()
        };
        egui::Grid::new("power_grid")
            .num_columns(4)
            .striped(true)
            .spacing([22.0, 5.0])
            .show(ui, |ui| {
                ui.label("");
                for h in [tr(fr, "Énergie", "Energy"), tr(fr, "Coût", "Cost"), ""] {
                    ui.label(RichText::new(h).strong().color(theme::accent()));
                }
                ui.end_row();
                let rows = [
                    (
                        tr(fr, "Depuis le démarrage du serveur", "Since the server started"),
                        tr(
                            fr,
                            "Énergie consommée par la carte pendant que le serveur tourne, \
                             depuis son dernier démarrage.",
                            "Energy the card used while the server runs, since its last start.",
                        ),
                        self.power.server_wh,
                        if tokens > 0 {
                            format!(
                                "{} / 1000 tokens",
                                power::energy(self.power.server_wh * 1000.0 / tokens as f64, fr)
                            )
                        } else {
                            String::new()
                        },
                    ),
                    (
                        tr(fr, "Depuis l'ouverture de l'application", "Since the app opened"),
                        tr(
                            fr,
                            "Toute la consommation de la carte depuis l'ouverture de \
                             l'application (ou la réinitialisation de la courbe), serveur \
                             arrêté compris.",
                            "All the card's consumption since the app opened (or the curve \
                             was reset), server stopped included.",
                        ),
                        self.power.session_wh,
                        String::new(),
                    ),
                    (
                        tr(fr, "Total cumulé", "Running total"),
                        tr(
                            fr,
                            "Cumul conservé d'une session à l'autre, depuis la date indiquée.",
                            "Total kept from one session to the next, since the date shown.",
                        ),
                        self.energy_total_wh,
                        format!("{} {since}", tr(fr, "depuis le", "since")),
                    ),
                ];
                for (label, tip, wh, extra) in rows {
                    ui.label(label).on_hover_text(tip);
                    ui.label(RichText::new(power::energy(wh, fr)).monospace().size(12.0));
                    ui.label(
                        RichText::new(power::cost(wh, self.cfg.kwh_price, &self.cfg.currency, fr))
                            .monospace()
                            .size(12.0)
                            .color(theme::yellow()),
                    );
                    ui.label(RichText::new(extra).color(theme::muted()).size(11.0));
                    ui.end_row();
                }
            });

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if let Some((min, mean, max)) = self.power.range() {
                ui.label(
                    RichText::new(format!(
                        "{} {} W · {} {} W · {} {} W",
                        tr(fr, "min", "min"),
                        spaced(min, 0, fr),
                        tr(fr, "moyenne", "average"),
                        spaced(mean, 0, fr),
                        tr(fr, "max", "max"),
                        spaced(max, 0, fr),
                    ))
                    .color(theme::muted())
                    .size(12.0),
                );
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .button(tr(fr, "Remettre le total à zéro", "Reset the total"))
                    .on_hover_text(tr(
                        fr,
                        "Remet à zéro le total cumulé, qui repart d'aujourd'hui.",
                        "Resets the running total, which starts again from today.",
                    ))
                    .clicked()
                {
                    actions.push(Action::ResetEnergyTotal);
                }
                if ui
                    .button(tr(fr, "Effacer la courbe", "Clear the curve"))
                    .on_hover_text(tr(
                        fr,
                        "Vide la courbe et le compteur depuis l'ouverture.",
                        "Empties the curve and the since-opening counter.",
                    ))
                    .clicked()
                {
                    actions.push(Action::ResetPowerCurve);
                }
            });
        });

        ui.add_space(6.0);
        let origin = self.power.origin_wall();
        let limit = self
            .gpus
            .iter()
            .filter_map(|g| g.power_limit_w)
            .map(f64::from)
            .sum::<f64>();
        let color = theme::orange();
        let points = self.power.decimated(2000);
        ui.label(RichText::new(tr(fr, "Puissance (W)", "Power (W)")).strong().size(12.0).color(color))
            .on_hover_text(tr(
                fr,
                "Relevé toutes les deux secondes, 24 h au plus. Molette + Ctrl pour zoomer, \
                 double clic pour revenir.",
                "Read every two seconds, 24 h at most. Ctrl + wheel to zoom, double click to \
                 reset.",
            ));
        let mut plot = egui_plot::Plot::new("power_plot")
            .height(ui.available_height().clamp(140.0, 400.0) - 10.0)
            .include_y(0.0)
            .allow_scroll(false)
            .y_axis_min_width(44.0)
            .y_grid_spacer(egui_plot::uniform_grid_spacer(|input| {
                let step = nice_step((input.bounds.1 - input.bounds.0) / 4.0);
                [step / 5.0, step, step * 5.0]
            }))
            // Graduations en minutes rondes : les puissances de 10 d'egui_plot
            // donneraient des pas de 10 ou 100 s, affichés « 14:05 » en double.
            .x_grid_spacer(egui_plot::uniform_grid_spacer(|input| {
                let span = input.bounds.1 - input.bounds.0;
                let step = [60.0, 120.0, 300.0, 600.0, 900.0, 1800.0, 3600.0, 7200.0, 14400.0]
                    .into_iter()
                    .find(|s| span / s <= 8.0)
                    .unwrap_or(21600.0);
                [step / 2.0, step, step * 4.0]
            }))
            .x_axis_formatter(move |mark, _| power::clock(origin, mark.value))
            .label_formatter(move |pos| {
                let p = match pos {
                    egui_plot::HoverPosition::NearDataPoint { position, .. } => position,
                    egui_plot::HoverPosition::Elsewhere { position } => position,
                };
                Some(format!("{} · {:.0} W", power::clock(origin, p.x), p.y))
            });
        if limit > 0.0 {
            plot = plot.include_y(limit);
        }
        plot.show(ui, |plot| {
            if limit > 0.0 {
                plot.hline(
                    egui_plot::HLine::new(tr(fr, "limite", "limit"), limit)
                        .color(theme::muted())
                        .width(1.0),
                );
            }
            plot.line(
                egui_plot::Line::new(tr(fr, "Puissance", "Power"), points)
                    .color(color)
                    .width(1.5)
                    .fill(0.0)
                    .fill_alpha(0.15),
            );
        });
    }

    fn models_tab(&mut self, ui: &mut Ui, actions: &mut Vec<Action>) {
        // Lecture des en-têtes en cours (premier lancement, nouveaux modèles).
        if self.meta_rx.is_some() {
            let (done, total) = self.meta_progress;
            ui.horizontal(|ui| {
                ui.add(egui::Spinner::new().size(14.0));
                ui.label(
                    RichText::new(format!(
                        "{} {done} / {total}",
                        self.t("Lecture des en-têtes GGUF…", "Reading GGUF headers…")
                    ))
                    .color(theme::muted())
                    .size(12.0),
                )
                .on_hover_text(self.t(
                    "L'en-tête de chaque nouveau modèle est lu en tâche de fond puis mémorisé \
                     (gguf-cache.json) : les colonnes Archi, Ctx max et Couches se remplissent \
                     au fil de la lecture.",
                    "Each new model's header is read in the background then remembered \
                     (gguf-cache.json): the Arch, Max ctx and Layers columns fill in as it goes.",
                ));
            });
        }
        ui.add_space(6.0);
        self.hub_row(ui, actions);
        ui.add_space(8.0);

        if self.models.is_empty() {
            ui.add_space(12.0);
            hint(
                ui,
                self.t(
                    "Choisissez un répertoire de modèles à gauche.",
                    "Pick a models directory on the left.",
                ),
            );
            return;
        }

        // Tableau virtualisé : seules les lignes visibles sont dessinées, même
        // avec des centaines de modèles. Pas de copie de la liste à chaque
        // image : on emprunte modèles et en-têtes.
        let fr = self.fr();
        let models = &self.models;
        let meta = &self.meta;
        let mut delete_request = None;
        let headers = [
            (
                tr(fr, "Nom", "Name"),
                tr(
                    fr,
                    "Nom du fichier sans son extension. Survolez une ligne pour le chemin complet \
                     et les métadonnées.",
                    "File name without its extension. Hover a row for the full path and the \
                     metadata.",
                ),
            ),
            ("Quant", tr(fr, "Quantisation déduite du nom de fichier.", "Quantization inferred from the file name.")),
            (
                tr(fr, "Taille", "Size"),
                tr(
                    fr,
                    "Taille sur le disque (toutes les parties d'un modèle scindé). Prévoyez un peu \
                     plus de VRAM que cette valeur, le cache KV s'y ajoute.",
                    "Size on disk (every part of a split model). Budget a bit more VRAM than this, \
                     the KV cache adds to it.",
                ),
            ),
            (
                tr(fr, "Archi", "Arch"),
                tr(
                    fr,
                    "Architecture déclarée dans l'en-tête GGUF (qwen3, llama, gemma3…), lue \
                     automatiquement.",
                    "Architecture declared in the GGUF header (qwen3, llama, gemma3…), read \
                     automatically.",
                ),
            ),
            (
                tr(fr, "Ctx max", "Max ctx"),
                tr(
                    fr,
                    "Contexte maximal pour lequel le modèle a été entraîné. Demander davantage via \
                     --ctx-size dégrade la qualité.",
                    "Largest context the model was trained for. Asking for more via --ctx-size \
                     degrades quality.",
                ),
            ),
            (
                tr(fr, "Couches", "Layers"),
                tr(
                    fr,
                    "Nombre de couches du modèle : c'est le maximum utile pour -ngl si vous ne \
                     voulez pas tout décharger sur le GPU.",
                    "Number of layers in the model: the useful maximum for -ngl if you do not \
                     want to offload everything to the GPU.",
                ),
            ),
            ("", ""),
        ];
        let mono = |text: String| RichText::new(text).monospace().size(12.0);

        // Écart entre colonnes, comme l'ancienne grille ; le nom prend la
        // place restante, les boutons une largeur fixe (sinon rognés).
        ui.spacing_mut().item_spacing.x = 14.0;
        egui_extras::TableBuilder::new(ui)
            .id_salt("models_table")
            .striped(true)
            .auto_shrink([false, false])
            .cell_layout(egui::Layout::left_to_right(egui::Align::Center))
            .column(egui_extras::Column::remainder().at_least(140.0).clip(true))
            .columns(egui_extras::Column::auto().at_least(40.0), 5)
            .column(egui_extras::Column::exact(66.0))
            .header(22.0, |mut header| {
                for (title, tip) in headers {
                    header.col(|ui| {
                        let label = ui.label(RichText::new(title).strong().color(theme::accent()));
                        if !tip.is_empty() {
                            label.on_hover_text(tip);
                        }
                    });
                }
            })
            .body(|body| {
                body.rows(24.0, models.len(), |mut row| {
                    let m = &models[row.index()];
                    let g = meta.get(&m.full_path).map(|c| &c.meta);
                    row.col(|ui| {
                        let mut tip = m.full_path.to_string_lossy().into_owned();
                        if let Some(Some(g)) = g {
                            if !g.name.is_empty() {
                                tip.push_str(&format!("\n{}", g.name));
                            }
                            if !g.size_label.is_empty() {
                                tip.push_str(&format!("\n{}", g.size_label));
                            }
                            tip.push_str(&format!("\n{} {}", g.tensor_count, tr(fr, "tenseurs", "tensors")));
                            if let Some(d) = g.embedding_length {
                                tip.push_str(&format!("\nembedding {d}"));
                            }
                            tip.push('\n');
                            tip.push_str(if g.has_chat_template {
                                tr(fr, "template de chat intégré", "built-in chat template")
                            } else {
                                tr(fr, "sans template de chat", "no chat template")
                            });
                        }
                        ui.label(mono(m.name.clone())).on_hover_text(tip);
                    });
                    row.col(|ui| {
                        ui.label(mono(m.quant.clone()));
                    });
                    row.col(|ui| {
                        ui.label(mono(m.size_human()));
                    });
                    let (arch, ctx_len, layers) = match g {
                        Some(Some(g)) => (
                            if g.architecture.is_empty() {
                                "-".to_string()
                            } else {
                                g.architecture.clone()
                            },
                            g.context_length.map(|c| c.to_string()).unwrap_or_else(|| "-".into()),
                            g.block_count.map(|c| c.to_string()).unwrap_or_else(|| "-".into()),
                        ),
                        Some(None) => ("?".into(), "?".into(), "?".into()),
                        None => ("…".into(), "…".into(), "…".into()),
                    };
                    for text in [arch, ctx_len, layers] {
                        row.col(|ui| {
                            ui.label(mono(text));
                        });
                    }
                    row.col(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        if ui
                            .small_button("📂")
                            .on_hover_text(tr(fr, "Ouvrir le dossier", "Open folder"))
                            .clicked()
                        {
                            actions.push(Action::RevealModel(m.full_path.clone()));
                        }
                        if ui
                            .small_button(RichText::new("🗑").color(theme::red()))
                            .on_hover_text(tr(fr, "Supprimer le fichier", "Delete the file"))
                            .clicked()
                        {
                            delete_request = Some(m.full_path.clone());
                        }
                    });
                });
            });
        if let Some(path) = delete_request {
            self.confirm_delete = Some(path);
        }
    }

    /// Hugging Face : tri, dépôt, fichier, lancement. Listes lues sur l'API,
    /// chargées à la première ouverture de l'onglet.
    fn hub_row(&mut self, ui: &mut Ui, actions: &mut Vec<Action>) {
        let fr = self.fr();
        if self.hub_repos.is_none() && self.hub_repos_rx.is_none() {
            actions.push(Action::HubLoad);
        }
        // Deux lignes explicites : avec un retour à la ligne automatique, la
        // liste des fichiers était poussée hors de l'écran par celle des dépôts.
        ui.horizontal(|ui| {
            ui.label(RichText::new("Hugging Face").color(theme::muted()).size(12.0))
                .on_hover_text(tr(
                    fr,
                    "Dépôts GGUF de modèles de langage (texte et vision), lus en direct sur \
                     Hugging Face. Les modèles d'image, d'audio ou de vidéo, que llama-server ne \
                     sait pas servir, sont écartés. Le fichier choisi est téléchargé dans le \
                     répertoire des modèles.",
                    "GGUF language-model repositories (text and vision), read live from Hugging \
                     Face. Image, audio and video models, which llama-server cannot serve, are \
                     left out.",
                ));

            // Tri
            let mut sort = self.hub_sort;
            ComboBox::from_id_salt("hub_sort")
                .width(130.0)
                .selected_text(sort.label(fr))
                .show_ui(ui, |ui| {
                    for s in hub::Sort::ALL {
                        ui.selectable_value(&mut sort, s, s.label(fr));
                    }
                })
                .response
                .on_hover_text(tr(
                    fr,
                    "Tendance : les plus en vue ces derniers jours. Plus téléchargés : les \
                     plus utilisés sur le dernier mois.",
                    "Trending: the most noticed in recent days. Most downloaded: the most used \
                     over the last month.",
                ));
            if sort != self.hub_sort {
                actions.push(Action::HubSort(sort));
            }

            // Dépôt
            match &self.hub_repos {
                None => {
                    ui.add(egui::Spinner::new());
                    ui.label(RichText::new(tr(fr, "chargement…", "loading…")).color(theme::muted()));
                }
                Some(Err(e)) => {
                    ui.label(
                        RichText::new(tr(fr, "Hugging Face injoignable", "Hugging Face unreachable"))
                            .color(theme::red()),
                    )
                    .on_hover_text(e.as_str());
                    if ui.button(tr(fr, "Réessayer", "Retry")).clicked() {
                        actions.push(Action::HubLoad);
                    }
                }
                Some(Ok(repos)) => {
                    let selected = repos.iter().find(|r| r.id == self.hub_repo);
                    ComboBox::from_id_salt("hub_repo")
                        .width(ui.available_width().clamp(160.0, 460.0))
                        .height(440.0)
                        .selected_text(selected.map(|r| r.id.clone()).unwrap_or_default())
                        .show_ui(ui, |ui| {
                            ui.set_min_width(460.0);
                            for r in repos {
                                let mut text = format!(
                                    "{}  ·  {} {}",
                                    r.id,
                                    hub::compact_count(r.downloads),
                                    tr(fr, "téléch.", "dl")
                                );
                                if r.vision {
                                    text.push_str(tr(fr, "  ·  vision", "  ·  vision"));
                                }
                                if ui.selectable_label(r.id == self.hub_repo, text).clicked() {
                                    actions.push(Action::HubPickRepo(r.id.clone()));
                                }
                            }
                        })
                        .response
                        .on_hover_text(tr(
                            fr,
                            "Téléchargements sur le dernier mois. « vision » : modèle qui lit \
                             aussi les images ; son projecteur est téléchargé avec lui.",
                            "Downloads over the last month. “vision”: a model that also reads \
                             images; its projector is downloaded with it.",
                        ));
                }
            }

        });

        // 2e ligne : fichier, rechargement, lancement.
        ui.horizontal(|ui| {
            ui.label(RichText::new(tr(fr, "Fichier", "File")).color(theme::muted()).size(12.0));
            match (&self.hub_files, self.hub_files_rx.is_some()) {
                (_, true) => {
                    ui.add(egui::Spinner::new());
                }
                (Some(Err(e)), _) => {
                    ui.label(RichText::new(tr(fr, "fichiers illisibles", "files unreadable")).color(theme::red()))
                        .on_hover_text(e.as_str());
                }
                (Some(Ok(files)), _) if files.is_empty() => {
                    ui.label(RichText::new(tr(fr, "aucun GGUF", "no GGUF")).color(theme::muted()));
                }
                (Some(Ok(files)), _) => {
                    let label = |f: &hub::HubFile| {
                        let mut t = f.quant.clone();
                        if !f.variant.is_empty() {
                            t.push_str(&format!("  ·  {}", f.variant));
                        }
                        t.push_str(&format!("  ·  {}", discovery::human_size(f.size)));
                        if f.parts > 1 {
                            t.push_str(&format!("  ·  {} {}", f.parts, tr(fr, "parties", "parts")));
                        }
                        t
                    };
                    let selected = files.iter().find(|f| f.file == self.hub_file);
                    let file = &mut self.hub_file;
                    ComboBox::from_id_salt("hub_file")
                        .width(210.0)
                        .height(440.0)
                        .selected_text(selected.map(label).unwrap_or_default())
                        .show_ui(ui, |ui| {
                            for f in files {
                                ui.selectable_value(file, f.file.clone(), label(f))
                                    .on_hover_text(f.file.as_str());
                            }
                        })
                        .response
                        .on_hover_text(tr(
                            fr,
                            "Fichier du modèle, par quantisation, du plus léger au plus lourd. \
                             Prévoir un peu plus de VRAM que sa taille.",
                            "Model file, by quantization, from lightest to heaviest. Budget a bit \
                             more VRAM than its size.",
                        ));
                }
                (None, false) => {}
            }

            if ui
                .small_button("⟳")
                .on_hover_text(tr(fr, "Recharger la liste.", "Reload the list."))
                .clicked()
            {
                actions.push(Action::HubLoad);
            }

            let models_set = Path::new(&self.cfg.models_directory).is_dir();
            let can = self.hub_download.is_none()
                && models_set
                && !self.hub_repo.is_empty()
                && !self.hub_file.is_empty();
            if ui
                .add_enabled(can, egui::Button::new(tr(fr, "Télécharger", "Download")))
                .on_hover_text(tr(
                    fr,
                    "Télécharge ce fichier dans le répertoire des modèles, rangé par auteur et \
                     par dépôt. Il apparaît ensuite dans la liste des modèles GGUF.",
                    "Downloads this file into the models directory, filed by author and \
                     repository. It then shows up in the GGUF model list.",
                ))
                .on_disabled_hover_text(if self.hub_download.is_some() {
                    tr(fr, "Un téléchargement est déjà en cours.", "A download is already running.")
                } else if !models_set {
                    tr(fr, "Choisissez d'abord le répertoire des modèles.", "Pick the models directory first.")
                } else {
                    tr(fr, "Choisissez un dépôt et un fichier.", "Pick a repository and a file.")
                })
                .clicked()
            {
                actions.push(Action::HubDownload);
            }
        });

        // Progression du téléchargement.
        if let Some(d) = &self.hub_download {
            let (done, total) = self.hub_progress;
            let fraction = if total == 0 { 0.0 } else { done as f32 / total as f32 };
            let secs = d.started.elapsed().as_secs_f64().max(0.001);
            let speed = done as f64 / secs;
            let eta = (speed > 0.0 && total > done)
                .then(|| format_elapsed(Duration::from_secs_f64((total - done) as f64 / speed)));
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                let label = format!(
                    "{:.0} %  ·  {} / {}  ·  {}/s{}",
                    fraction * 100.0,
                    discovery::human_size(done),
                    discovery::human_size(total),
                    discovery::human_size(speed as u64),
                    eta.map(|e| format!("  ·  {} {e}", tr(fr, "reste", "left")))
                        .unwrap_or_default()
                );
                let width = (ui.available_width() - 90.0).max(120.0);
                bar(ui, width, 20.0, fraction, theme::accent(), &label)
                    .on_hover_text(d.label.as_str());
                if ui
                    .button(tr(fr, "Annuler", "Cancel"))
                    .on_hover_text(tr(
                        fr,
                        "Arrête le téléchargement ; le fichier partiel est supprimé.",
                        "Stops the download; the partial file is deleted.",
                    ))
                    .clicked()
                {
                    actions.push(Action::HubCancelDownload);
                }
            });
        }
    }

    fn import_window(&mut self, ctx: &egui::Context, actions: &mut Vec<Action>) {
        if !self.import_open {
            return;
        }
        let mut open = true;
        egui::Window::new(self.t("Importer une commande", "Import a command"))
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_size([620.0, 340.0])
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                hint(
                    ui,
                    self.t(
                        "Collez une commande llama-server. Les continuations ^, ` et \\ sont acceptées. \
                         Une valeur absente d'une liste y est ajoutée automatiquement.",
                        "Paste a llama-server command. The ^, ` and \\ continuations are accepted. \
                         A value missing from a list is added to it automatically.",
                    ),
                );
                ui.add_space(6.0);
                let paste_tip = self.t(
                    "Seul champ de saisie libre de l'application, et pour cause : c'est du texte \
                     à coller, pas un paramètre. Les alias courts (-fa, -c, -ctk, -np) et la forme \
                     --option=valeur sont compris.",
                    "The only free-text field in the app, and for good reason: this is text to \
                     paste, not a parameter. Short aliases (-fa, -c, -ctk, -np) and the \
                     --option=value form are understood.",
                );
                let import_tip = self.t(
                    "Applique les paramètres trouvés. Tous les autres sont désactivés, afin que \
                     la commande obtenue soit exactement celle collée.",
                    "Applies the parameters found. Every other one is disabled, so the resulting \
                     command matches exactly what was pasted.",
                );
                let import_disabled_tip =
                    self.t("Collez d'abord une commande.", "Paste a command first.");
                let cancel_tip = self.t(
                    "Ferme sans rien changer.",
                    "Closes without changing anything.",
                );
                ui.add(
                    egui::TextEdit::multiline(&mut self.import_text)
                        .desired_width(f32::INFINITY)
                        .desired_rows(12)
                        .font(egui::TextStyle::Monospace)
                        .hint_text("llama-server.exe -ngl 999 --ctx-size 65536 ..."),
                )
                .on_hover_text(paste_tip);
                ui.add_space(8.0);
                let can_import = !self.import_text.trim().is_empty();
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(
                            can_import,
                            theme::action_button(
                                RichText::new(self.t("Importer", "Import")),
                                theme::accent(),
                            ),
                        )
                        .on_hover_text(import_tip)
                        .on_disabled_hover_text(import_disabled_tip)
                        .clicked()
                    {
                        actions.push(Action::ApplyImport);
                    }
                    if ui
                        .button(self.t("Annuler", "Cancel"))
                        .on_hover_text(cancel_tip)
                        .clicked()
                    {
                        self.import_open = false;
                    }
                });
            });
        if !open {
            self.import_open = false;
        }
    }

    fn update_window(&mut self, ctx: &egui::Context, actions: &mut Vec<Action>) {
        if !self.update_open {
            return;
        }
        let fr = self.fr();
        let mut open = true;
        let installed = updater::is_source(Path::new(&self.cfg.llama_source_directory));
        let busy = self.update_task.is_some();
        let title = if installed {
            tr(fr, "Mettre à jour llama.cpp", "Update llama.cpp")
        } else {
            tr(fr, "Installer llama.cpp", "Install llama.cpp")
        };
        egui::Window::new(title)
            .id(egui::Id::new("llama_cpp_window"))
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_width(620.0)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                if !installed && !busy {
                    self.install_panel(ui, actions);
                    return;
                }
                if !installed {
                    // Installation en cours : seule la progression compte.
                    self.progress_panel(ui, actions);
                    return;
                }
                // ---- Dépôt source ----
                dir_row(
                    ui,
                    tr(fr, "Dépôt source (git)", "Source repository (git)"),
                    &self.cfg.llama_source_directory,
                    tr(fr, "Parcourir…", "Browse…"),
                    tr(
                        fr,
                        "Clone git de llama.cpp. Détecté à partir du répertoire des versions ; \
                         les sources y sont mises à jour par git pull.",
                        "Git clone of llama.cpp. Detected from the versions directory; sources \
                         are updated there with git pull.",
                    ),
                    || actions.push(Action::BrowseSource),
                );
                match &self.update_git {
                    Some(Ok(g)) => hint(
                        ui,
                        &format!(
                            "{} {} · b{} ({})",
                            tr(fr, "Sources :", "Sources:"),
                            g.branch,
                            g.build,
                            g.commit
                        ),
                    ),
                    Some(Err(e)) => {
                        ui.label(RichText::new(e).color(theme::red()).size(11.0));
                    }
                    None => {}
                }
                ui.add_space(6.0);

                // ---- Dossier des releases ----
                dir_row(
                    ui,
                    tr(fr, "Dossier des releases", "Releases folder"),
                    &self.cfg.llama_releases_directory,
                    tr(fr, "Parcourir…", "Browse…"),
                    tr(
                        fr,
                        "Chaque mise à jour y crée un dossier llama-b<build>-<backend>. Ces \
                         releases apparaissent dans la liste des versions, à côté des \
                         autres.",
                        "Each update creates a llama-b<build>-<backend> folder here. These \
                         releases show up in the versions list, next to the others.",
                    ),
                    || actions.push(Action::BrowseReleases),
                );
                ui.add_space(6.0);

                // ---- Options de compilation ----
                field_label(
                    ui,
                    tr(fr, "Options de compilation", "Build options"),
                    tr(
                        fr,
                        "Reprises de votre build existant (backend, architecture GPU, \
                         générateur, DLL partagées), pour que la nouvelle release soit \
                         compilée comme l'actuelle. Compilation dans le dossier \
                         build-uillamacpp du dépôt, jamais dans votre dossier build.",
                        "Taken from your existing build (backend, GPU architecture, generator, \
                         shared DLLs), so the new release is built like the current one. Built \
                         in the repository's build-uillamacpp folder, never in your build \
                         folder.",
                    ),
                );
                match &self.update_config {
                    Some((from, cfg)) => {
                        ui.label(RichText::new(cfg.summary(fr)).color(theme::text()).size(12.0));
                        hint(ui, &format!("{} {from}", tr(fr, "depuis", "from")));
                    }
                    None => {
                        ui.label(
                            RichText::new(tr(
                                fr,
                                "Aucun build existant trouvé : compilation CPU par défaut.",
                                "No existing build found: default CPU build.",
                            ))
                            .color(theme::orange())
                            .size(11.0),
                        );
                    }
                }
                let source = Path::new(&self.cfg.llama_source_directory);
                let first_build = !source
                    .join(updater::BUILD_DIR_NAME)
                    .join("CMakeCache.txt")
                    .is_file();
                if first_build {
                    ui.label(
                        RichText::new(tr(
                            fr,
                            "Première compilation : elle part de zéro et peut durer de 20 à 60 min \
                             avec CUDA. Les suivantes sont incrémentales, donc bien plus rapides.",
                            "First build: it starts from scratch and can take 20 to 60 min with \
                             CUDA. Later ones are incremental, hence much faster.",
                        ))
                        .color(theme::yellow())
                        .size(11.0),
                    );
                }

                ui.separator();

                // ---- Vérification ----
                let checking = self.update_check_task.is_some();
                let busy = self.update_task.is_some();
                let source_ok = matches!(self.update_git, Some(Ok(_)));
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(
                            source_ok && !checking && !busy,
                            egui::Button::new(tr(
                                fr,
                                "Vérifier les mises à jour",
                                "Check for updates",
                            )),
                        )
                        .on_hover_text(tr(
                            fr,
                            "Interroge le dépôt distant (git fetch) sans rien modifier.",
                            "Queries the remote repository (git fetch) without changing anything.",
                        ))
                        .clicked()
                    {
                        actions.push(Action::CheckUpdates);
                    }
                    if checking {
                        ui.add(egui::Spinner::new());
                    }
                });

                // Une release de ce build existe-t-elle déjà ?
                let releases = Path::new(&self.cfg.llama_releases_directory);
                let released = |build: u32| {
                    self.versions
                        .iter()
                        .any(|v| v.build == Some(build) && v.dir.starts_with(releases))
                };
                let mut up_to_date_released = false;
                match &self.update_check {
                    Some(Ok(c)) if c.behind == 0 => {
                        up_to_date_released = released(c.current.build);
                        ui.label(
                            RichText::new(format!(
                                "{} b{} ({}) — {}",
                                tr(fr, "À jour :", "Up to date:"),
                                c.current.build,
                                c.current.commit,
                                c.upstream
                            ))
                            .color(theme::green()),
                        );
                    }
                    Some(Ok(c)) => {
                        ui.label(
                            RichText::new(format!(
                                "{} {} : b{} — b{}",
                                c.behind,
                                tr(fr, "nouveau(x) commit(s)", "new commit(s)"),
                                c.current.build,
                                c.remote_build
                            ))
                            .color(theme::accent())
                            .strong(),
                        );
                        egui::ScrollArea::vertical()
                            .id_salt("update_commits")
                            .max_height(150.0)
                            .show(ui, |ui| {
                                for line in &c.commits {
                                    ui.add(
                                        egui::Label::new(
                                            RichText::new(line).monospace().size(11.5),
                                        )
                                        .truncate(),
                                    );
                                }
                                if c.behind as usize > c.commits.len() {
                                    hint(
                                        ui,
                                        &format!(
                                            "… {} {}",
                                            c.behind as usize - c.commits.len(),
                                            tr(fr, "autre(s)", "more")
                                        ),
                                    );
                                }
                            });
                    }
                    Some(Err(e)) => {
                        ui.label(RichText::new(e).color(theme::red()).size(11.0));
                    }
                    None => {}
                }

                ui.separator();

                // ---- Compilation ----
                if busy {
                    self.progress_panel(ui, actions);
                    return;
                }
                {
                    let label = match &self.update_check {
                        Some(Ok(c)) if c.behind == 0 => {
                            tr(fr, "Compiler la version actuelle", "Build the current version")
                        }
                        _ => tr(fr, "Mettre à jour et compiler", "Update and build"),
                    };
                    // Par défaut l'ancienne version est conservée ; la supprimer
                    // est un choix explicite, refait à chaque mise à jour.
                    if let Some(v) = self.deletable_version() {
                        let label = format!(
                            "{} {} ({})",
                            tr(
                                fr,
                                "Supprimer l'ancienne version après la mise à jour :",
                                "Delete the old version after the update:"
                            ),
                            v.name,
                            v.tag()
                        );
                        let mut delete = self.update_delete_previous;
                        ui.checkbox(&mut delete, label).on_hover_text(tr(
                            fr,
                            "Décoché par défaut : l'ancienne version est conservée. Cochée, elle \
                             n'est supprimée qu'une fois la nouvelle compilée et vérifiée ; si \
                             elle est en cours d'utilisation, elle est simplement conservée.",
                            "Unchecked by default: the old version is kept. When checked, it is \
                             only deleted once the new one is built and verified; if it is in \
                             use, it is simply kept.",
                        ));
                        self.update_delete_previous = delete;
                    }
                    // Changer l'option FlashAttention justifie de recompiler
                    // la version déjà publiée.
                    let mut fa_changed = false;
                    if let Some((_, c)) = &self.update_config {
                        if c.backend() == "cuda" {
                            let mut fa = self.update_fa_all;
                            fa_all_checkbox(ui, &mut fa, fr);
                            self.update_fa_all = fa;
                            fa_changed = fa != c.fa_all();
                        }
                    }
                    let can = source_ok
                        && !self.cfg.llama_releases_directory.is_empty()
                        && (!up_to_date_released || fa_changed);
                    if ui
                        .add_enabled(
                            can,
                            theme::action_button(RichText::new(label).strong(), theme::green()),
                        )
                        .on_hover_text(tr(
                            fr,
                            "git pull, compilation, puis copie des exe et DLL dans une nouvelle \
                             release, qui est sélectionnée à la fin. Le serveur en cours n'est \
                             pas interrompu.",
                            "git pull, build, then copy of the exe and DLL files into a new \
                             release, selected at the end. The running server is not \
                             interrupted.",
                        ))
                        .on_disabled_hover_text(if up_to_date_released {
                            tr(
                                fr,
                                "Déjà à jour, et une release de ce build existe.",
                                "Already up to date, and a release of this build exists.",
                            )
                        } else {
                            tr(
                                fr,
                                "Choisissez le dépôt source et le dossier des releases.",
                                "Pick the source repository and the releases folder.",
                            )
                        })
                        .clicked()
                    {
                        actions.push(Action::StartUpdate);
                    }
                    match &self.update_result {
                        Some(Ok(r)) => {
                            ui.label(
                                RichText::new(format!(
                                    "✔ {} {} (b{})",
                                    tr(fr, "Release créée :", "Release created:"),
                                    r.name,
                                    r.build
                                ))
                                .color(theme::green()),
                            )
                            .on_hover_text(r.dir.to_string_lossy());
                        }
                        Some(Err(e)) => {
                            ui.label(RichText::new(format!("! {e}")).color(theme::red()));
                        }
                        None => {}
                    }
                }
            });
        if !open {
            self.update_open = false;
        }
    }

    /// Étape, progression, durée et annulation d'une installation ou mise à
    /// jour en cours.
    fn progress_panel(&mut self, ui: &mut Ui, actions: &mut Vec<Action>) {
        let fr = self.fr();
        if let Some((i, n, label)) = self.update_stage {
            ui.label(RichText::new(format!("{i}/{n} · {label}")).strong());
        }
        let elapsed = self
            .update_started
            .map(|t| format_elapsed(t.elapsed()))
            .unwrap_or_default();
        match self.update_progress {
            Some(p) => {
                let label = format!("{:.0} %  ·  {elapsed}", p * 100.0);
                bar(ui, ui.available_width(), 20.0, p, theme::accent(), &label);
            }
            None => {
                ui.horizontal(|ui| {
                    ui.add(egui::Spinner::new());
                    ui.label(RichText::new(elapsed).color(theme::muted()));
                });
            }
        }
        ui.add(
            egui::Label::new(
                RichText::new(&self.update_last_line)
                    .monospace()
                    .size(11.0)
                    .color(theme::muted()),
            )
            .truncate(),
        )
        .on_hover_text(tr(
            fr,
            "Dernière ligne de sortie ; le détail complet est dans la console.",
            "Last output line; the full detail is in the console.",
        ));
        if ui
            .add(
                theme::action_button(RichText::new(tr(fr, "Annuler", "Cancel")), theme::red()),
            )
            .on_hover_text(tr(
                fr,
                "Interrompt git ou la compilation. Aucune release n'est créée ; relancer \
                 reprend là où ça s'est arrêté.",
                "Stops git or the build. No release is created; running it again resumes \
                 where it stopped.",
            ))
            .clicked()
        {
            actions.push(Action::CancelUpdate);
        }
    }

    /// Installation depuis zéro : dossier, prérequis, backend.
    fn install_panel(&mut self, ui: &mut Ui, actions: &mut Vec<Action>) {
        let fr = self.fr();
        ui.label(tr(
            fr,
            "Aucune installation de llama.cpp trouvée. l'application peut télécharger ses \
             sources et les compiler pour votre machine.",
            "No llama.cpp installation found. the app can download its sources and \
             build them for your machine.",
        ));
        ui.add_space(6.0);

        // ---- Dossier ----
        dir_row(
            ui,
            tr(fr, "Dossier d'installation", "Install folder"),
            &self.install_dir,
            tr(fr, "Parcourir…", "Browse…"),
            tr(
                fr,
                "Les sources y sont clonées (llama.cpp) et les releases compilées publiées à \
                 côté (llama.cpp-releases). Prévoir environ 2 Go avec CUDA. C'est aussi le \
                 « Répertoire llama.cpp » de la colonne de gauche : changer l'un change l'autre.",
                "Sources are cloned there (llama.cpp) and built releases published next to \
                 them (llama.cpp-releases). Allow about 2 GB with CUDA. It is also the \
                 “llama.cpp directory” of the left column: changing one changes the other.",
            ),
            || actions.push(Action::BrowseInstallDir),
        );
        if !self.install_dir.trim().is_empty() {
            let root = PathBuf::from(self.install_dir.trim());
            hint(
                ui,
                &format!(
                    "{}  ·  {}",
                    root.join("llama.cpp").display(),
                    updater::default_releases_dir(&root.join("llama.cpp")).display()
                ),
            );
            // Windows limite un chemin à 260 caractères ; les fichiers
            // intermédiaires de compilation s'enfoncent de ~150 caractères
            // sous le dossier.
            if root.join("llama.cpp").to_string_lossy().len() > 80 {
                ui.label(
                    RichText::new(tr(
                        fr,
                        "Chemin long : la compilation risque de dépasser la limite de 260 \
                         caractères de Windows. Préférez un dossier court, par exemple C:\\llama.",
                        "Long path: the build may exceed Windows' 260-character limit. Prefer \
                         a short folder, for example C:\\llama.",
                    ))
                    .color(theme::orange())
                    .size(11.0),
                );
            }
        }
        ui.add_space(8.0);

        // ---- Prérequis ----
        ui.horizontal(|ui| {
            field_label(
                ui,
                tr(fr, "Prérequis", "Prerequisites"),
                tr(
                    fr,
                    "Outils nécessaires à la compilation. Pour un outil manquant, la commande \
                     winget indiquée l'installe ; cliquez ensuite sur « Revérifier », sans \
                     relancer l'application.",
                    "Tools needed to build. For a missing tool, the winget command shown \
                     installs it; then click “Check again”, without restarting the app.",
                ),
            );
            if self.prereq_rx.is_some() {
                ui.add(egui::Spinner::new().size(12.0));
            } else if ui
                .small_button(tr(fr, "Revérifier", "Check again"))
                .clicked()
            {
                actions.push(Action::RecheckPrereqs);
            }
        });

        let Some(p) = self.prereqs.clone() else {
            hint(ui, tr(fr, "Détection des outils…", "Detecting tools…"));
            return;
        };
        let rows: [(&str, &str, &Option<String>, bool); 6] = [
            ("git", "Git", &p.git, true),
            ("cmake", "CMake", &p.cmake, true),
            ("msvc", tr(fr, "Compilateur C++ (Build Tools)", "C++ compiler (Build Tools)"), &p.msvc, true),
            ("gpu", tr(fr, "GPU NVIDIA", "NVIDIA GPU"), &p.nvidia_gpu, false),
            ("cuda", "CUDA Toolkit", &p.cuda, false),
            ("vulkan", "Vulkan SDK", &p.vulkan, false),
        ];
        egui::Grid::new("prereqs")
            .num_columns(2)
            .spacing([10.0, 4.0])
            .show(ui, |ui| {
                for (key, name, found, required) in rows {
                    ui.horizontal(|ui| {
                        let color = match (found.is_some(), required) {
                            (true, _) => theme::green(),
                            (false, true) => theme::red(),
                            (false, false) => theme::muted(),
                        };
                        theme::dot(ui, color);
                        ui.label(name);
                    });
                    match found {
                        Some(v) => {
                            ui.label(RichText::new(v).color(theme::muted()).size(11.5));
                        }
                        None => {
                            let command = updater::install_hint(key);
                            ui.horizontal(|ui| {
                                ui.label(
                                    RichText::new(if required {
                                        tr(fr, "manquant", "missing")
                                    } else {
                                        tr(fr, "absent", "absent")
                                    })
                                    .color(if required { theme::red() } else { theme::muted() })
                                    .size(11.5),
                                );
                                if !command.is_empty()
                                    && ui
                                        .small_button(tr(fr, "Copier la commande", "Copy command"))
                                        .on_hover_text(command)
                                        .clicked()
                                {
                                    actions.push(Action::CopyText(command.to_string()));
                                }
                            });
                        }
                    }
                    ui.end_row();
                }
            });
        if p.msvc.is_none() {
            hint(
                ui,
                tr(
                    fr,
                    "Build Tools : installation de plusieurs Go, qui peut demander un redémarrage.",
                    "Build Tools: a multi-GB install that may require a reboot.",
                ),
            );
        }
        ui.add_space(8.0);

        // ---- Backend ----
        field_label(
            ui,
            "Backend",
            tr(
                fr,
                "Moteur de calcul compilé. CUDA : GPU NVIDIA, le plus rapide sur ces cartes. \
                 Vulkan : la plupart des GPU. CPU : fonctionne partout, bien plus lent. \
                 Un backend dont l'outil manque ne peut pas être choisi.",
                "Compute backend to build. CUDA: NVIDIA GPUs, the fastest on those cards. \
                 Vulkan: most GPUs. CPU: runs anywhere, much slower. A backend whose tool \
                 is missing cannot be picked.",
            ),
        );
        let mut chosen = self.install_backend;
        ComboBox::from_id_salt("install_backend")
            .width(260.0)
            .selected_text(chosen.label())
            .show_ui(ui, |ui| {
                for b in updater::Backend::ALL {
                    let ready = p.backend_ready(b);
                    let mut text = b.label().to_string();
                    if b == p.suggested() {
                        text.push_str(tr(fr, "  · conseillé", "  · suggested"));
                    }
                    if !ready {
                        text.push_str(tr(fr, "  · outil manquant", "  · tool missing"));
                    }
                    ui.add_enabled_ui(ready, |ui| {
                        ui.selectable_value(&mut chosen, b, text);
                    });
                }
            });
        if chosen != self.install_backend {
            self.install_backend = chosen;
            self.install_backend_chosen = true;
        }
        if self.install_backend == updater::Backend::Cuda {
            hint(
                ui,
                tr(
                    fr,
                    "Compilation CUDA : 20 à 60 min selon la machine.",
                    "CUDA build: 20 to 60 min depending on the machine.",
                ),
            );
            let mut fa = self.update_fa_all;
            fa_all_checkbox(ui, &mut fa, fr);
            self.update_fa_all = fa;
        }

        ui.add_space(10.0);
        let ready = p.can_build()
            && p.backend_ready(self.install_backend)
            && !self.install_dir.trim().is_empty();
        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    ready,
                    theme::action_button(
                        RichText::new(tr(fr, "Télécharger et compiler", "Download and build"))
                            .strong(),
                        theme::green(),
                    ),
                )
                .on_hover_text(tr(
                    fr,
                    "Clone llama.cpp depuis GitHub, le compile avec le backend choisi et \
                     publie la première release, sélectionnée à la fin.",
                    "Clones llama.cpp from GitHub, builds it with the chosen backend and \
                     publishes the first release, selected at the end.",
                ))
                .on_disabled_hover_text(tr(
                    fr,
                    "Installez d'abord les outils marqués en rouge.",
                    "Install the tools marked in red first.",
                ))
                .clicked()
            {
                actions.push(Action::StartInstall);
            }
            if ui
                .button(tr(fr, "Utiliser un clone existant…", "Use an existing clone…"))
                .on_hover_text(tr(
                    fr,
                    "Vous avez déjà cloné llama.cpp : choisissez son dossier pour passer \
                     directement aux mises à jour.",
                    "You already cloned llama.cpp: pick its folder to go straight to updates.",
                ))
                .clicked()
            {
                actions.push(Action::BrowseSource);
            }
        });
        if let Some(Err(e)) = &self.update_result {
            ui.label(RichText::new(format!("! {e}")).color(theme::red()));
        }
    }

    fn confirm_version_window(&mut self, ctx: &egui::Context, actions: &mut Vec<Action>) {
        let Some(dir) = self.confirm_version_delete.clone() else {
            return;
        };
        let fr = self.fr();
        let version = self.versions.iter().find(|v| v.dir == dir).cloned();
        let others = self.versions.len().saturating_sub(1);
        egui::Window::new(tr(fr, "Supprimer cette version ?", "Delete this version?"))
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                if let Some(v) = &version {
                    ui.label(RichText::new(format!("{}  ·  {}", v.tag(), v.name)).strong());
                }
                ui.label(RichText::new(dir.to_string_lossy()).monospace().size(11.5));
                ui.add_space(4.0);
                ui.label(
                    RichText::new(tr(
                        fr,
                        "Le dossier et ses exécutables seront supprimés du disque. Les résultats \
                         de benchmark et les profils qui la mentionnent sont conservés.",
                        "The folder and its executables will be deleted from disk. Benchmark \
                         results and profiles mentioning it are kept.",
                    ))
                    .color(theme::orange()),
                );
                if others == 0 {
                    ui.label(
                        RichText::new(tr(
                            fr,
                            "C'est la dernière version : il faudra réinstaller llama.cpp.",
                            "This is the last version: llama.cpp will need reinstalling.",
                        ))
                        .color(theme::red()),
                    );
                }
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui
                        .add(
                            theme::action_button(
                                RichText::new(tr(fr, "Supprimer", "Delete")),
                                theme::red(),
                            ),
                        )
                        .clicked()
                    {
                        actions.push(Action::DeleteVersion(dir.clone()));
                    }
                    if ui.button(tr(fr, "Annuler", "Cancel")).clicked() {
                        self.confirm_version_delete = None;
                    }
                });
            });
    }

    fn confirm_window(&mut self, ctx: &egui::Context, actions: &mut Vec<Action>) {
        let Some(path) = self.confirm_delete.clone() else {
            return;
        };
        egui::Window::new(self.t("Supprimer le modèle ?", "Delete the model?"))
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.label(RichText::new(path.to_string_lossy()).monospace().size(12.0));
                ui.add_space(4.0);
                ui.label(
                    RichText::new(self.t(
                        "Le fichier sera définitivement supprimé du disque.",
                        "The file will be permanently deleted from disk.",
                    ))
                    .color(theme::orange()),
                );
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui
                        .add(
                            theme::action_button(
                                RichText::new(self.t("Supprimer", "Delete")),
                                theme::red(),
                            ),
                        )
                        .on_hover_text(self.t(
                            "Supprime le fichier .gguf du disque. Il ne passe pas par la corbeille \
                             et l'opération est irréversible.",
                            "Deletes the .gguf file from disk. It does not go to the recycle bin \
                             and the operation cannot be undone.",
                        ))
                        .clicked()
                    {
                        actions.push(Action::DeleteModel(path.clone()));
                    }
                    if ui
                        .button(self.t("Annuler", "Cancel"))
                        .on_hover_text(self.t("Ne supprime rien.", "Deletes nothing."))
                        .clicked()
                    {
                        self.confirm_delete = None;
                    }
                });
            });
    }
}

// --------------------------------------------------------------- cartes profil

/// Édition en cours d'une carte de profil.
struct ProfileEdit {
    stem: String,
    name: String,
    /// Le champ de renommage doit prendre le focus à sa première image.
    ///
    /// On ne peut pas le demander au moment du clic sur ✏ : le champ n'existe
    /// pas encore, et accesskit panique sur un focus visant un widget absent
    /// de son arbre (« Focused ID … is not in the node list »).
    focus_pending: bool,
}

struct CardState {
    active: bool,
    dirty: bool,
    now: chrono::DateTime<chrono::Local>,
    fr: bool,
}

/// Une carte de profil, façon liste de conversations. Toute la carte est
/// cliquable (charge le profil) ; ses boutons restent prioritaires car la
/// détection du clic de la carte est placée sous ses enfants.
fn profile_card(
    ui: &mut Ui,
    entry: &ProfileEntry,
    state: CardState,
    edit: &mut Option<ProfileEdit>,
    delete: &mut Option<String>,
    actions: &mut Vec<Action>,
) {
    let CardState {
        active,
        dirty,
        now,
        fr,
    } = state;
    let stem = &entry.stem;
    let p = &entry.profile;
    let editing = matches!(edit, Some(e) if e.stem == *stem);
    let deleting = delete.as_deref() == Some(stem.as_str());
    let rename_id = egui::Id::new(("profile_rename", stem));

    let builder = egui::UiBuilder::new()
        .id_salt(("profile_card", stem))
        .sense(egui::Sense::click());
    let card = ui.scope_builder(builder, |ui| {
        let hovered = ui.response().hovered() && !editing && !deleting;
        let (fill, stroke) = if deleting {
            (theme::danger_card(), theme::red())
        } else if active {
            (theme::active_card(), theme::accent())
        } else if hovered {
            (theme::card_hi(), theme::hover_border())
        } else {
            (theme::card(), theme::border())
        };

        egui::Frame::new()
            .fill(fill)
            .stroke(egui::Stroke::new(1.0, stroke))
            .corner_radius(6)
            .inner_margin(egui::Margin::symmetric(10, 8))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());

                // ---- Titre + actions ----
                if editing {
                    let mut commit = None;
                    let mut cancel = false;
                    if let Some(ProfileEdit {
                        name,
                        focus_pending,
                        ..
                    }) = edit.as_mut()
                    {
                        let field = ui
                            .add(
                                egui::TextEdit::singleline(name)
                                    .id(rename_id)
                                    .desired_width(f32::INFINITY),
                            )
                            .on_hover_text(tr(
                                fr,
                                "Nouveau nom. Entrée pour renommer, Échap pour annuler.",
                                "New name. Enter to rename, Esc to cancel.",
                            ));
                        if *focus_pending {
                            field.request_focus();
                            *focus_pending = false;
                        }
                        if field.lost_focus() {
                            if ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                                commit = Some(name.trim().to_string());
                            } else if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                                cancel = true;
                            }
                        }
                        ui.add_space(4.0);
                        ui.horizontal_wrapped(|ui| {
                            let changed =
                                !name.trim().is_empty() && name.trim() != entry.display_name();
                            if ui
                                .add_enabled(changed, egui::Button::new(tr(fr, "Renommer", "Rename")))
                                .on_hover_text(tr(
                                    fr,
                                    "Change seulement le nom : réglages et date sont conservés.",
                                    "Changes the name only: settings and date are kept.",
                                ))
                                .on_disabled_hover_text(tr(
                                    fr,
                                    "Saisissez un nom différent.",
                                    "Type a different name.",
                                ))
                                .clicked()
                            {
                                commit = Some(name.trim().to_string());
                            }
                            if ui
                                .button(tr(fr, "Remplacer par l'actuel", "Replace with current"))
                                .on_hover_text(tr(
                                    fr,
                                    "Écrase les réglages du profil par ceux affichés à l'écran \
                                     (paramètres, hôte, port, modèle, version).",
                                    "Overwrites the profile's settings with the ones on screen \
                                     (parameters, host, port, model, version).",
                                ))
                                .clicked()
                            {
                                actions.push(Action::UpdateProfile(stem.clone()));
                            }
                            if ui
                                .button(tr(fr, "Annuler", "Cancel"))
                                .on_hover_text(tr(fr, "Ne change rien.", "Changes nothing."))
                                .clicked()
                            {
                                cancel = true;
                            }
                        });
                    }
                    if let Some(name) = commit.filter(|n| !n.is_empty()) {
                        if name == entry.display_name() {
                            *edit = None;
                        } else {
                            actions.push(Action::RenameProfile {
                                stem: stem.clone(),
                                name,
                            });
                        }
                    } else if cancel {
                        *edit = None;
                    }
                    return;
                }

                ui.horizontal(|ui| {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if !deleting {
                            if ui
                                .small_button(RichText::new("🗑").color(theme::red()))
                                .on_hover_text(tr(
                                    fr,
                                    "Supprimer ce profil (confirmation demandée).",
                                    "Delete this profile (asks for confirmation).",
                                ))
                                .clicked()
                            {
                                *delete = Some(stem.clone());
                                *edit = None;
                            }
                            if ui
                                .small_button("✏")
                                .on_hover_text(tr(
                                    fr,
                                    "Éditer : renommer, ou remplacer ses réglages par ceux \
                                     affichés.",
                                    "Edit: rename, or replace its settings with the ones shown.",
                                ))
                                .clicked()
                            {
                                *edit = Some(ProfileEdit {
                                    stem: stem.clone(),
                                    name: entry.display_name().to_string(),
                                    focus_pending: true,
                                });
                                *delete = None;
                            }
                        }
                        ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                            ui.add(
                                egui::Label::new(
                                    RichText::new(entry.display_name())
                                        .strong()
                                        .size(13.5)
                                        .color(if active { theme::strong() } else { theme::text() }),
                                )
                                .truncate()
                                .selectable(false),
                            );
                        });
                    });
                });

                // ---- Modèle, version, adresse ----
                let model_line = match (p.model_stem(), p.version.is_empty()) {
                    (Some(m), false) => format!("{m}  ·  {}", p.version),
                    (Some(m), true) => m,
                    (None, false) => p.version.clone(),
                    (None, true) => tr(fr, "modèle non mémorisé", "model not stored").to_string(),
                };
                ui.add(
                    egui::Label::new(RichText::new(model_line).color(theme::muted()).size(11.0))
                        .truncate()
                        .selectable(false),
                )
                .on_hover_text(if p.model.is_empty() {
                    tr(
                        fr,
                        "Profil créé avant la mémorisation du modèle : seuls les paramètres \
                         seront restaurés.",
                        "Profile created before models were stored: only the parameters will be \
                         restored.",
                    )
                    .to_string()
                } else {
                    format!(
                        "{}\n{}",
                        p.model,
                        tr(
                            fr,
                            "Modèle et version sont resélectionnés au chargement s'ils existent \
                             encore.",
                            "Model and version are re-selected on load if they still exist.",
                        )
                    )
                });

                // ---- Tags ----
                let tags = profiles::tags(p);
                if !tags.is_empty() {
                    ui.add_space(3.0);
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing = egui::vec2(4.0, 4.0);
                        for tag in &tags {
                            chip(ui, &tag.text, tag.primary).on_hover_text(if tag.primary {
                                tr(
                                    fr,
                                    "Taille du contexte (--ctx-size) : la longueur maximale de \
                                     conversation que le serveur garde en mémoire.",
                                    "Context size (--ctx-size): the longest conversation the \
                                     server keeps in memory.",
                                )
                            } else {
                                tr(
                                    fr,
                                    "Réglage clé enregistré dans ce profil.",
                                    "Key setting stored in this profile.",
                                )
                            });
                        }
                    });
                }

                // ---- Pied : nombre d'options, date, état ----
                ui.add_space(3.0);
                ui.horizontal(|ui| {
                    let count = p.parameters.len();
                    let mut footer = if fr {
                        format!("{count} option{}", if count > 1 { "s" } else { "" })
                    } else {
                        format!("{count} option{}", if count == 1 { "" } else { "s" })
                    };
                    if !p.host.is_empty() && !p.port.is_empty() {
                        footer.push_str(&format!("  ·  {}:{}", p.host, p.port));
                    }
                    if let Some(when) = entry.updated {
                        footer.push_str("  ·  ");
                        footer.push_str(&profiles::relative(when, now, fr));
                    }
                    ui.add(
                        egui::Label::new(RichText::new(footer).color(theme::muted()).size(10.5))
                            .truncate()
                            .selectable(false),
                    )
                    .on_hover_text(
                        entry
                            .updated
                            .map(|w| {
                                format!(
                                    "{} {}",
                                    tr(fr, "Mis à jour le", "Updated on"),
                                    w.format("%d/%m/%Y %H:%M")
                                )
                            })
                            .unwrap_or_default(),
                    );
                });

                if dirty {
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        theme::dot(ui, theme::orange());
                        ui.label(
                            RichText::new(tr(fr, "modifié", "modified"))
                                .color(theme::orange())
                                .size(11.0),
                        )
                        .on_hover_text(tr(
                            fr,
                            "Les réglages affichés diffèrent de ceux enregistrés dans ce profil.",
                            "The settings shown differ from the ones saved in this profile.",
                        ));
                        if ui
                            .small_button(tr(fr, "Mettre à jour", "Update"))
                            .on_hover_text(tr(
                                fr,
                                "Enregistre les réglages actuels dans ce profil.",
                                "Saves the current settings into this profile.",
                            ))
                            .clicked()
                        {
                            actions.push(Action::UpdateProfile(stem.clone()));
                        }
                    });
                }

                // ---- Confirmation de suppression ----
                if deleting {
                    ui.add_space(6.0);
                    ui.label(
                        RichText::new(tr(
                            fr,
                            "Supprimer définitivement ce profil ?",
                            "Delete this profile for good?",
                        ))
                        .color(theme::text())
                        .size(12.0),
                    );
                    ui.horizontal(|ui| {
                        if ui
                            .add(
                                theme::action_button(
                                    RichText::new(tr(fr, "Supprimer", "Delete")),
                                    theme::red(),
                                ),
                            )
                            .on_hover_text(tr(
                                fr,
                                "Supprime le fichier du profil. Les réglages affichés ne changent \
                                 pas.",
                                "Deletes the profile file. The settings shown are left alone.",
                            ))
                            .clicked()
                        {
                            actions.push(Action::DeleteProfile(stem.clone()));
                        }
                        if ui
                            .button(tr(fr, "Annuler", "Cancel"))
                            .on_hover_text(tr(fr, "Garde le profil.", "Keeps the profile."))
                            .clicked()
                        {
                            *delete = None;
                        }
                    });
                }
            });
    });

    if !editing && !deleting {
        let response = card.response.on_hover_text(if active {
            tr(
                fr,
                "Profil actif. Cliquer pour le recharger et annuler les modifications.",
                "Active profile. Click to reload it and discard changes.",
            )
        } else {
            tr(fr, "Cliquer pour charger ce profil.", "Click to load this profile.")
        });
        if response.clicked() {
            actions.push(Action::LoadProfile(stem.clone()));
        }
    }
}

/// Lignes « Host » et « Port » de la section Serveur (dans sa grille).
///
/// Le port se choisit dans la liste ou, via « Personnalisé… », dans un champ
/// numérique borné à 1–65535 : un nombre, jamais un texte invalide.
fn address_rows(ui: &mut Ui, host: &mut String, port: &mut String, fr: bool) {
    let host_tip = tr(
        fr,
        "Interface sur laquelle le serveur écoute. 127.0.0.1 : accessible seulement depuis \
         cette machine. 0.0.0.0 : accessible depuis le réseau local — n'exposez pas le serveur \
         sans pare-feu ni clé API.",
        "Interface the server listens on. 127.0.0.1: reachable from this machine only. \
         0.0.0.0: reachable from the local network — do not expose the server without a \
         firewall or an API key.",
    );
    ui.label("Host").on_hover_text(host_tip);
    ComboBox::from_id_salt("host")
        .width(ui.available_width().max(120.0))
        .selected_text(host.clone())
        .show_ui(ui, |ui| {
            for h in params::HOSTS {
                ui.selectable_value(host, h.to_string(), *h);
            }
        })
        .response
        .on_hover_text(host_tip);
    ui.end_row();

    let port_tip = tr(
        fr,
        "Port TCP d'écoute. Choisissez « Personnalisé… » pour un autre numéro (1 à 65535). \
         Changez-le si un autre service occupe déjà le port : l'erreur apparaît alors dans la \
         console.",
        "TCP port to listen on. Pick “Custom…” for another number (1 to 65535). Change it if \
         another service already holds the port: the error then shows up in the console.",
    );
    ui.label("Port").on_hover_text(port_tip);
    // Mode personnalisé : choisi explicitement, ou port hors liste (profil,
    // import, ancienne configuration).
    let custom_id = egui::Id::new("port_custom");
    let mut custom = ui.data(|d| d.get_temp::<bool>(custom_id)).unwrap_or(false)
        || !params::PORTS.contains(&port.as_str());
    let custom_label = tr(fr, "Personnalisé…", "Custom…");
    ui.horizontal(|ui| {
        ComboBox::from_id_salt("port")
            .width(120.0)
            .selected_text(if custom { custom_label.to_string() } else { port.clone() })
            .show_ui(ui, |ui| {
                for p in params::PORTS {
                    if ui.selectable_label(!custom && port.as_str() == *p, *p).clicked() {
                        *port = p.to_string();
                        custom = false;
                    }
                }
                ui.separator();
                if ui.selectable_label(custom, custom_label).clicked() {
                    custom = true;
                }
            })
            .response
            .on_hover_text(port_tip);
        if custom {
            let mut number: u16 = port.trim().parse().ok().filter(|n| *n > 0).unwrap_or(8080);
            ui.add(egui::DragValue::new(&mut number).range(1..=65535).speed(1.0))
                .on_hover_text(tr(
                    fr,
                    "Cliquez pour taper le numéro, ou faites glisser. Limité à 1–65535.",
                    "Click to type the number, or drag. Limited to 1–65535.",
                ));
            // Toujours un nombre valide dans la configuration.
            *port = number.to_string();
        }
    });
    ui.data_mut(|d| d.insert_temp(custom_id, custom));
    ui.end_row();
}

/// Infobulle d'un paramètre : nom exact de l'option, rôle, et ce que
/// « désactivé » implique.
fn param_tip(def: &params::ParamDef, fr: bool) -> String {
    format!(
        "{}\n\n{}\n\n{}",
        def.name,
        if fr { def.help_fr } else { def.help_en },
        match (def.kind, def.secret, fr) {
            (Kind::Flag, _, true) => "Drapeau : présent ou absent de la commande, sans valeur.",
            (Kind::Flag, _, false) => "Flag: present or absent from the command, with no value.",
            (_, true, true) => "Champ vide : option absente de la commande.",
            (_, true, false) => "Empty field: the option is left out of the command.",
            (_, false, true) => {
                "« désactivé » retire complètement l'option : llama-server applique alors son \
                 propre défaut."
            }
            (_, false, false) => {
                "“disabled” drops the option entirely: llama-server then applies its own \
                 default."
            }
        }
    )
}

/// Grille libellé / contrôle des paramètres d'un groupe, utilisée par le
/// panneau central.
fn param_grid(ui: &mut Ui, group: Group, states: &mut [ParamState], fr: bool, fa: Option<&FaSupport>) {
    param_grid_with(ui, group, states, fr, fa, |_| {});
}

/// Paramètre de type de cache KV : (nom du paramètre partenaire, est-ce K ?).
fn kv_partner(name: &str) -> Option<(&'static str, bool)> {
    match name {
        "--cache-type-k" => Some(("--cache-type-v", true)),
        "--cache-type-v" => Some(("--cache-type-k", false)),
        "--cache-type-k-draft" => Some(("--cache-type-v-draft", true)),
        "--cache-type-v-draft" => Some(("--cache-type-k-draft", false)),
        _ => None,
    }
}

/// Même grille, précédée de lignes propres à la section (hôte et port pour
/// Serveur), pour que tout reste aligné sur les mêmes colonnes.
fn param_grid_with(
    ui: &mut Ui,
    group: Group,
    states: &mut [ParamState],
    fr: bool,
    fa: Option<&FaSupport>,
    leading: impl FnOnce(&mut Ui),
) {
    let value_of = |states: &[ParamState], name: &str| {
        params::index_of(name).map(|i| states[i].value.clone()).unwrap_or_default()
    };
    // FlashAttention désactivé : les paires K/V ne comptent plus.
    let fa = fa.filter(|_| value_of(states, "--flash-attn") != "off");
    egui::Grid::new(format!("grid_{group:?}"))
        .num_columns(2)
        .spacing([10.0, 7.0])
        .show(ui, |ui| {
            leading(ui);
            for (i, def) in params::PARAMS.iter().enumerate() {
                if def.group != group {
                    continue;
                }
                // Types de cache K/V : valeurs dont la paire n'est pas compilée.
                let mut kv = KvCheck::default();
                if let (Some(fa), Some((partner, is_k))) = (fa, kv_partner(def.name)) {
                    let other = value_of(states, partner);
                    let pair = |mine: &str| {
                        if is_k { (mine.to_string(), other.clone()) } else { (other.clone(), mine.to_string()) }
                    };
                    let state = &states[i];
                    kv.bad = std::iter::once(&String::new())
                        .chain(state.options.iter())
                        .filter(|o| {
                            let (k, v) = pair(o);
                            !fa.supports(&k, &v)
                        })
                        .cloned()
                        .collect();
                    if kv.bad.contains(&state.value) {
                        let (k, v) = pair(&state.value);
                        let name = |t: &str| if t.is_empty() { "f16".to_string() } else { t.to_string() };
                        kv.warning = Some(format!(
                            "{} {}-{} {}\n{} {}\n\n{}",
                            tr(fr, "⚠ Paire K/V", "⚠ K/V pair"),
                            name(&k),
                            name(&v),
                            tr(
                                fr,
                                "non compilée dans cette release : FlashAttention passera sur le \
                                 CPU (on) ou sera désactivé (auto), bien plus lent.",
                                "not compiled in this release: FlashAttention will run on the \
                                 CPU (on) or be disabled (auto), much slower.",
                            ),
                            tr(fr, "Paires compilées :", "Compiled pairs:"),
                            fa.describe(fr),
                            tr(
                                fr,
                                "Choisissez une paire compilée, ou recompilez avec « FlashAttention \
                                 pour tous les types de cache KV » (⚙ Options › Mettre à jour).",
                                "Pick a compiled pair, or rebuild with “FlashAttention for every \
                                 KV cache type” (⚙ Options › Update).",
                            ),
                        ));
                    }
                }
                let state = &mut states[i];
                let tip = param_tip(def, fr);
                // Les rares libellés du catalogue écrits en français ont leur
                // version anglaise ici.
                let label = match (def.name, fr) {
                    ("--api-key", false) => "API key",
                    ("--sleep-idle-seconds", false) => "Idle sleep (s)",
                    _ => def.label,
                };
                ui.label(RichText::new(label).color(if state.enabled() {
                    theme::text()
                } else {
                    theme::muted()
                }))
                .on_hover_text(&tip);

                if def.secret {
                    secret_field(ui, def, &mut state.value, &tip, fr);
                } else {
                    match def.kind {
                        Kind::Flag => flag_field(ui, &mut state.value, &tip, fr),
                        Kind::Choice => choice_field(ui, def, state, &tip, fr, &kv),
                    }
                }
                ui.end_row();
            }
        });
}

/// Case à cocher d'un drapeau, avec son propre libellé (hors grille).
fn flag_field_labeled(ui: &mut Ui, value: &mut String, label: &str, tip: &str) {
    let mut on = !value.trim().is_empty();
    if ui.checkbox(&mut on, label).on_hover_text(tip).changed() {
        *value = if on { "on".into() } else { String::new() };
    }
}

fn flag_field(ui: &mut Ui, value: &mut String, tip: &str, fr: bool) {
    let mut on = !value.trim().is_empty();
    let text = match (on, fr) {
        (true, true) => "activé",
        (true, false) => "on",
        (false, true) => "désactivé",
        (false, false) => "off",
    };
    if ui.checkbox(&mut on, text).on_hover_text(tip).changed() {
        *value = if on { "on".into() } else { String::new() };
    }
}

/// Valeurs d'un type de cache K/V dont la paire FlashAttention n'est pas
/// compilée dans la release, et l'avertissement si la valeur choisie en est.
#[derive(Default)]
struct KvCheck {
    bad: Vec<String>,
    warning: Option<String>,
}

fn choice_field(
    ui: &mut Ui,
    def: &params::ParamDef,
    state: &mut ParamState,
    tip: &str,
    fr: bool,
    kv: &KvCheck,
) {
    let disabled = tr(fr, "— désactivé —", "— disabled —");
    let mark = |opt: &str, text: String| -> RichText {
        if kv.bad.iter().any(|b| b == opt) {
            RichText::new(format!("⚠ {text}")).color(theme::orange())
        } else {
            RichText::new(text)
        }
    };
    let text = mark(
        &state.value,
        if state.value.is_empty() {
            disabled.to_string()
        } else {
            display_value(def, &state.value, fr)
        },
    );
    let unmarked_tip;
    let tip = match &kv.warning {
        Some(w) => {
            unmarked_tip = format!("{w}\n\n{tip}");
            unmarked_tip.as_str()
        }
        None => tip,
    };
    let bad_tip = tr(
        fr,
        "Paire K/V non compilée pour FlashAttention dans la release sélectionnée.",
        "K/V pair not compiled for FlashAttention in the selected release.",
    );
    let value = &mut state.value;
    let options = &state.options;
    ComboBox::from_id_salt(def.name)
        .width(ui.available_width().max(120.0))
        // Assez haut pour les longues listes (contexte : 27 valeurs) : la
        // hauteur par défaut n'en montre que 5.
        .height(440.0)
        .selected_text(text)
        .show_ui(ui, |ui| {
            ui.selectable_value(value, String::new(), mark("", disabled.to_string()));
            ui.separator();
            if options.is_empty() {
                ui.label(
                    RichText::new(if def.name == "--mmproj" {
                        tr(
                            fr,
                            "Aucun fichier mmproj dans le répertoire des modèles.",
                            "No mmproj file in the models directory.",
                        )
                    } else {
                        tr(
                            fr,
                            "Aucune valeur : importez une commande qui en contient une.",
                            "No value: import a command that contains one.",
                        )
                    })
                    .color(theme::muted())
                    .italics()
                    .size(11.0),
                );
            }
            for opt in options {
                let shown = display_value(def, opt, fr);
                let text = if opt == def.default {
                    format!("{shown}   ·  {}", tr(fr, "défaut", "default"))
                } else {
                    shown
                };
                let bad = kv.bad.contains(opt);
                ui.selectable_value(value, opt.clone(), mark(opt, text))
                    .on_hover_text(if bad {
                        format!("{opt}\n{bad_tip}")
                    } else {
                        opt.clone()
                    });
            }
        })
        .response
        .on_hover_text(tip);
}

/// Champ masqué, seule saisie libre de paramètre de l'application : une clé
/// API ne peut pas venir d'une liste. Le bouton 👁 l'affiche le temps de la
/// vérifier.
fn secret_field(ui: &mut Ui, def: &params::ParamDef, value: &mut String, tip: &str, fr: bool) {
    let reveal_id = egui::Id::new(("reveal_secret", def.name));
    let mut reveal = ui.data(|d| d.get_temp::<bool>(reveal_id)).unwrap_or(false);
    ui.horizontal(|ui| {
        let width = (ui.available_width() - 34.0).max(80.0);
        ui.add(
            egui::TextEdit::singleline(value)
                .password(!reveal)
                .desired_width(width)
                .hint_text(tr(fr, "aucune", "none")),
        )
        .on_hover_text(tip);
        if ui
            .selectable_label(reveal, "👁")
            .on_hover_text(tr(fr, "Afficher / masquer la clé.", "Show / hide the key."))
            .clicked()
        {
            reveal = !reveal;
        }
    });
    ui.data_mut(|d| d.insert_temp(reveal_id, reveal));
}

/// Options importées que le catalogue ne connaît pas. Elles sont transmises
/// telles quelles ; on ne peut que les retirer, jamais les saisir.
fn extra_args_section(ui: &mut Ui, extra: &[Vec<String>], fr: bool, actions: &mut Vec<Action>) {
    let title = format!(
        "{}  ({})",
        tr(fr, "Hors catalogue", "Uncatalogued"),
        extra.len()
    );
    egui::CollapsingHeader::new(RichText::new(title).strong().color(theme::yellow()))
        .id_salt("group_extra")
        .default_open(true)
        .show_background(false)
        .show(ui, |ui| {
            for (i, tokens) in extra.iter().enumerate() {
                ui.horizontal(|ui| {
                    if ui
                        .small_button(RichText::new("🗑").color(theme::red()))
                        .on_hover_text(tr(
                            fr,
                            "Retire cette option de la commande.",
                            "Removes this option from the command.",
                        ))
                        .clicked()
                    {
                        actions.push(Action::RemoveExtraArg(i));
                    }
                    ui.add(
                        egui::Label::new(
                            RichText::new(tokens.join(" ")).monospace().size(12.0),
                        )
                        .truncate(),
                    );
                });
            }
        })
        .header_response
        .on_hover_text(tr(
            fr,
            "Options présentes dans la dernière commande importée mais absentes du catalogue. \
             Elles sont transmises telles quelles à llama-server, pour que la commande lancée \
             soit celle qui a été collée. On peut les retirer, pas les modifier.",
            "Options from the last imported command that the catalog does not know. They are \
             passed to llama-server as-is, so the command that runs is the one that was pasted. \
             They can be removed, not edited.",
        ));
}

/// Valeur telle qu'affichée dans une liste : masquée si secrète, sens de
/// « 0 » pour le contexte, chemin réduit au nom du fichier (le chemin complet
/// est en infobulle).
fn display_value(def: &params::ParamDef, value: &str, fr: bool) -> String {
    if def.secret {
        return params::mask(value);
    }
    // Contexte : 0 n'est pas une taille, c'est « celle du modèle ».
    if def.name == "--ctx-size" && value == "0" {
        return format!("0  ·  {}", tr(fr, "contexte du modèle", "model's context"));
    }
    if value.contains('\\') || value.contains('/') {
        if let Some(name) = std::path::Path::new(value).file_name() {
            return name.to_string_lossy().into_owned();
        }
    }
    value.to_string()
}

/// Jauge horizontale dessinée : fond et bord visibles dans les deux thèmes,
/// remplissage rectangulaire. (`ProgressBar` d'egui n'a pas de bord, son fond
/// se confond avec le panneau en thème clair, et ses coins très arrondis
/// transforment un faible remplissage en pastille.)
fn gauge(ui: &mut Ui, fraction: f32, fill: Color32, text: &str) -> egui::Response {
    bar(ui, 130.0, TOP_BAR_ITEM_HEIGHT, fraction, fill, text)
}

/// Hauteur des boutons de la barre du haut (`interact_size.y`), reprise par
/// l'état du serveur et les jauges pour qu'ils s'alignent.
const TOP_BAR_ITEM_HEIGHT: f32 = 24.0;

/// État du serveur : pastille et texte dans un cadre de la hauteur des
/// boutons, au même style que les jauges.
fn status_chip(ui: &mut Ui, dot: Color32, text: &str) -> egui::Response {
    let font = egui::FontId::proportional(12.0);
    let galley = ui.painter().layout_no_wrap(text.to_owned(), font, theme::text());
    let size = egui::vec2(galley.size().x + 30.0, TOP_BAR_ITEM_HEIGHT);
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::hover());
    let painter = ui.painter();
    let radius = egui::CornerRadius::same(3);
    painter.rect_filled(rect, radius, theme::code_bg());
    painter.rect_stroke(
        rect,
        radius,
        egui::Stroke::new(1.0, theme::hover_border()),
        egui::StrokeKind::Inside,
    );
    painter.circle_filled(egui::pos2(rect.left() + 12.0, rect.center().y), 4.0, dot);
    let pos = egui::pos2(rect.left() + 22.0, rect.center().y - galley.size().y / 2.0);
    painter.galley(pos, galley, theme::text());
    response
}

/// Barre de progression dessinée, de largeur et hauteur données. Remplace
/// `egui::ProgressBar` partout, pour le même rendu dans les deux thèmes.
fn bar(ui: &mut Ui, width: f32, height: f32, fraction: f32, fill: Color32, text: &str) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
    let painter = ui.painter();
    let radius = egui::CornerRadius::same(3);
    painter.rect_filled(rect, radius, theme::code_bg());
    let width = rect.width() * fraction.clamp(0.0, 1.0);
    if width > 0.0 {
        let filled = egui::Rect::from_min_size(rect.min, egui::vec2(width.max(3.0), rect.height()));
        painter.rect_filled(filled, radius, fill.gamma_multiply(0.85));
    }
    painter.rect_stroke(
        rect,
        radius,
        egui::Stroke::new(1.0, theme::hover_border()),
        egui::StrokeKind::Inside,
    );
    painter.text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        text,
        egui::FontId::proportional(11.0),
        theme::strong(),
    );
    response
}

/// Case « FlashAttention pour tous les types de cache KV » (mise à jour et
/// installation, backend CUDA).
fn fa_all_checkbox(ui: &mut Ui, on: &mut bool, fr: bool) {
    ui.checkbox(
        on,
        tr(
            fr,
            "FlashAttention pour tous les types de cache KV (GGML_CUDA_FA_QUANTS=all)",
            "FlashAttention for every KV cache type (GGML_CUDA_FA_QUANTS=all)",
        ),
    )
    .on_hover_text(tr(
        fr,
        "Par défaut, llama.cpp ne compile les noyaux FlashAttention CUDA que pour les paires \
         K/V f16-f16, q4_0-q4_0, q8_0-q8_0 et bf16-bf16. Cochée, la release les compile pour \
         toutes les paires de f16, bf16, q8_0, q5_1, q5_0, q4_1 et q4_0, y compris mixtes (K \
         q8_0 + V q4_0). La compilation de ces noyaux est nettement plus longue. iq4_nl et f32 \
         n'ont pas de noyau FlashAttention CUDA, quelle que soit l'option. La release porte le \
         suffixe -fa-all.",
        "By default llama.cpp only compiles the CUDA FlashAttention kernels for the K/V pairs \
         f16-f16, q4_0-q4_0, q8_0-q8_0 and bf16-bf16. When checked, the release compiles them \
         for every pair of f16, bf16, q8_0, q5_1, q5_0, q4_1 and q4_0, mixed ones included (K \
         q8_0 + V q4_0). Building those kernels takes notably longer. iq4_nl and f32 have no \
         CUDA FlashAttention kernel, whatever the option. The release gets the -fa-all suffix.",
    ));
}

/// Réglage du benchmark : libellé et liste fermée (deux cellules de grille).
fn bench_combo(
    ui: &mut Ui,
    id: &str,
    label: &str,
    tip: &str,
    value: &mut String,
    choices: &[&str],
    enabled: bool,
) {
    ui.label(RichText::new(label).color(theme::muted()).size(12.0))
        .on_hover_text(tip);
    ui.add_enabled_ui(enabled, |ui| {
        ComboBox::from_id_salt(id)
            .width(72.0)
            .selected_text(value.as_str())
            .show_ui(ui, |ui| {
                for c in choices {
                    ui.selectable_value(value, c.to_string(), *c);
                }
            })
            .response
            .on_hover_text(tip);
    });
}

/// Liste à cocher du benchmark (versions ou modèles) : titre avec le compte,
/// boutons Tout / Aucun, cases dans un cadre défilant. `items` : (clé,
/// libellé) ; `selected` : les clés cochées.
#[allow(clippy::too_many_arguments)]
fn selection_list(
    ui: &mut Ui,
    id: &str,
    title: &str,
    tip: &str,
    empty: &str,
    items: &[(String, String)],
    selected: &mut Vec<String>,
    enabled: bool,
    fr: bool,
) {
    let count = items.iter().filter(|(k, _)| selected.contains(k)).count();
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(format!("{title} ({count}/{})", items.len()))
                .strong()
                .size(12.0),
        )
        .on_hover_text(tip);
        ui.add_enabled_ui(enabled && !items.is_empty(), |ui| {
            if ui.small_button(tr(fr, "Tout", "All")).clicked() {
                for (k, _) in items {
                    if !selected.contains(k) {
                        selected.push(k.clone());
                    }
                }
            }
            if ui.small_button(tr(fr, "Aucun", "None")).clicked() {
                selected.clear();
            }
        });
    });
    egui::Frame::new()
        .fill(theme::code_bg())
        .stroke(egui::Stroke::new(1.0, theme::border()))
        .corner_radius(3)
        .inner_margin(6)
        .show(ui, |ui| {
            egui::ScrollArea::vertical()
                .id_salt(id)
                .max_height(120.0)
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    if items.is_empty() {
                        hint(ui, empty);
                    }
                    for (key, label) in items {
                        let mut on = selected.contains(key);
                        if ui
                            .add_enabled(enabled, egui::Checkbox::new(&mut on, label.as_str()))
                            .changed()
                        {
                            if on {
                                selected.push(key.clone());
                            } else {
                                selected.retain(|k| k != key);
                            }
                        }
                    }
                });
        });
}

/// Débit selon le contexte : un point par requête, reliés dans l'ordre du
/// contexte pour montrer la tendance. `points` : triés par contexte, chacun
/// avec le détail de sa requête, affiché au survol.
fn context_plot(
    ui: &mut Ui,
    id: &str,
    title: &str,
    points: Vec<([f64; 2], String)>,
    color: Color32,
    height: f32,
    fr: bool,
) {
    ui.label(RichText::new(title).strong().size(12.0).color(color))
        .on_hover_text(tr(
            fr,
            "Un point par requête : en abscisse, les tokens dans le contexte à la fin de la \
             requête ; en ordonnée, le débit. Le débit baisse d'ordinaire quand le contexte \
             grandit. Survolez un point pour la taille du prompt et le détail de la requête ; \
             molette + Ctrl pour zoomer, double clic pour revenir.",
            "One point per request: horizontally, the tokens in context at the end of the \
             request; vertically, the throughput. Throughput usually drops as the context \
             grows. Hover a point for the prompt size and the request's detail; Ctrl + wheel \
             to zoom, double click to reset.",
        ));
    let (xy, details): (Vec<[f64; 2]>, Vec<String>) = points.into_iter().unzip();
    let axis = tr(fr, "contexte", "context");
    egui_plot::Plot::new(id)
        .height(height)
        .include_x(0.0)
        .include_y(0.0)
        .allow_scroll(false)
        // Largeur réservée aux graduations : sinon, calculée sur l'image
        // précédente, elle masquait parfois toutes les valeurs de l'axe.
        .y_axis_min_width(44.0)
        // Pas « rond » visant quatre graduations : l'espacement par puissances
        // de 10 d'egui_plot ne laissait sur un graphique bas que le « 0 ».
        .y_grid_spacer(egui_plot::uniform_grid_spacer(|input| {
            let step = nice_step((input.bounds.1 - input.bounds.0) / 4.0);
            [step / 5.0, step, step * 5.0]
        }))
        .x_axis_label(tr(fr, "contexte (tokens)", "context (tokens)"))
        .label_formatter(move |pos| match pos {
            egui_plot::HoverPosition::NearDataPoint { index, .. } => details.get(*index).cloned(),
            egui_plot::HoverPosition::Elsewhere { position } => Some(format!(
                "{axis} {}\n{} t/s",
                spaced(position.x, 0, fr),
                spaced(position.y, 1, fr)
            )),
        })
        .show(ui, |plot| {
            plot.line(
                egui_plot::Line::new(title, xy.clone())
                    .color(color.gamma_multiply(0.45))
                    .width(1.0),
            );
            plot.points(
                egui_plot::Points::new(title, xy)
                    .color(color)
                    .radius(3.0)
                    .filled(true),
            );
        });
}

/// `125125.5` → `125 125,5` en français (`125 125.5` en anglais) : les
/// milliers séparés par une espace, plus lisibles.
fn spaced(value: f64, decimals: usize, fr: bool) -> String {
    let text = format!("{:.*}", decimals, value.abs());
    let (int, frac) = text.split_once('.').unwrap_or((&text, ""));
    let mut grouped = String::new();
    for (i, c) in int.chars().enumerate() {
        if i > 0 && (int.len() - i) % 3 == 0 {
            grouped.push(' ');
        }
        grouped.push(c);
    }
    if value < 0.0 && value.abs() >= 0.5 * 10f64.powi(-(decimals as i32)) {
        grouped.insert(0, '-');
    }
    if !frac.is_empty() {
        grouped.push(if fr { ',' } else { '.' });
        grouped.push_str(frac);
    }
    grouped
}

/// Pas de graduation 1, 2 ou 5 × 10^n, le plus proche au-dessus de `raw`.
fn nice_step(raw: f64) -> f64 {
    if !raw.is_finite() || raw <= 0.0 {
        return 1.0;
    }
    let power = 10f64.powf(raw.log10().floor());
    let unit = raw / power;
    let nice = if unit <= 1.0 {
        1.0
    } else if unit <= 2.0 {
        2.0
    } else if unit <= 5.0 {
        5.0
    } else {
        10.0
    };
    nice * power
}

/// `12:34`, ou `1:02:03` au-delà d'une heure.
fn format_elapsed(d: Duration) -> String {
    let s = d.as_secs();
    if s >= 3600 {
        format!("{}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
    } else {
        format!("{}:{:02}", s / 60, s % 60)
    }
}

/// Petite étiquette arrondie.
fn chip(ui: &mut Ui, text: &str, primary: bool) -> egui::Response {
    let (fill, color) = if primary {
        theme::chip_primary()
    } else {
        theme::chip()
    };
    // Un seul widget de taille connue d'avance : `horizontal_wrapped` peut
    // alors le passer à la ligne. Un Frame contenant un libellé ne se replie
    // pas : la rangée de tags débordait de la carte et du panneau.
    let galley = ui.painter().layout_no_wrap(text.to_owned(), egui::FontId::proportional(11.0), color);
    let size = galley.size() + egui::vec2(14.0, 2.0);
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::hover());
    if ui.is_rect_visible(rect) {
        ui.painter().rect_filled(rect, egui::CornerRadius::same(9), fill);
        let pos = rect.center() - galley.size() / 2.0;
        ui.painter().galley(pos, galley, color);
    }
    response
}

/// Traduction en ligne hors de `App`, pour les fonctions libres.
fn tr(fr: bool, fr_text: &'static str, en_text: &'static str) -> &'static str {
    if fr {
        fr_text
    } else {
        en_text
    }
}

// ---------------------------------------------------------------------- outils

fn panel_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(theme::bg())
        .inner_margin(egui::Margin::same(12))
}

/// En-tête de section pliable de la colonne de gauche : même flèche que les
/// groupes de paramètres, titre cliquable, compteur éventuel. Renvoie `true`
/// si la section est dépliée ; l'état est retenu par egui pour la session.
fn fold_header(ui: &mut Ui, id: &str, title: &str, tip: &str, count: Option<String>) -> bool {
    let id = ui.make_persistent_id(id);
    let mut state =
        egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, true);
    ui.horizontal(|ui| {
        state.show_toggle_button(ui, egui::collapsing_header::paint_default_icon);
        let label = ui
            .add(
                egui::Label::new(
                    RichText::new(title).color(theme::text()).strong().size(14.0),
                )
                .selectable(false)
                .sense(egui::Sense::click()),
            )
            .on_hover_text(tip);
        if label.clicked() {
            state.toggle(ui);
        }
        if let Some(count) = count {
            ui.label(RichText::new(count).color(theme::muted()).size(11.0));
        }
    });
    let open = state.is_open();
    state.store(ui.ctx());
    if open {
        ui.add_space(6.0);
    }
    open
}

/// Libellé de champ, avec son explication au survol.
fn field_label(ui: &mut Ui, text: &str, tip: &str) {
    ui.label(RichText::new(text).color(theme::muted()).size(11.0))
        .on_hover_text(tip);
}

fn hint(ui: &mut Ui, text: &str) {
    ui.label(RichText::new(text).color(theme::muted()).size(11.0).italics());
}

/// Comme `hint`, mais explique en détail au survol.
fn hint_tip(ui: &mut Ui, text: &str, tip: &str) {
    ui.label(RichText::new(text).color(theme::muted()).size(11.0).italics())
        .on_hover_text(tip);
}

fn dir_row(
    ui: &mut Ui,
    label: &str,
    value: &str,
    button: &str,
    tip: &str,
    mut on_browse: impl FnMut(),
) {
    field_label(ui, label, tip);
    ui.horizontal(|ui| {
        if ui.button(button).on_hover_text(tip).clicked() {
            on_browse();
        }
        let shown = if value.is_empty() { "—" } else { value };
        ui.label(
            RichText::new(shorten(shown, 34))
                .monospace()
                .size(11.0)
                .color(theme::muted()),
        )
        .on_hover_text(format!("{shown}\n\n{tip}"));
    });
}

/// Tronque au milieu pour garder le début et la fin d'un chemin.
fn shorten(text: &str, max: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= max {
        return text.to_string();
    }
    let head: String = chars[..max / 2 - 2].iter().collect();
    let tail: String = chars[chars.len() - (max / 2 - 1)..].iter().collect();
    format!("{head}…{tail}")
}

fn line_color(line: &str) -> Color32 {
    let lower = line.to_lowercase();
    if line.starts_with('!') || lower.contains("error") || lower.contains("failed") {
        theme::red()
    } else if line.starts_with('$') {
        theme::accent()
    } else if line.starts_with('▶') || line.starts_with('✔') || line.contains('✔') {
        theme::green()
    } else if line.starts_with('■') || line.starts_with("——") || line.starts_with('↻') {
        theme::orange()
    } else if lower.contains("warn") {
        theme::yellow()
    } else {
        theme::text()
    }
}

fn pick_folder(current: &str) -> Option<PathBuf> {
    let mut dialog = rfd::FileDialog::new();
    let path = PathBuf::from(current);
    if !current.is_empty() && path.is_dir() {
        dialog = dialog.set_directory(path);
    }
    dialog.pick_folder()
}

fn reveal(path: &std::path::Path) {
    #[cfg(windows)]
    {
        let _ = std::process::Command::new("explorer")
            .arg("/select,")
            .arg(path)
            .spawn();
    }
    #[cfg(not(windows))]
    {
        if let Some(dir) = path.parent() {
            let _ = std::process::Command::new("xdg-open").arg(dir).spawn();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thousands_are_spaced() {
        assert_eq!(spaced(125125.0, 0, true), "125 125");
        assert_eq!(spaced(1234567.34, 1, true), "1 234 567,3");
        assert_eq!(spaced(1234.5, 1, false), "1 234.5");
        assert_eq!(spaced(999.0, 0, true), "999");
        assert_eq!(spaced(-1500.0, 0, true), "-1 500");
    }

    #[test]
    fn nice_steps_are_round() {
        assert_eq!(nice_step(212.5), 500.0);
        assert_eq!(nice_step(180.0), 200.0);
        assert_eq!(nice_step(10_000.0), 10_000.0);
        assert!((nice_step(0.03) - 0.05).abs() < 1e-12);
        assert_eq!(nice_step(0.0), 1.0);
    }

    #[test]
    fn shorten_keeps_both_ends() {
        let s = shorten("D:/models/very/long/path/to/a/model-file.gguf", 20);
        assert!(s.starts_with("D:/mod"), "{s}");
        assert!(s.ends_with(".gguf"), "{s}");
        assert!(s.contains('…'), "{s}");
        assert!(s.chars().count() <= 20);
    }

    #[test]
    fn shorten_leaves_short_text_alone() {
        assert_eq!(shorten("abc", 20), "abc");
    }
}
