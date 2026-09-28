//! Mise à jour de llama.cpp depuis les sources.
//!
//! Chaque mise à jour publie une **nouvelle release** dans son propre dossier
//! (`llama-b10512-cuda`) : les binaires en cours d'utilisation ne sont jamais
//! écrasés — Windows refuserait d'ailleurs de remplacer un exe qui tourne — et
//! les versions précédentes restent disponibles, y compris pour le benchmark.
//!
//! Étapes : `git pull --ff-only`, configuration CMake (une fois, en reprenant
//! les options du build existant de l'utilisateur), compilation incrémentale
//! dans un dossier dédié, copie des exe et DLL vers la release, vérification
//! du numéro de build.
//!
//! Sans installation existante, `install` clone d'abord le dépôt dans le
//! dossier choisi puis enchaîne les mêmes étapes, avec le backend choisi.

use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};

#[cfg(windows)]
use std::os::windows::process::CommandExt;
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// Dossier de compilation dédié, dans le dépôt (ignoré par son `.gitignore`
/// grâce au motif `/build*`).
pub const BUILD_DIR_NAME: &str = "build-uillamacpp";

/// Dépôt officiel, cloné lors d'une installation depuis zéro.
pub const REPO_URL: &str = "https://github.com/ggml-org/llama.cpp";

// --------------------------------------------------------------------- outils

const GIT: &str = if cfg!(windows) { "git.exe" } else { "git" };
const CMAKE: &str = if cfg!(windows) { "cmake.exe" } else { "cmake" };
const NVCC: &str = if cfg!(windows) { "nvcc.exe" } else { "nvcc" };

/// Chemin d'un outil : le PATH, puis ses emplacements d'installation
/// habituels. Un outil installé après le démarrage de l'application n'est
/// pas dans le PATH hérité par celle-ci ; sans ce repli, il faudrait la
/// relancer après chaque installation.
pub fn find_tool(exe: &str) -> Option<PathBuf> {
    let from_path = std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|d| d.join(exe))
            .find(|p| p.is_file())
    });
    from_path.or_else(|| known_locations(exe).into_iter().find(|p| p.is_file()))
}

fn known_locations(exe: &str) -> Vec<PathBuf> {
    let pf = PathBuf::from(std::env::var("ProgramFiles").unwrap_or_else(|_| r"C:\Program Files".into()));
    match exe {
        "git.exe" => vec![pf.join(r"Git\cmd\git.exe")],
        "cmake.exe" => vec![pf.join(r"CMake\bin\cmake.exe")],
        "nvcc.exe" => {
            let mut out: Vec<PathBuf> = std::env::var("CUDA_PATH")
                .map(|p| vec![PathBuf::from(p).join(r"bin\nvcc.exe")])
                .unwrap_or_default();
            // Toolkit le plus récent installé, même si CUDA_PATH est absent.
            let root = pf.join(r"NVIDIA GPU Computing Toolkit\CUDA");
            if let Ok(rd) = std::fs::read_dir(root) {
                let mut dirs: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
                dirs.sort();
                out.extend(dirs.into_iter().rev().map(|d| d.join(r"bin\nvcc.exe")));
            }
            out
        }
        _ => Vec::new(),
    }
}

fn tool(exe: &str) -> Command {
    Command::new(find_tool(exe).unwrap_or_else(|| PathBuf::from(exe)))
}

/// Première ligne non vide de la sortie d'une commande.
fn first_line(mut cmd: Command) -> Option<String> {
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);
    let out = cmd.stdin(Stdio::null()).output().ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .map(str::to_string)
}

// --------------------------------------------------------------- prérequis

/// Backends proposés pour une installation depuis zéro.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Backend {
    Cpu,
    Cuda,
    Vulkan,
}

impl Backend {
    pub const ALL: [Backend; 3] = [Backend::Cuda, Backend::Vulkan, Backend::Cpu];

    pub fn label(self) -> &'static str {
        match self {
            Backend::Cpu => "CPU",
            Backend::Cuda => "CUDA (NVIDIA)",
            Backend::Vulkan => "Vulkan",
        }
    }
}

/// Outils présents sur la machine ; `None` = absent.
#[derive(Clone, Debug, Default)]
pub struct Prereqs {
    pub git: Option<String>,
    pub cmake: Option<String>,
    /// Compilateur C++ (Visual Studio ou Build Tools avec l'outillage C++).
    pub msvc: Option<String>,
    pub cuda: Option<String>,
    pub vulkan: Option<String>,
    pub nvidia_gpu: Option<String>,
}

impl Prereqs {
    /// Outils indispensables quel que soit le backend.
    pub fn can_build(&self) -> bool {
        self.git.is_some() && self.cmake.is_some() && self.msvc.is_some()
    }

    pub fn backend_ready(&self, backend: Backend) -> bool {
        match backend {
            Backend::Cpu => true,
            Backend::Cuda => self.cuda.is_some(),
            Backend::Vulkan => self.vulkan.is_some(),
        }
    }

    /// Backend conseillé : le GPU s'il est exploitable, sinon le CPU.
    pub fn suggested(&self) -> Backend {
        if self.nvidia_gpu.is_some() && self.cuda.is_some() {
            Backend::Cuda
        } else if self.vulkan.is_some() {
            Backend::Vulkan
        } else {
            Backend::Cpu
        }
    }
}

/// Détecte les outils ; lance quelques exe, à appeler hors du rendu.
pub fn detect_prereqs() -> Prereqs {
    let version = |exe: &str, prefix: &str| -> Option<String> {
        let mut cmd = tool(exe);
        cmd.arg("--version");
        first_line(cmd).map(|l| l.trim_start_matches(prefix).trim().to_string())
    };
    let git = find_tool(GIT).and_then(|_| version(GIT, "git version"));
    let cmake = find_tool(CMAKE).and_then(|_| version(CMAKE, "cmake version"));

    let msvc = if cfg!(windows) {
        let pf86 = std::env::var("ProgramFiles(x86)")
            .unwrap_or_else(|_| r"C:\Program Files (x86)".into());
        let vswhere = PathBuf::from(pf86).join(r"Microsoft Visual Studio\Installer\vswhere.exe");
        vswhere.is_file().then(|| {
            let mut cmd = Command::new(&vswhere);
            cmd.args([
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "displayName",
            ]);
            first_line(cmd)
        })
        .flatten()
    } else {
        Some("compilateur système".to_string())
    };

    let cuda = find_tool(NVCC).and_then(|nvcc| {
        let mut cmd = Command::new(nvcc);
        cmd.arg("--version");
        #[cfg(windows)]
        cmd.creation_flags(CREATE_NO_WINDOW);
        let out = cmd.output().ok()?;
        let text = String::from_utf8_lossy(&out.stdout).into_owned();
        // « Cuda compilation tools, release 13.3, V13.3.73 »
        let i = text.find("release ")? + "release ".len();
        Some(text[i..].chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect())
    });

    // VULKAN_SDK, ou le SDK le plus récent de C:\VulkanSDK : la variable
    // n'est pas visible d'une application lancée avant l'installation.
    let vulkan_root = std::env::var("VULKAN_SDK").ok().map(PathBuf::from).or_else(|| {
        let mut dirs: Vec<PathBuf> = std::fs::read_dir(r"C:\VulkanSDK")
            .ok()?
            .flatten()
            .map(|e| e.path())
            .collect();
        dirs.sort();
        dirs.pop()
    });
    let vulkan = vulkan_root.and_then(|sdk| {
        let glslc = if cfg!(windows) { r"Bin\glslc.exe" } else { "bin/glslc" };
        sdk.join(glslc).is_file().then(|| {
            sdk.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "SDK".into())
        })
    });

    let nvidia_gpu = {
        let mut cmd = Command::new("nvidia-smi");
        cmd.args(["--query-gpu=name", "--format=csv,noheader"]);
        first_line(cmd)
    };

    Prereqs {
        git,
        cmake,
        msvc,
        cuda,
        vulkan,
        nvidia_gpu,
    }
}

/// Commande winget qui installe un prérequis manquant.
pub fn install_hint(what: &str) -> &'static str {
    match what {
        "git" => "winget install --id Git.Git -e",
        "cmake" => "winget install --id Kitware.CMake -e",
        "msvc" => {
            "winget install --id Microsoft.VisualStudio.2022.BuildTools -e --override \"--passive --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended\""
        }
        "cuda" => "winget install --id Nvidia.CUDA -e",
        "vulkan" => "winget install --id KhronosGroup.VulkanSDK -e",
        _ => "",
    }
}

// ------------------------------------------------------------------ détection

/// Dépôt llama.cpp contenant `from` (ou l'un de ses parents).
pub fn detect_source(from: &Path) -> Option<PathBuf> {
    from.ancestors()
        .find(|d| is_source(d))
        .map(Path::to_path_buf)
}

pub fn is_source(dir: &Path) -> bool {
    dir.join(".git").exists()
        && dir.join("CMakeLists.txt").is_file()
        && dir.join("include").join("llama.h").is_file()
}

/// Cache CMake du build existant, dont on reprend les options : celui d'un
/// parent du dossier de version, sinon `<source>/build`.
pub fn detect_reference_cache(from: &Path, source: &Path) -> Option<PathBuf> {
    from.ancestors()
        .map(|d| d.join("CMakeCache.txt"))
        .find(|p| p.is_file())
        .or_else(|| {
            let p = source.join("build").join("CMakeCache.txt");
            p.is_file().then_some(p)
        })
}

/// Emplacement proposé pour les releases : à côté du dépôt, pas dedans.
pub fn default_releases_dir(source: &Path) -> PathBuf {
    let name = source
        .file_name()
        .map(|n| format!("{}-releases", n.to_string_lossy()))
        .unwrap_or_else(|| "llama.cpp-releases".into());
    source.parent().unwrap_or(source).join(name)
}

// ----------------------------------------------------------- options de build

/// Options de compilation relues dans un `CMakeCache.txt`.
#[derive(Clone, Default, Debug)]
pub struct BuildConfig {
    pub generator: String,
    pub instance: String,
    pub platform: String,
    pub toolset: String,
    /// (nom, type, valeur) des options llama.cpp / ggml à reproduire.
    pub defines: Vec<(String, String, String)>,
}

impl BuildConfig {
    /// Options d'une installation neuve : celles de llama.cpp par défaut, le
    /// backend choisi, et sans les tests (inutiles ici, et longs à compiler).
    pub fn for_backend(backend: Backend) -> Self {
        let mut defines = vec![(
            "LLAMA_BUILD_TESTS".to_string(),
            "BOOL".to_string(),
            "OFF".to_string(),
        )];
        match backend {
            Backend::Cpu => {}
            Backend::Cuda => {
                defines.push(("GGML_CUDA".into(), "BOOL".into(), "ON".into()));
                // Uniquement l'architecture du GPU présent : bien plus rapide
                // à compiler que toutes les architectures.
                defines.push((
                    "CMAKE_CUDA_ARCHITECTURES".into(),
                    "UNINITIALIZED".into(),
                    "native".into(),
                ));
            }
            Backend::Vulkan => defines.push(("GGML_VULKAN".into(), "BOOL".into(), "ON".into())),
        }
        BuildConfig {
            defines,
            ..Default::default()
        }
    }

    pub fn read(cache: &Path) -> anyhow::Result<Self> {
        Ok(Self::parse(&std::fs::read_to_string(cache)?))
    }

    pub fn parse(text: &str) -> Self {
        let mut cfg = BuildConfig::default();
        for line in text.lines() {
            if line.starts_with('#') || line.starts_with("//") {
                continue;
            }
            let Some((key, value)) = line.split_once('=') else {
                continue;
            };
            let Some((name, kind)) = key.split_once(':') else {
                continue;
            };
            match name {
                "CMAKE_GENERATOR" => cfg.generator = value.to_string(),
                "CMAKE_GENERATOR_INSTANCE" => cfg.instance = value.to_string(),
                "CMAKE_GENERATOR_PLATFORM" => cfg.platform = value.to_string(),
                "CMAKE_GENERATOR_TOOLSET" => cfg.toolset = value.to_string(),
                _ => {
                    // Les options du projet et les quelques variables CMake
                    // qui changent le résultat. Les chemins détectés
                    // (FILEPATH, PATH) et l'interne sont recalculés.
                    let wanted = name.starts_with("GGML_")
                        || name.starts_with("LLAMA_")
                        || matches!(
                            name,
                            "BUILD_SHARED_LIBS"
                                | "CMAKE_CUDA_ARCHITECTURES"
                                | "CMAKE_HIP_ARCHITECTURES"
                        );
                    let kind_ok = matches!(kind, "BOOL" | "STRING" | "UNINITIALIZED");
                    if wanted && kind_ok && !value.ends_with("-NOTFOUND") {
                        cfg.defines
                            .push((name.to_string(), kind.to_string(), value.to_string()));
                    }
                }
            }
        }
        cfg
    }

    /// FlashAttention CUDA compilé pour toutes les paires de types K/V ?
    /// Par défaut llama.cpp ne compile que f16, q4_0, q8_0 et bf16, chacun
    /// avec lui-même (`GGML_CUDA_FA_QUANTS`).
    pub fn fa_all(&self) -> bool {
        self.is_on("GGML_CUDA_FA_ALL_QUANTS")
            || self
                .defines
                .iter()
                .any(|(n, _, v)| n == "GGML_CUDA_FA_QUANTS" && v.trim().eq_ignore_ascii_case("all"))
    }

    /// Demande (ou retire) toutes les paires K/V. `GGML_CUDA_FA_QUANTS=all`
    /// pour les sources récentes, `GGML_CUDA_FA_ALL_QUANTS` pour les
    /// anciennes, qui n'ont que cette option (les récentes l'acceptent encore,
    /// avec un avertissement).
    pub fn set_fa_all(&mut self, on: bool) {
        self.defines
            .retain(|(n, ..)| n != "GGML_CUDA_FA_QUANTS" && n != "GGML_CUDA_FA_ALL_QUANTS");
        self.defines.extend(fa_defines(on).into_iter().filter_map(|d| {
            let (name, value) = d.strip_prefix("-D")?.split_once('=')?;
            let (name, kind) = name.split_once(':')?;
            Some((name.to_string(), kind.to_string(), value.to_string()))
        }));
    }

    fn is_on(&self, name: &str) -> bool {
        self.defines.iter().any(|(n, _, v)| {
            n == name && matches!(v.to_uppercase().as_str(), "ON" | "1" | "TRUE" | "YES")
        })
    }

    /// Backend principal, pour nommer la release.
    pub fn backend(&self) -> &'static str {
        for (option, name) in [
            ("GGML_CUDA", "cuda"),
            ("GGML_HIP", "rocm"),
            ("GGML_SYCL", "sycl"),
            ("GGML_VULKAN", "vulkan"),
            ("GGML_METAL", "metal"),
            ("GGML_OPENCL", "opencl"),
            ("GGML_MUSA", "musa"),
            ("GGML_CANN", "cann"),
        ] {
            if self.is_on(option) {
                return name;
            }
        }
        "cpu"
    }

    /// Résumé lisible : « CUDA · Visual Studio 17 2022 · DLL partagées ».
    pub fn summary(&self, fr: bool) -> String {
        let mut parts = vec![self.backend().to_uppercase()];
        if !self.generator.is_empty() {
            parts.push(self.generator.clone());
        }
        if self.is_on("BUILD_SHARED_LIBS") {
            parts.push(if fr { "DLL partagées" } else { "shared DLLs" }.to_string());
        }
        if let Some((_, _, arch)) = self
            .defines
            .iter()
            .find(|(n, ..)| n == "CMAKE_CUDA_ARCHITECTURES")
        {
            parts.push(format!("arch {arch}"));
        }
        parts.join(" · ")
    }

    fn configure_args(&self, source: &Path, build: &Path) -> Vec<String> {
        let mut args = vec![
            "-S".to_string(),
            source.to_string_lossy().into_owned(),
            "-B".to_string(),
            build.to_string_lossy().into_owned(),
        ];
        if !self.generator.is_empty() {
            args.extend(["-G".to_string(), self.generator.clone()]);
        }
        if !self.platform.is_empty() {
            args.extend(["-A".to_string(), self.platform.clone()]);
        }
        if !self.toolset.is_empty() {
            args.extend(["-T".to_string(), self.toolset.clone()]);
        }
        if !self.instance.is_empty() {
            args.push(format!("-DCMAKE_GENERATOR_INSTANCE={}", self.instance));
        }
        for (name, kind, value) in &self.defines {
            if kind == "UNINITIALIZED" {
                args.push(format!("-D{name}={value}"));
            } else {
                args.push(format!("-D{name}:{kind}={value}"));
            }
        }
        args
    }
}

/// Arguments CMake qui activent (ou retirent) toutes les paires K/V de
/// FlashAttention. Sans elles, la liste par défaut de llama.cpp s'applique
/// (`-U` efface la valeur mise en cache).
fn fa_defines(on: bool) -> Vec<String> {
    if on {
        vec![
            "-DGGML_CUDA_FA_ALL_QUANTS:BOOL=ON".into(),
            "-DGGML_CUDA_FA_QUANTS:STRING=all".into(),
        ]
    } else {
        vec!["-DGGML_CUDA_FA_ALL_QUANTS:BOOL=OFF".into()]
    }
}

// ------------------------------------------------------------------------ git

#[derive(Clone, Debug)]
pub struct GitState {
    pub branch: String,
    /// Numéro de build llama.cpp : nombre de commits depuis l'origine.
    pub build: u32,
    pub commit: String,
}

fn git(source: &Path, args: &[&str]) -> anyhow::Result<String> {
    let mut cmd = tool(GIT);
    cmd.arg("-C").arg(source).args(args);
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);
    let out = cmd.output().map_err(|e| anyhow::anyhow!("git introuvable : {e}"))?;
    if !out.status.success() {
        anyhow::bail!(
            "git {} : {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

pub fn git_state(source: &Path) -> anyhow::Result<GitState> {
    Ok(GitState {
        branch: git(source, &["rev-parse", "--abbrev-ref", "HEAD"])?,
        build: git(source, &["rev-list", "--count", "HEAD"])?.parse()?,
        commit: git(source, &["rev-parse", "--short", "HEAD"])?,
    })
}

/// Résultat de « Vérifier les mises à jour ».
#[derive(Clone, Debug)]
pub struct Check {
    pub current: GitState,
    pub upstream: String,
    pub behind: u32,
    pub remote_build: u32,
    /// `hash sujet`, du plus récent au plus ancien.
    pub commits: Vec<String>,
}

fn check_now(source: &Path) -> anyhow::Result<Check> {
    let current = git_state(source)?;
    git(source, &["fetch", "--quiet"])?;
    let upstream = git(source, &["rev-parse", "--abbrev-ref", "@{u}"]).map_err(|_| {
        anyhow::anyhow!(
            "la branche « {} » ne suit aucune branche distante",
            current.branch
        )
    })?;
    let behind = git(source, &["rev-list", "--count", "HEAD..@{u}"])?.parse()?;
    let remote_build = git(source, &["rev-list", "--count", "@{u}"])?.parse()?;
    let commits = git(source, &["log", "--format=%h %s", "-n", "15", "HEAD..@{u}"])?
        .lines()
        .map(str::to_string)
        .collect();
    Ok(Check {
        current,
        upstream,
        behind,
        remote_build,
        commits,
    })
}

// ---------------------------------------------------------------- tâches

pub enum Event {
    Line(String),
    Stage {
        index: usize,
        total: usize,
        label: &'static str,
    },
    /// Avancement de la compilation, s'il est mesurable.
    Progress(f32),
    Checked(Result<Check, String>),
    Done(Result<Release, String>),
}

#[derive(Clone, Debug)]
pub struct Release {
    pub name: String,
    pub dir: PathBuf,
    pub build: u32,
    /// Ancienne version supprimée à la demande, si la suppression a réussi.
    pub removed: Option<String>,
}

/// Tâche de fond (vérification ou mise à jour), annulable.
pub struct Task {
    pub rx: Receiver<Event>,
    cancel: Arc<AtomicBool>,
    pid: Arc<Mutex<Option<u32>>>,
}

impl Task {
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
        if let Some(pid) = self.pid.lock().ok().and_then(|g| *g) {
            kill_tree(pid);
        }
    }
}

fn kill_tree(pid: u32) {
    let mut cmd = Command::new("taskkill");
    cmd.args(["/PID", &pid.to_string(), "/T", "/F"])
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);
    let _ = cmd.status();
}

struct Ctx {
    tx: Sender<Event>,
    cancel: Arc<AtomicBool>,
    pid: Arc<Mutex<Option<u32>>>,
}

impl Ctx {
    fn line(&self, text: impl Into<String>) {
        let _ = self.tx.send(Event::Line(text.into()));
    }

    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }
}

fn spawn_task(work: impl FnOnce(&Ctx) + Send + 'static) -> Task {
    let (tx, rx) = channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let pid = Arc::new(Mutex::new(None));
    let ctx = Ctx {
        tx,
        cancel: cancel.clone(),
        pid: pid.clone(),
    };
    std::thread::spawn(move || work(&ctx));
    Task { rx, cancel, pid }
}

/// Interroge le dépôt distant (réseau) sans rien modifier.
pub fn check(source: PathBuf) -> Task {
    spawn_task(move |ctx| {
        let result = check_now(&source).map_err(|e| e.to_string());
        let _ = ctx.tx.send(Event::Checked(result));
    })
}

pub struct Plan {
    pub source: PathBuf,
    /// Cache CMake dont reprendre les options ; `None` = `backend`.
    pub reference: Option<PathBuf>,
    /// Backend d'une première configuration sans build de référence.
    pub backend: Backend,
    pub releases_dir: PathBuf,
    /// Cloner le dépôt au lieu de le mettre à jour (installation depuis zéro).
    pub clone: bool,
    /// Release à supprimer une fois la nouvelle vérifiée. `None` par défaut :
    /// une mise à jour conserve toujours l'ancienne version.
    pub delete_previous: Option<PathBuf>,
    /// FlashAttention CUDA pour toutes les paires K/V : `Some` impose le
    /// choix (reconfiguration si le build existant diffère), `None` garde
    /// celui du build.
    pub fa_all: Option<bool>,
}

/// Met à jour les sources, compile et publie une nouvelle release.
pub fn update(plan: Plan) -> Task {
    spawn_task(move |ctx| {
        let result = run_update(&plan, ctx).map_err(|e| {
            if ctx.cancelled() {
                "annulé".to_string()
            } else {
                e.to_string()
            }
        });
        let _ = ctx.tx.send(Event::Done(result));
    })
}

/// Clone llama.cpp dans `plan.source`, le compile et publie sa première
/// release.
pub fn install(plan: Plan) -> Task {
    update(Plan {
        clone: true,
        delete_previous: None,
        ..plan
    })
}

const STAGES: usize = 5;

fn stage(ctx: &Ctx, index: usize, label: &'static str) -> anyhow::Result<()> {
    anyhow::ensure!(!ctx.cancelled(), "annulé");
    let _ = ctx.tx.send(Event::Stage {
        index,
        total: STAGES,
        label,
    });
    ctx.line(format!("—— {index}/{STAGES} {label} ——"));
    Ok(())
}

fn run_update(plan: &Plan, ctx: &Ctx) -> anyhow::Result<Release> {
    let source = &plan.source;
    let build_dir = source.join(BUILD_DIR_NAME);

    // 1. Sources.
    if plan.clone {
        stage(ctx, 1, "Téléchargement des sources (git clone)")?;
        anyhow::ensure!(
            !source.exists() || std::fs::read_dir(source)?.next().is_none(),
            "{} existe déjà et n'est pas vide",
            source.display()
        );
        if let Some(parent) = source.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // Clone partiel : tout l'historique des commits (le numéro de build
        // en dépend), mais le contenu des anciennes révisions n'est
        // téléchargé qu'à la demande.
        //
        // core.longpaths : le plus long chemin du dépôt dépasse 160
        // caractères ; sans cette option, tout dossier d'installation un peu
        // profond fait échouer l'extraction (« Filename too long »). Passée à
        // `clone`, l'option est aussi enregistrée dans le dépôt pour les pull.
        let mut clone = tool(GIT);
        clone
            .args([
                "clone",
                "-c",
                "core.longpaths=true",
                "--progress",
                "--filter=blob:none",
                REPO_URL,
            ])
            .arg(source);
        let tx = ctx.tx.clone();
        let cloned = run_streamed(clone, ctx, |line| match git_progress(line) {
            Some(p) => {
                let _ = tx.send(Event::Progress(p));
                // Les mises à jour de pourcentage ne vont pas dans la
                // console, seulement la ligne finale.
                !line.contains("done")
            }
            None => false,
        });
        if let Err(e) = cloned {
            // Le dossier était absent ou vide : un clone partiel ne doit pas
            // bloquer la tentative suivante.
            let _ = std::fs::remove_dir_all(source);
            return Err(e);
        }
    } else {
        stage(ctx, 1, "Récupération des sources (git pull)")?;
        anyhow::ensure!(is_source(source), "{} n'est pas un dépôt llama.cpp", source.display());
        let mut pull = tool(GIT);
        pull.args(["-c", "core.longpaths=true", "-C"])
            .arg(source)
            .args(["pull", "--ff-only"]);
        run_streamed(pull, ctx, |_| false)?;
    }
    anyhow::ensure!(is_source(source), "{} n'est pas un dépôt llama.cpp", source.display());
    let state = git_state(source)?;
    ctx.line(format!("sources : {} · b{} ({})", state.branch, state.build, state.commit));

    // 2. Configuration, seulement la première fois : les suivantes sont
    //    incrémentales et CMake se reconfigure seul si besoin.
    stage(ctx, 2, "Configuration CMake")?;
    if build_dir.join("CMakeCache.txt").is_file() {
        ctx.line(format!("configuration existante réutilisée : {}", build_dir.display()));
        let current = BuildConfig::read(&build_dir.join("CMakeCache.txt"))?;
        if let Some(want) = plan.fa_all.filter(|w| current.backend() == "cuda" && *w != current.fa_all()) {
            ctx.line(if want {
                "FlashAttention : toutes les paires de types K/V (compilation plus longue)"
            } else {
                "FlashAttention : paires K/V par défaut de llama.cpp"
            });
            let mut cmake = tool(CMAKE);
            cmake.arg("-S").arg(source).arg("-B").arg(&build_dir).args(fa_defines(want));
            if !want {
                cmake.arg("-UGGML_CUDA_FA_QUANTS");
            }
            run_streamed(cmake, ctx, |_| false)?;
        }
    } else {
        let mut config = match &plan.reference {
            Some(cache) => {
                ctx.line(format!("options reprises de {}", cache.display()));
                BuildConfig::read(cache)?
            }
            None => {
                ctx.line(format!("backend choisi : {}", plan.backend.label()));
                BuildConfig::for_backend(plan.backend)
            }
        };
        if let Some(want) = plan.fa_all.filter(|_| config.backend() == "cuda") {
            config.set_fa_all(want);
        }
        let mut cmake = tool(CMAKE);
        cmake.args(config.configure_args(source, &build_dir));
        run_streamed(cmake, ctx, |_| false)?;
    }
    let built = BuildConfig::read(&build_dir.join("CMakeCache.txt"))?;

    // 3. Compilation (toutes les cibles : les DLL partagées doivent rester
    //    cohérentes avec chaque exe de la release).
    stage(ctx, 3, "Compilation")?;
    let total = count_projects(&build_dir);
    let mut done = 0usize;
    let jobs = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let mut cmake = tool(CMAKE);
    cmake
        .arg("--build")
        .arg(&build_dir)
        .args(["--config", "Release", "--parallel", &jobs.to_string()])
        // Messages MSBuild en anglais : stables et sans problème d'encodage.
        .env("VSLANG", "1033");
    let tx = ctx.tx.clone();
    run_streamed(cmake, ctx, |line| {
        if let Some(fraction) = build_progress(line, &mut done, total) {
            let _ = tx.send(Event::Progress(fraction));
        }
        false
    })?;

    // 4. Publication dans un nouveau dossier.
    stage(ctx, 4, "Copie de la release")?;
    let bin = [build_dir.join("bin").join("Release"), build_dir.join("bin")]
        .into_iter()
        .find(|d| d.join(crate::discovery::SERVER_EXE).is_file())
        .ok_or_else(|| anyhow::anyhow!("llama-server introuvable après compilation"))?;
    std::fs::create_dir_all(&plan.releases_dir)?;
    // « -fa-all » : la release compile FlashAttention pour toutes les paires
    // K/V, et se distingue ainsi d'une compilation par défaut du même build.
    let base = format!(
        "llama-b{}-{}{}",
        state.build,
        built.backend(),
        if built.backend() == "cuda" && built.fa_all() { "-fa-all" } else { "" }
    );
    let name = unique_name(&plan.releases_dir, &base);
    let target = plan.releases_dir.join(&name);
    let copied = copy_binaries(&bin, &target)?;
    ctx.line(format!("{copied} fichiers copiés vers {}", target.display()));

    // 5. Vérification.
    stage(ctx, 5, "Vérification")?;
    match crate::discovery::probe_version(&target.join(crate::discovery::SERVER_EXE)) {
        Some((build, commit)) if build == state.build => {
            ctx.line(format!("✔ llama-server build {build} ({commit})"));
        }
        Some((build, _)) => ctx.line(format!(
            "attention : la release annonce le build {build}, les sources sont au build {}",
            state.build
        )),
        None => anyhow::bail!("la nouvelle release ne démarre pas (llama-server --version)"),
    }

    // Seulement maintenant que la nouvelle version est vérifiée. Un échec
    // n'annule pas la mise à jour : l'ancienne version reste, voilà tout.
    let mut removed = None;
    if let Some(previous) = plan.delete_previous.as_ref().filter(|p| **p != target) {
        let label = previous
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        match delete_release(previous, &plan.releases_dir) {
            Ok(()) => {
                ctx.line(format!("ancienne version supprimée : {label}"));
                removed = Some(label);
            }
            Err(e) => ctx.line(format!("! ancienne version conservée : {e}")),
        }
    }

    Ok(Release {
        name,
        dir: target,
        build: state.build,
        removed,
    })
}

/// `dir` est-il une release du dossier des releases, donc supprimable ?
pub fn is_release(dir: &Path, releases_dir: &Path) -> bool {
    if releases_dir.as_os_str().is_empty() {
        return false;
    }
    let (Ok(dir), Ok(root)) = (dir.canonicalize(), releases_dir.canonicalize()) else {
        return false;
    };
    dir.parent() == Some(root.as_path()) && dir.join(crate::discovery::SERVER_EXE).is_file()
}

/// Supprime une release. Refuse tout ce qui n'est pas un dossier direct du
/// dossier des releases contenant llama-server : un build de l'utilisateur
/// situé ailleurs n'est jamais touché.
///
/// Le dossier est d'abord renommé : si l'un de ses fichiers est utilisé (un
/// llama-server lancé depuis cette version), Windows refuse le renommage et
/// rien n'est supprimé — plutôt qu'une suppression à moitié faite qui
/// laisserait une version cassée.
pub fn delete_release(dir: &Path, releases_dir: &Path) -> anyhow::Result<()> {
    anyhow::ensure!(
        is_release(dir, releases_dir),
        "{} n'est pas une release de {}",
        dir.display(),
        releases_dir.display()
    );
    let target = dir.canonicalize()?;
    let name = target
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let trash = target.with_file_name(format!(".{name}.deleting"));
    if trash.exists() {
        std::fs::remove_dir_all(&trash)?;
    }
    std::fs::rename(&target, &trash).map_err(|e| {
        anyhow::anyhow!(
            "{name} est en cours d'utilisation (un llama-server tourne-t-il depuis cette \
             version ?) : {e}"
        )
    })?;
    std::fs::remove_dir_all(&trash)?;
    Ok(())
}

/// `llama-b10512-cuda`, ou `llama-b10512-cuda-2` si le premier existe déjà.
fn unique_name(dir: &Path, base: &str) -> String {
    if !dir.join(base).exists() {
        return base.to_string();
    }
    (2..)
        .map(|n| format!("{base}-{n}"))
        .find(|n| !dir.join(n).exists())
        .unwrap_or_else(|| base.to_string())
}

/// Copie exe et DLL dans un dossier temporaire puis le renomme : une release
/// n'apparaît jamais à moitié copiée.
fn copy_binaries(from: &Path, target: &Path) -> anyhow::Result<usize> {
    let tmp = target.with_file_name(format!(
        ".{}.tmp",
        target.file_name().unwrap_or_default().to_string_lossy()
    ));
    if tmp.exists() {
        std::fs::remove_dir_all(&tmp)?;
    }
    std::fs::create_dir_all(&tmp)?;
    let mut count = 0;
    for entry in std::fs::read_dir(from)?.flatten() {
        let path = entry.path();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_lowercase);
        if path.is_file() && matches!(ext.as_deref(), Some("exe" | "dll")) {
            std::fs::copy(&path, tmp.join(entry.file_name()))?;
            count += 1;
        }
    }
    std::fs::rename(&tmp, target)?;
    Ok(count)
}

/// Nombre de projets MSBuild à compiler, pour mesurer l'avancement.
fn count_projects(build: &Path) -> usize {
    fn walk(dir: &Path, depth: usize, n: &mut usize) {
        if depth > 6 {
            return;
        }
        let Ok(rd) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in rd.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, depth + 1, n);
            } else if path.extension().and_then(|e| e.to_str()) == Some("vcxproj") {
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                if !matches!(
                    stem,
                    "ALL_BUILD" | "ZERO_CHECK" | "INSTALL" | "RUN_TESTS" | "PACKAGE"
                ) {
                    *n += 1;
                }
            }
        }
    }
    let mut n = 0;
    walk(build, 0, &mut n);
    n
}

/// Avancement lu dans une ligne de sortie : `[12/345]` (Ninja, Makefiles) ou
/// `projet.vcxproj -> …` (MSBuild, compté sur le nombre de projets).
fn build_progress(line: &str, done: &mut usize, total: usize) -> Option<f32> {
    let t = line.trim_start();
    if let Some(rest) = t.strip_prefix('[') {
        let (counts, _) = rest.split_once(']')?;
        let (a, b) = counts.split_once('/')?;
        let (a, b): (f32, f32) = (a.trim().parse().ok()?, b.trim().parse().ok()?);
        return (b > 0.0).then(|| (a / b).clamp(0.0, 1.0));
    }
    if total > 0 && t.contains(".vcxproj -> ") {
        *done += 1;
        return Some((*done as f32 / total as f32).min(1.0));
    }
    None
}

/// Pourcentage d'une ligne de progression git (`Receiving objects:  45% (…)`).
fn git_progress(line: &str) -> Option<f32> {
    if !(line.contains("objects:") || line.contains("deltas:") || line.contains("files:")) {
        return None;
    }
    let pct = line.find('%')?;
    let digits: String = line[..pct]
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_digit())
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    digits.parse::<f32>().ok().map(|p| (p / 100.0).clamp(0.0, 1.0))
}

/// Découpe un flux en lignes sur `\n` **et** `\r` : git réécrit sa ligne de
/// progression avec `\r`, qu'une lecture par `\n` ne rendrait qu'à la fin.
fn read_lines(stream: impl Read, mut f: impl FnMut(String)) {
    let mut reader = BufReader::new(stream);
    let mut current = Vec::new();
    loop {
        let consumed = match reader.fill_buf() {
            Ok([]) | Err(_) => break,
            Ok(buf) => {
                for &b in buf {
                    if b == b'\n' || b == b'\r' {
                        let text = String::from_utf8_lossy(&current).trim_end().to_string();
                        current.clear();
                        if !text.is_empty() {
                            f(text);
                        }
                    } else {
                        current.push(b);
                    }
                }
                buf.len()
            }
        };
        reader.consume(consumed);
    }
    let text = String::from_utf8_lossy(&current).trim_end().to_string();
    if !text.is_empty() {
        f(text);
    }
}

/// Lance une commande en relayant stdout et stderr ligne par ligne.
///
/// `on_line` voit chaque ligne des deux flux ; s'il renvoie `true`, la ligne
/// n'est pas recopiée dans la console (progression répétitive).
fn run_streamed(
    mut cmd: Command,
    ctx: &Ctx,
    mut on_line: impl FnMut(&str) -> bool,
) -> anyhow::Result<()> {
    let program = Path::new(cmd.get_program())
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);
    let mut child = cmd
        .spawn()
        .map_err(|e| anyhow::anyhow!("{program} introuvable : {e}"))?;
    if let Ok(mut guard) = ctx.pid.lock() {
        *guard = Some(child.id());
    }

    // Les deux flux sont lus en parallèle et réunis ici, pour que `on_line`
    // voie aussi stderr (où git écrit sa progression).
    let (line_tx, line_rx) = channel::<String>();
    let readers: Vec<_> = [
        child.stdout.take().map(|s| Box::new(s) as Box<dyn Read + Send>),
        child.stderr.take().map(|s| Box::new(s) as Box<dyn Read + Send>),
    ]
    .into_iter()
    .flatten()
    .map(|stream| {
        let tx = line_tx.clone();
        std::thread::spawn(move || read_lines(stream, |l| {
            let _ = tx.send(l);
        }))
    })
    .collect();
    drop(line_tx);
    // Garde la ligne qui explique un échec : la dernière erreur signalée,
    // sinon la dernière ligne tout court.
    let mut reason = String::new();
    let mut last = String::new();
    for text in line_rx {
        let lower = text.to_lowercase();
        if lower.starts_with("fatal:") || lower.starts_with("error") || lower.contains(" error ")
            || lower.contains("cmake error")
        {
            reason = text.clone();
        }
        if !on_line(&text) {
            last = text.clone();
            ctx.line(text);
        }
    }
    for handle in readers {
        let _ = handle.join();
    }
    let status = child.wait()?;
    if let Ok(mut guard) = ctx.pid.lock() {
        *guard = None;
    }
    anyhow::ensure!(!ctx.cancelled(), "annulé");
    if !status.success() {
        let code = status.code().map(|c| c.to_string()).unwrap_or_else(|| "?".into());
        let why = if reason.is_empty() { last } else { reason };
        anyhow::bail!("{program} a échoué (code {code}) : {why}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const CACHE: &str = "\
# This is the CMakeCache file.
//Build shared libraries
BUILD_SHARED_LIBS:BOOL=ON
CMAKE_CUDA_ARCHITECTURES:UNINITIALIZED=native
GGML_CUDA:BOOL=ON
GGML_VULKAN:BOOL=OFF
GGML_CCACHE_FOUND:FILEPATH=C:/tools/ccache.exe
GGML_SCCACHE_FOUND:FILEPATH=GGML_SCCACHE_FOUND-NOTFOUND
GGML_SYCL_TARGET:STRING=INTEL
LLAMA_BUILD_TESTS:BOOL=OFF
CMAKE_CXX_FLAGS:STRING=/DWIN32
CMAKE_GENERATOR:INTERNAL=Visual Studio 17 2022
CMAKE_GENERATOR_INSTANCE:INTERNAL=C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools
CMAKE_GENERATOR_PLATFORM:INTERNAL=
";

    #[test]
    fn reads_backend_generator_and_options() {
        let cfg = BuildConfig::parse(CACHE);
        assert_eq!(cfg.backend(), "cuda");
        assert_eq!(cfg.generator, "Visual Studio 17 2022");
        let names: Vec<&str> = cfg.defines.iter().map(|(n, ..)| n.as_str()).collect();
        // Les chemins détectés et les drapeaux du compilateur ne sont pas repris.
        assert!(!names.contains(&"GGML_CCACHE_FOUND"));
        assert!(!names.contains(&"CMAKE_CXX_FLAGS"));
        assert!(names.contains(&"GGML_CUDA") && names.contains(&"LLAMA_BUILD_TESTS"));
        assert_eq!(
            cfg.summary(true),
            "CUDA · Visual Studio 17 2022 · DLL partagées · arch native"
        );
    }

    #[test]
    fn fa_all_quants_round_trip() {
        let mut c = BuildConfig::for_backend(Backend::Cuda);
        assert!(!c.fa_all());
        c.set_fa_all(true);
        assert!(c.fa_all());
        let args = c.configure_args(Path::new("src"), Path::new("build"));
        assert!(args.contains(&"-DGGML_CUDA_FA_QUANTS:STRING=all".to_string()));
        assert!(args.contains(&"-DGGML_CUDA_FA_ALL_QUANTS:BOOL=ON".to_string()));
        c.set_fa_all(false);
        assert!(!c.fa_all());
        assert!(!c.configure_args(Path::new("s"), Path::new("b")).iter().any(|a| a.contains("FA_QUANTS")));
        // Relu dans un CMakeCache récent.
        let cache = BuildConfig::parse(
            "GGML_CUDA:BOOL=ON\nGGML_CUDA_FA_ALL_QUANTS:BOOL=OFF\nGGML_CUDA_FA_QUANTS:STRING=all\n",
        );
        assert!(cache.fa_all());
        let default = BuildConfig::parse(
            "GGML_CUDA:BOOL=ON\nGGML_CUDA_FA_QUANTS:STRING=q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16\n",
        );
        assert!(!default.fa_all());
    }

    #[test]
    fn configure_args_reproduce_the_reference_build() {
        let cfg = BuildConfig::parse(CACHE);
        let args = cfg.configure_args(Path::new("C:/src"), Path::new("C:/src/build-uillamacpp"));
        assert_eq!(&args[..6], ["-S", "C:/src", "-B", "C:/src/build-uillamacpp", "-G", "Visual Studio 17 2022"]);
        assert!(args.contains(&"-DCMAKE_GENERATOR_INSTANCE=C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools".to_string()));
        assert!(args.contains(&"-DGGML_CUDA:BOOL=ON".to_string()));
        // Passée sans type sur la ligne de commande d'origine : reprise telle quelle.
        assert!(args.contains(&"-DCMAKE_CUDA_ARCHITECTURES=native".to_string()));
        // Plateforme vide : pas de -A.
        assert!(!args.contains(&"-A".to_string()));
    }

    #[test]
    fn default_config_builds_cpu() {
        assert_eq!(BuildConfig::default().backend(), "cpu");
    }

    #[test]
    fn fresh_install_options_follow_the_chosen_backend() {
        let cuda = BuildConfig::for_backend(Backend::Cuda);
        assert_eq!(cuda.backend(), "cuda");
        let args = cuda.configure_args(Path::new("s"), Path::new("b"));
        assert!(args.contains(&"-DCMAKE_CUDA_ARCHITECTURES=native".to_string()));
        assert!(args.contains(&"-DLLAMA_BUILD_TESTS:BOOL=OFF".to_string()));
        assert_eq!(BuildConfig::for_backend(Backend::Vulkan).backend(), "vulkan");
        assert_eq!(BuildConfig::for_backend(Backend::Cpu).backend(), "cpu");
    }

    #[test]
    fn suggests_the_gpu_only_when_usable() {
        let mut p = Prereqs::default();
        assert_eq!(p.suggested(), Backend::Cpu);
        p.nvidia_gpu = Some("RTX".into());
        // GPU présent mais pas de toolkit CUDA : pas de CUDA.
        assert_eq!(p.suggested(), Backend::Cpu);
        p.cuda = Some("13.3".into());
        assert_eq!(p.suggested(), Backend::Cuda);
    }

    #[test]
    fn reads_git_progress_and_splits_on_carriage_returns() {
        assert_eq!(git_progress("Receiving objects:  45% (450/1000)"), Some(0.45));
        assert_eq!(git_progress("Receiving objects: 100% (1000/1000), done."), Some(1.0));
        assert_eq!(git_progress("Cloning into 'llama.cpp'..."), None);
        let mut lines = Vec::new();
        read_lines(&b"a: 1%\ra: 50%\ra: 100%, done.\nfin"[..], |l| lines.push(l));
        assert_eq!(lines, ["a: 1%", "a: 50%", "a: 100%, done.", "fin"]);
    }

    #[test]
    fn progress_from_ninja_and_msbuild() {
        let mut done = 0;
        assert_eq!(build_progress("[50/200] Building CXX object", &mut done, 0), Some(0.25));
        assert_eq!(
            build_progress("  ggml-cuda.vcxproj -> C:\\b\\ggml-cuda.dll", &mut done, 4),
            Some(0.25)
        );
        assert_eq!(build_progress("  llama.vcxproj -> C:\\b\\llama.dll", &mut done, 4), Some(0.5));
        assert_eq!(build_progress("warning C4244: conversion", &mut done, 4), None);
    }

    #[test]
    fn release_is_copied_whole_under_a_free_name() {
        let root = std::env::temp_dir().join(format!("uillamacpp-test-{}", std::process::id()));
        let bin = root.join("bin");
        let releases = root.join("releases");
        std::fs::create_dir_all(&bin).unwrap();
        for f in ["llama-server.exe", "ggml-cuda.dll", "notes.txt"] {
            std::fs::write(bin.join(f), b"x").unwrap();
        }
        std::fs::create_dir_all(releases.join("llama-b1-cuda")).unwrap();

        // Le nom est pris : suffixe, sans toucher à la release existante.
        let name = unique_name(&releases, "llama-b1-cuda");
        assert_eq!(name, "llama-b1-cuda-2");
        let copied = copy_binaries(&bin, &releases.join(&name)).unwrap();
        assert_eq!(copied, 2, "seuls les exe et dll sont copiés");
        assert!(releases.join(&name).join("ggml-cuda.dll").is_file());
        assert!(!releases.join(&name).join("notes.txt").exists());
        // Aucun dossier temporaire ne reste.
        let leftovers: Vec<_> = std::fs::read_dir(&releases)
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty());
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// Lance la vraie configuration CMake avec les options reprises d'un
    /// build existant, dans un dossier temporaire (rien n'est compilé).
    ///
    /// ```text
    /// set LLAMA_SOURCE=C:\...\llama.cpp
    /// cargo test configure_like_reference -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "nécessite un dépôt llama.cpp configuré (LLAMA_SOURCE) et CMake"]
    fn configure_like_reference() {
        let source = PathBuf::from(std::env::var("LLAMA_SOURCE").expect("LLAMA_SOURCE"));
        let reference = source.join("build").join("CMakeCache.txt");
        let wanted = BuildConfig::read(&reference).unwrap();
        let build = std::env::temp_dir().join(format!("uillamacpp-cfg-{}", std::process::id()));

        let out = Command::new("cmake")
            .args(wanted.configure_args(&source, &build))
            .output()
            .unwrap();
        let log = String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success(), "cmake a échoué :
{log}");

        let got = BuildConfig::read(&build.join("CMakeCache.txt")).unwrap();
        println!("référence : {}
obtenu    : {}", wanted.summary(true), got.summary(true));
        assert_eq!(got.backend(), wanted.backend());
        assert_eq!(got.generator, wanted.generator);
        assert_eq!(got.summary(true), wanted.summary(true));
        let _ = std::fs::remove_dir_all(&build);
    }

    /// Installation complète depuis zéro, backend CPU : clone GitHub,
    /// configuration, compilation, release, vérification. Plusieurs minutes.
    ///
    /// ```text
    /// set UILLAMACPP_INSTALL_TEST=D:\tmp\essai   (dossier vide ou absent)
    /// cargo test install_from_scratch -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "réseau + compilation complète (UILLAMACPP_INSTALL_TEST)"]
    fn install_from_scratch_cpu() {
        let root = PathBuf::from(std::env::var("UILLAMACPP_INSTALL_TEST").unwrap());
        let source = root.join("llama.cpp");
        let task = install(Plan {
            releases_dir: default_releases_dir(&source),
            source,
            reference: None,
            backend: Backend::Cpu,
            clone: false,
            delete_previous: None,
            fa_all: None,
        });
        let started = std::time::Instant::now();
        let release = loop {
            match task.rx.recv().expect("tâche interrompue") {
                Event::Stage { index, label, .. } => {
                    println!("[{:>5}s] étape {index} : {label}", started.elapsed().as_secs())
                }
                Event::Done(result) => break result.expect("installation échouée"),
                Event::Line(l) if l.starts_with("——") || l.contains("fichiers copiés") || l.starts_with('✔') => {
                    println!("[{:>5}s] {l}", started.elapsed().as_secs())
                }
                _ => {}
            }
        };
        println!("release : {} (b{}) en {} s", release.name, release.build, started.elapsed().as_secs());
        assert!(release.name.starts_with("llama-b") && release.name.ends_with("-cpu"));
        assert!(release.dir.join(crate::discovery::SERVER_EXE).is_file());
        assert!(release.dir.join(crate::discovery::BENCH_EXE).is_file());
        // Clone partiel : l'historique complet est bien là, le numéro de build
        // est donc réaliste et non « 1 ».
        assert!(release.build > 10_000, "build {}", release.build);
    }

    fn scratch(tag: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("uillamacpp-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    fn fake_version(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join(crate::discovery::SERVER_EXE), b"x").unwrap();
        std::fs::write(dir.join("ggml.dll"), b"x").unwrap();
    }

    #[test]
    fn deletes_a_release_and_nothing_else() {
        let root = scratch("delete");
        let releases = root.join("llama.cpp-releases");
        let old = releases.join("llama-b1-cuda");
        let keep = releases.join("llama-b2-cuda");
        let own_build = root.join("mon-build").join("Release");
        fake_version(&old);
        fake_version(&keep);
        fake_version(&own_build);

        delete_release(&old, &releases).unwrap();
        assert!(!old.exists());
        assert!(keep.join(crate::discovery::SERVER_EXE).is_file(), "voisine intacte");

        // Un build situé hors du dossier des releases est refusé.
        assert!(delete_release(&own_build, &releases).is_err());
        assert!(own_build.exists());
        // Le dossier des releases lui-même aussi.
        assert!(delete_release(&releases, &releases).is_err());
        assert!(keep.exists());
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// Un fichier ouvert sans partage (comme l'exe d'un serveur en cours)
    /// empêche la suppression, et la version reste entière.
    #[cfg(windows)]
    #[test]
    fn a_release_in_use_is_left_whole() {
        use std::os::windows::fs::OpenOptionsExt;
        let root = scratch("locked");
        let releases = root.join("releases");
        let busy = releases.join("llama-b3-cuda");
        fake_version(&busy);
        let lock = std::fs::OpenOptions::new()
            .read(true)
            .share_mode(0)
            .open(busy.join(crate::discovery::SERVER_EXE))
            .unwrap();

        let err = delete_release(&busy, &releases).unwrap_err().to_string();
        assert!(err.contains("en cours d'utilisation"), "{err}");
        assert!(busy.join(crate::discovery::SERVER_EXE).is_file());
        assert!(busy.join("ggml.dll").is_file(), "rien n'a été supprimé à moitié");

        drop(lock);
        delete_release(&busy, &releases).unwrap();
        assert!(!busy.exists());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn releases_live_next_to_the_repository() {
        assert_eq!(
            default_releases_dir(Path::new("C:/src/repos/llama.cpp")),
            PathBuf::from("C:/src/repos/llama.cpp-releases")
        );
    }
}
