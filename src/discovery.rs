//! Découverte des versions de llama.cpp et des modèles GGUF sur le disque.

use std::path::{Path, PathBuf};

pub const SERVER_EXE: &str = if cfg!(windows) {
    "llama-server.exe"
} else {
    "llama-server"
};
pub const BENCH_EXE: &str = if cfg!(windows) {
    "llama-bench.exe"
} else {
    "llama-bench"
};

#[derive(Clone, PartialEq)]
pub struct VersionInfo {
    pub name: String,
    pub dir: PathBuf,
    /// Numéro de build llama.cpp, lu dans le nom du dossier (`llama-b10512-…`)
    /// ou, à défaut, via `llama-server --version` en tâche de fond.
    pub build: Option<u32>,
    pub commit: Option<String>,
    /// Backend déduit des DLL présentes (`ggml-cuda.dll` → CUDA).
    pub backend: &'static str,
    /// `llama-bench` présent, relevé à la découverte : l'onglet Benchmark
    /// le demandait au disque à chaque image.
    has_bench: bool,
}

impl VersionInfo {
    fn new(name: String, dir: PathBuf) -> Self {
        Self {
            build: build_from_name(&name),
            commit: None,
            backend: backend_of(&dir),
            has_bench: dir.join(BENCH_EXE).is_file(),
            name,
            dir,
        }
    }

    /// « b10475 · CUDA » ; « ? » tant que le numéro n'est pas connu.
    pub fn tag(&self) -> String {
        match self.build {
            Some(b) => format!("b{b} · {}", self.backend),
            None => format!("? · {}", self.backend),
        }
    }

    pub fn server(&self) -> PathBuf {
        self.dir.join(SERVER_EXE)
    }
    pub fn bench(&self) -> PathBuf {
        self.dir.join(BENCH_EXE)
    }
    pub fn has_bench(&self) -> bool {
        self.has_bench
    }
}

/// Sous-répertoires (ou le répertoire lui-même) contenant `llama-server`.
/// On regarde aussi les emplacements de build usuels : `build/bin`, `bin`.
pub fn find_versions(root: &str) -> Vec<VersionInfo> {
    let root = Path::new(root);
    if root.as_os_str().is_empty() || !root.is_dir() {
        return Vec::new();
    }

    let mut out = Vec::new();

    if let Some(dir) = binary_dir(root) {
        let name = root
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(".")
            .to_string();
        out.push(VersionInfo::new(name, dir));
    }

    if let Ok(rd) = std::fs::read_dir(root) {
        let mut entries: Vec<PathBuf> = rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        entries.sort();
        for entry in entries {
            if let Some(dir) = binary_dir(&entry) {
                let name = entry
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or_default()
                    .to_string();
                out.push(VersionInfo::new(name, dir));
            }
        }
    }

    out
}

/// `llama-b10512-cuda` ou `b10512` → 10512.
pub fn build_from_name(name: &str) -> Option<u32> {
    name.split(['-', '_', ' ', '.'])
        .filter_map(|part| part.strip_prefix('b'))
        .find_map(|digits| {
            (!digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()))
                .then(|| digits.parse().ok())
                .flatten()
        })
}

/// Backend d'une version, d'après les DLL ggml qu'elle embarque.
pub fn backend_of(dir: &Path) -> &'static str {
    for (dll, name) in [
        ("ggml-cuda", "CUDA"),
        ("ggml-hip", "ROCm"),
        ("ggml-sycl", "SYCL"),
        ("ggml-vulkan", "Vulkan"),
        ("ggml-metal", "Metal"),
        ("ggml-opencl", "OpenCL"),
    ] {
        if dir.join(format!("{dll}.dll")).is_file() || dir.join(format!("lib{dll}.so")).is_file() {
            return name;
        }
    }
    "CPU"
}

/// Numéro de build et commit annoncés par `llama-server --version`
/// (« version: 0.1.1-dev (build 10475, commit ed1c3a20f) »).
pub fn probe_version(server: &Path) -> Option<(u32, String)> {
    use std::process::{Command, Stdio};
    // Un faux exe ferait afficher par Windows une boîte d'erreur bloquante
    // (« Application 16 bits non prise en charge »).
    if !is_executable(server) {
        return None;
    }
    let mut cmd = Command::new(server);
    cmd.arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(dir) = server.parent() {
        cmd.current_dir(dir);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    let mut child = cmd.spawn().ok()?;
    // Garde-fou : un exe qui ne rend pas la main ne doit pas bloquer.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if std::time::Instant::now() < deadline => {
                std::thread::sleep(std::time::Duration::from_millis(30))
            }
            _ => {
                let _ = child.kill();
                return None;
            }
        }
    }
    let out = child.wait_with_output().ok()?;
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    parse_version(&text)
}

/// Vrai exécutable Windows (en-têtes MZ puis PE) ? Sur les autres systèmes,
/// simple test d'existence.
///
/// Lancer un fichier `.exe` corrompu ou tronqué fait afficher par Windows une
/// boîte de dialogue modale au lieu de renvoyer une erreur : on vérifie
/// l'en-tête avant de lancer quoi que ce soit.
pub fn is_executable(path: &Path) -> bool {
    use std::io::{Read, Seek, SeekFrom};
    if !cfg!(windows) {
        return path.is_file();
    }
    let Ok(mut f) = std::fs::File::open(path) else {
        return false;
    };
    let mut dos = [0u8; 64];
    if f.read_exact(&mut dos).is_err() || &dos[..2] != b"MZ" {
        return false;
    }
    let pe_offset = u32::from_le_bytes([dos[60], dos[61], dos[62], dos[63]]) as u64;
    let mut sig = [0u8; 4];
    f.seek(SeekFrom::Start(pe_offset)).is_ok()
        && f.read_exact(&mut sig).is_ok()
        && &sig == b"PE\0\0"
}

pub fn parse_version(text: &str) -> Option<(u32, String)> {
    let after = |key: &str| -> Option<String> {
        let i = text.find(key)? + key.len();
        Some(
            text[i..]
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric())
                .collect(),
        )
    };
    let build = after("build ")?.parse().ok()?;
    let commit = after("commit ").unwrap_or_default();
    Some((build, commit))
}

fn binary_dir(base: &Path) -> Option<PathBuf> {
    for sub in ["", "bin", "build/bin", "build/bin/Release"] {
        let dir = if sub.is_empty() {
            base.to_path_buf()
        } else {
            base.join(sub)
        };
        if dir.join(SERVER_EXE).is_file() {
            return Some(dir);
        }
    }
    None
}

#[derive(Clone, PartialEq)]
pub struct ModelInfo {
    pub name: String,
    pub full_path: PathBuf,
    pub size_bytes: u64,
    pub quant: String,
}

impl ModelInfo {
    pub fn size_human(&self) -> String {
        human_size(self.size_bytes)
    }
}

pub fn human_size(bytes: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    let b = bytes as f64;
    if b >= GIB {
        format!("{:.2} GiB", b / GIB)
    } else if b >= MIB {
        format!("{:.0} MiB", b / MIB)
    } else {
        format!("{bytes} B")
    }
}

/// Modèles GGUF d'un répertoire (récursif, profondeur limitée) et
/// projecteurs multimodaux (`*mmproj*.gguf`, proposés pour `--mmproj`), en
/// un seul parcours : le répertoire était parcouru deux fois à chaque
/// rafraîchissement.
///
/// Sont écartés des modèles : les `mmproj` et toutes les parties d'un
/// modèle scindé sauf la première (`-00001-of-000NN`), dont la taille
/// affichée est celle de toutes les parties réunies.
pub fn scan_models(root: &str) -> (Vec<ModelInfo>, Vec<PathBuf>) {
    let root = Path::new(root);
    let mut files = Vec::new();
    if !root.as_os_str().is_empty() && root.is_dir() {
        walk(root, 0, &mut files);
    }

    let mut models = Vec::new();
    let mut mmproj = Vec::new();
    for (path, size) in &files {
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let lower = stem.to_lowercase();
        if lower.contains("mmproj") {
            mmproj.push(path.clone());
            continue;
        }
        if is_non_first_shard(&lower) {
            continue;
        }
        // Première partie d'un modèle scindé : ajoute la taille des autres.
        let size_bytes = match shard_prefix(&lower) {
            Some(prefix) => files
                .iter()
                .filter(|(p, _)| p.parent() == path.parent())
                .filter(|(p, _)| {
                    p.file_stem()
                        .and_then(|s| s.to_str())
                        .is_some_and(|s| shard_prefix(&s.to_lowercase()).as_deref() == Some(prefix.as_str()))
                })
                .map(|(_, len)| len)
                .sum(),
            None => *size,
        };
        models.push(ModelInfo {
            name: stem.to_string(),
            quant: detect_quant(stem),
            size_bytes,
            full_path: path.clone(),
        });
    }
    models.sort_by_key(|m| m.name.to_lowercase());
    mmproj.sort();
    (models, mmproj)
}

/// Fichiers `.gguf` (chemin, taille), sur quatre niveaux au plus.
fn walk(dir: &Path, depth: usize, out: &mut Vec<(PathBuf, u64)>) {
    if depth > 4 {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in rd.flatten() {
        let path = entry.path();
        let Ok(meta) = entry.metadata() else { continue };
        if meta.is_dir() {
            walk(&path, depth + 1, out);
        } else if path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
        {
            out.push((path, meta.len()));
        }
    }
}

/// `model-00001-of-00003` → `model-?-of-00003` : commun à toutes les parties
/// d'un même modèle scindé ; `None` pour un fichier unique.
fn shard_prefix(lower_stem: &str) -> Option<String> {
    let pos = lower_stem.rfind("-of-")?;
    let before = &lower_stem[..pos];
    let dash = before.rfind('-')?;
    let index = &before[dash + 1..];
    if index.is_empty() || !index.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some(format!("{}-?{}", &before[..dash], &lower_stem[pos..]))
}

/// `model-00002-of-00003` → true ; `model-00001-of-00003` → false.
fn is_non_first_shard(lower_stem: &str) -> bool {
    let Some(pos) = lower_stem.rfind("-of-") else {
        return false;
    };
    let before = &lower_stem[..pos];
    let Some(dash) = before.rfind('-') else {
        return false;
    };
    let index = &before[dash + 1..];
    if index.is_empty() || !index.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    index.parse::<u32>().map(|n| n != 1).unwrap_or(false)
}

/// Quantisation déduite du nom de fichier (correspondance la plus longue).
pub fn detect_quant(stem: &str) -> String {
    const QUANTS: &[&str] = &[
        "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ3_XXS", "IQ3_XS", "IQ3_S",
        "IQ3_M", "IQ4_XS", "IQ4_NL", "Q2_K_S", "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q3_K",
        "Q4_K_S", "Q4_K_M", "Q4_K", "Q4_0", "Q4_1", "Q5_K_S", "Q5_K_M", "Q5_K", "Q5_0", "Q5_1",
        "Q6_K", "Q8_0", "MXFP4", "BF16", "F16", "F32",
    ];
    let upper = stem.to_ascii_uppercase();
    let mut best = "";
    for q in QUANTS {
        if upper.contains(q) && q.len() > best.len() {
            best = q;
        }
    }
    if best.is_empty() {
        "-".to_string()
    } else {
        best.to_string()
    }
}

/// Nom « de base » du modèle : le stem privé de sa quantisation, utilisé
/// comme clé dans `benchmark.md`.
///
/// Majuscules ASCII seulement : l'indice trouvé dans `upper` doit valoir
/// dans `stem`, ce que `to_uppercase` ne garantit pas (« ﬁ » → « FI »).
pub fn base_name(stem: &str) -> String {
    let quant = detect_quant(stem);
    if quant == "-" {
        return stem.to_string();
    }
    let upper = stem.to_ascii_uppercase();
    match upper.find(&quant) {
        Some(pos) => stem[..pos].trim_end_matches(['-', '_', '.']).to_string(),
        None => stem.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_models_show_their_total_size() {
        let dir = std::env::temp_dir().join(format!("uillamacpp-shards-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("sub/M-Q4_K_M-00001-of-00002.gguf"), [0u8; 10]).unwrap();
        std::fs::write(dir.join("sub/M-Q4_K_M-00002-of-00002.gguf"), [0u8; 7]).unwrap();
        std::fs::write(dir.join("sub/mmproj-M.gguf"), [0u8; 3]).unwrap();
        std::fs::write(dir.join("single-Q8_0.gguf"), [0u8; 5]).unwrap();
        let (models, mmproj) = scan_models(dir.to_str().unwrap());
        let sizes: Vec<(&str, u64)> = models.iter().map(|m| (m.name.as_str(), m.size_bytes)).collect();
        assert_eq!(sizes, [("M-Q4_K_M-00001-of-00002", 17), ("single-Q8_0", 5)]);
        assert_eq!(mmproj.len(), 1);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn non_ascii_names_do_not_panic() {
        // La ligature « ﬁ » (3 octets) devient « FI » (2 octets) avec
        // to_uppercase : l'indice décalé tombait au milieu d'un caractère.
        assert_eq!(base_name("ﬁﬁ-Q4_K_M"), "ﬁﬁ");
        assert_eq!(detect_quant("İstanbul-q8_0"), "Q8_0");
    }

    #[test]
    fn build_number_from_folder_name() {
        assert_eq!(build_from_name("llama-b10512-cuda"), Some(10512));
        assert_eq!(build_from_name("llama-b10512-cuda-2"), Some(10512));
        assert_eq!(build_from_name("b9553"), Some(9553));
        assert_eq!(build_from_name("Release"), None);
        assert_eq!(build_from_name("llama-bench"), None);
    }

    #[test]
    fn recognises_real_executables_only() {
        let dir = std::env::temp_dir().join(format!("uillamacpp-exe-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let fake = dir.join("llama-server.exe");
        std::fs::write(&fake, b"x").unwrap();
        assert!(!is_executable(&fake), "fichier texte nommé .exe");
        std::fs::write(&fake, b"MZ tronque").unwrap();
        assert!(!is_executable(&fake), "en-tête MZ sans PE");
        assert!(!is_executable(&dir.join("absent.exe")));
        // Le binaire de test lui-même est un vrai exécutable.
        assert!(is_executable(&std::env::current_exe().unwrap()));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn parses_llama_server_version_line() {
        let out = "version: 0.1.1-dev (build 10475, commit ed1c3a20f)\nbuilt with MSVC 19.44 for x64";
        assert_eq!(parse_version(out), Some((10475, "ed1c3a20f".to_string())));
        assert_eq!(parse_version("usage: llama-server"), None);
    }
}
