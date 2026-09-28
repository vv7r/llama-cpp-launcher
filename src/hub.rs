//! Hugging Face : dépôts GGUF utilisables par llama-server, et leurs fichiers.
//!
//! La liste proposée dans l'onglet Modèles vient de l'API Hugging Face,
//! plus d'une liste écrite en dur qui vieillissait. Elle reste une liste
//! fermée : on choisit un dépôt puis un fichier, rien ne se tape.
//!
//! Le filtre `gguf` seul ramène aussi des modèles d'image, d'audio ou de
//! vidéo que llama-server ne sait pas servir. Filtrer par type de tâche ne
//! suffit pas : de gros dépôts de modèles de langage n'en déclarent aucun
//! (`unsloth/Qwen3.8-27B-GGUF`, 2e plus téléchargé). On garde donc les tâches
//! texte et vision, plus les dépôts sans tâche marqués `conversational` ou
//! `text-generation` ; tout le reste est écarté.

use crate::discovery::detect_quant;
use serde::Deserialize;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver};
use std::sync::Arc;
use std::time::{Duration, Instant};

const API: &str = "https://huggingface.co/api/models";
/// Dépôts demandés, avant filtrage.
const FETCHED: usize = 100;
/// Dépôts proposés après filtrage.
const SHOWN: usize = 50;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Sort {
    Trending,
    Downloads,
}

impl Sort {
    pub const ALL: [Sort; 2] = [Sort::Trending, Sort::Downloads];

    fn param(self) -> &'static str {
        match self {
            Sort::Trending => "trendingScore",
            Sort::Downloads => "downloads",
        }
    }

    pub fn label(self, fr: bool) -> &'static str {
        match (self, fr) {
            (Sort::Trending, true) => "Tendance",
            (Sort::Trending, false) => "Trending",
            (Sort::Downloads, true) => "Plus téléchargés",
            (Sort::Downloads, false) => "Most downloaded",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Repo {
    pub id: String,
    pub downloads: u64,
    pub likes: u64,
    /// Modèle de vision (image + texte) : llama-server télécharge aussi son
    /// projecteur mmproj.
    pub vision: bool,
    trending: f64,
}

/// Un modèle téléchargeable d'un dépôt : un fichier, ou le premier fichier
/// d'un modèle scindé (llama.cpp charge les autres parties seul).
#[derive(Clone, Debug, PartialEq)]
pub struct HubFile {
    /// Chemin dans le dépôt de la (première) partie : identifie le choix.
    pub file: String,
    /// Toutes les parties à télécharger, dans l'ordre (une seule en général).
    pub members: Vec<(String, u64)>,
    /// Quantisation lue dans le nom, ou le nom du fichier à défaut.
    pub quant: String,
    /// Ce qui suit la quantisation dans le nom (« mtp » pour
    /// `…-IQ2_S-mtp.gguf`) : distingue deux fichiers de même quantisation.
    pub variant: String,
    /// Taille totale, parties comprises.
    pub size: u64,
    pub parts: usize,
}

#[derive(Deserialize)]
struct ApiModel {
    id: String,
    #[serde(default)]
    downloads: u64,
    #[serde(default)]
    likes: u64,
    #[serde(default, rename = "trendingScore")]
    trending: f64,
    #[serde(default)]
    pipeline_tag: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
}

impl ApiModel {
    /// Modèle de langage servable par llama-server ?
    fn is_language_model(&self) -> bool {
        match self.pipeline_tag.as_deref() {
            Some("text-generation" | "image-text-to-text") => true,
            Some(_) => false,
            None => self
                .tags
                .iter()
                .any(|t| t == "conversational" || t == "text-generation"),
        }
    }
}

#[derive(Deserialize)]
struct ApiRepo {
    #[serde(default)]
    siblings: Vec<Sibling>,
}

#[derive(Deserialize)]
struct Sibling {
    rfilename: String,
    #[serde(default)]
    size: Option<u64>,
}

fn tls() -> ureq::tls::TlsConfig {
    ureq::tls::TlsConfig::builder()
        .provider(ureq::tls::TlsProvider::NativeTls)
        .root_certs(ureq::tls::RootCerts::PlatformVerifier)
        .build()
}

fn get(url: &str) -> anyhow::Result<String> {
    // TLS de Windows et ses certificats (ureq prend Rustls par défaut, non
    // compilé ici) : fonctionne aussi derrière un proxy d'entreprise dont le
    // certificat est installé dans le système.
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .tls_config(tls())
        .timeout_global(Some(Duration::from_secs(20)))
        .user_agent(concat!("Uillamacpp/", env!("CARGO_PKG_VERSION")))
        .build()
        .into();
    Ok(agent.get(url).call()?.body_mut().read_to_string()?)
}

/// Dépôts GGUF de modèles de langage (texte et vision), triés.
pub fn list_repos(sort: Sort) -> anyhow::Result<Vec<Repo>> {
    let url = format!(
        "{API}?filter=gguf&sort={}&direction=-1&limit={FETCHED}",
        sort.param()
    );
    parse_repos(&get(&url)?, sort)
}

/// Garde les modèles de langage et trie.
pub fn parse_repos(text: &str, sort: Sort) -> anyhow::Result<Vec<Repo>> {
    let mut repos: Vec<Repo> = serde_json::from_str::<Vec<ApiModel>>(text)?
        .into_iter()
        .filter(ApiModel::is_language_model)
        .map(|m| Repo {
            vision: m.pipeline_tag.as_deref() == Some("image-text-to-text"),
            id: m.id,
            downloads: m.downloads,
            likes: m.likes,
            trending: m.trending,
        })
        .collect();
    match sort {
        Sort::Trending => repos.sort_by(|a, b| b.trending.total_cmp(&a.trending)),
        Sort::Downloads => repos.sort_by_key(|r| std::cmp::Reverse(r.downloads)),
    }
    repos.truncate(SHOWN);
    Ok(repos)
}

/// Fichiers GGUF d'un dépôt, du plus léger au plus lourd.
pub fn list_files(repo: &str) -> anyhow::Result<Vec<HubFile>> {
    parse_files(&get(&format!("{API}/{repo}?blobs=true"))?)
}

pub fn parse_files(text: &str) -> anyhow::Result<Vec<HubFile>> {
    let repo: ApiRepo = serde_json::from_str(text)?;
    let mut files: Vec<HubFile> = Vec::new();
    for s in repo.siblings {
        let name = s.rfilename;
        let lower = name.to_lowercase();
        if !lower.ends_with(".gguf") || lower.contains("mmproj") || is_imatrix(&lower) {
            continue;
        }
        let size = s.size.unwrap_or(0);
        // Modèle scindé : `…-00002-of-00003.gguf` s'ajoute à sa première
        // partie, seule proposée.
        if let Some(first) = first_shard(&name) {
            match files.iter_mut().find(|f| f.file == first) {
                Some(f) => {
                    f.size += size;
                    f.parts += 1;
                    f.members.push((name, size));
                }
                None => {
                    let (quant, variant) = quant_of(&first);
                    files.push(HubFile {
                        quant,
                        variant,
                        file: first,
                        members: vec![(name, size)],
                        size,
                        parts: 1,
                    })
                }
            }
            continue;
        }
        let (quant, variant) = quant_of(&name);
        files.push(HubFile {
            quant,
            variant,
            members: vec![(name.clone(), size)],
            file: name,
            size,
            parts: 1,
        });
    }
    for f in files.iter_mut() {
        f.members.sort();
    }
    files.sort_by_key(|f| f.size);
    Ok(files)
}

/// Nom de la première partie d'un fichier scindé, ou `None`.
fn first_shard(name: &str) -> Option<String> {
    let stem = name.strip_suffix(".gguf").or_else(|| name.strip_suffix(".GGUF"))?;
    let (head, total) = stem.rsplit_once("-of-")?;
    let (prefix, index) = head.rsplit_once('-')?;
    let digits = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit());
    (digits(index) && digits(total)).then(|| {
        format!("{prefix}-{}-of-{total}.gguf", "0".repeat(index.len() - 1) + "1")
    })
}

/// Données de calibration de quantisation (`imatrix-qwen.gguf`), pas un
/// modèle. Seuls le début et la fin du nom comptent : un vrai modèle peut
/// contenir « iMatrix » au milieu (`Qwen3.8-27B-iMatrix-NVFP4-MTP.gguf`).
fn is_imatrix(lower_path: &str) -> bool {
    let file = lower_path.rsplit('/').next().unwrap_or(lower_path);
    let stem = file.strip_suffix(".gguf").unwrap_or(file);
    stem.starts_with("imatrix") || stem.ends_with("imatrix")
}

/// (« Q4_K_M », « mtp ») ; à défaut de quantisation reconnue, le nom du
/// fichier sans dossier ni extension, sans variante.
fn quant_of(path: &str) -> (String, String) {
    let file = path.rsplit('/').next().unwrap_or(path);
    let stem = file.strip_suffix(".gguf").unwrap_or(file);
    // Suffixe de partie d'un modèle scindé (`-00001-of-00002`) : pas une
    // variante.
    let stem = match stem.rsplit_once("-of-") {
        Some((head, total)) if total.chars().all(|c| c.is_ascii_digit()) => head
            .rsplit_once('-')
            .filter(|(_, i)| !i.is_empty() && i.chars().all(|c| c.is_ascii_digit()))
            .map_or(stem, |(prefix, _)| prefix),
        _ => stem,
    };
    let quant = detect_quant(stem);
    if quant == "-" {
        return (stem.to_string(), String::new());
    }
    // `to_ascii_uppercase` : même longueur en octets, l'indice vaut dans
    // `stem` (`to_uppercase` change la longueur de « ﬁ » et faisait paniquer).
    let variant = stem
        .to_ascii_uppercase()
        .rfind(&quant)
        .map(|i| stem[i + quant.len()..].trim_matches(['-', '_', '.']).to_string())
        .unwrap_or_default();
    (quant, variant)
}

// --------------------------------------------------------------- download

/// Emplacement local d'un fichier du dépôt : `<modèles>/<auteur>/<dépôt>/…`,
/// la disposition qu'utilise déjà LM Studio.
pub fn local_path(models_dir: &Path, repo: &str, file: &str) -> PathBuf {
    let mut path = models_dir.to_path_buf();
    for part in repo.split('/').chain(file.split('/')) {
        path.push(part);
    }
    path
}

pub enum DownloadEvent {
    /// Octets reçus sur le total, parties comprises.
    Progress { done: u64, total: u64 },
    /// Reprise, nouvelle tentative ou redémarrage : pour la console.
    Notice(String),
    /// Chemin local de la (première) partie.
    Done(Result<PathBuf, String>),
}

/// Téléchargement en cours, annulable.
pub struct Download {
    pub rx: Receiver<DownloadEvent>,
    pub started: Instant,
    pub label: String,
    cancel: Arc<AtomicBool>,
}

impl Download {
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
    }
}

/// Télécharge un modèle (toutes ses parties) dans le répertoire des
/// modèles. Chaque fichier est écrit en `.part` puis renommé une fois
/// complet : un téléchargement interrompu n'apparaît jamais comme modèle.
///
/// Une coupure ne fait pas tout reprendre : les octets reçus restent dans le
/// `.part` et le téléchargement repart de là, automatiquement puis au
/// prochain lancement si les tentatives s'épuisent. Si la reprise est
/// impossible ou incohérente, le `.part` est supprimé et le fichier
/// retéléchargé en entier. Une annulation supprime le `.part`.
pub fn download(repo: String, file: HubFile, models_dir: PathBuf) -> Download {
    let (tx, rx) = channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let stop = cancel.clone();
    let label = format!("{repo} · {}", file.file);
    std::thread::spawn(move || {
        let result = fetch_all(
            &repo,
            &file,
            &models_dir,
            &stop,
            &mut |done, total| {
                let _ = tx.send(DownloadEvent::Progress { done, total });
            },
            &mut |msg| {
                let _ = tx.send(DownloadEvent::Notice(msg));
            },
        );
        let _ = tx.send(DownloadEvent::Done(result.map_err(|e| e.to_string())));
    });
    Download {
        rx,
        started: Instant::now(),
        label,
        cancel,
    }
}

fn fetch_all(
    repo: &str,
    file: &HubFile,
    models_dir: &Path,
    stop: &AtomicBool,
    progress: &mut dyn FnMut(u64, u64),
    notice: &mut dyn FnMut(String),
) -> anyhow::Result<PathBuf> {
    anyhow::ensure!(models_dir.is_dir(), "répertoire des modèles introuvable : {}", models_dir.display());
    // Pas de délai global sur l'agent : chaque plage demandée a le sien
    // (`CHUNK_TIMEOUT`).
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .tls_config(tls())
        .timeout_connect(Some(Duration::from_secs(20)))
        .timeout_recv_response(Some(Duration::from_secs(60)))
        .user_agent(concat!("Uillamacpp/", env!("CARGO_PKG_VERSION")))
        .build()
        .into();

    let total = file.size;
    // Octets des parties déjà terminées.
    let mut before = 0u64;
    for (name, size) in &file.members {
        let target = local_path(models_dir, repo, name);
        // Déjà là, complet : rien à refaire.
        if std::fs::metadata(&target).map(|m| m.len() == *size && *size > 0).unwrap_or(false) {
            before += size;
            progress(before, total);
            continue;
        }
        if let Some(dir) = target.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let part = target.with_extension("gguf.part");
        let url = format!(
            "https://huggingface.co/{repo}/resolve/main/{}",
            name.replace(' ', "%20")
        );
        let fetched = fetch_member(
            &agent,
            &url,
            &part,
            *size,
            stop,
            &mut |n| progress(before + n, total),
            notice,
        );
        if let Err(e) = fetched {
            // Annulé : choix explicite, le fichier partiel part avec. Sinon
            // il reste, pour reprendre au prochain téléchargement.
            if stop.load(Ordering::SeqCst) {
                let _ = std::fs::remove_file(&part);
            }
            return Err(e);
        }
        std::fs::rename(&part, &target)?;
        before += size;
        progress(before, total);
    }
    Ok(local_path(models_dir, repo, &file.file))
}

/// Échecs d'affilée, sans aucun octet reçu, avant d'abandonner.
const MAX_FAILURES: u32 = 5;
/// Délai d'une requête de plage, corps compris. Une connexion bloquée sans
/// erreur (ureq n'a pas de délai d'inactivité) est ainsi relancée ; les
/// octets déjà reçus sont gardés.
const CHUNK_TIMEOUT: Duration = Duration::from_secs(120);
const MIB: u64 = 1 << 20;

/// Taille de la prochaine plage : ≈ 20 s au débit mesuré, pour qu'une
/// requête finisse bien avant `CHUNK_TIMEOUT` sans multiplier les allers-
/// retours sur une bonne connexion.
fn chunk_len(speed: f64) -> u64 {
    if speed <= 0.0 {
        16 * MIB
    } else {
        ((speed * 20.0) as u64).clamp(4 * MIB, 1024 * MIB)
    }
}

/// Vérifie l'en-tête `Content-Range: bytes a-b/total` d'une réponse 206 :
/// la plage doit commencer où le `.part` s'arrête, et le fichier distant
/// avoir la taille attendue (sinon il a changé depuis le début du
/// téléchargement, et les deux morceaux ne vont pas ensemble).
fn check_content_range(header: Option<&str>, have: u64, size: u64) -> Result<(), String> {
    let parse = || -> Option<(u64, Option<u64>)> {
        let rest = header?.trim().strip_prefix("bytes ")?;
        let (range, total) = rest.split_once('/')?;
        let start = range.split_once('-')?.0.trim().parse().ok()?;
        Some((start, total.trim().parse().ok()))
    };
    let Some((start, total)) = parse() else {
        return Err("réponse de reprise illisible".into());
    };
    if start != have {
        return Err(format!("reprise à l'octet {start} au lieu de {have}"));
    }
    if let Some(t) = total {
        if size > 0 && t != size {
            return Err(format!("le fichier a changé sur le serveur ({t} octets au lieu de {size})"));
        }
    }
    Ok(())
}

enum Failure {
    /// Inutile de réessayer (dépôt protégé, disque plein, annulation…).
    Fatal(anyhow::Error),
    /// Coupure : on reprend là où on en est.
    Retry(String),
    /// Reprise incohérente : supprimer le `.part` et tout retélécharger.
    Restart(String),
    /// Le serveur ignore les plages : téléchargement d'un seul tenant.
    NoRanges,
}

fn fetch_member(
    agent: &ureq::Agent,
    url: &str,
    part: &Path,
    size: u64,
    stop: &AtomicBool,
    progress: &mut dyn FnMut(u64),
    notice: &mut dyn FnMut(String),
) -> anyhow::Result<()> {
    let name = part.file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let mut have = std::fs::metadata(part).map(|m| m.len()).unwrap_or(0);
    // Taille inconnue : pas de plages, une seule requête sans délai.
    let mut ranged = size > 0;
    if have > 0 && (!ranged || have > size) {
        std::fs::remove_file(part)?;
        notice(format!("{name} : fichier partiel inutilisable, supprimé ; nouveau téléchargement"));
        have = 0;
    } else if have > 0 {
        notice(format!(
            "{name} : reprise du téléchargement à {} sur {}",
            crate::discovery::human_size(have),
            crate::discovery::human_size(size)
        ));
    }
    progress(have);

    let mut speed = 0.0;
    let mut failures = 0u32;
    loop {
        anyhow::ensure!(!stop.load(Ordering::SeqCst), "annulé");
        if ranged && have >= size {
            let len = std::fs::metadata(part).map(|m| m.len()).unwrap_or(0);
            if len == size {
                return Ok(());
            }
            failures += 1;
            anyhow::ensure!(failures < MAX_FAILURES, "{name} : taille incorrecte après téléchargement");
            let _ = std::fs::remove_file(part);
            notice(format!("{name} : taille incorrecte ({len} octets), nouveau téléchargement"));
            have = 0;
            progress(0);
            continue;
        }

        let start = have;
        let step = if ranged {
            fetch_range(agent, url, part, &mut have, size, &mut speed, stop, progress)
        } else {
            fetch_whole(agent, url, part, &mut have, stop, progress)
        };
        // Sans plages, chaque tentative repart de zéro : seul un
        // téléchargement par plages progresse vraiment.
        let progressed = ranged && have > start;
        if progressed {
            failures = 0;
        }
        match step {
            Ok(()) if ranged => {}
            Ok(()) => {
                let len = std::fs::metadata(part).map(|m| m.len()).unwrap_or(0);
                if size == 0 || len == size {
                    return Ok(());
                }
                failures += 1;
                anyhow::ensure!(failures < MAX_FAILURES, "{name} : taille incorrecte après téléchargement");
                notice(format!("{name} : taille incorrecte ({len} octets), nouveau téléchargement"));
                have = 0;
            }
            Err(Failure::Fatal(e)) => return Err(e),
            Err(Failure::NoRanges) => {
                let _ = std::fs::remove_file(part);
                notice(format!(
                    "{name} : le serveur ne permet pas la reprise, téléchargement complet"
                ));
                ranged = false;
                have = 0;
                progress(0);
            }
            Err(Failure::Restart(reason)) => {
                failures += 1;
                anyhow::ensure!(failures < MAX_FAILURES, "{name} : {reason}");
                let _ = std::fs::remove_file(part);
                notice(format!("{name} : {reason} ; fichier partiel supprimé, nouveau téléchargement"));
                have = 0;
                progress(0);
            }
            Err(Failure::Retry(e)) => {
                // Une plage coupée après avoir reçu des octets repart tout
                // de suite ; sans progrès, on attend de plus en plus.
                if progressed {
                    continue;
                }
                failures += 1;
                if failures >= MAX_FAILURES {
                    anyhow::bail!(
                        "{name} : {e} ({MAX_FAILURES} tentatives). {}",
                        if ranged && have > 0 {
                            "Le fichier partiel est conservé : relancez le téléchargement pour reprendre."
                        } else {
                            "Relancez le téléchargement."
                        }
                    );
                }
                let delay = 2u64.pow(failures).min(30);
                notice(format!(
                    "{name} : {e} ; nouvelle tentative dans {delay} s ({failures}/{})",
                    MAX_FAILURES - 1
                ));
                let until = Instant::now() + Duration::from_secs(delay);
                while Instant::now() < until {
                    anyhow::ensure!(!stop.load(Ordering::SeqCst), "annulé");
                    std::thread::sleep(Duration::from_millis(100));
                }
                if !ranged {
                    have = 0;
                }
            }
        }
    }
}

/// Erreur HTTP d'une requête : définitive ou à retenter.
fn http_failure(e: ureq::Error, url: &str) -> Failure {
    match e {
        ureq::Error::StatusCode(401 | 403) => Failure::Fatal(anyhow::anyhow!(
            "dépôt protégé : acceptez sa licence sur huggingface.co ({url})"
        )),
        ureq::Error::StatusCode(404) => {
            Failure::Fatal(anyhow::anyhow!("fichier introuvable : {url}"))
        }
        ureq::Error::StatusCode(416) => {
            Failure::Restart("le serveur refuse la plage demandée".into())
        }
        ureq::Error::StatusCode(c) if c == 408 || c == 429 || c >= 500 => {
            Failure::Retry(format!("HTTP {c}"))
        }
        ureq::Error::StatusCode(c) => Failure::Fatal(anyhow::anyhow!("HTTP {c} : {url}")),
        other => Failure::Retry(other.to_string()),
    }
}

/// Demande la plage suivante et l'ajoute au `.part`.
#[allow(clippy::too_many_arguments)]
fn fetch_range(
    agent: &ureq::Agent,
    url: &str,
    part: &Path,
    have: &mut u64,
    size: u64,
    speed: &mut f64,
    stop: &AtomicBool,
    progress: &mut dyn FnMut(u64),
) -> Result<(), Failure> {
    let end = (*have + chunk_len(*speed)).min(size) - 1;
    let response = agent
        .get(url)
        .header("Range", format!("bytes={}-{end}", *have))
        .config()
        .timeout_global(Some(CHUNK_TIMEOUT))
        .build()
        .call()
        .map_err(|e| http_failure(e, url))?;
    if response.status().as_u16() != 206 {
        return Err(Failure::NoRanges);
    }
    let range = response.headers().get("content-range").and_then(|v| v.to_str().ok());
    check_content_range(range, *have, size).map_err(Failure::Restart)?;
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(part)
        .map_err(|e| Failure::Fatal(e.into()))?;
    let started = Instant::now();
    let first = *have;
    let result = copy_body(response, file, have, stop, progress);
    let secs = started.elapsed().as_secs_f64();
    if secs > 0.5 {
        *speed = (*have - first) as f64 / secs;
    }
    result
}

/// Tout le fichier en une requête, sans délai global (serveur sans plages,
/// ou taille inconnue).
fn fetch_whole(
    agent: &ureq::Agent,
    url: &str,
    part: &Path,
    have: &mut u64,
    stop: &AtomicBool,
    progress: &mut dyn FnMut(u64),
) -> Result<(), Failure> {
    let response = agent.get(url).call().map_err(|e| http_failure(e, url))?;
    let file = std::fs::File::create(part).map_err(|e| Failure::Fatal(e.into()))?;
    *have = 0;
    copy_body(response, file, have, stop, progress)
}

/// Recopie le corps de la réponse dans le fichier en comptant les octets.
/// Ce qui a été reçu avant une coupure est écrit : rien ne se perd.
fn copy_body(
    response: ureq::http::Response<ureq::Body>,
    file: std::fs::File,
    have: &mut u64,
    stop: &AtomicBool,
    progress: &mut dyn FnMut(u64),
) -> Result<(), Failure> {
    let mut reader = response.into_body().into_reader();
    let mut out = std::io::BufWriter::with_capacity(MIB as usize, file);
    let mut buf = vec![0u8; MIB as usize];
    let mut last = Instant::now();
    let result = loop {
        if stop.load(Ordering::SeqCst) {
            break Err(Failure::Fatal(anyhow::anyhow!("annulé")));
        }
        match reader.read(&mut buf) {
            Ok(0) => break Ok(()),
            Ok(n) => {
                if let Err(e) = out.write_all(&buf[..n]) {
                    break Err(Failure::Fatal(e.into()));
                }
                *have += n as u64;
                if last.elapsed() >= Duration::from_millis(200) {
                    progress(*have);
                    last = Instant::now();
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => break Err(Failure::Retry(e.to_string())),
        }
    };
    out.flush().map_err(|e| Failure::Fatal(e.into()))?;
    progress(*have);
    result
}

/// 1 234 567 → « 1,2 M ».
pub fn compact_count(n: u64) -> String {
    match n {
        n if n >= 1_000_000 => format!("{:.1} M", n as f64 / 1e6),
        n if n >= 1_000 => format!("{:.0} k", n as f64 / 1e3),
        n => n.to_string(),
    }
    .replace('.', ",")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Extrait d'une vraie réponse de l'API (septembre 2026).
    const MODELS: &str = r#"[
      {"id":"unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF","downloads":10762547,"likes":900,"trendingScore":40,"pipeline_tag":"text-generation","tags":["gguf"]},
      {"id":"unsloth/Qwen3.8-27B-GGUF","downloads":6651662,"likes":1200,"trendingScore":90,"tags":["gguf","qwen3_5","conversational"]},
      {"id":"mudler/locate-anything.cpp-gguf","downloads":5591572,"likes":10,"trendingScore":5,"pipeline_tag":"object-detection","tags":["gguf"]},
      {"id":"abenzerps/Qwen-Image-2.1-Uncensored-GGUF","downloads":964220,"likes":2000,"trendingScore":1730,"pipeline_tag":"text-to-image","tags":["gguf"]},
      {"id":"ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF","downloads":1608439,"likes":1749,"trendingScore":267,"pipeline_tag":"image-text-to-text","tags":["gguf"]},
      {"id":"someone/unlabelled-GGUF","downloads":999999,"likes":1,"trendingScore":1,"tags":["gguf"]}
    ]"#;

    #[test]
    fn keeps_language_models_only_including_untagged_conversational() {
        let by_dl = parse_repos(MODELS, Sort::Downloads).unwrap();
        let ids: Vec<&str> = by_dl.iter().map(|r| r.id.as_str()).collect();
        // Image, détection d'objets et dépôt sans aucune indication : écartés.
        // Qwen3.8 sans type de tâche mais « conversational » : gardé.
        assert_eq!(
            ids,
            [
                "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
                "unsloth/Qwen3.8-27B-GGUF",
                "ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF"
            ]
        );
        assert!(by_dl[2].vision && !by_dl[0].vision);

        let by_trend = parse_repos(MODELS, Sort::Trending).unwrap();
        assert_eq!(by_trend[0].id, "ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF");
    }

    #[test]
    fn groups_shards_skips_mmproj_and_sorts_by_size() {
        let json = r#"{"siblings":[
          {"rfilename":"README.md","size":100},
          {"rfilename":"Qwen3-14B-Q8_0.gguf","size":15000},
          {"rfilename":"Qwen3-14B-Q4_K_M.gguf","size":9000},
          {"rfilename":"mmproj-Qwen3-14B-F16.gguf","size":800},
          {"rfilename":"BF16/Qwen3-14B-BF16-00002-of-00002.gguf","size":14000},
          {"rfilename":"BF16/Qwen3-14B-BF16-00001-of-00002.gguf","size":15000},
          {"rfilename":"Qwen3-14B-GSQ-RCO.gguf","size":7000},
          {"rfilename":"imatrix-qwen3-14b.gguf","size":13},
          {"rfilename":"Qwen3-14B-Q4_K_M-mtp.gguf","size":9400}
        ]}"#;
        let files = parse_files(json).unwrap();
        let summary: Vec<(&str, &str, &str, u64, usize)> = files
            .iter()
            .map(|f| (f.file.as_str(), f.quant.as_str(), f.variant.as_str(), f.size, f.parts))
            .collect();
        // imatrix écarté ; la variante -mtp distinguée de la version normale.
        assert_eq!(
            summary,
            [
                ("Qwen3-14B-GSQ-RCO.gguf", "Qwen3-14B-GSQ-RCO", "", 7000, 1),
                ("Qwen3-14B-Q4_K_M.gguf", "Q4_K_M", "", 9000, 1),
                ("Qwen3-14B-Q4_K_M-mtp.gguf", "Q4_K_M", "mtp", 9400, 1),
                ("Qwen3-14B-Q8_0.gguf", "Q8_0", "", 15000, 1),
                ("BF16/Qwen3-14B-BF16-00001-of-00002.gguf", "BF16", "", 29000, 2),
            ]
        );
    }

    /// Attend la fin d'un téléchargement, en annulant au premier signe de
    /// progression si demandé.
    fn wait(d: &Download, cancel_early: bool) -> Result<PathBuf, String> {
        wait_notes(d, cancel_early).0
    }

    /// Comme `wait`, avec les messages de reprise reçus.
    fn wait_notes(d: &Download, cancel_early: bool) -> (Result<PathBuf, String>, Vec<String>) {
        let mut notes = Vec::new();
        loop {
            match d.rx.recv().unwrap() {
                DownloadEvent::Progress { done, .. } if cancel_early && done > 0 => d.cancel(),
                DownloadEvent::Progress { .. } => {}
                DownloadEvent::Notice(msg) => {
                    println!("   · {msg}");
                    notes.push(msg);
                }
                DownloadEvent::Done(r) => return (r, notes),
            }
        }
    }

    #[test]
    fn content_range_must_continue_the_partial_file() {
        assert_eq!(check_content_range(Some("bytes 100-199/1000"), 100, 1000), Ok(()));
        // Taille totale inconnue (« * ») : acceptée.
        assert_eq!(check_content_range(Some("bytes 100-199/*"), 100, 1000), Ok(()));
        // Mauvais point de départ, fichier changé, en-tête absent ou illisible.
        assert!(check_content_range(Some("bytes 0-99/1000"), 100, 1000).is_err());
        assert!(check_content_range(Some("bytes 100-199/2000"), 100, 1000)
            .unwrap_err()
            .contains("changé"));
        assert!(check_content_range(None, 100, 1000).is_err());
        assert!(check_content_range(Some("octets 100-199/1000"), 100, 1000).is_err());
    }

    #[test]
    fn chunks_follow_measured_speed() {
        assert_eq!(chunk_len(0.0), 16 * MIB);
        // 50 Mo/s : 20 s de transfert, soit 1 000 Mo.
        assert_eq!(chunk_len(50e6), 1_000_000_000);
        // Bornes : ni minuscule sur une connexion lente, ni démesurée.
        assert_eq!(chunk_len(1_000.0), 4 * MIB);
        assert_eq!(chunk_len(1e10), 1024 * MIB);
    }

    /// Vrai téléchargement (88 Mo), relance, annulation et dépôt protégé :
    /// `cargo test live_download -- --ignored --nocapture`.
    #[test]
    #[ignore = "réseau : télécharge un petit modèle depuis huggingface.co"]
    fn live_download() {
        let dir = std::env::temp_dir().join(format!("uillamacpp-dl-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let repo = "unsloth/SmolLM2-135M-Instruct-GGUF";
        let files = list_files(repo).unwrap();
        let small = files.iter().find(|f| f.quant == "Q2_K").unwrap().clone();

        // 1. Téléchargement complet.
        let t = Instant::now();
        let path = wait(&download(repo.into(), small.clone(), dir.clone()), false).unwrap();
        let len = std::fs::metadata(&path).unwrap().len();
        let mut magic = [0u8; 4];
        std::fs::File::open(&path).unwrap().read_exact(&mut magic).unwrap();
        println!("1. {} : {} octets en {:.1} s", path.display(), len, t.elapsed().as_secs_f64());
        assert_eq!(len, small.size, "taille exacte");
        assert_eq!(&magic, b"GGUF");
        assert_eq!(path, local_path(&dir, repo, &small.file));

        // 2. Déjà présent : rien n'est retéléchargé.
        let t = Instant::now();
        wait(&download(repo.into(), small.clone(), dir.clone()), false).unwrap();
        println!("2. relance : {:.2} s", t.elapsed().as_secs_f64());
        assert!(t.elapsed() < Duration::from_secs(3));

        // 2b. Reprise : un .part coupé à mi-chemin repart de là, et le
        // fichier final est identique octet pour octet.
        let full = std::fs::read(&path).unwrap();
        let dir2 = dir.join("reprise");
        let target2 = local_path(&dir2, repo, &small.file);
        std::fs::create_dir_all(target2.parent().unwrap()).unwrap();
        let part2 = target2.with_extension("gguf.part");
        std::fs::write(&part2, &full[..full.len() / 2]).unwrap();
        let (r, notes) = wait_notes(&download(repo.into(), small.clone(), dir2.clone()), false);
        assert_eq!(std::fs::read(r.unwrap()).unwrap(), full, "reprise identique");
        assert!(notes.iter().any(|n| n.contains("reprise")), "{notes:?}");
        assert!(!part2.exists());

        // 2c. .part plus long que le fichier : supprimé, tout retéléchargé.
        std::fs::remove_file(&target2).unwrap();
        let mut too_long = full.clone();
        too_long.extend_from_slice(&[0u8; 100]);
        std::fs::write(&part2, &too_long).unwrap();
        let (r, notes) = wait_notes(&download(repo.into(), small.clone(), dir2.clone()), false);
        assert_eq!(std::fs::read(r.unwrap()).unwrap(), full, "retéléchargé");
        assert!(notes.iter().any(|n| n.contains("inutilisable")), "{notes:?}");

        // 2d. Fichier distant différent (taille attendue fausse) : la reprise
        // est refusée, le .part supprimé, et l'échec expliqué.
        std::fs::remove_file(&target2).unwrap();
        std::fs::write(&part2, &full[..1000]).unwrap();
        let mut wrong = small.clone();
        wrong.size += 1;
        wrong.members[0].1 += 1;
        let (r, notes) = wait_notes(&download(repo.into(), wrong, dir2.clone()), false);
        println!("2d. {r:?}");
        assert!(notes.iter().any(|n| n.contains("changé")), "{notes:?}");
        assert!(r.is_err());

        // 3. Annulation : erreur « annulé », aucun fichier partiel.
        let other = files.iter().find(|f| f.quant == "Q4_K_M").unwrap().clone();
        let err = wait(&download(repo.into(), other.clone(), dir.clone()), true).unwrap_err();
        let target = local_path(&dir, repo, &other.file);
        println!("3. annulation : {err}");
        assert!(err.contains("annulé"));
        assert!(!target.exists() && !target.with_extension("gguf.part").exists());

        // 4. Dépôt protégé : message explicite.
        let gated = HubFile {
            file: "x.gguf".into(),
            members: vec![("x.gguf".into(), 1)],
            quant: "x".into(),
            variant: String::new(),
            size: 1,
            parts: 1,
        };
        let err = wait(
            &download("HuggingFaceTB/SmolLM2-135M-Instruct-GGUF".into(), gated, dir.clone()),
            false,
        )
        .unwrap_err();
        println!("4. protégé : {err}");
        assert!(err.contains("protégé") || err.contains("introuvable"), "{err}");

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn non_ascii_file_names_do_not_panic() {
        assert_eq!(quant_of("ﬁ-ﬁ-Q4_K_M-mtp.gguf"), ("Q4_K_M".to_string(), "mtp".to_string()));
    }

    #[test]
    fn split_model_lists_every_part_in_order() {
        let json = r#"{"siblings":[
          {"rfilename":"BF16/M-BF16-00002-of-00002.gguf","size":14},
          {"rfilename":"BF16/M-BF16-00001-of-00002.gguf","size":15}
        ]}"#;
        let files = parse_files(json).unwrap();
        assert_eq!(
            files[0].members,
            [
                ("BF16/M-BF16-00001-of-00002.gguf".to_string(), 15),
                ("BF16/M-BF16-00002-of-00002.gguf".to_string(), 14)
            ]
        );
    }

    #[test]
    fn local_path_mirrors_the_repository() {
        let p = local_path(Path::new("C:/models"), "unsloth/Qwen3-14B-GGUF", "BF16/Qwen3-14B-BF16-00001-of-00002.gguf");
        assert_eq!(
            p,
            Path::new("C:/models/unsloth/Qwen3-14B-GGUF/BF16/Qwen3-14B-BF16-00001-of-00002.gguf")
        );
    }

    #[test]
    fn imatrix_is_recognised_by_name_edges_only() {
        assert!(is_imatrix("imatrix-qwen3.8-27b.gguf"));
        assert!(is_imatrix("sub/qwen3.8-27b.imatrix.gguf"));
        // Un vrai modèle qui contient « iMatrix » au milieu de son nom.
        assert!(!is_imatrix("qwen3.8-27b-imatrix-nvfp4-mtp.gguf"));
    }

    /// Interroge la vraie API (réseau) : `cargo test live_hub -- --ignored --nocapture`.
    #[test]
    #[ignore = "réseau : interroge huggingface.co"]
    fn live_hub() {
        for sort in Sort::ALL {
            let repos = list_repos(sort).unwrap();
            println!("{:?} : {} dépôts", sort, repos.len());
            for r in repos.iter().take(4) {
                println!("  {} · {} téléch.{}", r.id, compact_count(r.downloads), if r.vision { " · vision" } else { "" });
            }
            assert!(repos.len() > 10);
        }
        let files = list_files("unsloth/Qwen3-14B-GGUF").unwrap();
        println!("unsloth/Qwen3-14B-GGUF : {} fichiers", files.len());
        for f in files.iter().take(5) {
            println!("  {} · {} · {:.1} Go · {} partie(s)", f.quant, f.file, f.size as f64 / 1e9, f.parts);
        }
        assert!(files.iter().any(|f| f.quant == "Q4_K_M" && f.size > 1_000_000_000));
    }

    #[test]
    fn compact_counts_read_well() {
        assert_eq!(compact_count(3_343_748), "3,3 M");
        assert_eq!(compact_count(28_802), "29 k");
        assert_eq!(compact_count(313), "313");
    }
}
