//! Exécution de `llama-bench` sur les combinaisons modèle × version,
//! et persistance des résultats dans `benchmark.md`.

use crate::config::app_dir;
use crate::discovery::{base_name, ModelInfo, VersionInfo};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};

#[cfg(windows)]
use std::os::windows::process::CommandExt;
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// Réglages d'une série de mesures, choisis dans des listes fermées
/// (valeurs vérifiées contre `llama-bench --help`). Enregistrés dans
/// `config.json` ; chaque ligne de résultat garde les siens (`label`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct BenchSettings {
    pub ngl: String,
    /// Type du cache KV, appliqué à K et V.
    pub cache_type: String,
    pub flash_attn: String,
    pub prompt: String,
    pub gen: String,
    /// Contexte déjà rempli avant la mesure (`-d`) : le débit baisse quand
    /// le contexte grandit.
    pub depth: String,
    pub repetitions: String,
}

/// Les réglages d'avant leur introduction : les lignes à huit colonnes de
/// `benchmark.md` ont été mesurées ainsi.
impl Default for BenchSettings {
    fn default() -> Self {
        Self {
            ngl: "999".into(),
            cache_type: "f16".into(),
            flash_attn: "auto".into(),
            prompt: "512".into(),
            gen: "128".into(),
            depth: "0".into(),
            repetitions: "3".into(),
        }
    }
}

pub const NGL_CHOICES: &[&str] = &["0", "8", "16", "24", "32", "48", "64", "99", "999"];
pub const FLASH_ATTN_CHOICES: &[&str] = &["auto", "on", "off"];
pub const PROMPT_CHOICES: &[&str] = &["128", "256", "512", "1024", "2048", "4096"];
pub const GEN_CHOICES: &[&str] = &["32", "64", "128", "256", "512"];
pub const DEPTH_CHOICES: &[&str] = &[
    "0", "1024", "2048", "4096", "8192", "16384", "32768", "65536", "131072",
];
pub const REPETITION_CHOICES: &[&str] = &["1", "2", "3", "5", "10"];

impl BenchSettings {
    /// Options passées à llama-bench (hors modèle et format de sortie).
    pub fn args(&self) -> Vec<String> {
        let mut args: Vec<String> = [
            "-ngl",
            &self.ngl,
            "-ctk",
            &self.cache_type,
            "-ctv",
            &self.cache_type,
            "-p",
            &self.prompt,
            "-n",
            &self.gen,
            "-r",
            &self.repetitions,
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        // Valeurs par défaut de llama-bench : omises, ce qui ménage aussi
        // les versions plus anciennes.
        if self.flash_attn != "auto" {
            args.extend(["-fa".to_string(), self.flash_attn.clone()]);
        }
        if self.depth != "0" {
            args.extend(["-d".to_string(), self.depth.clone()]);
        }
        args
    }

    /// `ngl 999 · f16 · fa auto · pp512 · tg128 · r3`, plus `· d4096` avec
    /// une profondeur : la colonne Config de `benchmark.md`.
    pub fn label(&self) -> String {
        let mut s = format!(
            "ngl {} · {} · fa {} · pp{} · tg{} · r{}",
            self.ngl, self.cache_type, self.flash_attn, self.prompt, self.gen, self.repetitions
        );
        if self.depth != "0" {
            s.push_str(&format!(" · d{}", self.depth));
        }
        s
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BenchRow {
    pub model: String,
    pub quant: String,
    pub version: String,
    pub backend: String,
    pub size: String,
    pub params: String,
    pub pp: String,
    pub tg: String,
    /// Réglages de la mesure (`BenchSettings::label`).
    pub config: String,
}

impl BenchRow {
    /// Une mesure par modèle, version et réglages.
    pub fn key(&self) -> (String, String, String, String) {
        (
            self.model.to_lowercase(),
            self.quant.to_lowercase(),
            self.version.to_lowercase(),
            self.config.to_lowercase(),
        )
    }

    /// Valeur numérique du débit prompt, pour le tri.
    pub fn pp_value(&self) -> f64 {
        self.pp
            .split('±')
            .next()
            .unwrap_or("")
            .trim()
            .parse()
            .unwrap_or(0.0)
    }
}

pub enum BenchEvent {
    Log(String),
    Progress { done: usize, total: usize, label: String },
    Row(BenchRow),
    Done { completed: usize, failed: usize },
}

pub struct BenchHandle {
    pub rx: Receiver<BenchEvent>,
    cancel: Arc<AtomicBool>,
    current_pid: Arc<Mutex<Option<u32>>>,
}

impl BenchHandle {
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
        if let Ok(guard) = self.current_pid.lock() {
            if let Some(pid) = *guard {
                kill_tree(pid);
            }
        }
    }
}

fn kill_tree(pid: u32) {
    #[cfg(windows)]
    {
        let _ = Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .creation_flags(CREATE_NO_WINDOW)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    #[cfg(not(windows))]
    {
        let _ = Command::new("kill").arg("-9").arg(pid.to_string()).status();
    }
}

/// Couples (version, modèle) à mesurer.
pub type Job = (VersionInfo, ModelInfo);

/// Lance les benchmarks en tâche de fond.
pub fn run(jobs: Vec<Job>, settings: BenchSettings) -> BenchHandle {
    let (tx, rx) = channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let current_pid = Arc::new(Mutex::new(None));

    let worker_cancel = cancel.clone();
    let worker_pid = current_pid.clone();
    std::thread::spawn(move || {
        let total = jobs.len();
        let mut completed = 0usize;
        let mut failed = 0usize;

        for (index, (version, model)) in jobs.into_iter().enumerate() {
            if worker_cancel.load(Ordering::SeqCst) {
                let _ = tx.send(BenchEvent::Log("— benchmark interrompu —".into()));
                break;
            }
            let label = format!("{} · {}", version.name, model.name);
            let _ = tx.send(BenchEvent::Progress {
                done: index,
                total,
                label: label.clone(),
            });
            let _ = tx.send(BenchEvent::Log(format!(
                "▶ [{}/{}] {label}",
                index + 1,
                total
            )));

            match run_one(&version, &model, &settings, &tx, &worker_pid) {
                Ok(rows) if !rows.is_empty() => {
                    completed += 1;
                    for row in rows {
                        let _ = tx.send(BenchEvent::Row(row));
                    }
                }
                Ok(_) => {
                    failed += 1;
                    let _ = tx.send(BenchEvent::Log(
                        "  ! aucun résultat exploitable dans la sortie".into(),
                    ));
                }
                Err(e) => {
                    failed += 1;
                    let _ = tx.send(BenchEvent::Log(format!("  ! échec : {e}")));
                }
            }
        }

        let _ = tx.send(BenchEvent::Progress {
            done: total,
            total,
            label: String::new(),
        });
        let _ = tx.send(BenchEvent::Done { completed, failed });
    });

    BenchHandle {
        rx,
        cancel,
        current_pid,
    }
}

fn run_one(
    version: &VersionInfo,
    model: &ModelInfo,
    settings: &BenchSettings,
    tx: &Sender<BenchEvent>,
    pid_slot: &Arc<Mutex<Option<u32>>>,
) -> anyhow::Result<Vec<BenchRow>> {
    let exe = version.bench();
    anyhow::ensure!(exe.is_file(), "{} introuvable", exe.display());

    let mut cmd = Command::new(&exe);
    cmd.arg("-m")
        .arg(&model.full_path)
        .args(settings.args())
        .args(["-o", "md"]);
    let _ = tx.send(BenchEvent::Log(format!("  {}", settings.label())));
    cmd.current_dir(&version.dir)
    .stdin(Stdio::null())
    .stdout(Stdio::piped())
    .stderr(Stdio::piped());
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);

    let mut child = cmd.spawn()?;
    if let Ok(mut guard) = pid_slot.lock() {
        *guard = Some(child.id());
    }

    // stderr : progression, relayée telle quelle dans la console.
    let stderr_thread = child.stderr.take().map(|err| {
        let tx = tx.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(err).split(b'\n').flatten() {
                let text = String::from_utf8_lossy(&line).trim_end().to_string();
                if !text.trim().is_empty() {
                    let _ = tx.send(BenchEvent::Log(format!("  {text}")));
                }
            }
        })
    });

    let mut stdout_text = String::new();
    if let Some(mut out) = child.stdout.take() {
        let mut buf = Vec::new();
        out.read_to_end(&mut buf)?;
        stdout_text = String::from_utf8_lossy(&buf).into_owned();
    }
    let status = child.wait()?;
    if let Some(handle) = stderr_thread {
        let _ = handle.join();
    }
    if let Ok(mut guard) = pid_slot.lock() {
        *guard = None;
    }

    if !status.success() {
        anyhow::bail!("llama-bench a quitté avec le code {:?}", status.code());
    }

    Ok(parse_table(&stdout_text, &version.name, model, &settings.label()))
}

/// Analyse la table Markdown produite par `llama-bench -o md`.
pub fn parse_table(output: &str, version: &str, model: &ModelInfo, config: &str) -> Vec<BenchRow> {
    let mut header: Option<Vec<String>> = None;
    let mut pp = String::new();
    let mut tg = String::new();
    let mut backend = String::new();
    let mut size = String::new();
    let mut params = String::new();

    for line in output.lines() {
        let line = line.trim();
        if !line.starts_with('|') {
            continue;
        }
        let cells: Vec<String> = line
            .trim_matches('|')
            .split('|')
            .map(|c| c.trim().to_string())
            .collect();

        // Ligne de séparation `| --- | ---: |`
        if cells
            .iter()
            .all(|c| !c.is_empty() && c.chars().all(|ch| ch == '-' || ch == ':'))
        {
            continue;
        }

        let Some(cols) = &header else {
            header = Some(cells.iter().map(|c| c.to_lowercase()).collect());
            continue;
        };

        let get = |name: &str| -> String {
            cols.iter()
                .position(|c| c == name)
                .and_then(|i| cells.get(i))
                .cloned()
                .unwrap_or_default()
        };

        let test = get("test");
        let tps = get("t/s");
        if backend.is_empty() {
            backend = get("backend");
        }
        if size.is_empty() {
            size = get("size");
        }
        if params.is_empty() {
            params = get("params");
        }

        if test.starts_with("pp") {
            pp = tps;
        } else if test.starts_with("tg") {
            tg = tps;
        }
    }

    if pp.is_empty() && tg.is_empty() {
        return Vec::new();
    }

    vec![BenchRow {
        model: base_name(&model.name),
        quant: model.quant.clone(),
        version: version.to_string(),
        backend: if backend.is_empty() { "-".into() } else { backend },
        size: if size.is_empty() {
            model.size_human()
        } else {
            size
        },
        params: if params.is_empty() { "-".into() } else { params },
        pp: if pp.is_empty() { "-".into() } else { pp },
        tg: if tg.is_empty() { "-".into() } else { tg },
        config: config.to_string(),
    }]
}

pub fn markdown_path() -> PathBuf {
    app_dir().join("benchmark.md")
}

/// Relit `benchmark.md` pour connaître les combinaisons déjà mesurées.
pub fn load_results() -> Vec<BenchRow> {
    std::fs::read_to_string(markdown_path())
        .map(|text| parse_results(&text))
        .unwrap_or_default()
}

/// Lignes d'un `benchmark.md` : huit colonnes (avant les réglages) ou neuf
/// (avec la colonne Config).
pub fn parse_results(text: &str) -> Vec<BenchRow> {
    let mut rows = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if !line.starts_with('|') {
            continue;
        }
        let cells: Vec<String> = line
            .trim_matches('|')
            .split('|')
            .map(|c| c.trim().to_string())
            .collect();
        if cells.len() < 8 {
            continue;
        }
        let lower = cells[0].to_lowercase();
        if lower == "modèle" || lower == "model" {
            continue;
        }
        if cells
            .iter()
            .all(|c| !c.is_empty() && c.chars().all(|ch| ch == '-' || ch == ':'))
        {
            continue;
        }
        rows.push(BenchRow {
            model: cells[0].clone(),
            quant: cells[1].clone(),
            version: cells[2].clone(),
            backend: cells[3].clone(),
            size: cells[4].clone(),
            params: cells[5].clone(),
            pp: cells[6].clone(),
            tg: cells[7].clone(),
            // Neuvième colonne absente : fichier d'avant les réglages,
            // mesuré avec les valeurs par défaut.
            config: cells
                .get(8)
                .filter(|c| !c.is_empty())
                .cloned()
                .unwrap_or_else(|| BenchSettings::default().label()),
        });
    }
    rows
}

/// Rend le tableau Markdown, trié par débit de prompt décroissant.
pub fn render_markdown(rows: &[BenchRow]) -> String {
    let mut sorted = rows.to_vec();
    sorted.sort_by(|a, b| {
        b.pp_value()
            .partial_cmp(&a.pp_value())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut out = String::from("# Benchmark Results\n\n");
    out.push_str(&format!(
        "**Date de dernière mise à jour :** {}\n\n",
        chrono::Local::now().format("%Y-%m-%d %H:%M:%S")
    ));
    out.push_str(
        "**Paramètres :** colonne Config (couches GPU, cache K/V, flash attention, \
         tokens de prompt et générés, répétitions, profondeur de contexte).\n\n",
    );
    out.push_str("## Résultats\n\n");
    out.push_str(
        "| Modèle | Quant | Version | Backend | Size | Params | PP (t/s) | TG (t/s) | Config |\n",
    );
    out.push_str(
        "|--------|-------|---------|---------|------|--------|----------|----------|--------|\n",
    );
    for r in &sorted {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            r.model, r.quant, r.version, r.backend, r.size, r.params, r.pp, r.tg, r.config
        ));
    }
    out
}

pub fn save_results(rows: &[BenchRow]) -> anyhow::Result<PathBuf> {
    let path = markdown_path();
    std::fs::write(&path, render_markdown(rows))?;
    Ok(path)
}

/// Fusionne un résultat dans la liste : une combinaison déjà présente est
/// remplacée par la mesure la plus récente.
pub fn upsert(rows: &mut Vec<BenchRow>, row: BenchRow) {
    match rows.iter().position(|r| r.key() == row.key()) {
        Some(i) => rows[i] = row,
        None => rows.push(row),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn model() -> ModelInfo {
        ModelInfo {
            name: "qwen3-8B-Q4_K_M".into(),
            full_path: PathBuf::from("qwen3-8B-Q4_K_M.gguf"),
            size_bytes: 0,
            quant: "Q4_K_M".into(),
        }
    }

    #[test]
    fn parses_llama_bench_markdown() {
        let out = "\
| model              |       size |     params | backend | ngl |  test |              t/s |
| ------------------ | ---------: | ---------: | ------- | --: | ----: | ---------------: |
| qwen3 8B Q4_K      |   4.36 GiB |     7.62 B | Vulkan  | 999 | pp512 |     85.32 ± 1.23 |
| qwen3 8B Q4_K      |   4.36 GiB |     7.62 B | Vulkan  | 999 | tg128 |     15.67 ± 0.45 |
";
        let rows = parse_table(out, "llama-b9553-vulkan", &model(), "c");
        assert_eq!(rows.len(), 1);
        let r = &rows[0];
        assert_eq!(r.backend, "Vulkan");
        assert_eq!(r.size, "4.36 GiB");
        assert_eq!(r.pp, "85.32 ± 1.23");
        assert_eq!(r.tg, "15.67 ± 0.45");
        assert_eq!(r.quant, "Q4_K_M");
        assert_eq!(r.model, "qwen3-8B");
        assert!((r.pp_value() - 85.32).abs() < 1e-6);
    }

    #[test]
    fn upsert_replaces_same_combination() {
        let mut rows = vec![];
        let mut a = parse_table(
            "| model | test | t/s |\n| - | - | - |\n| m | pp512 | 10.0 ± 0 |",
            "v1",
            &model(),
            "c",
        );
        upsert(&mut rows, a.remove(0));
        let mut b = parse_table(
            "| model | test | t/s |\n| - | - | - |\n| m | pp512 | 20.0 ± 0 |",
            "v1",
            &model(),
            "c",
        );
        upsert(&mut rows, b.remove(0));
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].pp, "20.0 ± 0");

        // Mêmes modèle et version, autres réglages : une ligne de plus.
        let mut c = parse_table(
            "| model | test | t/s |\n| - | - | - |\n| m | pp512 | 30.0 ± 0 |",
            "v1",
            &model(),
            "autre",
        );
        upsert(&mut rows, c.remove(0));
        assert_eq!(rows.len(), 2);
    }

    /// Sortie réelle de llama-bench b11235 avec profondeur et flash attention.
    #[test]
    fn parses_depth_tests() {
        let out = "\
| model                          |       size |     params | backend    | ngl | type_k | type_v |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | --: | --------------: | -------------------: |
| llama 256M Q2_K - Medium       |  82.41 MiB |   134.52 M | CUDA       | 999 |   q8_0 |   q8_0 |   1 |    pp128 @ d512 |   12333.34 ± 3711.33 |
| llama 256M Q2_K - Medium       |  82.41 MiB |   134.52 M | CUDA       | 999 |   q8_0 |   q8_0 |   1 |     tg32 @ d512 |      683.74 ± 122.40 |
";
        let rows = parse_table(out, "v", &model(), "c");
        assert_eq!((rows[0].pp.as_str(), rows[0].tg.as_str()), ("12333.34 ± 3711.33", "683.74 ± 122.40"));
    }

    #[test]
    fn settings_args_and_label() {
        let d = BenchSettings::default();
        assert_eq!(
            d.args().join(" "),
            "-ngl 999 -ctk f16 -ctv f16 -p 512 -n 128 -r 3",
            "les valeurs par défaut de llama-bench (fa auto, d 0) ne sont pas passées"
        );
        assert_eq!(d.label(), "ngl 999 · f16 · fa auto · pp512 · tg128 · r3");
        let s = BenchSettings {
            flash_attn: "on".into(),
            depth: "4096".into(),
            cache_type: "q8_0".into(),
            ..d
        };
        assert_eq!(
            s.args().join(" "),
            "-ngl 999 -ctk q8_0 -ctv q8_0 -p 512 -n 128 -r 3 -fa on -d 4096"
        );
        assert!(s.label().ends_with("· d4096"));
        // Chaque valeur proposée existe dans les listes.
        assert!(crate::params::CACHE_TYPES.contains(&"q8_0"));
        for (list, default) in [
            (NGL_CHOICES, "999"),
            (FLASH_ATTN_CHOICES, "auto"),
            (PROMPT_CHOICES, "512"),
            (GEN_CHOICES, "128"),
            (DEPTH_CHOICES, "0"),
            (REPETITION_CHOICES, "3"),
        ] {
            assert!(list.contains(&default));
        }
    }

    /// `benchmark.md` d'avant les réglages (huit colonnes) : relu avec les
    /// réglages par défaut ; le nouveau format se relit à l'identique.
    #[test]
    fn reads_old_and_new_benchmark_files() {
        let old = "\
# Benchmark Results

| Modèle | Quant | Version | Backend | Size | Params | PP (t/s) | TG (t/s) |
|--------|-------|---------|---------|------|--------|----------|----------|
| qwen3-8B | Q4_K_M | llama-b1 | CUDA | 4.36 GiB | 7.62 B | 85.32 ± 1.23 | 15.67 ± 0.45 |
";
        let rows = parse_results(old);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].config, BenchSettings::default().label());

        let mut new = rows[0].clone();
        new.config = "ngl 99 · q8_0 · fa on · pp512 · tg128 · r3 · d4096".into();
        let all = vec![rows[0].clone(), new];
        assert_eq!(parse_results(&render_markdown(&all)), all);
    }
}
