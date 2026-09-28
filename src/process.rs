//! Pilotage du processus enfant `llama-server` (une seule instance à la fois).

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;

#[cfg(windows)]
use std::os::windows::process::CommandExt;
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

pub enum ProcEvent {
    Line(String),
}

/// Réveille l'interface : appelé à chaque ligne reçue et à la fermeture
/// d'un flux, pour qu'elle n'ait pas à se redessiner en boucle.
pub type Wake = Arc<dyn Fn() + Send + Sync>;

pub struct ServerProcess {
    child: Child,
    pub rx: Receiver<ProcEvent>,
    /// Nombre de flux (stdout/stderr) encore ouverts.
    open_streams: Arc<AtomicUsize>,
    pub command_line: String,
}

impl ServerProcess {
    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    /// Arrêt de l'arborescence du processus (llama-server peut lancer des
    /// enfants selon le backend).
    pub fn stop(&mut self) {
        let pid = self.child.id();
        #[cfg(windows)]
        {
            let killed = Command::new("taskkill")
                .args(["/PID", &pid.to_string(), "/T", "/F"])
                .creation_flags(CREATE_NO_WINDOW)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            if killed {
                let _ = self.child.wait();
                return;
            }
        }
        #[cfg(not(windows))]
        let _ = pid;
        let _ = self.child.kill();
        let _ = self.child.wait();
    }

    pub fn try_wait(&mut self) -> Option<Option<i32>> {
        match self.child.try_wait() {
            Ok(Some(status)) => Some(status.code()),
            _ => None,
        }
    }

    pub fn streams_open(&self) -> bool {
        self.open_streams.load(Ordering::SeqCst) > 0
    }
}

/// Provenance du modèle : un fichier local. (Les modèles Hugging Face sont
/// téléchargés dans le répertoire des modèles, pas lancés depuis le dépôt.)
pub enum ModelSource {
    File(PathBuf),
}

impl ModelSource {
    fn to_args(&self) -> anyhow::Result<Vec<String>> {
        match self {
            ModelSource::File(path) => {
                anyhow::ensure!(path.is_file(), "modèle introuvable : {}", path.display());
                Ok(vec!["-m".into(), path.to_string_lossy().into_owned()])
            }
        }
    }
}

/// Lance `llama-server` ; les lignes de stdout et stderr sont poussées
/// dans le canal au fil de l'eau.
pub fn start(
    server_exe: &Path,
    model: &ModelSource,
    host: &str,
    port: &str,
    extra_args: &[String],
    wake: Wake,
) -> anyhow::Result<ServerProcess> {
    anyhow::ensure!(
        server_exe.is_file(),
        "exécutable introuvable : {}",
        server_exe.display()
    );
    anyhow::ensure!(
        crate::discovery::is_executable(server_exe),
        "{} n'est pas un exécutable valide (fichier corrompu ou copie incomplète ?)",
        server_exe.display()
    );

    let mut args: Vec<String> = vec![
        "--host".into(),
        host.to_string(),
        "--port".into(),
        port.to_string(),
    ];
    args.extend(model.to_args()?);
    args.extend(extra_args.iter().cloned());

    let command_line = format!("{} {}", server_exe.display(), args.join(" "));

    let mut cmd = Command::new(server_exe);
    cmd.args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    // Les DLL du backend (Vulkan, CUDA…) sont à côté de l'exécutable.
    if let Some(dir) = server_exe.parent() {
        cmd.current_dir(dir);
    }
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);

    let mut child = cmd.spawn()?;
    let (tx, rx) = channel();
    let open_streams = Arc::new(AtomicUsize::new(0));

    if let Some(out) = child.stdout.take() {
        spawn_reader(out, tx.clone(), open_streams.clone(), wake.clone());
    }
    if let Some(err) = child.stderr.take() {
        spawn_reader(err, tx, open_streams.clone(), wake);
    }

    Ok(ServerProcess {
        child,
        rx,
        open_streams,
        command_line,
    })
}

fn spawn_reader<R: std::io::Read + Send + 'static>(
    stream: R,
    tx: Sender<ProcEvent>,
    open: Arc<AtomicUsize>,
    wake: Wake,
) {
    open.fetch_add(1, Ordering::SeqCst);
    std::thread::spawn(move || {
        let reader = BufReader::new(stream);
        // `split` plutôt que `lines` : la sortie de llama.cpp n'est pas
        // toujours de l'UTF-8 strict.
        for chunk in reader.split(b'\n') {
            match chunk {
                Ok(bytes) => {
                    let line = String::from_utf8_lossy(&bytes)
                        .trim_end_matches('\r')
                        .to_string();
                    if tx.send(ProcEvent::Line(line)).is_err() {
                        break;
                    }
                    wake();
                }
                Err(_) => break,
            }
        }
        open.fetch_sub(1, Ordering::SeqCst);
        wake();
    });
}

/// Vitesses extraites des lignes de log de llama-server.
#[derive(Default, Clone, Copy)]
pub struct Speeds {
    pub prompt: Option<f64>,
    pub generation: Option<f64>,
}

impl Speeds {
    /// Reconnaît les lignes du type
    /// `prompt eval time = … (   85.32 tokens per second)`.
    pub fn observe(&mut self, line: &str) {
        let Some(value) = extract_tps(line) else {
            return;
        };
        let lower = line.to_ascii_lowercase();
        if lower.contains("prompt eval time") {
            self.prompt = Some(value);
        } else if lower.contains("eval time") {
            self.generation = Some(value);
        }
    }
}

fn extract_tps(line: &str) -> Option<f64> {
    // `to_ascii_lowercase` garde la longueur en octets de chaque caractère :
    // l'indice trouvé vaut aussi dans `line`. Avec `to_lowercase`, un
    // caractère non ASCII qui change de longueur (« İ ») décalait l'indice et
    // la découpe paniquait au milieu d'un caractère.
    let lower = line.to_ascii_lowercase();
    let marker = lower.find("tokens per second")?;
    let head = &line[..marker];
    let number: String = head
        .chars()
        .rev()
        .skip_while(|c| c.is_whitespace() || *c == ',')
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    let number: String = number.chars().rev().collect();
    number.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_tokens_per_second() {
        let mut s = Speeds::default();
        s.observe("prompt eval time =  1200.00 ms /  512 tokens (   85.32 tokens per second)");
        s.observe("       eval time =  8000.00 ms /  128 runs   (   15.67 tokens per second)");
        assert_eq!(s.prompt, Some(85.32));
        assert_eq!(s.generation, Some(15.67));
        // Caractères non ASCII avant le motif : pas de panique.
        s.observe("İİİ modèle ß — eval time = 1.0 ms / 1 tokens (   42.5 tokens per second)");
        assert_eq!(s.generation, Some(42.5));
    }
}
