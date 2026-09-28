//! Occupation de la mémoire vidéo et consommation électrique, lues par
//! `nvidia-smi`.
//!
//! Interrogé toutes les deux secondes en tâche de fond (≈ 40 ms par appel).
//! Sans pilote NVIDIA, `nvidia-smi` est absent : le suivi s'arrête et rien
//! n'est affiché.

use std::process::{Command, Stdio};
use std::sync::mpsc::{channel, Receiver};
use std::time::{Duration, Instant};

#[derive(Clone, Debug, PartialEq)]
pub struct GpuMemory {
    pub name: String,
    pub used_mib: u64,
    pub total_mib: u64,
    /// Charge de calcul, en %.
    pub load: u64,
    /// Consommation instantanée et limite de puissance, en watts. `None`
    /// quand la carte ne les expose pas (`[N/A]`).
    pub power_w: Option<f32>,
    pub power_limit_w: Option<f32>,
}

impl GpuMemory {
    pub fn fraction(&self) -> f32 {
        if self.total_mib == 0 {
            0.0
        } else {
            (self.used_mib as f32 / self.total_mib as f32).clamp(0.0, 1.0)
        }
    }
}

/// `0, NVIDIA GeForce RTX 5080, 852, 16303, 0, 17.01, 360.00` → une entrée
/// par GPU.
pub fn parse(text: &str) -> Vec<GpuMemory> {
    text.lines()
        .filter_map(|line| {
            let cells: Vec<&str> = line.split(',').map(str::trim).collect();
            let [_, name, used, total, load, power, limit] = cells.as_slice() else {
                return None;
            };
            Some(GpuMemory {
                name: name.to_string(),
                used_mib: used.parse().ok()?,
                total_mib: total.parse().ok()?,
                load: load.parse().unwrap_or(0),
                power_w: power.parse().ok(),
                power_limit_w: limit.parse().ok(),
            })
        })
        .collect()
}

fn query() -> Option<Vec<GpuMemory>> {
    let mut cmd = Command::new("nvidia-smi");
    cmd.args([
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw,power.limit",
        "--format=csv,noheader,nounits",
    ])
    .stdin(Stdio::null())
    .stderr(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    let out = cmd.output().ok()?;
    out.status
        .success()
        .then(|| parse(&String::from_utf8_lossy(&out.stdout)))
        .filter(|gpus| !gpus.is_empty())
}

/// Un relevé et son heure : l'interface peut le lire bien plus tard
/// (fenêtre réduite, elle ne se redessine plus), l'énergie doit être
/// intégrée sur les vrais intervalles.
pub type Sample = (Instant, Vec<GpuMemory>);

/// Lance le suivi ; le fil s'arrête quand le récepteur est abandonné, ou
/// d'emblée si `nvidia-smi` ne répond pas (pas de carte NVIDIA). Un échec
/// passager ensuite (pilote occupé, mise en veille) n'arrête plus le suivi.
pub fn monitor() -> Receiver<Sample> {
    let (tx, rx) = channel();
    std::thread::spawn(move || {
        let Some(first) = query() else {
            return;
        };
        if tx.send((Instant::now(), first)).is_err() {
            return;
        }
        loop {
            std::thread::sleep(Duration::from_secs(2));
            if let Some(gpus) = query() {
                if tx.send((Instant::now(), gpus)).is_err() {
                    break;
                }
            }
        }
    });
    rx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nvidia_smi_csv() {
        let gpus = parse(
            "0, NVIDIA GeForce RTX 5080, 852, 16303, 7, 17.01, 360.00\n\
             1, NVIDIA RTX A4000, 100, 16376, 0, [N/A], [N/A]\n",
        );
        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[0].name, "NVIDIA GeForce RTX 5080");
        assert_eq!((gpus[0].used_mib, gpus[0].total_mib, gpus[0].load), (852, 16303, 7));
        assert!((gpus[0].fraction() - 852.0 / 16303.0).abs() < 1e-6);
        assert_eq!((gpus[0].power_w, gpus[0].power_limit_w), (Some(17.01), Some(360.0)));
        // Puissance non exposée : la carte reste suivie, sans watts.
        assert_eq!((gpus[1].power_w, gpus[1].power_limit_w), (None, None));
        // Valeur non numérique (GPU en erreur) : ligne ignorée.
        assert!(parse("0, GPU, [N/A], 16303, 0, 10, 100").is_empty());
    }
}
