//! Consommation de la carte graphique : courbe des watts relevés par
//! `gpu.rs` (toutes les deux secondes) et énergie consommée, intégrée au fil
//! des relevés.

use chrono::{DateTime, Local, Timelike};
use std::time::{Duration, Instant};

/// Relevés gardés : 24 h à un relevé toutes les deux secondes.
const KEPT: usize = 43_200;
/// Au-delà, un trou entre deux relevés (mise en veille…) n'est pas compté.
const MAX_GAP_SECS: f64 = 10.0;

pub struct PowerLog {
    origin: Instant,
    /// Début de la minute de l'ouverture : les abscisses comptent depuis une
    /// minute ronde, pour que les graduations tombent sur `14:05`, `14:05:30`.
    origin_wall: DateTime<Local>,
    /// Secondes écoulées dans cette minute à l'instant `origin`.
    origin_offset: f64,
    last: Option<Instant>,
    last_cpu: Option<Instant>,
    /// Début du compteur de session : ouverture de l'application, ou dernière
    /// réinitialisation de la courbe.
    session_start: Instant,
    /// (secondes depuis l'origine, watts) de la carte graphique.
    pub samples: Vec<[f64; 2]>,
    /// Même chose pour le processeur (`sysmon`), sur la même échelle de
    /// temps.
    pub cpu_samples: Vec<[f64; 2]>,
    /// Énergie depuis l'ouverture de l'application (ou la réinitialisation
    /// de la courbe), carte graphique et processeur, en Wh.
    pub session_wh: f64,
    /// Énergie depuis le démarrage du serveur, carte et processeur, en Wh.
    pub server_wh: f64,
    /// Part du processeur dans `session_wh` et `server_wh`.
    pub session_cpu_wh: f64,
    pub server_cpu_wh: f64,
    /// Temps de fonctionnement du serveur compté avec son énergie, par les
    /// relevés de la carte et par ceux du processeur (chacun le sien : l'un
    /// peut manquer), en secondes.
    server_secs: f64,
    server_cpu_secs: f64,
}

impl PowerLog {
    pub fn new() -> Self {
        let wall = Local::now();
        let origin_offset = f64::from(wall.second()) + f64::from(wall.nanosecond() % 1_000_000_000) / 1e9;
        Self {
            origin: Instant::now(),
            origin_wall: wall.with_second(0).and_then(|w| w.with_nanosecond(0)).unwrap_or(wall),
            origin_offset,
            last: None,
            last_cpu: None,
            session_start: Instant::now(),
            samples: Vec::new(),
            cpu_samples: Vec::new(),
            session_wh: 0.0,
            server_wh: 0.0,
            session_cpu_wh: 0.0,
            server_cpu_wh: 0.0,
            server_secs: 0.0,
            server_cpu_secs: 0.0,
        }
    }

    /// Ajoute un relevé ; renvoie l'énergie comptée depuis le précédent, en
    /// Wh (le débit relevé, tenu pendant l'intervalle écoulé).
    pub fn record(&mut self, watts: f64, server_running: bool, now: Instant) -> f64 {
        let dt = match self.last {
            Some(prev) => now.saturating_duration_since(prev).as_secs_f64(),
            None => 0.0,
        };
        let dt = if dt <= MAX_GAP_SECS { dt } else { 0.0 };
        let wh = watts * dt / 3600.0;
        self.last = Some(now);
        self.session_wh += wh;
        if server_running {
            self.server_wh += wh;
            self.server_secs += dt;
        }
        let secs = now.saturating_duration_since(self.origin).as_secs_f64() + self.origin_offset;
        self.samples.push([secs, watts]);
        if self.samples.len() > KEPT {
            self.samples.remove(0);
        }
        wh
    }

    /// Relevé de la consommation du processeur, pour sa courbe.
    /// Relevé du processeur : sa courbe, et son énergie ajoutée aux
    /// compteurs (renvoyée en Wh, comme `record`).
    pub fn record_cpu(&mut self, watts: f64, server_running: bool, now: Instant) -> f64 {
        let dt = match self.last_cpu {
            Some(prev) => now.saturating_duration_since(prev).as_secs_f64(),
            None => 0.0,
        };
        let dt = if dt <= MAX_GAP_SECS { dt } else { 0.0 };
        let wh = watts * dt / 3600.0;
        self.last_cpu = Some(now);
        self.session_wh += wh;
        self.session_cpu_wh += wh;
        if server_running {
            self.server_wh += wh;
            self.server_cpu_wh += wh;
            self.server_cpu_secs += dt;
        }
        let secs = now.saturating_duration_since(self.origin).as_secs_f64() + self.origin_offset;
        self.cpu_samples.push([secs, watts]);
        if self.cpu_samples.len() > KEPT {
            self.cpu_samples.remove(0);
        }
        wh
    }

    fn series(&self, series: Series) -> &[[f64; 2]] {
        match series {
            Series::Gpu => &self.samples,
            Series::Cpu => &self.cpu_samples,
        }
    }

    /// Vide les courbes et le compteur de session.
    pub fn reset(&mut self) {
        self.samples.clear();
        self.cpu_samples.clear();
        self.session_wh = 0.0;
        self.session_cpu_wh = 0.0;
        self.session_start = Instant::now();
    }

    /// Temps écoulé depuis le début du compteur de session.
    pub fn session_duration(&self) -> Duration {
        self.session_start.elapsed()
    }

    pub fn reset_server(&mut self) {
        self.server_wh = 0.0;
        self.server_cpu_wh = 0.0;
        self.server_secs = 0.0;
        self.server_cpu_secs = 0.0;
    }

    /// Temps pendant lequel le serveur a tourné depuis son démarrage (ou la
    /// réinitialisation des statistiques), compté avec son énergie.
    pub fn server_duration(&self) -> Duration {
        Duration::from_secs_f64(self.server_secs.max(self.server_cpu_secs))
    }

    pub fn origin_wall(&self) -> DateTime<Local> {
        self.origin_wall
    }

    /// La courbe réduite à `max` points au plus pour l'affichage : 24 h de
    /// relevés font 43 200 points, copiés et tracés à chaque image. Chaque
    /// tranche garde son minimum et son maximum, pour que les pics restent
    /// visibles.
    pub fn decimated(&self, series: Series, max: usize) -> Vec<[f64; 2]> {
        let samples = self.series(series);
        let n = samples.len();
        if n <= max || max < 2 {
            return samples.to_vec();
        }
        let bucket = n.div_ceil(max / 2);
        let mut out = Vec::with_capacity(max);
        for chunk in samples.chunks(bucket) {
            let lo = chunk.iter().min_by(|a, b| a[1].total_cmp(&b[1])).unwrap();
            let hi = chunk.iter().max_by(|a, b| a[1].total_cmp(&b[1])).unwrap();
            if lo[0] <= hi[0] {
                out.extend([*lo, *hi]);
            } else {
                out.extend([*hi, *lo]);
            }
        }
        out
    }

    /// Minimum, moyenne et maximum des watts relevés.
    pub fn range(&self, series: Series) -> Option<(f64, f64, f64)> {
        let samples = self.series(series);
        if samples.is_empty() {
            return None;
        }
        let w = samples.iter().map(|s| s[1]);
        let min = w.clone().fold(f64::INFINITY, f64::min);
        let max = w.clone().fold(f64::NEG_INFINITY, f64::max);
        let mean = w.sum::<f64>() / samples.len() as f64;
        Some((min, mean, max))
    }
}

/// Courbe de consommation : carte graphique ou processeur.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Series {
    Gpu,
    Cpu,
}

/// Heure murale d'un point de la courbe : `14:05`, ou `14:05:30` avec les
/// secondes.
pub fn clock(origin: DateTime<Local>, secs: f64, seconds: bool) -> String {
    (origin + chrono::Duration::milliseconds((secs * 1000.0) as i64))
        .format(if seconds { "%H:%M:%S" } else { "%H:%M" })
        .to_string()
}

/// Durée lisible : `45 s`, `12 min`, `2 h 05 min`.
pub fn duration(d: Duration) -> String {
    let secs = d.as_secs();
    match secs {
        s if s < 60 => format!("{s} s"),
        s if s < 3600 => format!("{} min", s / 60),
        s => format!("{} h {:02} min", s / 3600, s / 60 % 60),
    }
}

/// `12.3 Wh`, ou `1.234 kWh` à partir de 1 000 Wh.
pub fn energy(wh: f64, fr: bool) -> String {
    let text = if wh >= 1000.0 {
        format!("{:.3} kWh", wh / 1000.0)
    } else {
        format!("{wh:.1} Wh")
    };
    if fr { text.replace('.', ",") } else { text }
}

/// Coût d'une énergie au prix donné par kWh (`0.37 €`).
pub fn cost(wh: f64, price_per_kwh: f64, currency: &str, fr: bool) -> String {
    let value = wh / 1000.0 * price_per_kwh;
    let digits = if value < 1.0 { 3 } else { 2 };
    let text = format!("{value:.digits$}");
    let text = if fr { text.replace('.', ",") } else { text };
    if currency == "€" || currency == "CHF" {
        format!("{text} {currency}")
    } else {
        format!("{currency}{text}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn integrates_watts_over_time() {
        let mut log = PowerLog::new();
        let t0 = Instant::now();
        // 360 W pendant 2 s = 0.2 Wh ; le premier relevé ne compte rien.
        assert_eq!(log.record(360.0, true, t0), 0.0);
        let wh = log.record(360.0, true, t0 + Duration::from_secs(2));
        assert!((wh - 0.2).abs() < 1e-9);
        // Serveur arrêté : compté dans la session seulement.
        log.record(180.0, false, t0 + Duration::from_secs(4));
        assert!((log.session_wh - 0.3).abs() < 1e-9);
        assert!((log.server_wh - 0.2).abs() < 1e-9);
        assert_eq!(log.server_duration(), Duration::from_secs(2));
        // Un trou d'une minute (veille) n'est pas compté.
        assert_eq!(log.record(300.0, true, t0 + Duration::from_secs(64)), 0.0);
        assert_eq!(log.range(Series::Gpu), Some((180.0, 300.0, 360.0)));
        // Processeur : sa courbe, et son énergie ajoutée aux compteurs
        // (720 W pendant 2 s = 0,4 Wh, serveur en marche).
        assert_eq!(log.record_cpu(40.0, true, t0 + Duration::from_secs(2)), 0.0);
        let wh = log.record_cpu(720.0, true, t0 + Duration::from_secs(4));
        assert!((wh - 0.4).abs() < 1e-9);
        assert_eq!(log.range(Series::Cpu), Some((40.0, 380.0, 720.0)));
        assert!((log.session_wh - 0.7).abs() < 1e-9);
        assert!((log.session_cpu_wh - 0.4).abs() < 1e-9);
        assert!((log.server_wh - 0.6).abs() < 1e-9);
        assert!((log.server_cpu_wh - 0.4).abs() < 1e-9);
        log.reset();
        assert!(log.samples.is_empty() && log.cpu_samples.is_empty() && log.session_wh == 0.0);
        assert_eq!(log.session_cpu_wh, 0.0);
    }

    #[test]
    fn decimation_keeps_peaks() {
        let mut log = PowerLog::new();
        log.samples = (0..10_000).map(|i| [i as f64, if i == 5_000 { 400.0 } else { 20.0 }]).collect();
        let d = log.decimated(Series::Gpu, 1000);
        assert!(d.len() <= 1000);
        assert!(d.iter().any(|p| p[1] == 400.0), "le pic reste visible");
        assert!(d.windows(2).all(|w| w[0][0] <= w[1][0]), "abscisses croissantes");
    }

    #[test]
    fn formats_durations() {
        assert_eq!(duration(Duration::from_secs(45)), "45 s");
        assert_eq!(duration(Duration::from_secs(12 * 60 + 30)), "12 min");
        assert_eq!(duration(Duration::from_secs(2 * 3600 + 5 * 60)), "2 h 05 min");
        assert_eq!(duration(Duration::from_secs(26 * 3600)), "26 h 00 min");
    }

    #[test]
    fn formats_energy_and_cost() {
        assert_eq!(energy(12.34, true), "12,3 Wh");
        assert_eq!(energy(1234.0, false), "1.234 kWh");
        // 2 kWh à 0,25 €/kWh.
        assert_eq!(cost(2000.0, 0.25, "€", true), "0,500 €");
        assert_eq!(cost(10_000.0, 0.25, "$", false), "$2.50");
    }
}
