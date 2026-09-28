//! Consommation de la carte graphique : courbe des watts relevés par
//! `gpu.rs` (toutes les deux secondes) et énergie consommée, intégrée au fil
//! des relevés.

use chrono::{DateTime, Local};
use std::time::Instant;

/// Relevés gardés : 24 h à un relevé toutes les deux secondes.
const KEPT: usize = 43_200;
/// Au-delà, un trou entre deux relevés (mise en veille…) n'est pas compté.
const MAX_GAP_SECS: f64 = 10.0;

pub struct PowerLog {
    origin: Instant,
    origin_wall: DateTime<Local>,
    last: Option<Instant>,
    /// (secondes depuis l'origine, watts).
    pub samples: Vec<[f64; 2]>,
    /// Énergie depuis l'ouverture de l'application (ou la réinitialisation
    /// de la courbe), en Wh.
    pub session_wh: f64,
    /// Énergie depuis le démarrage du serveur, en Wh.
    pub server_wh: f64,
}

impl PowerLog {
    pub fn new() -> Self {
        Self {
            origin: Instant::now(),
            origin_wall: Local::now(),
            last: None,
            samples: Vec::new(),
            session_wh: 0.0,
            server_wh: 0.0,
        }
    }

    /// Ajoute un relevé ; renvoie l'énergie comptée depuis le précédent, en
    /// Wh (le débit relevé, tenu pendant l'intervalle écoulé).
    pub fn record(&mut self, watts: f64, server_running: bool, now: Instant) -> f64 {
        let wh = match self.last {
            Some(prev) => {
                let dt = now.saturating_duration_since(prev).as_secs_f64();
                if dt <= MAX_GAP_SECS { watts * dt / 3600.0 } else { 0.0 }
            }
            None => 0.0,
        };
        self.last = Some(now);
        self.session_wh += wh;
        if server_running {
            self.server_wh += wh;
        }
        let secs = now.saturating_duration_since(self.origin).as_secs_f64();
        self.samples.push([secs, watts]);
        if self.samples.len() > KEPT {
            self.samples.remove(0);
        }
        wh
    }

    /// Vide la courbe et le compteur de session.
    pub fn reset(&mut self) {
        self.samples.clear();
        self.session_wh = 0.0;
    }

    pub fn reset_server(&mut self) {
        self.server_wh = 0.0;
    }

    pub fn origin_wall(&self) -> DateTime<Local> {
        self.origin_wall
    }

    /// La courbe réduite à `max` points au plus pour l'affichage : 24 h de
    /// relevés font 43 200 points, copiés et tracés à chaque image. Chaque
    /// tranche garde son minimum et son maximum, pour que les pics restent
    /// visibles.
    pub fn decimated(&self, max: usize) -> Vec<[f64; 2]> {
        let n = self.samples.len();
        if n <= max || max < 2 {
            return self.samples.clone();
        }
        let bucket = n.div_ceil(max / 2);
        let mut out = Vec::with_capacity(max);
        for chunk in self.samples.chunks(bucket) {
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
    pub fn range(&self) -> Option<(f64, f64, f64)> {
        if self.samples.is_empty() {
            return None;
        }
        let w = self.samples.iter().map(|s| s[1]);
        let min = w.clone().fold(f64::INFINITY, f64::min);
        let max = w.clone().fold(f64::NEG_INFINITY, f64::max);
        let mean = w.sum::<f64>() / self.samples.len() as f64;
        Some((min, mean, max))
    }
}

/// Heure murale d'un point de la courbe (`14:05`).
pub fn clock(origin: DateTime<Local>, secs: f64) -> String {
    (origin + chrono::Duration::milliseconds((secs * 1000.0) as i64))
        .format("%H:%M")
        .to_string()
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
        // Un trou d'une minute (veille) n'est pas compté.
        assert_eq!(log.record(300.0, true, t0 + Duration::from_secs(64)), 0.0);
        assert_eq!(log.range(), Some((180.0, 300.0, 360.0)));
        log.reset();
        assert!(log.samples.is_empty() && log.session_wh == 0.0);
    }

    #[test]
    fn decimation_keeps_peaks() {
        let mut log = PowerLog::new();
        log.samples = (0..10_000).map(|i| [i as f64, if i == 5_000 { 400.0 } else { 20.0 }]).collect();
        let d = log.decimated(1000);
        assert!(d.len() <= 1000);
        assert!(d.iter().any(|p| p[1] == 400.0), "le pic reste visible");
        assert!(d.windows(2).all(|w| w[0][0] <= w[1][0]), "abscisses croissantes");
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
