//! Mémoire vive, charge du processeur et consommation du processeur (W).
//!
//! Relevés toutes les deux secondes en tâche de fond, comme `gpu.rs`.
//! Windows : `GlobalMemoryStatusEx` et les compteurs de performance (PDH) —
//! « % Processor Utility », la charge affichée par le Gestionnaire des
//! tâches, et « Energy Meter », les compteurs RAPL que Windows expose sans
//! pilote ni droits administrateur. Linux : `/proc/meminfo`, `/proc/stat` et
//! `/sys/class/powercap` (RAPL, souvent réservé à root : pas de watts alors).
//! Une mesure indisponible vaut `None` et n'est pas affichée.

use std::sync::mpsc::{channel, Receiver};
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SysSample {
    /// Charge du processeur, en %.
    pub cpu_percent: Option<f32>,
    /// Mémoire vive utilisée et totale, en octets.
    pub ram_used: u64,
    pub ram_total: u64,
    /// Consommation du processeur (boîtier entier), en watts.
    pub cpu_power_w: Option<f32>,
}

impl SysSample {
    pub fn ram_fraction(&self) -> f32 {
        if self.ram_total == 0 {
            0.0
        } else {
            (self.ram_used as f32 / self.ram_total as f32).clamp(0.0, 1.0)
        }
    }
}

/// Un relevé et son heure (voir `gpu::Sample`).
pub type Sample = (Instant, SysSample);

/// Lance le suivi ; le fil s'arrête quand le récepteur est abandonné.
pub fn monitor() -> Receiver<Sample> {
    let (tx, rx) = channel();
    std::thread::spawn(move || {
        let mut probe = Probe::new();
        loop {
            let sample = probe.sample();
            if sample.ram_total > 0 && tx.send((Instant::now(), sample)).is_err() {
                break;
            }
            std::thread::sleep(Duration::from_secs(2));
        }
    });
    rx
}

// ------------------------------------------------------------------ Windows

#[cfg(windows)]
mod win {
    use super::SysSample;

    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }

    /// `PDH_FMT_COUNTERVALUE` : statut puis union (ici le `double`).
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct FmtValue {
        status: u32,
        value: f64,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct FmtItem {
        name: *const u16,
        value: FmtValue,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(status: *mut MemoryStatusEx) -> i32;
    }

    #[link(name = "pdh")]
    unsafe extern "system" {
        fn PdhOpenQueryW(source: *const u16, user: usize, query: *mut isize) -> u32;
        fn PdhAddEnglishCounterW(query: isize, path: *const u16, user: usize, counter: *mut isize) -> u32;
        fn PdhCollectQueryData(query: isize) -> u32;
        fn PdhGetFormattedCounterArrayW(
            counter: isize,
            format: u32,
            size: *mut u32,
            count: *mut u32,
            items: *mut FmtItem,
        ) -> u32;
        fn PdhCloseQuery(query: isize) -> u32;
    }

    const PDH_FMT_DOUBLE: u32 = 0x0000_0200;
    const PDH_MORE_DATA: u32 = 0x8000_07D2;
    const PDH_CSTATUS_VALID_DATA: u32 = 0;
    const PDH_CSTATUS_NEW_DATA: u32 = 1;

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    pub struct Probe {
        query: isize,
        cpu: Option<isize>,
        energy: Option<isize>,
    }

    impl Probe {
        pub fn new() -> Self {
            let mut query = 0isize;
            if unsafe { PdhOpenQueryW(std::ptr::null(), 0, &mut query) } != 0 {
                return Probe { query: 0, cpu: None, energy: None };
            }
            // Noms anglais : les noms affichés sont traduits (« Compteur
            // Énergie » sous Windows en français).
            let add = |path: &str| {
                let mut counter = 0isize;
                let path = wide(path);
                (unsafe { PdhAddEnglishCounterW(query, path.as_ptr(), 0, &mut counter) } == 0)
                    .then_some(counter)
            };
            let cpu = add(r"\Processor Information(_Total)\% Processor Utility")
                .or_else(|| add(r"\Processor(_Total)\% Processor Time"));
            let energy = add(r"\Energy Meter(*)\Power");
            // Premier relevé : les compteurs de taux en exigent deux.
            unsafe { PdhCollectQueryData(query) };
            Probe { query, cpu, energy }
        }

        /// Valeurs de toutes les instances d'un compteur : (nom, valeur).
        fn values(&self, counter: isize) -> Vec<(String, f64)> {
            let (mut size, mut count) = (0u32, 0u32);
            let status = unsafe {
                PdhGetFormattedCounterArrayW(counter, PDH_FMT_DOUBLE, &mut size, &mut count, std::ptr::null_mut())
            };
            if status != PDH_MORE_DATA || size == 0 {
                return Vec::new();
            }
            // Tampon aligné sur 8 octets : les éléments puis leurs noms.
            let mut buf = vec![0u64; (size as usize).div_ceil(8)];
            let items = buf.as_mut_ptr() as *mut FmtItem;
            if unsafe { PdhGetFormattedCounterArrayW(counter, PDH_FMT_DOUBLE, &mut size, &mut count, items) } != 0 {
                return Vec::new();
            }
            (0..count as usize)
                .filter_map(|i| {
                    let item = unsafe { *items.add(i) };
                    if !matches!(item.value.status, PDH_CSTATUS_VALID_DATA | PDH_CSTATUS_NEW_DATA) {
                        return None;
                    }
                    let mut len = 0;
                    while unsafe { *item.name.add(len) } != 0 {
                        len += 1;
                    }
                    let name = String::from_utf16_lossy(unsafe { std::slice::from_raw_parts(item.name, len) });
                    Some((name, item.value.value))
                })
                .collect()
        }

        pub fn sample(&mut self) -> SysSample {
            let mut mem = MemoryStatusEx {
                length: std::mem::size_of::<MemoryStatusEx>() as u32,
                memory_load: 0,
                total_phys: 0,
                avail_phys: 0,
                total_page_file: 0,
                avail_page_file: 0,
                total_virtual: 0,
                avail_virtual: 0,
                avail_extended_virtual: 0,
            };
            let mut out = SysSample::default();
            if unsafe { GlobalMemoryStatusEx(&mut mem) } != 0 {
                out.ram_total = mem.total_phys;
                out.ram_used = mem.total_phys.saturating_sub(mem.avail_phys);
            }
            if self.query == 0 || unsafe { PdhCollectQueryData(self.query) } != 0 {
                return out;
            }
            out.cpu_percent = self
                .cpu
                .and_then(|c| self.values(c).first().map(|(_, v)| v.clamp(0.0, 100.0) as f32));
            // Instances RAPL : un boîtier (« …_PKG ») par processeur, et ses
            // cœurs, déjà compris dans le boîtier. En milliwatts.
            out.cpu_power_w = self.energy.and_then(|c| super::package_watts(&self.values(c)));
            out
        }
    }

    impl Drop for Probe {
        fn drop(&mut self) {
            if self.query != 0 {
                unsafe { PdhCloseQuery(self.query) };
            }
        }
    }
}

#[cfg(windows)]
use win::Probe;

/// Somme des boîtiers RAPL (`RAPL_Package0_PKG`…), en milliwatts → watts.
#[cfg_attr(not(windows), allow(dead_code))]
fn package_watts(values: &[(String, f64)]) -> Option<f32> {
    let packages: Vec<f64> = values
        .iter()
        .filter(|(name, _)| name.to_ascii_lowercase().ends_with("_pkg"))
        .map(|(_, mw)| *mw)
        .collect();
    // 0 : premier relevé, pris juste après l'ouverture des compteurs, avant
    // toute mesure ; un processeur ne consomme jamais 0 W.
    let watts = packages.iter().sum::<f64>() / 1000.0;
    (!packages.is_empty() && watts > 0.0).then_some(watts as f32)
}

// -------------------------------------------------------------------- Linux

#[cfg(not(windows))]
struct Probe {
    /// (inactif, total) du relevé précédent de `/proc/stat`.
    cpu_prev: Option<(u64, u64)>,
    /// (énergie en µJ, instant) du relevé précédent, par boîtier RAPL.
    energy_prev: Vec<(u64, Instant)>,
}

#[cfg(not(windows))]
impl Probe {
    fn new() -> Self {
        Probe { cpu_prev: None, energy_prev: Vec::new() }
    }

    fn sample(&mut self) -> SysSample {
        let mut out = SysSample::default();
        if let Ok(text) = std::fs::read_to_string("/proc/meminfo") {
            if let Some((total, used)) = parse_meminfo(&text) {
                out.ram_total = total;
                out.ram_used = used;
            }
        }
        if let Some(now) = std::fs::read_to_string("/proc/stat").ok().as_deref().and_then(parse_proc_stat) {
            if let Some(prev) = self.cpu_prev {
                out.cpu_percent = cpu_percent(prev, now);
            }
            self.cpu_prev = Some(now);
        }
        out.cpu_power_w = self.rapl_watts();
        out
    }

    /// Boîtiers RAPL (`intel-rapl:0`, `intel-rapl:1`… ; AMD Zen compris) :
    /// différence d'énergie entre deux relevés. Illisible sans root sur la
    /// plupart des distributions depuis 2020.
    fn rapl_watts(&mut self) -> Option<f32> {
        let now = Instant::now();
        let mut current = Vec::new();
        for i in 0.. {
            let dir = std::path::PathBuf::from(format!("/sys/class/powercap/intel-rapl:{i}"));
            if !dir.is_dir() {
                break;
            }
            let read = |f: &str| std::fs::read_to_string(dir.join(f)).ok()?.trim().parse::<u64>().ok();
            current.push((read("energy_uj")?, read("max_energy_range_uj").unwrap_or(u64::MAX)));
        }
        if current.is_empty() {
            return None;
        }
        let watts = (self.energy_prev.len() == current.len()).then(|| {
            current
                .iter()
                .zip(&self.energy_prev)
                .map(|(&(uj, range), &(prev, at))| {
                    // Le compteur repart de zéro après `max_energy_range_uj`.
                    let delta = if uj >= prev { uj - prev } else { range - prev + uj };
                    delta as f64 / 1e6 / now.duration_since(at).as_secs_f64().max(1e-3)
                })
                .sum::<f64>() as f32
        });
        self.energy_prev = current.iter().map(|&(uj, _)| (uj, now)).collect();
        watts
    }
}

/// `MemTotal` et `MemTotal - MemAvailable`, en octets.
#[cfg_attr(windows, allow(dead_code))]
fn parse_meminfo(text: &str) -> Option<(u64, u64)> {
    let field = |name: &str| {
        text.lines()
            .find_map(|l| l.strip_prefix(name))
            .and_then(|rest| rest.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
            .map(|kb| kb * 1024)
    };
    let total = field("MemTotal:")?;
    let available = field("MemAvailable:")?;
    Some((total, total.saturating_sub(available)))
}

/// Première ligne de `/proc/stat` : (temps inactif, temps total).
#[cfg_attr(windows, allow(dead_code))]
fn parse_proc_stat(text: &str) -> Option<(u64, u64)> {
    let fields: Vec<u64> = text
        .lines()
        .next()?
        .strip_prefix("cpu ")?
        .split_whitespace()
        .filter_map(|f| f.parse().ok())
        .collect();
    // user nice system idle iowait irq softirq steal
    let total: u64 = fields.iter().take(8).sum();
    let idle = fields.get(3)? + fields.get(4).unwrap_or(&0);
    Some((idle, total))
}

#[cfg_attr(windows, allow(dead_code))]
fn cpu_percent((idle0, total0): (u64, u64), (idle1, total1): (u64, u64)) -> Option<f32> {
    let total = total1.checked_sub(total0).filter(|t| *t > 0)?;
    let idle = idle1.saturating_sub(idle0).min(total);
    Some(100.0 * (total - idle) as f32 / total as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Relevé réel : `cargo test sysmon_live -- --ignored --nocapture`.
    #[test]
    #[ignore = "lit la machine"]
    fn sysmon_live() {
        let mut probe = Probe::new();
        for _ in 0..3 {
            std::thread::sleep(Duration::from_secs(1));
            println!("{:?}", probe.sample());
        }
    }

    #[test]
    fn sums_rapl_packages_only() {
        // Relevé réel (Ryzen 7 9800X3D, Windows 11) : boîtier et cœurs, en mW.
        let values = vec![
            ("rapl_package0_core0_core".to_string(), 960.0),
            ("rapl_package0_core1_core".to_string(), 692.0),
            ("rapl_package0_pkg".to_string(), 25630.7),
            ("_total".to_string(), 0.0),
        ];
        assert_eq!(package_watts(&values), Some(25.6307));
        assert_eq!(package_watts(&values[..2]), None);
        assert_eq!(package_watts(&[("rapl_package0_pkg".to_string(), 0.0)]), None);
    }

    #[test]
    fn reads_linux_memory_and_load() {
        let meminfo = "MemTotal:       32768000 kB\nMemFree:         1000000 kB\nMemAvailable:   24576000 kB\n";
        assert_eq!(parse_meminfo(meminfo), Some((32768000 * 1024, 8192000 * 1024)));
        let a = parse_proc_stat("cpu  100 0 100 700 100 0 0 0 0 0\ncpu0 1 2 3\n").unwrap();
        assert_eq!(a, (800, 1000));
        let b = parse_proc_stat("cpu  200 0 200 1100 100 0 0 0 0 0\n").unwrap();
        // 600 de plus, dont 400 inactifs : 33 %.
        assert!((cpu_percent(a, b).unwrap() - 100.0 / 3.0).abs() < 0.01);
        assert_eq!(cpu_percent(b, b), None);
    }
}
