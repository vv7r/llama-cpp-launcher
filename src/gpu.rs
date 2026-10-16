//! Occupation de la mémoire vidéo, charge et consommation des cartes
//! graphiques.
//!
//! Interrogées toutes les deux secondes en tâche de fond. Trois sources :
//! - `nvidia-smi` (≈ 40 ms), cartes NVIDIA, Windows et Linux ;
//! - Windows, toutes marques : nom et VRAM totale par DXGI, VRAM utilisée et
//!   charge par les compteurs de performance (« GPU Adapter Memory », « GPU
//!   Engine », ceux du Gestionnaire des tâches). Pas de watts : Windows n'en
//!   donne pas pour les cartes AMD ou Intel ;
//! - Linux : pilote `amdgpu` (`/sys/class/drm/cardN/device` : VRAM, charge,
//!   watts par `hwmon`) et cartes Intel dédiées (watts par `hwmon`).
//!
//! Sans aucune carte reconnue au démarrage, le suivi s'arrête et rien n'est
//! affiché. Un échec passager ensuite n'arrête plus le suivi.

use std::process::{Command, Stdio};
use std::sync::mpsc::{channel, Receiver};
use std::time::{Duration, Instant};

#[derive(Clone, Debug, PartialEq)]
pub struct GpuMemory {
    pub name: String,
    /// Mémoire vidéo utilisée et totale ; `total_mib` à 0 : inconnue (pas de
    /// jauge, les watts seuls).
    pub used_mib: u64,
    pub total_mib: u64,
    /// Charge de calcul, en %.
    pub load: Option<u64>,
    /// Consommation instantanée et limite de puissance, en watts. `None`
    /// quand la carte ne les expose pas (`[N/A]`, cartes AMD ou Intel sous
    /// Windows).
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

/// Carte de moins d'1 Gio de mémoire dédiée : puce intégrée (la Radeon d'un
/// Ryzen, une Intel UHD), qui ne sert pas aux modèles. Ignorée.
const MIN_DEDICATED: u64 = 1 << 30;

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
                load: load.parse().ok(),
                power_w: power.parse().ok(),
                power_limit_w: limit.parse().ok(),
            })
        })
        .collect()
}

fn nvidia_smi() -> Option<Vec<GpuMemory>> {
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

/// Toutes les sources de la machine.
struct Sources {
    nvidia: bool,
    #[cfg(windows)]
    windows: Option<win::Adapters>,
    #[cfg(not(windows))]
    cards: Vec<linux::Card>,
}

impl Sources {
    fn detect() -> Sources {
        let nvidia = nvidia_smi().is_some();
        Sources {
            nvidia,
            // Les cartes NVIDIA vont par nvidia-smi, plus complet (watts) ;
            // sans lui, elles passent par la source générique.
            #[cfg(windows)]
            windows: win::Adapters::open(nvidia),
            #[cfg(not(windows))]
            cards: linux::cards(),
        }
    }

    fn is_empty(&self) -> bool {
        #[cfg(windows)]
        let others = self.windows.is_none();
        #[cfg(not(windows))]
        let others = self.cards.is_empty();
        !self.nvidia && others
    }

    fn query(&mut self) -> Vec<GpuMemory> {
        let mut out = if self.nvidia { nvidia_smi().unwrap_or_default() } else { Vec::new() };
        #[cfg(windows)]
        if let Some(adapters) = &self.windows {
            out.extend(adapters.query());
        }
        #[cfg(not(windows))]
        out.extend(self.cards.iter_mut().filter_map(linux::Card::read));
        out
    }
}

/// Un relevé et son heure : l'interface peut le lire bien plus tard
/// (fenêtre réduite, elle ne se redessine plus), l'énergie doit être
/// intégrée sur les vrais intervalles.
pub type Sample = (Instant, Vec<GpuMemory>);

/// Lance le suivi ; le fil s'arrête quand le récepteur est abandonné, ou
/// d'emblée si aucune carte n'est reconnue.
pub fn monitor() -> Receiver<Sample> {
    let (tx, rx) = channel();
    std::thread::spawn(move || {
        let mut sources = Sources::detect();
        if sources.is_empty() {
            return;
        }
        loop {
            let gpus = sources.query();
            if !gpus.is_empty() && tx.send((Instant::now(), gpus)).is_err() {
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
    use super::{GpuMemory, MIN_DEDICATED};
    use crate::pdh::{Counter, Query};
    use std::ffi::c_void;

    #[repr(C)]
    struct Guid {
        d1: u32,
        d2: u16,
        d3: u16,
        d4: [u8; 8],
    }

    /// `DXGI_ADAPTER_DESC1`.
    #[repr(C)]
    struct AdapterDesc1 {
        description: [u16; 128],
        vendor_id: u32,
        device_id: u32,
        sub_sys_id: u32,
        revision: u32,
        dedicated_video_memory: usize,
        dedicated_system_memory: usize,
        shared_system_memory: usize,
        luid_low: u32,
        luid_high: i32,
        flags: u32,
    }

    #[link(name = "dxgi")]
    unsafe extern "system" {
        fn CreateDXGIFactory1(riid: *const Guid, factory: *mut *mut c_void) -> i32;
    }

    /// IID_IDXGIFactory1.
    const IID_FACTORY1: Guid = Guid {
        d1: 0x770a_ae78,
        d2: 0xf26f,
        d3: 0x4dba,
        d4: [0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87],
    };
    const DXGI_ADAPTER_FLAG_SOFTWARE: u32 = 2;
    const VENDOR_NVIDIA: u32 = 0x10de;

    /// Méthode n° `index` de la table virtuelle d'un objet COM.
    unsafe fn method(object: *mut c_void, index: usize) -> *const c_void {
        let vtable = unsafe { *(object as *const *const *const c_void) };
        unsafe { *vtable.add(index) }
    }

    unsafe fn release(object: *mut c_void) {
        // IUnknown::Release : 3e méthode.
        let f: unsafe extern "system" fn(*mut c_void) -> u32 = unsafe { std::mem::transmute(method(object, 2)) };
        unsafe { f(object) };
    }

    struct Adapter {
        name: String,
        total: u64,
        /// `luid_0x00000000_0x0000d1a3` : début des noms d'instance PDH.
        luid: String,
    }

    /// Cartes dédiées vues par DXGI (sauf NVIDIA si nvidia-smi s'en charge).
    fn adapters(skip_nvidia: bool) -> Vec<Adapter> {
        let mut factory: *mut c_void = std::ptr::null_mut();
        if unsafe { CreateDXGIFactory1(&IID_FACTORY1, &mut factory) } < 0 || factory.is_null() {
            return Vec::new();
        }
        // IDXGIFactory1::EnumAdapters1 : IUnknown (3), IDXGIObject (4),
        // IDXGIFactory (5), puis EnumAdapters1.
        let enum_adapters: unsafe extern "system" fn(*mut c_void, u32, *mut *mut c_void) -> i32 =
            unsafe { std::mem::transmute(method(factory, 12)) };
        let mut out = Vec::new();
        for index in 0..16 {
            let mut adapter: *mut c_void = std::ptr::null_mut();
            if unsafe { enum_adapters(factory, index, &mut adapter) } < 0 || adapter.is_null() {
                break;
            }
            // IDXGIAdapter1::GetDesc1 : IUnknown (3), IDXGIObject (4),
            // IDXGIAdapter (3), puis GetDesc1.
            let get_desc: unsafe extern "system" fn(*mut c_void, *mut AdapterDesc1) -> i32 =
                unsafe { std::mem::transmute(method(adapter, 10)) };
            let mut desc: AdapterDesc1 = unsafe { std::mem::zeroed() };
            let ok = unsafe { get_desc(adapter, &mut desc) } >= 0;
            unsafe { release(adapter) };
            if !ok
                || desc.flags & DXGI_ADAPTER_FLAG_SOFTWARE != 0
                || (desc.dedicated_video_memory as u64) < MIN_DEDICATED
                || (skip_nvidia && desc.vendor_id == VENDOR_NVIDIA)
            {
                continue;
            }
            let len = desc.description.iter().position(|&c| c == 0).unwrap_or(128);
            out.push(Adapter {
                name: String::from_utf16_lossy(&desc.description[..len]).trim().to_string(),
                total: desc.dedicated_video_memory as u64,
                luid: super::luid_prefix(desc.luid_high as u32, desc.luid_low),
            });
        }
        unsafe { release(factory) };
        out
    }

    pub struct Adapters {
        list: Vec<Adapter>,
        query: Query,
        memory: Counter,
        engines: Option<Counter>,
    }

    impl Adapters {
        pub fn open(skip_nvidia: bool) -> Option<Adapters> {
            let list = adapters(skip_nvidia);
            if list.is_empty() {
                return None;
            }
            let query = Query::open()?;
            let memory = query.add(r"\GPU Adapter Memory(*)\Dedicated Usage")?;
            let engines = query.add(r"\GPU Engine(*)\Utilization Percentage");
            query.collect();
            Some(Adapters { list, query, memory, engines })
        }

        pub fn query(&self) -> Vec<GpuMemory> {
            if !self.query.collect() {
                return Vec::new();
            }
            let memory = self.query.values(self.memory);
            let engines = self.engines.map(|c| self.query.values(c)).unwrap_or_default();
            self.list
                .iter()
                .map(|a| GpuMemory {
                    name: a.name.clone(),
                    used_mib: super::sum_for(&memory, &a.luid) as u64 >> 20,
                    total_mib: a.total >> 20,
                    load: super::engine_load(&engines, &a.luid),
                    power_w: None,
                    power_limit_w: None,
                })
                .collect()
        }
    }
}

/// Début des noms d'instance PDH d'une carte : `luid_0x00000000_0x0000d1a3`.
#[cfg_attr(not(windows), allow(dead_code))]
fn luid_prefix(high: u32, low: u32) -> String {
    format!("luid_0x{high:08x}_0x{low:08x}")
}

/// Somme des instances d'une carte (`luid_…_phys_0`, parfois plusieurs).
#[cfg_attr(not(windows), allow(dead_code))]
fn sum_for(values: &[(String, f64)], luid: &str) -> f64 {
    values
        .iter()
        .filter(|(name, _)| name.to_ascii_lowercase().starts_with(luid))
        .map(|(_, v)| v)
        .sum()
}

/// Charge d'une carte comme le Gestionnaire des tâches : par type de moteur
/// (`…_engtype_3D`, `…_engtype_Compute`), somme des processus ; la charge est
/// le type le plus occupé.
#[cfg_attr(not(windows), allow(dead_code))]
fn engine_load(values: &[(String, f64)], luid: &str) -> Option<u64> {
    let mut by_type: Vec<(String, f64)> = Vec::new();
    for (name, v) in values {
        let lower = name.to_ascii_lowercase();
        let Some(i) = lower.find(luid) else {
            continue;
        };
        let Some(engtype) = lower[i..].split("_engtype_").nth(1) else {
            continue;
        };
        match by_type.iter_mut().find(|(t, _)| t == engtype) {
            Some((_, sum)) => *sum += v,
            None => by_type.push((engtype.to_string(), *v)),
        }
    }
    by_type
        .iter()
        .map(|(_, v)| *v)
        .reduce(f64::max)
        .map(|v| v.clamp(0.0, 100.0).round() as u64)
}

// -------------------------------------------------------------------- Linux

#[cfg_attr(windows, allow(dead_code))]
mod linux {
    use super::{GpuMemory, MIN_DEDICATED};
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    #[derive(Debug, PartialEq)]
    enum Kind {
        /// Pilote `amdgpu` : VRAM, charge et watts.
        Amd,
        /// Carte Intel dédiée (i915, xe) : watts seulement.
        Intel,
    }

    pub struct Card {
        /// `/sys/class/drm/cardN/device`.
        device: PathBuf,
        hwmon: Option<PathBuf>,
        kind: Kind,
        name: String,
        /// Énergie (µJ) et instant du relevé précédent, pour les cartes qui
        /// ne donnent que l'énergie cumulée.
        energy_prev: Option<(u64, Instant)>,
    }

    fn read_u64(path: &Path) -> Option<u64> {
        std::fs::read_to_string(path).ok()?.trim().parse().ok()
    }

    fn hwmon_dir(device: &Path) -> Option<PathBuf> {
        std::fs::read_dir(device.join("hwmon")).ok()?.flatten().map(|e| e.path()).min()
    }

    /// Cartes AMD et Intel dédiées ; les NVIDIA passent par nvidia-smi.
    pub fn cards() -> Vec<Card> {
        cards_in(Path::new("/sys/class/drm"))
    }

    pub(super) fn cards_in(drm: &Path) -> Vec<Card> {
        let Ok(entries) = std::fs::read_dir(drm) else {
            return Vec::new();
        };
        let mut names: Vec<String> = entries
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            // card0, card1… ; pas les connecteurs (card0-DP-1).
            .filter(|n| n.strip_prefix("card").is_some_and(|i| !i.is_empty() && i.bytes().all(|b| b.is_ascii_digit())))
            .collect();
        names.sort();
        let mut out = Vec::new();
        for n in names {
            let device = drm.join(&n).join("device");
            let vendor = std::fs::read_to_string(device.join("vendor")).unwrap_or_default();
            let hwmon = hwmon_dir(&device);
            let kind = match vendor.trim() {
                "0x1002" if read_u64(&device.join("mem_info_vram_total")).unwrap_or(0) >= MIN_DEDICATED => Kind::Amd,
                // Seules les cartes Intel dédiées ont un hwmon d'énergie.
                "0x8086" if hwmon.as_ref().is_some_and(|h| read_u64(&h.join("energy1_input")).is_some()) => {
                    Kind::Intel
                }
                _ => continue,
            };
            let name = card_name(&device, &kind);
            out.push(Card { device, hwmon, kind, name, energy_prev: None });
        }
        out
    }

    /// Nom de la carte : `product_name` du pilote, sinon celui de `lspci`
    /// pour son emplacement PCI, sinon générique.
    fn card_name(device: &Path, kind: &Kind) -> String {
        if let Some(name) = std::fs::read_to_string(device.join("product_name"))
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
        {
            return name;
        }
        let slot = std::fs::canonicalize(device)
            .ok()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()));
        let from_lspci = slot.and_then(|slot| {
            let out = std::process::Command::new("lspci").args(["-s", &slot]).output().ok()?;
            let text = String::from_utf8_lossy(&out.stdout);
            let desc = text.lines().next()?.split_once(": ")?.1.to_string();
            Some(super::short_gpu_name(&desc))
        });
        from_lspci.unwrap_or_else(|| match kind {
            Kind::Amd => "AMD Radeon".into(),
            Kind::Intel => "Intel Arc".into(),
        })
    }

    impl Card {
        pub fn read(&mut self) -> Option<GpuMemory> {
            let micro = |file: &str| {
                self.hwmon
                    .as_ref()
                    .and_then(|h| read_u64(&h.join(file)))
                    .map(|uw| uw as f32 / 1e6)
            };
            match self.kind {
                Kind::Amd => {
                    let total = read_u64(&self.device.join("mem_info_vram_total"))?;
                    let used = read_u64(&self.device.join("mem_info_vram_used")).unwrap_or(0);
                    Some(GpuMemory {
                        name: self.name.clone(),
                        used_mib: used >> 20,
                        total_mib: total >> 20,
                        load: read_u64(&self.device.join("gpu_busy_percent")),
                        // power1_average (anciens noyaux), power1_input (RDNA3).
                        power_w: micro("power1_average").or_else(|| micro("power1_input")),
                        power_limit_w: micro("power1_cap"),
                    })
                }
                Kind::Intel => {
                    let now = Instant::now();
                    let energy = self.hwmon.as_ref().and_then(|h| read_u64(&h.join("energy1_input")))?;
                    let power_w = self.energy_prev.and_then(|(prev, at)| {
                        let dt = now.duration_since(at).as_secs_f64();
                        (energy >= prev && dt > 0.0).then(|| ((energy - prev) as f64 / 1e6 / dt) as f32)
                    });
                    self.energy_prev = Some((energy, now));
                    Some(GpuMemory {
                        name: self.name.clone(),
                        used_mib: 0,
                        total_mib: 0,
                        load: None,
                        power_w,
                        power_limit_w: micro("power1_max").or_else(|| micro("power1_cap")),
                    })
                }
            }
        }
    }
}

/// `Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [Radeon RX 7900 XT/7900 XTX] (rev c8)`
/// → `AMD Radeon RX 7900 XT/7900 XTX` ; `Intel Corporation DG2 [Arc A770] (rev 08)`
/// → `Intel Arc A770`.
#[cfg_attr(windows, allow(dead_code))]
fn short_gpu_name(desc: &str) -> String {
    let desc = match desc.rfind(" (rev ") {
        Some(i) => &desc[..i],
        None => desc,
    };
    let last = desc
        .rfind('[')
        .and_then(|i| desc[i + 1..].strip_suffix(']'))
        .filter(|s| !s.contains("AMD/ATI"));
    let vendor = if desc.starts_with("Intel") {
        "Intel"
    } else if desc.starts_with("Advanced Micro Devices") {
        "AMD"
    } else {
        ""
    };
    match last {
        Some(model) if !vendor.is_empty() => format!("{vendor} {model}"),
        Some(model) => model.to_string(),
        None => desc.to_string(),
    }
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
        assert_eq!((gpus[0].used_mib, gpus[0].total_mib, gpus[0].load), (852, 16303, Some(7)));
        assert!((gpus[0].fraction() - 852.0 / 16303.0).abs() < 1e-6);
        assert_eq!((gpus[0].power_w, gpus[0].power_limit_w), (Some(17.01), Some(360.0)));
        // Puissance non exposée : la carte reste suivie, sans watts.
        assert_eq!((gpus[1].power_w, gpus[1].power_limit_w), (None, None));
        // Valeur non numérique (GPU en erreur) : ligne ignorée.
        assert!(parse("0, GPU, [N/A], 16303, 0, 10, 100").is_empty());
    }

    #[test]
    fn reads_windows_counters_per_adapter() {
        let luid = luid_prefix(0, 0xD1A3);
        assert_eq!(luid, "luid_0x00000000_0x0000d1a3");
        let memory = vec![
            ("luid_0x00000000_0x0000D1A3_phys_0".to_string(), 2.0 * 1024.0 * 1024.0 * 1024.0),
            ("luid_0x00000000_0x0000AAAA_phys_0".to_string(), 1.0),
        ];
        assert_eq!(sum_for(&memory, &luid) as u64 >> 20, 2048);
        let engines = vec![
            ("pid_100_luid_0x00000000_0x0000D1A3_phys_0_eng_0_engtype_3D".to_string(), 20.0),
            ("pid_200_luid_0x00000000_0x0000D1A3_phys_0_eng_0_engtype_3D".to_string(), 15.0),
            ("pid_200_luid_0x00000000_0x0000D1A3_phys_0_eng_3_engtype_Compute".to_string(), 30.0),
            ("pid_300_luid_0x00000000_0x0000AAAA_phys_0_eng_0_engtype_3D".to_string(), 90.0),
        ];
        // 3D : 20 + 15 = 35 ; Compute : 30 ; l'autre carte ne compte pas.
        assert_eq!(engine_load(&engines, &luid), Some(35));
        assert_eq!(engine_load(&[], &luid), None);
    }

    #[test]
    fn shortens_lspci_names() {
        assert_eq!(
            short_gpu_name("Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [Radeon RX 7900 XT/7900 XTX] (rev c8)"),
            "AMD Radeon RX 7900 XT/7900 XTX"
        );
        assert_eq!(short_gpu_name("Intel Corporation DG2 [Arc A770] (rev 08)"), "Intel Arc A770");
        assert_eq!(short_gpu_name("Some GPU"), "Some GPU");
    }

    /// Arborescence `/sys/class/drm` simulée : une Radeon dédiée, sa puce
    /// intégrée (512 Mio, ignorée), une NVIDIA (pour nvidia-smi) et un
    /// connecteur.
    #[test]
    fn reads_amd_cards_from_sysfs() {
        let root = std::env::temp_dir().join(format!("uillamacpp-drm-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let card = |n: &str, vendor: &str, files: &[(&str, &str)]| {
            let dev = root.join(n).join("device");
            std::fs::create_dir_all(dev.join("hwmon/hwmon3")).unwrap();
            std::fs::write(dev.join("vendor"), vendor).unwrap();
            std::fs::write(dev.join("product_name"), "Radeon RX 7900 XTX").unwrap();
            for (f, v) in files {
                std::fs::write(dev.join(f), v).unwrap();
            }
        };
        card("card0", "0x1002\n", &[
            ("mem_info_vram_total", "25753026560\n"),
            ("mem_info_vram_used", "1073741824\n"),
            ("gpu_busy_percent", "42\n"),
            ("hwmon/hwmon3/power1_average", "61000000\n"),
            ("hwmon/hwmon3/power1_cap", "339000000\n"),
        ]);
        card("card1", "0x1002\n", &[("mem_info_vram_total", "536870912\n")]);
        card("card2", "0x10de\n", &[]);
        std::fs::create_dir_all(root.join("card0-DP-1")).unwrap();

        let mut cards = linux::cards_in(&root);
        assert_eq!(cards.len(), 1);
        let g = cards[0].read().unwrap();
        assert_eq!(g.name, "Radeon RX 7900 XTX");
        assert_eq!((g.used_mib, g.total_mib, g.load), (1024, 24560, Some(42)));
        assert_eq!((g.power_w, g.power_limit_w), (Some(61.0), Some(339.0)));
        let _ = std::fs::remove_dir_all(&root);
    }
}
