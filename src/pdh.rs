//! Compteurs de performance de Windows (PDH), par FFI sans dépendance.
//!
//! Les chemins sont donnés en anglais (`PdhAddEnglishCounterW`) : les noms
//! affichés sont traduits (« Compteur Énergie » sous Windows en français).
//! Les compteurs de taux exigent deux relevés : la première valeur, prise à
//! l'ouverture, n'est pas significative.

#[repr(C)]
#[derive(Clone, Copy)]
struct FmtValue {
    status: u32,
    value: f64,
}

/// `PDH_FMT_COUNTERVALUE_ITEM_W` : nom de l'instance puis valeur.
#[repr(C)]
#[derive(Clone, Copy)]
struct FmtItem {
    name: *const u16,
    value: FmtValue,
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

pub fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// Requête PDH ouverte et ses compteurs.
pub struct Query {
    handle: isize,
}

/// Compteur ajouté à une requête.
#[derive(Clone, Copy)]
pub struct Counter(isize);

impl Query {
    pub fn open() -> Option<Query> {
        let mut handle = 0isize;
        (unsafe { PdhOpenQueryW(std::ptr::null(), 0, &mut handle) } == 0).then_some(Query { handle })
    }

    /// Ajoute un compteur par son chemin anglais (`\Energy Meter(*)\Power`).
    pub fn add(&self, path: &str) -> Option<Counter> {
        let mut counter = 0isize;
        let path = wide(path);
        (unsafe { PdhAddEnglishCounterW(self.handle, path.as_ptr(), 0, &mut counter) } == 0)
            .then_some(Counter(counter))
    }

    /// Relève tous les compteurs de la requête.
    pub fn collect(&self) -> bool {
        unsafe { PdhCollectQueryData(self.handle) == 0 }
    }

    /// Valeurs de toutes les instances d'un compteur : (nom, valeur).
    pub fn values(&self, counter: Counter) -> Vec<(String, f64)> {
        let (mut size, mut count) = (0u32, 0u32);
        let status = unsafe {
            PdhGetFormattedCounterArrayW(counter.0, PDH_FMT_DOUBLE, &mut size, &mut count, std::ptr::null_mut())
        };
        if status != PDH_MORE_DATA || size == 0 {
            return Vec::new();
        }
        // Tampon aligné sur 8 octets : les éléments puis leurs noms.
        let mut buf = vec![0u64; (size as usize).div_ceil(8)];
        let items = buf.as_mut_ptr() as *mut FmtItem;
        if unsafe { PdhGetFormattedCounterArrayW(counter.0, PDH_FMT_DOUBLE, &mut size, &mut count, items) } != 0 {
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
}

impl Drop for Query {
    fn drop(&mut self) {
        unsafe { PdhCloseQuery(self.handle) };
    }
}
