//! Lecture des métadonnées de l'en-tête d'un fichier GGUF.
//!
//! Seul l'en-tête est lu : on saute le contenu des tableaux (vocabulaire,
//! merges…) par `seek`, la lecture reste donc quasi instantanée même sur
//! un modèle de plusieurs dizaines de gigaoctets.

use std::fs::File;
use std::io::{BufReader, Read, Seek};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct GgufMeta {
    pub architecture: String,
    pub name: String,
    pub size_label: String,
    pub context_length: Option<u64>,
    pub block_count: Option<u64>,
    pub embedding_length: Option<u64>,
    pub tensor_count: u64,
    pub has_chat_template: bool,
}

const T_UINT8: u32 = 0;
const T_INT8: u32 = 1;
const T_UINT16: u32 = 2;
const T_INT16: u32 = 3;
const T_UINT32: u32 = 4;
const T_INT32: u32 = 5;
const T_FLOAT32: u32 = 6;
const T_BOOL: u32 = 7;
const T_STRING: u32 = 8;
const T_ARRAY: u32 = 9;
const T_UINT64: u32 = 10;
const T_INT64: u32 = 11;
const T_FLOAT64: u32 = 12;

/// Taille en octets d'un type scalaire ; `None` pour string/array.
fn scalar_size(t: u32) -> Option<u64> {
    match t {
        T_UINT8 | T_INT8 | T_BOOL => Some(1),
        T_UINT16 | T_INT16 => Some(2),
        T_UINT32 | T_INT32 | T_FLOAT32 => Some(4),
        T_UINT64 | T_INT64 | T_FLOAT64 => Some(8),
        _ => None,
    }
}

struct Reader<R> {
    inner: R,
}

impl<R: Read + Seek> Reader<R> {
    fn u32(&mut self) -> anyhow::Result<u32> {
        let mut b = [0u8; 4];
        self.inner.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    fn u64(&mut self) -> anyhow::Result<u64> {
        let mut b = [0u8; 8];
        self.inner.read_exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    fn string(&mut self) -> anyhow::Result<String> {
        let len = self.u64()?;
        // Une clé ou une valeur de plus de 16 Mio signale un fichier corrompu.
        anyhow::ensure!(len <= 16 << 20, "chaîne GGUF invalide ({len} octets)");
        let mut buf = vec![0u8; len as usize];
        self.inner.read_exact(&mut buf)?;
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    /// `seek_relative` garde le tampon quand la cible y est déjà : un
    /// `seek` le vidait à chaque chaîne sautée, soit des dizaines de milliers
    /// d'appels système et de relectures pour le vocabulaire (≈ 100 ms par
    /// modèle au lieu de quelques ms).
    fn skip(&mut self, n: u64) -> anyhow::Result<()> {
        let n = i64::try_from(n).map_err(|_| anyhow::anyhow!("saut GGUF invalide"))?;
        self.inner.seek_relative(n)?;
        Ok(())
    }

    fn skip_string(&mut self) -> anyhow::Result<()> {
        let len = self.u64()?;
        anyhow::ensure!(len <= 16 << 20, "chaîne GGUF invalide");
        self.skip(len)
    }

    /// Lit une valeur scalaire et la rend sous forme numérique si possible.
    fn scalar_as_u64(&mut self, t: u32) -> anyhow::Result<Option<u64>> {
        let v = match t {
            T_UINT8 | T_BOOL => {
                let mut b = [0u8; 1];
                self.inner.read_exact(&mut b)?;
                Some(b[0] as u64)
            }
            T_INT8 => {
                let mut b = [0u8; 1];
                self.inner.read_exact(&mut b)?;
                Some(b[0] as i8 as i64 as u64)
            }
            T_UINT16 | T_INT16 => {
                let mut b = [0u8; 2];
                self.inner.read_exact(&mut b)?;
                Some(u16::from_le_bytes(b) as u64)
            }
            T_UINT32 | T_INT32 => Some(self.u32()? as u64),
            T_FLOAT32 => {
                let _ = self.u32()?;
                None
            }
            T_UINT64 | T_INT64 => Some(self.u64()?),
            T_FLOAT64 => {
                let _ = self.u64()?;
                None
            }
            _ => None,
        };
        Ok(v)
    }

    /// Lit un entier ; consomme la valeur quel que soit son type afin de
    /// ne jamais désynchroniser le flux.
    fn number(&mut self, t: u32) -> anyhow::Result<Option<u64>> {
        if scalar_size(t).is_some() {
            self.scalar_as_u64(t)
        } else {
            self.skip_value(t)?;
            Ok(None)
        }
    }

    fn skip_value(&mut self, t: u32) -> anyhow::Result<()> {
        if let Some(n) = scalar_size(t) {
            return self.skip(n);
        }
        match t {
            T_STRING => self.skip_string(),
            T_ARRAY => {
                let elem = self.u32()?;
                let count = self.u64()?;
                if let Some(n) = scalar_size(elem) {
                    self.skip(n.saturating_mul(count))
                } else if elem == T_STRING {
                    for _ in 0..count {
                        self.skip_string()?;
                    }
                    Ok(())
                } else {
                    anyhow::bail!("type de tableau GGUF non géré: {elem}")
                }
            }
            other => anyhow::bail!("type GGUF non géré: {other}"),
        }
    }
}

pub fn read(path: &Path) -> anyhow::Result<GgufMeta> {
    let file = File::open(path)?;
    let mut r = Reader {
        inner: BufReader::with_capacity(1 << 16, file),
    };

    let mut magic = [0u8; 4];
    r.inner.read_exact(&mut magic)?;
    anyhow::ensure!(&magic == b"GGUF", "ce n'est pas un fichier GGUF");

    let version = r.u32()?;
    anyhow::ensure!((2..=3).contains(&version), "version GGUF non gérée: {version}");

    let mut meta = GgufMeta {
        tensor_count: r.u64()?,
        ..Default::default()
    };
    let kv_count = r.u64()?;
    anyhow::ensure!(kv_count <= 100_000, "en-tête GGUF invalide");

    for _ in 0..kv_count {
        let key = r.string()?;
        let vtype = r.u32()?;

        match key.as_str() {
            "general.architecture" if vtype == T_STRING => meta.architecture = r.string()?,
            "general.name" if vtype == T_STRING => meta.name = r.string()?,
            "general.size_label" if vtype == T_STRING => meta.size_label = r.string()?,
            "tokenizer.chat_template" => {
                meta.has_chat_template = true;
                r.skip_value(vtype)?;
            }
            _ if key.ends_with(".context_length") => {
                meta.context_length = r.number(vtype)?.or(meta.context_length);
            }
            _ if key.ends_with(".block_count") => {
                meta.block_count = r.number(vtype)?.or(meta.block_count);
            }
            _ if key.ends_with(".embedding_length") => {
                meta.embedding_length = r.number(vtype)?.or(meta.embedding_length);
            }
            _ => r.skip_value(vtype)?,
        }
    }

    Ok(meta)
}

/// En-tête mémorisé dans `gguf-cache.json`, avec la taille et la date du
/// fichier lu : un fichier remplacé est relu, les autres jamais.
#[derive(Clone, Serialize, Deserialize)]
pub struct Cached {
    pub size: u64,
    pub modified: u64,
    /// `None` : fichier illisible (en-tête invalide).
    pub meta: Option<GgufMeta>,
}

/// Taille et date de modification (secondes) d'un fichier.
pub fn stamp(path: &Path) -> Option<(u64, u64)> {
    let m = std::fs::metadata(path).ok()?;
    let modified = m
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs();
    Some((m.len(), modified))
}

/// Lit l'en-tête et l'horodate.
pub fn read_cached(path: &Path) -> Cached {
    let (size, modified) = stamp(path).unwrap_or((0, 0));
    Cached {
        size,
        modified,
        meta: read(path).ok(),
    }
}

/// L'entrée correspond-elle encore au fichier sur le disque ?
pub fn is_fresh(entry: &Cached, path: &Path) -> bool {
    stamp(path) == Some((entry.size, entry.modified))
}

fn cache_path() -> PathBuf {
    crate::config::app_dir().join("gguf-cache.json")
}

pub fn load_cache() -> HashMap<PathBuf, Cached> {
    std::fs::read_to_string(cache_path())
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

/// Enregistre le cache, sans les fichiers qui n'existent plus.
pub fn save_cache(cache: &HashMap<PathBuf, Cached>) {
    let kept: HashMap<&PathBuf, &Cached> = cache.iter().filter(|(p, _)| p.is_file()).collect();
    if let Ok(json) = serde_json::to_string_pretty(&kept) {
        let _ = std::fs::write(cache_path(), json);
    }
}

#[cfg(test)]
mod timing {
    /// Durée de lecture d'un vrai en-tête : `GGUF_FILE=<modèle> cargo test
    /// header_read_time -- --ignored --nocapture`.
    #[test]
    #[ignore = "lit un modèle local (GGUF_FILE)"]
    fn header_read_time() {
        let path = std::path::PathBuf::from(std::env::var("GGUF_FILE").unwrap());
        let t = std::time::Instant::now();
        for _ in 0..20 {
            super::read(&path).unwrap();
        }
        println!("{:?} par lecture", t.elapsed() / 20);
    }
}
