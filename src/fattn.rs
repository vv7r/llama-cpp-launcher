//! Paires de types K/V pour lesquelles une release CUDA a compilé ses noyaux
//! FlashAttention.
//!
//! Par défaut llama.cpp n'en compile que quatre (f16-f16, q4_0-q4_0,
//! q8_0-q8_0, bf16-bf16) ; `GGML_CUDA_FA_QUANTS=all` les compile toutes. Une
//! paire absente fait passer l'attention sur le CPU (flash-attn on) ou
//! désactive FlashAttention (auto) : bien plus lent.
//!
//! Depuis 2026, llama.cpp inscrit la liste dans `ggml-cuda.dll`, parmi les
//! « features » du backend : `FA_QUANTS` suivi de sa valeur (`all`, ou
//! `q4_0-q4_0,q8_0-q8_0,f16-f16,bf16-bf16`). Les versions plus anciennes
//! ne l'indiquent pas : rien n'est alors signalé.

use std::path::Path;

/// Types qui ont des noyaux FlashAttention CUDA (`FA_TYPES` du CMake de
/// ggml). f32 et iq4_nl n'en ont pas, quelle que soit la compilation.
pub const FA_TYPES: &[&str] = &["f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"];

#[derive(Clone, Debug, PartialEq)]
pub enum FaSupport {
    /// Toutes les paires de `FA_TYPES`.
    All,
    /// Les paires (K, V) listées ; f16-f16 est toujours compilée.
    Pairs(Vec<(String, String)>),
}

impl FaSupport {
    /// `all`, ou une liste `k-v` séparée par des virgules (ou des
    /// points-virgules, forme du CMakeCache).
    pub fn parse(value: &str) -> Option<FaSupport> {
        let value = value.trim();
        if value.eq_ignore_ascii_case("all") {
            return Some(FaSupport::All);
        }
        let mut pairs = Vec::new();
        for item in value.split([',', ';']) {
            let (k, v) = item.trim().split_once('-')?;
            if !FA_TYPES.contains(&k) || !FA_TYPES.contains(&v) {
                return None;
            }
            pairs.push((k.to_string(), v.to_string()));
        }
        (!pairs.is_empty()).then_some(FaSupport::Pairs(pairs))
    }

    /// La paire K/V a-t-elle ses noyaux ? Un type vide vaut f16, le défaut
    /// de llama-server.
    pub fn supports(&self, k: &str, v: &str) -> bool {
        let k = if k.is_empty() { "f16" } else { k };
        let v = if v.is_empty() { "f16" } else { v };
        if !FA_TYPES.contains(&k) || !FA_TYPES.contains(&v) {
            return false;
        }
        match self {
            FaSupport::All => true,
            FaSupport::Pairs(pairs) => {
                (k == "f16" && v == "f16") || pairs.iter().any(|(a, b)| a == k && b == v)
            }
        }
    }

    /// `toutes les paires` ou `f16-f16, q8_0-q8_0…`, pour une infobulle.
    pub fn describe(&self, fr: bool) -> String {
        match self {
            FaSupport::All => {
                if fr { "toutes les paires".into() } else { "every pair".into() }
            }
            FaSupport::Pairs(pairs) => pairs
                .iter()
                .map(|(k, v)| format!("{k}-{v}"))
                .collect::<Vec<_>>()
                .join(", "),
        }
    }
}

/// Paires compilées par la release `dir` ; `None` sans `ggml-cuda.dll`, ou
/// si la version ne l'indique pas. Lit la DLL entière (≈ 35 Mo) : en tâche
/// de fond seulement.
pub fn read(dir: &Path) -> Option<FaSupport> {
    let bytes = std::fs::read(dir.join("ggml-cuda.dll")).ok()?;
    find_in(&bytes)
}

/// Cherche parmi les chaînes terminées par un zéro celle qui suit
/// `FA_QUANTS`, ou à défaut une liste de paires valide.
fn find_in(bytes: &[u8]) -> Option<FaSupport> {
    let mut after_marker = false;
    let mut fallback = None;
    for chunk in bytes.split(|b| *b == 0) {
        if chunk.is_empty() {
            continue;
        }
        if chunk.len() > 2048 || !chunk.is_ascii() {
            after_marker = false;
            continue;
        }
        let text = std::str::from_utf8(chunk).unwrap_or("");
        if after_marker {
            if let Some(found) = FaSupport::parse(text) {
                return Some(found);
            }
        }
        if fallback.is_none() && text.contains('-') {
            fallback = FaSupport::parse(text).filter(|f| matches!(f, FaSupport::Pairs(_)));
        }
        after_marker = text == "FA_QUANTS";
    }
    fallback
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_pairs_and_all() {
        let d = FaSupport::parse("q4_0-q4_0,q8_0-q8_0,f16-f16,bf16-bf16").unwrap();
        assert!(d.supports("q8_0", "q8_0"));
        assert!(d.supports("", ""), "vide = f16");
        assert!(!d.supports("q8_0", "q4_0"));
        assert!(!d.supports("q5_0", "q5_0"));
        let all = FaSupport::parse("all").unwrap();
        assert!(all.supports("q8_0", "q4_0"));
        assert!(all.supports("q5_1", "bf16"));
        // Pas de noyau, quelle que soit la compilation.
        assert!(!all.supports("iq4_nl", "iq4_nl"));
        assert!(!all.supports("f32", "f16"));
        // Forme du CMakeCache.
        assert_eq!(FaSupport::parse("q8_0-q4_0;f16-f16").unwrap().describe(true), "q8_0-q4_0, f16-f16");
        assert!(FaSupport::parse("fattn-vec").is_none());
        assert!(FaSupport::parse("q9_9-q8_0").is_none());
    }

    #[test]
    fn found_after_the_feature_name() {
        // Extrait de ggml-cuda.dll b11235.
        let dll = b"\0\0ARCHS\0USE_GRAPHS\0FA_QUANTS\0q4_0-q4_0,q8_0-q8_0,f16-f16,bf16-bf16\0BLACKWELL_NATIVE_FP4\0";
        assert_eq!(
            find_in(dll),
            FaSupport::parse("q4_0-q4_0,q8_0-q8_0,f16-f16,bf16-bf16")
        );
        assert_eq!(find_in(b"\0FA_QUANTS\0\0all\0x\0"), Some(FaSupport::All));
        // « all » seul, loin de FA_QUANTS : ignoré.
        assert_eq!(find_in(b"\0all\0FA\0"), None);
    }

    /// Relit la vraie DLL d'une release : `FA_RELEASE=<dossier> cargo test
    /// fa_release -- --ignored --nocapture`.
    #[test]
    #[ignore = "lit une release CUDA locale (FA_RELEASE)"]
    fn fa_release() {
        let dir = std::env::var("FA_RELEASE").unwrap();
        let t = std::time::Instant::now();
        let found = read(Path::new(&dir));
        println!("{found:?} en {:?}", t.elapsed());
        assert!(found.is_some());
    }
}
