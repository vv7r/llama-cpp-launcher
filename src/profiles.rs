//! Profils de lancement : `profiles/<nom>.json`.
//!
//! Un profil mémorise les paramètres actifs, l'hôte, le port et — depuis
//! l'affichage en cartes — le modèle et la version en cours à
//! l'enregistrement, ainsi que sa date de mise à jour. Ces champs sont
//! optionnels : les profils écrits par les versions précédentes se relisent.

use crate::config::app_dir;
use chrono::{DateTime, Local};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Serialize, Deserialize, Default)]
pub struct LaunchProfile {
    pub name: String,
    pub parameters: BTreeMap<String, String>,
    #[serde(default)]
    pub host: String,
    #[serde(default)]
    pub port: String,
    /// Chemin complet du modèle sélectionné à l'enregistrement.
    #[serde(default)]
    pub model: String,
    /// Nom de la version de llama.cpp sélectionnée à l'enregistrement.
    #[serde(default)]
    pub version: String,
    /// Date de dernière mise à jour (RFC 3339). Un renommage ne la modifie
    /// pas, comme pour une conversation.
    #[serde(default)]
    pub updated: String,
    /// Options importées absentes du catalogue, transmises telles quelles.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extra_args: Vec<Vec<String>>,
}

impl LaunchProfile {
    /// Nom du fichier du modèle, sans extension.
    pub fn model_stem(&self) -> Option<String> {
        Path::new(&self.model)
            .file_stem()
            .and_then(|s| s.to_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    }
}

/// Un profil tel qu'affiché dans la liste : son identifiant (le nom du
/// fichier), son contenu et sa date.
#[derive(Clone)]
pub struct ProfileEntry {
    pub stem: String,
    pub profile: LaunchProfile,
    pub updated: Option<DateTime<Local>>,
}

impl ProfileEntry {
    pub fn display_name(&self) -> &str {
        if self.profile.name.trim().is_empty() {
            &self.stem
        } else {
            &self.profile.name
        }
    }
}

pub fn profiles_dir() -> PathBuf {
    app_dir().join("profiles")
}

/// Tous les profils, du plus récemment modifié au plus ancien.
pub fn entries() -> Vec<ProfileEntry> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(profiles_dir()) else {
        return out;
    };
    for entry in rd.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(profile) = serde_json::from_str::<LaunchProfile>(&text) else {
            continue;
        };
        // Repli sur la date du fichier pour les profils plus anciens.
        let updated = DateTime::parse_from_rfc3339(&profile.updated)
            .map(|d| d.with_timezone(&Local))
            .ok()
            .or_else(|| {
                entry
                    .metadata()
                    .and_then(|m| m.modified())
                    .ok()
                    .map(DateTime::<Local>::from)
            });
        out.push(ProfileEntry {
            stem: stem.to_string(),
            profile,
            updated,
        });
    }
    out.sort_by_key(|e| std::cmp::Reverse(e.updated));
    out
}

fn path_for(name: &str) -> PathBuf {
    profiles_dir().join(format!("{}.json", sanitize(name)))
}

/// Enregistre le profil en datant la mise à jour de maintenant.
pub fn save(profile: &LaunchProfile) -> anyhow::Result<()> {
    let mut profile = profile.clone();
    profile.updated = Local::now().to_rfc3339();
    write(&profile)
}

fn write(profile: &LaunchProfile) -> anyhow::Result<()> {
    std::fs::create_dir_all(profiles_dir())?;
    let json = serde_json::to_string_pretty(profile)?;
    std::fs::write(path_for(&profile.name), json)?;
    Ok(())
}

pub fn load(stem: &str) -> anyhow::Result<LaunchProfile> {
    let text = std::fs::read_to_string(path_for(stem))?;
    Ok(serde_json::from_str(&text)?)
}

pub fn exists(name: &str) -> bool {
    path_for(name).is_file()
}

pub fn delete(stem: &str) -> anyhow::Result<()> {
    std::fs::remove_file(path_for(stem))?;
    Ok(())
}

/// Renomme un profil sans toucher à ses paramètres ni à sa date.
/// Refuse d'écraser un autre profil existant.
pub fn rename(stem: &str, new_name: &str) -> anyhow::Result<String> {
    let new_name = new_name.trim();
    anyhow::ensure!(!new_name.is_empty(), "nom vide");
    let new_stem = sanitize(new_name);
    let same_file = new_stem.to_lowercase() == stem.to_lowercase();
    anyhow::ensure!(
        same_file || !exists(&new_stem),
        "un profil « {new_name} » existe déjà"
    );

    let mut profile = load(stem)?;
    profile.name = new_name.to_string();
    if profile.updated.is_empty() {
        // Fige la date d'origine avant que la réécriture ne change celle
        // du fichier.
        if let Ok(modified) = std::fs::metadata(path_for(stem)).and_then(|m| m.modified()) {
            profile.updated = DateTime::<Local>::from(modified).to_rfc3339();
        }
    }
    write(&profile)?;
    if !same_file {
        std::fs::remove_file(path_for(stem))?;
    }
    Ok(new_stem)
}

/// Retire les caractères interdits dans un nom de fichier Windows.
pub fn sanitize(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| {
            if r#"\/:*?"<>|"#.contains(c) || c.is_control() {
                '_'
            } else {
                c
            }
        })
        .collect();
    let trimmed = cleaned.trim().trim_end_matches('.').to_string();
    if trimmed.is_empty() {
        "profil".into()
    } else {
        trimmed
    }
}

// ------------------------------------------------------------------ affichage

/// Étiquette affichée sur la carte d'un profil.
#[derive(Debug, PartialEq)]
pub struct Tag {
    pub text: String,
    /// Le tag de contexte est mis en avant.
    pub primary: bool,
}

/// Tags résumant un profil : le contexte d'abord, puis les réglages qui
/// changent le plus le comportement du serveur.
pub fn tags(profile: &LaunchProfile) -> Vec<Tag> {
    let p = &profile.parameters;
    let get = |k: &str| p.get(k).map(String::as_str).filter(|v| !v.is_empty());
    let mut out = Vec::new();

    if let Some(ctx) = get("--ctx-size") {
        out.push(Tag {
            text: format!("ctx {}", compact_tokens(ctx)),
            primary: true,
        });
    }
    if let Some(ngl) = get("-ngl") {
        out.push(Tag {
            text: format!("ngl {ngl}"),
            primary: false,
        });
    }
    match (get("--cache-type-k"), get("--cache-type-v")) {
        (Some(k), Some(v)) if k == v => out.push(Tag {
            text: format!("KV {k}"),
            primary: false,
        }),
        (Some(k), Some(v)) => out.push(Tag {
            text: format!("KV {k}/{v}"),
            primary: false,
        }),
        (Some(k), None) => out.push(Tag {
            text: format!("K {k}"),
            primary: false,
        }),
        (None, Some(v)) => out.push(Tag {
            text: format!("V {v}"),
            primary: false,
        }),
        (None, None) => {}
    }
    if get("--flash-attn") == Some("on") {
        out.push(Tag {
            text: "FA".into(),
            primary: false,
        });
    }
    if let Some(np) = get("--parallel").filter(|n| *n != "1") {
        out.push(Tag {
            text: format!("×{np} parallel"),
            primary: false,
        });
    }
    if let Some(spec) = get("--spec-type").filter(|s| *s != "none") {
        let n = get("--spec-draft-n-max").map(|n| format!(" ×{n}")).unwrap_or_default();
        out.push(Tag {
            text: format!("spec {spec}{n}"),
            primary: false,
        });
        match (get("--cache-type-k-draft"), get("--cache-type-v-draft")) {
            (Some(k), Some(v)) if k == v => out.push(Tag {
                text: format!("draft KV {k}"),
                primary: false,
            }),
            (Some(k), Some(v)) => out.push(Tag {
                text: format!("draft KV {k}/{v}"),
                primary: false,
            }),
            _ => {}
        }
    }
    if let Some(effort) = get("--reasoning-effort") {
        out.push(Tag {
            text: format!("reasoning {effort}"),
            primary: false,
        });
    }
    if let Some(format) = get("--reasoning-format") {
        out.push(Tag {
            text: format!("think {format}"),
            primary: false,
        });
    }
    if get("--mmproj").is_some() {
        out.push(Tag {
            text: "vision".into(),
            primary: false,
        });
    }
    if let Some(t) = get("--temp") {
        out.push(Tag {
            text: format!("T {t}"),
            primary: false,
        });
    }
    for (option, label) in [
        ("--top-p", "top-p"),
        ("--top-k", "top-k"),
        ("--min-p", "min-p"),
        ("--n-predict", "max"),
    ] {
        if let Some(v) = get(option) {
            out.push(Tag {
                text: format!("{label} {v}"),
                primary: false,
            });
        }
    }
    if get("--jinja").is_some() {
        out.push(Tag {
            text: "jinja".into(),
            primary: false,
        });
    }
    if get("--chat-template-kwargs").is_some() {
        out.push(Tag {
            text: "kwargs".into(),
            primary: false,
        });
    }
    if get("--api-key").is_some() {
        out.push(Tag {
            text: "API key".into(),
            primary: false,
        });
    }
    out
}

/// `65536` → `64k`, `131072` → `128k`, `1000` → `1000`.
pub fn compact_tokens(value: &str) -> String {
    match value.parse::<u64>() {
        Ok(n) if n >= 1024 && n % 1024 == 0 => format!("{}k", n / 1024),
        _ => value.to_string(),
    }
}

/// Date relative façon messagerie : « à l'instant », « il y a 5 min »,
/// « hier », puis la date.
pub fn relative(when: DateTime<Local>, now: DateTime<Local>, fr: bool) -> String {
    let secs = (now - when).num_seconds().max(0);
    let mins = secs / 60;
    let hours = mins / 60;
    let days = (now.date_naive() - when.date_naive()).num_days();
    match (fr, secs, mins, hours, days) {
        (true, s, ..) if s < 60 => "à l'instant".into(),
        (false, s, ..) if s < 60 => "just now".into(),
        (true, _, m, ..) if m < 60 => format!("il y a {m} min"),
        (false, _, m, ..) if m < 60 => format!("{m} min ago"),
        (true, _, _, h, 0) => format!("il y a {h} h"),
        (false, _, _, h, 0) => format!("{h} h ago"),
        (true, .., 1) => format!("hier, {}", when.format("%H:%M")),
        (false, .., 1) => format!("yesterday, {}", when.format("%H:%M")),
        (true, .., d) if d < 7 => format!("il y a {d} jours"),
        (false, .., d) if d < 7 => format!("{d} days ago"),
        _ => when.format("%d/%m/%Y").to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn profile(pairs: &[(&str, &str)]) -> LaunchProfile {
        LaunchProfile {
            parameters: pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn context_tag_comes_first_and_is_primary() {
        let t = tags(&profile(&[
            ("-ngl", "999"),
            ("--ctx-size", "65536"),
            ("--cache-type-k", "q8_0"),
            ("--cache-type-v", "q8_0"),
            ("--flash-attn", "on"),
        ]));
        assert_eq!(
            t[0],
            Tag {
                text: "ctx 64k".into(),
                primary: true
            }
        );
        let texts: Vec<&str> = t.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(texts, ["ctx 64k", "ngl 999", "KV q8_0", "FA"]);
    }

    #[test]
    fn mixed_cache_types_and_defaults_are_summarised() {
        let t = tags(&profile(&[
            ("--cache-type-k", "q8_0"),
            ("--cache-type-v", "f16"),
            ("--parallel", "1"),
            ("--flash-attn", "off"),
        ]));
        let texts: Vec<&str> = t.iter().map(|t| t.text.as_str()).collect();
        // parallel 1 et flash-attn off sont les réglages de base : pas de tag.
        assert_eq!(texts, ["KV q8_0/f16"]);
    }

    #[test]
    fn reasoning_speculation_and_sampling_are_tagged() {
        let t = tags(&profile(&[
            ("--spec-type", "draft-mtp"),
            ("--spec-draft-n-max", "2"),
            ("--cache-type-k-draft", "q8_0"),
            ("--cache-type-v-draft", "q8_0"),
            ("--reasoning-effort", "high"),
            ("--mmproj", "mmproj.gguf"),
            ("--top-k", "40"),
            ("--api-key", "secret"),
        ]));
        let texts: Vec<&str> = t.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(
            texts,
            ["spec draft-mtp ×2", "draft KV q8_0", "reasoning high", "vision", "top-k 40", "API key"]
        );
    }

    #[test]
    fn compact_tokens_rounds_only_exact_multiples() {
        assert_eq!(compact_tokens("65536"), "64k");
        assert_eq!(compact_tokens("131072"), "128k");
        assert_eq!(compact_tokens("1000"), "1000");
        assert_eq!(compact_tokens("abc"), "abc");
    }

    #[test]
    fn relative_dates_read_like_a_chat_list() {
        let now = Local.with_ymd_and_hms(2026, 9, 26, 15, 0, 0).unwrap();
        let ago = |s: i64| now - chrono::Duration::seconds(s);
        assert_eq!(relative(ago(10), now, true), "à l'instant");
        assert_eq!(relative(ago(5 * 60), now, true), "il y a 5 min");
        assert_eq!(relative(ago(3 * 3600), now, true), "il y a 3 h");
        let yesterday = Local.with_ymd_and_hms(2026, 9, 25, 22, 30, 0).unwrap();
        assert_eq!(relative(yesterday, now, true), "hier, 22:30");
        let last_week = Local.with_ymd_and_hms(2026, 9, 22, 9, 0, 0).unwrap();
        assert_eq!(relative(last_week, now, true), "il y a 4 jours");
        let old = Local.with_ymd_and_hms(2026, 8, 1, 9, 0, 0).unwrap();
        assert_eq!(relative(old, now, true), "01/08/2026");
    }

    #[test]
    fn old_profiles_without_new_fields_still_parse() {
        let json = r#"{"name":"ancien","parameters":{"-ngl":"999"},"host":"","port":""}"#;
        let p: LaunchProfile = serde_json::from_str(json).unwrap();
        assert_eq!(p.name, "ancien");
        assert!(p.model.is_empty() && p.updated.is_empty());
        assert_eq!(p.model_stem(), None);
    }
}

