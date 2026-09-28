//! Import d'une ligne de commande `llama-server` existante.
//!
//! Le texte collé est découpé en jetons (les continuations `^`, `` ` `` et
//! `\` sont ignorées, les guillemets respectés), puis chaque option est
//! rapportée à son paramètre via le catalogue (nom, alias ou forme
//! négative). Une valeur absente des listes préenregistrées est ajoutée à
//! la liste du paramètre concerné.
//!
//! Une option absente du catalogue n'est jamais perdue : elle est conservée
//! telle quelle et transmise au serveur, pour que la commande lancée soit
//! celle qui a été collée.

use crate::params::{self, Kind, Resolved};
use std::collections::BTreeMap;

#[derive(Default)]
pub struct Imported {
    /// Paramètres du catalogue, sous leur nom canonique ; une chaîne vide
    /// désactive (forme négative comme `--no-jinja`).
    pub params: BTreeMap<String, String>,
    pub host: Option<String>,
    pub port: Option<String>,
    pub model: Option<String>,
    /// Options hors catalogue, chacune avec ses jetons (option + valeur).
    pub unknown: Vec<Vec<String>>,
}

/// `-x`, `--xx` ; pas un nombre négatif (`-1`, `-.5`).
fn is_option(token: &str) -> bool {
    token.starts_with('-')
        && token.len() > 1
        && !token[1..].starts_with(|c: char| c.is_ascii_digit() || c == '.')
}

pub fn parse(input: &str) -> Imported {
    let tokens = tokenize(input);
    let mut out = Imported::default();

    let mut i = 0;
    while i < tokens.len() {
        let token = &tokens[i];
        if !is_option(token) {
            i += 1;
            continue;
        }

        // Forme `--option=valeur`.
        let (opt, inline_value) = match token.split_once('=') {
            Some((o, v)) => (o.to_string(), Some(v.to_string())),
            None => (token.clone(), None),
        };

        let next_is_value = inline_value.is_none()
            && i + 1 < tokens.len()
            && !is_option(&tokens[i + 1]);

        let had_inline = inline_value.is_some();
        let value = match inline_value {
            Some(v) => Some(v),
            None if next_is_value => Some(tokens[i + 1].clone()),
            None => None,
        };
        let mut consumed = if had_inline || !next_is_value { 1 } else { 2 };

        match opt.as_str() {
            "-m" | "--model" => out.model = value,
            "--host" => out.host = value,
            "--port" => out.port = value,
            _ => match params::resolve(&opt) {
                Some(Resolved::Param(idx)) => {
                    let def = params::def(idx);
                    let v = match def.kind {
                        // Un drapeau n'a jamais de valeur : on rend le jeton
                        // suivant au parseur s'il en a pris un.
                        Kind::Flag => {
                            consumed = 1;
                            "on".to_string()
                        }
                        Kind::Choice => value.clone().unwrap_or_else(|| "on".to_string()),
                    };
                    out.params.insert(def.name.to_string(), v);
                }
                Some(Resolved::Off(idx)) => {
                    consumed = 1;
                    out.params.insert(params::def(idx).name.to_string(), String::new());
                }
                None => {
                    let mut group = vec![opt.clone()];
                    group.extend(value.clone());
                    out.unknown.push(group);
                }
            },
        }

        i += consumed;
    }

    out
}

/// Découpe en jetons façon shell, en ignorant les continuations de ligne.
pub fn tokenize(input: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut has_token = false;
    let mut quote: Option<char> = None;
    let mut chars = input.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            '\\' if quote != Some('\'') => {
                // Séquence d'échappement, ou continuation en fin de ligne.
                // Hors guillemets, `\\` reste tel quel : c'est le début d'un
                // chemin réseau Windows (`\\serveur\partage`), pas un échappement.
                match chars.peek() {
                    Some('\n') | Some('\r') => {
                        chars.next();
                    }
                    Some(&next) if next == '"' || (next == '\\' && quote.is_some()) => {
                        current.push(next);
                        has_token = true;
                        chars.next();
                    }
                    _ => {
                        current.push('\\');
                        has_token = true;
                    }
                }
            }
            '^' | '`' if quote.is_none() => {
                // Continuation cmd.exe / PowerShell : ignorée.
                if matches!(chars.peek(), Some('\n') | Some('\r')) {
                    chars.next();
                } else if chars.peek().is_none() {
                    // fin de saisie
                } else {
                    current.push(c);
                    has_token = true;
                }
            }
            '"' | '\'' => match quote {
                Some(q) if q == c => quote = None,
                Some(_) => {
                    current.push(c);
                    has_token = true;
                }
                None => {
                    quote = Some(c);
                    has_token = true;
                }
            },
            c if c.is_whitespace() && quote.is_none() => {
                if has_token {
                    tokens.push(std::mem::take(&mut current));
                    has_token = false;
                }
            }
            c => {
                current.push(c);
                has_token = true;
            }
        }
    }
    if has_token {
        tokens.push(current);
    }

    // Le premier jeton est souvent l'exécutable : on l'écarte.
    if let Some(first) = tokens.first() {
        let lower = first.to_lowercase();
        if lower.contains("llama-server") || lower.contains("llama_server") {
            tokens.remove(0);
        }
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_readme_example() {
        let cmd = r#"llama-server.exe ^
-ngl 999 ^
--parallel 1 ^
--ctx-size 65536 ^
--flash-attn on ^
--cache-type-k q8_0 ^
--batch-size 2048 ^
--top-k 40 ^
--jinja ^
--chat-template-kwargs "{\"preserve_thinking\": true}""#;
        let got = parse(cmd);
        assert_eq!(got.params.get("-ngl").map(String::as_str), Some("999"));
        assert_eq!(got.params.get("--ctx-size").map(String::as_str), Some("65536"));
        assert_eq!(got.params.get("--flash-attn").map(String::as_str), Some("on"));
        assert_eq!(got.params.get("--jinja").map(String::as_str), Some("on"));
        assert_eq!(
            got.params.get("--chat-template-kwargs").map(String::as_str),
            Some(r#"{"preserve_thinking": true}"#)
        );
    }

    #[test]
    fn flag_does_not_swallow_following_positional() {
        let got = parse("--jinja --temp 0.7");
        assert_eq!(got.params.get("--jinja").map(String::as_str), Some("on"));
        assert_eq!(got.params.get("--temp").map(String::as_str), Some("0.7"));
    }

    #[test]
    fn handles_aliases_equals_and_host_port() {
        let got = parse("-fa on -c 8192 --host=0.0.0.0 --port 1234 -m D:/models/a.gguf -ngl 40");
        assert_eq!(got.params.get("--ctx-size").map(String::as_str), Some("8192"));
        assert_eq!(got.params.get("--flash-attn").map(String::as_str), Some("on"));
        assert_eq!(got.host.as_deref(), Some("0.0.0.0"));
        assert_eq!(got.port.as_deref(), Some("1234"));
        assert_eq!(got.model.as_deref(), Some("D:/models/a.gguf"));
        assert_eq!(got.params.get("-ngl").map(String::as_str), Some("40"));
    }

    #[test]
    fn negative_numbers_are_values_not_options() {
        let got = parse("--temp -1");
        assert_eq!(got.params.get("--temp").map(String::as_str), Some("-1"));
        let got = parse("--temp -.5");
        assert_eq!(got.params.get("--temp").map(String::as_str), Some("-.5"));
    }

    #[test]
    fn network_paths_keep_their_backslashes() {
        let got = parse(r"-m \\nas\models\a.gguf -mm \\nas\models\mmproj.gguf");
        assert_eq!(got.model.as_deref(), Some(r"\\nas\models\a.gguf"));
        assert_eq!(
            got.params.get("--mmproj").map(String::as_str),
            Some(r"\\nas\models\mmproj.gguf")
        );
    }

    /// Les huit options d'une vraie commande qui étaient ignorées à l'import.
    #[test]
    fn user_command_imports_without_losing_anything() {
        let cmd = r#"llama-server.exe -m C:\models\Qwen3.8-27B.gguf ^
  -mm C:\models\mmproj-Qwen3.8-27B.gguf ^
  --no-mmproj-offload ^
  --api-key secret-key-42 ^
  -n 98304 ^
  --sleep-idle-seconds 300 ^
  -ctkd q8_0 ^
  -ctvd q8_0 ^
  --reasoning-effort low"#;
        let got = parse(cmd);
        assert!(got.unknown.is_empty(), "options perdues : {:?}", got.unknown);
        let get = |k: &str| got.params.get(k).map(String::as_str);
        assert_eq!(get("--mmproj"), Some(r"C:\models\mmproj-Qwen3.8-27B.gguf"));
        assert_eq!(get("--no-mmproj-offload"), Some("on"));
        assert_eq!(get("--api-key"), Some("secret-key-42"));
        assert_eq!(get("--n-predict"), Some("98304"));
        assert_eq!(get("--sleep-idle-seconds"), Some("300"));
        assert_eq!(get("--cache-type-k-draft"), Some("q8_0"));
        assert_eq!(get("--cache-type-v-draft"), Some("q8_0"));
        assert_eq!(get("--reasoning-effort"), Some("low"));
    }

    #[test]
    fn spec_draft_p_min_and_its_alias() {
        for cmd in ["--spec-draft-p-min 0.75", "--draft-p-min 0.75"] {
            let got = parse(cmd);
            assert!(got.unknown.is_empty(), "{cmd}");
            assert_eq!(got.params.get("--spec-draft-p-min").map(String::as_str), Some("0.75"));
        }
    }

    #[test]
    fn negative_forms_disable_flags() {
        let got = parse("--no-jinja --mmap --mmproj-offload");
        assert!(got.unknown.is_empty());
        for flag in ["--jinja", "--no-mmap", "--no-mmproj-offload"] {
            assert_eq!(got.params.get(flag).map(String::as_str), Some(""), "{flag}");
        }
    }

    #[test]
    fn unknown_options_are_kept_with_their_value() {
        let got = parse("--option-future 12 --drapeau-futur -ngl 999");
        assert_eq!(
            got.unknown,
            vec![
                vec!["--option-future".to_string(), "12".to_string()],
                vec!["--drapeau-futur".to_string()],
            ]
        );
        assert_eq!(got.params.get("-ngl").map(String::as_str), Some("999"));
    }
}
