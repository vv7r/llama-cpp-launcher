//! Statistiques par requête, relevées dans les logs de llama-server.
//!
//! Pour chaque tâche, llama-server écrit (b11235) :
//!
//! ```text
//! slot print_timing: id  3 | task 201 | prompt eval time =  11.88 ms /  23 tokens (… 1935.38 tokens per second)
//! slot print_timing: id  3 | task 201 |        eval time = 336.77 ms / 200 tokens (…  590.91 tokens per second)
//! slot print_timing: id  3 | task 201 |       total time = 348.65 ms / 223 tokens
//! slot      release: id  3 | task 201 | stop processing: n_tokens = 246, truncated = 0
//! ```
//!
//! et, avec le décodage spéculatif, `draft acceptance = 0.56295 ( 2835
//! accepted / 5036 generated)`. La requête est close à « total time » ; les
//! lignes suivantes de la même tâche (contexte, brouillon) la complètent.

/// Une requête terminée.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Request {
    pub task: i64,
    pub prompt_tokens: u64,
    /// Durée de lecture du prompt (prefill), en ms.
    pub prompt_ms: f64,
    /// Débit de lecture du prompt (tokens/s).
    pub prompt_tps: Option<f64>,
    pub gen_tokens: u64,
    /// Durée de génération, en ms.
    pub gen_ms: f64,
    /// Débit de génération (tokens/s).
    pub gen_tps: Option<f64>,
    /// Tokens dans le contexte en fin de requête (`n_tokens`).
    pub context: Option<u64>,
    /// Brouillon : tokens acceptés, tokens proposés.
    pub draft: Option<(u64, u64)>,
}

/// Minimum, moyenne pondérée, médiane, maximum et dernière valeur d'une
/// série de débits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spread {
    pub min: f64,
    /// Total des tokens divisé par le temps total : une requête de quelques
    /// tokens (prompt déjà en cache) pèse selon ses tokens, pas comme une
    /// requête entière. La moyenne simple des débits la surestimait.
    pub weighted: f64,
    pub median: f64,
    pub max: f64,
    pub last: f64,
}

impl Spread {
    /// `samples` : (débit, tokens, durée en ms) de chaque requête.
    fn of(samples: &[(f64, u64, f64)]) -> Option<Spread> {
        let last = samples.last()?.0;
        let values: Vec<f64> = samples.iter().map(|s| s.0).collect();
        let tokens: u64 = samples.iter().map(|s| s.1).sum();
        let ms: f64 = samples.iter().map(|s| s.2).sum();
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let n = sorted.len();
        let median = if n % 2 == 1 {
            sorted[n / 2]
        } else {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        };
        Some(Spread {
            min: sorted[0],
            weighted: if ms > 0.0 {
                tokens as f64 * 1000.0 / ms
            } else {
                sorted.iter().sum::<f64>() / n as f64
            },
            median,
            max: sorted[n - 1],
            last,
        })
    }
}

#[derive(Default)]
pub struct TokenStats {
    /// Requête en cours de relevé (avant « total time »).
    current: Option<Request>,
    pub requests: Vec<Request>,
}

/// Requêtes gardées : au-delà, les plus anciennes sont oubliées.
const KEPT: usize = 5000;

impl TokenStats {
    /// Lit une ligne de log ; vrai si elle a apporté une information.
    pub fn observe(&mut self, line: &str) -> bool {
        let task = task_id(line);
        if let Some(rest) = after(line, "draft acceptance") {
            let (Some(a), Some(g)) = (before_word(rest, "accepted"), before_word(rest, "generated"))
            else {
                return false;
            };
            // Sans numéro de tâche : la requête la plus récente.
            let target = match task {
                Some(t) => self.find(t),
                None => self.current.as_mut().or(self.requests.last_mut()),
            };
            return target.map(|r| r.draft = Some((a, g))).is_some();
        }
        let Some(task) = task else {
            return false;
        };
        if let Some(rest) = after(line, "prompt eval time =") {
            let r = self.open(task);
            r.prompt_tokens = tokens(rest).unwrap_or(0);
            r.prompt_ms = leading_number(rest).unwrap_or(0.0);
            r.prompt_tps = tps(rest);
            true
        } else if let Some(rest) = after(line, "eval time =") {
            let r = self.open(task);
            r.gen_tokens = tokens(rest).unwrap_or(0);
            r.gen_ms = leading_number(rest).unwrap_or(0.0);
            r.gen_tps = tps(rest);
            true
        } else if after(line, "total time =").is_some() {
            if let Some(r) = self.current.take() {
                self.requests.push(r);
                if self.requests.len() > KEPT {
                    self.requests.remove(0);
                }
            }
            true
        } else if let Some(rest) = after(line, "n_tokens =") {
            let n = leading_number(rest).map(|n| n as u64);
            self.find(task).map(|r| r.context = n).is_some()
        } else {
            false
        }
    }

    pub fn clear(&mut self) {
        *self = TokenStats::default();
    }

    fn open(&mut self, task: i64) -> &mut Request {
        if self.current.as_ref().is_none_or(|r| r.task != task) {
            // Une requête laissée sans « total time » est gardée telle quelle.
            if let Some(r) = self.current.take() {
                self.requests.push(r);
            }
            self.current = Some(Request {
                task,
                ..Default::default()
            });
        }
        self.current.as_mut().unwrap()
    }

    /// La requête de cette tâche : en cours, ou la plus récente terminée.
    fn find(&mut self, task: i64) -> Option<&mut Request> {
        if self.current.as_ref().is_some_and(|r| r.task == task) {
            return self.current.as_mut();
        }
        self.requests.iter_mut().rev().find(|r| r.task == task)
    }

    pub fn prompt(&self) -> Option<Spread> {
        Spread::of(
            &self
                .requests
                .iter()
                .filter_map(|r| Some((r.prompt_tps?, r.prompt_tokens, r.prompt_ms)))
                .collect::<Vec<_>>(),
        )
    }

    pub fn generation(&self) -> Option<Spread> {
        Spread::of(
            &self
                .requests
                .iter()
                .filter_map(|r| Some((r.gen_tps?, r.gen_tokens, r.gen_ms)))
                .collect::<Vec<_>>(),
        )
    }

    pub fn prompt_tokens(&self) -> u64 {
        self.requests.iter().map(|r| r.prompt_tokens).sum()
    }

    pub fn gen_tokens(&self) -> u64 {
        self.requests.iter().map(|r| r.gen_tokens).sum()
    }

    /// Brouillon cumulé : acceptés, proposés.
    pub fn draft(&self) -> Option<(u64, u64)> {
        let (a, g) = self
            .requests
            .iter()
            .filter_map(|r| r.draft)
            .fold((0, 0), |(a, g), (x, y)| (a + x, g + y));
        (g > 0).then_some((a, g))
    }

    /// Points (contexte, débit) des requêtes au contexte connu, avec leur
    /// requête (pour l'infobulle), triés par contexte.
    pub fn by_context(&self, pick: impl Fn(&Request) -> Option<f64>) -> Vec<([f64; 2], &Request)> {
        let mut points: Vec<([f64; 2], &Request)> = self
            .requests
            .iter()
            .filter_map(|r| Some(([r.context? as f64, pick(r)?], r)))
            .collect();
        points.sort_by(|a, b| a.0[0].total_cmp(&b.0[0]));
        points
    }
}

/// `… | task 201 | …` → 201. Les lignes sans tâche (`task -1`) sont ignorées.
fn task_id(line: &str) -> Option<i64> {
    let rest = after(line, "| task ")?;
    let id = leading_number(rest)? as i64;
    (id >= 0).then_some(id)
}

fn after<'a>(line: &'a str, marker: &str) -> Option<&'a str> {
    line.find(marker).map(|i| &line[i + marker.len()..])
}

/// Premier nombre (éventuellement décimal) en tête, espaces ignorés.
fn leading_number(s: &str) -> Option<f64> {
    let s = s.trim_start();
    let end = s
        .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'))
        .unwrap_or(s.len());
    s[..end].parse().ok()
}

/// `59.57 ms /    38 tokens (…)` → 38.
fn tokens(rest: &str) -> Option<u64> {
    leading_number(after(rest, "/")?).map(|n| n as u64)
}

/// `(… 637.89 tokens per second)` → 637.89.
fn tps(rest: &str) -> Option<f64> {
    let i = rest.find("tokens per second")?;
    let head = rest[..i].trim_end();
    let start = head.rfind([' ', ',', '('])? + 1;
    head[start..].parse().ok()
}

/// `( 2835 accepted / 5036 generated)` : le nombre juste avant `word`.
fn before_word(rest: &str, word: &str) -> Option<u64> {
    let i = rest.find(word)?;
    let head = rest[..i].trim_end();
    let start = head.rfind(|c: char| !c.is_ascii_digit()).map_or(0, |j| j + 1);
    head[start..].parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Extrait réel d'un llama-server b11235 (deux requêtes).
    const LOG: &str = "\
0.00.398.025 I slot get_availabl: id  3 | task -1 | selected slot by LRU, t_last = -1
0.00.398.278 I slot launch_slot_: id  3 | task 0 | processing task, is_child = 0
0.00.774.891 I slot print_timing: id  3 | task 0 | prompt eval time =      59.57 ms /    38 tokens (    1.57 ms per token,   637.89 tokens per second)
0.00.774.896 I slot print_timing: id  3 | task 0 |        eval time =     317.02 ms /   200 tokens (    1.59 ms per token,   627.72 tokens per second)
0.00.774.897 I slot print_timing: id  3 | task 0 |       total time =     376.59 ms /   238 tokens
0.00.774.898 I slot print_timing: id  3 | task 0 |    graphs reused =        199
0.00.774.932 I slot      release: id  3 | task 0 | stop processing: n_tokens = 237, truncated = 0
0.00.804.161 I slot launch_slot_: id  3 | task 201 | processing task, is_child = 0
0.01.152.833 I slot print_timing: id  3 | task 201 | prompt eval time =      11.88 ms /    23 tokens (    0.52 ms per token,  1935.38 tokens per second)
0.01.152.837 I slot print_timing: id  3 | task 201 |        eval time =     336.77 ms /   200 tokens (    1.69 ms per token,   590.91 tokens per second)
0.01.152.838 I slot print_timing: id  3 | task 201 |       total time =     348.65 ms /   223 tokens
0.01.152.838 I slot print_timing: id  3 | task 201 |    graphs reused =        397
0.01.152.854 I slot      release: id  3 | task 201 | stop processing: n_tokens = 246, truncated = 0
";

    #[test]
    fn reads_real_server_log() {
        let mut s = TokenStats::default();
        for line in LOG.lines() {
            s.observe(line);
        }
        assert_eq!(s.requests.len(), 2);
        assert_eq!(
            s.requests[0],
            Request {
                task: 0,
                prompt_tokens: 38,
                prompt_ms: 59.57,
                prompt_tps: Some(637.89),
                gen_tokens: 200,
                gen_ms: 317.02,
                gen_tps: Some(627.72),
                context: Some(237),
                draft: None,
            }
        );
        assert_eq!(s.requests[1].context, Some(246));
        assert_eq!((s.prompt_tokens(), s.gen_tokens()), (61, 400));

        let tg = s.generation().unwrap();
        assert_eq!((tg.min, tg.max, tg.last), (590.91, 627.72, 590.91));
        // Pondérée : 400 tokens en 317.02 + 336.77 ms.
        assert!((tg.weighted - 400_000.0 / (317.02 + 336.77)).abs() < 1e-9);
        assert!((tg.median - (627.72 + 590.91) / 2.0).abs() < 1e-9);
        let points: Vec<[f64; 2]> = s.by_context(|r| r.gen_tps).iter().map(|p| p.0).collect();
        assert_eq!(points, vec![[237.0, 627.72], [246.0, 590.91]]);

        // PP : la requête de 23 tokens (1935 t/s) et celle de 38 (637 t/s)
        // pèsent selon leurs tokens et leur durée.
        let pp = s.prompt().unwrap();
        assert!((pp.weighted - 61_000.0 / (59.57 + 11.88)).abs() < 1e-9);

        s.clear();
        assert!(s.requests.is_empty() && s.generation().is_none());
    }

    #[test]
    fn draft_acceptance_is_attached_to_its_task() {
        let mut s = TokenStats::default();
        for line in [
            "slot print_timing: id  0 | task 7 | prompt eval time = 10.00 ms / 100 tokens (0.10 ms per token, 10000.00 tokens per second)",
            "slot print_timing: id  0 | task 7 |        eval time = 1000.00 ms / 50 tokens (20.00 ms per token, 50.00 tokens per second)",
            "slot print_timing: id  0 | task 7 |       total time = 1010.00 ms / 150 tokens",
            "slot print_timing: id  0 | task 7 | draft acceptance = 0.56295 ( 2835 accepted / 5036 generated)",
        ] {
            assert!(s.observe(line), "{line}");
        }
        assert_eq!(s.requests[0].draft, Some((2835, 5036)));
        assert_eq!(s.draft(), Some((2835, 5036)));
        // Sans contexte connu, la requête n'apparaît pas sur le graphique.
        assert!(s.by_context(|r| r.gen_tps).is_empty());
        // Ligne sans numéro de tâche : rattachée à la dernière requête.
        assert!(s.observe("draft acceptance = 0.50000 ( 10 accepted / 20 generated)"));
        assert_eq!(s.requests[0].draft, Some((10, 20)));
    }

    #[test]
    fn median_of_odd_series() {
        let s = Spread::of(&[(3.0, 3, 1000.0), (1.0, 1, 1000.0), (2.0, 2, 1000.0)]).unwrap();
        assert_eq!((s.min, s.median, s.max, s.last), (1.0, 2.0, 3.0, 2.0));
        // 6 tokens en 3 s.
        assert_eq!(s.weighted, 2.0);
        // Une petite requête rapide pèse peu : 10 tokens à 1000 t/s contre
        // 1000 tokens à 100 t/s donnent ~101 t/s, pas une moyenne de 550.
        let s = Spread::of(&[(1000.0, 10, 10.0), (100.0, 1000, 10_000.0)]).unwrap();
        assert!((s.weighted - 1010.0 / 10.01).abs() < 1e-9);
        assert!(Spread::of(&[]).is_none());
    }
}
