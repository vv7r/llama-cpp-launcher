//! Catalogue des paramètres de `llama-server`.
//!
//! Différence majeure avec le projet d'origine : aucune saisie libre.
//! Chaque paramètre expose une liste de valeurs préenregistrées et
//! l'utilisateur choisit dans cette liste (ou « désactivé »).
//!
//! Les noms, alias et valeurs admises suivent `llama-server --help` ; le
//! test ignoré `catalog_matches_llama_server_help` les y confronte.

use std::collections::BTreeMap;

/// Valeur spéciale signifiant « paramètre non passé à llama-server ».
pub const DISABLED: &str = "";

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Liste déroulante de valeurs préenregistrées : `--nom valeur`
    Choice,
    /// Case à cocher : `--nom` seul, sans valeur.
    Flag,
}

pub struct ParamDef {
    pub name: &'static str,
    pub label: &'static str,
    pub kind: Kind,
    /// Valeurs proposées dans la liste déroulante.
    pub presets: &'static [&'static str],
    /// Valeur sélectionnée au premier lancement ("" = désactivé).
    pub default: &'static str,
    pub help_fr: &'static str,
    pub help_en: &'static str,
    /// Regroupement dans l'UI.
    pub group: Group,
    /// Autres orthographes reconnues à l'import (`-c` pour `--ctx-size`).
    pub aliases: &'static [&'static str],
    /// Pour un drapeau : formes qui le désactivent explicitement
    /// (`--mmap` pour `--no-mmap`).
    pub off_aliases: &'static [&'static str],
    /// Valeur masquée à l'écran (clé API).
    pub secret: bool,
}

/// Valeurs communes, complétées par chaque entrée du catalogue.
const BASE: ParamDef = ParamDef {
    name: "",
    label: "",
    kind: Kind::Choice,
    presets: &[],
    default: DISABLED,
    help_fr: "",
    help_en: "",
    group: Group::Performance,
    aliases: &[],
    off_aliases: &[],
    secret: false,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Group {
    Performance,
    Memory,
    Sampling,
    Speculation,
    Template,
    Multimodal,
    Server,
}

impl Group {
    /// Groupes du panneau « Paramètres de lancement ». Multimodal et Serveur
    /// sont dans la colonne de gauche, avec le modèle et l'adresse qu'ils
    /// complètent.
    pub const CENTER: [Group; 5] = [
        Group::Performance,
        Group::Memory,
        Group::Sampling,
        Group::Speculation,
        Group::Template,
    ];

    pub fn title(self, fr: bool) -> &'static str {
        match (self, fr) {
            (Group::Performance, _) => "Performance",
            (Group::Memory, true) => "Mémoire / Cache",
            (Group::Memory, false) => "Memory / Cache",
            (Group::Sampling, true) => "Échantillonnage",
            (Group::Sampling, false) => "Sampling",
            (Group::Speculation, true) => "Spéculation",
            (Group::Speculation, false) => "Speculation",
            (Group::Template, true) => "Template de chat",
            (Group::Template, false) => "Chat template",
            (Group::Multimodal, _) => "Multimodal",
            (Group::Server, true) => "Serveur",
            (Group::Server, false) => "Server",
        }
    }

    /// Infobulle du groupe : à quoi servent ces options.
    pub fn help(self, fr: bool) -> &'static str {
        match (self, fr) {
            (Group::Performance, true) => {
                "Répartition du calcul entre GPU et CPU, et taille des lots. \
                 C'est ici que se joue la vitesse."
            }
            (Group::Performance, false) => {
                "How the work is split between GPU and CPU, and batch sizes. \
                 This is where speed is decided."
            }
            (Group::Memory, true) => {
                "Longueur du contexte et quantisation du cache KV : le principal \
                 poste de consommation de VRAM après le modèle lui-même."
            }
            (Group::Memory, false) => {
                "Context length and KV cache quantization: the biggest VRAM consumer \
                 after the model itself."
            }
            (Group::Sampling, true) => {
                "Valeurs par défaut du serveur pour le tirage des tokens. Un client \
                 API peut les redéfinir requête par requête."
            }
            (Group::Sampling, false) => {
                "Server defaults for token sampling. An API client can override them \
                 per request."
            }
            (Group::Speculation, true) => {
                "Décodage spéculatif : un brouillon rapide est vérifié par le modèle \
                 principal. Accélère la génération quand le modèle le supporte."
            }
            (Group::Speculation, false) => {
                "Speculative decoding: a fast draft is verified by the main model. \
                 Speeds up generation when the model supports it."
            }
            (Group::Template, true) => {
                "Mise en forme des conversations et du raisonnement avant l'envoi au modèle."
            }
            (Group::Template, false) => {
                "How conversations and reasoning are formatted before the model sees them."
            }
            (Group::Multimodal, true) => {
                "Vision : le projecteur (mmproj) permet au modèle de lire des images. \
                 Les fichiers mmproj du répertoire des modèles sont proposés."
            }
            (Group::Multimodal, false) => {
                "Vision: the projector (mmproj) lets the model read images. The mmproj \
                 files found in the models directory are offered."
            }
            (Group::Server, true) => {
                "Accès et comportement du serveur HTTP : authentification et mise en veille."
            }
            (Group::Server, false) => "HTTP server access and behaviour: authentication and sleep.",
        }
    }
}

/// Listes préenregistrées communes.
const NGL: &[&str] = &["0", "8", "16", "24", "32", "40", "48", "64", "80", "99", "999"];
const PARALLEL: &[&str] = &["-1", "1", "2", "3", "4", "6", "8", "12", "16", "32"];
/// 0 = contexte prévu par le modèle (valeur par défaut de llama-server).
/// Grille sans trou : paliers usuels jusqu'à 32768, puis tous les 4096
/// jusqu'à 131072, tous les 8192 jusqu'à 262144, tous les 32768 jusqu'à
/// 524288 et tous les 131072 jusqu'à 1048576.
const CTX: &[&str] = &[
    "0", "512", "1024", "2048", "3072", "4096", "6144", "8192", "10240", "12288", "14336", "16384",
    "20480", "24576", "28672", "32768", "36864", "40960", "45056", "49152", "53248", "57344",
    "61440", "65536", "69632", "73728", "77824", "81920", "86016", "90112", "94208", "98304",
    "102400", "106496", "110592", "114688", "118784", "122880", "126976", "131072", "139264",
    "147456", "155648", "163840", "172032", "180224", "188416", "196608", "204800", "212992",
    "221184", "229376", "237568", "245760", "253952", "262144", "294912", "327680", "360448",
    "393216", "425984", "458752", "491520", "524288", "655360", "786432", "917504", "1048576",
];
const ON_OFF_AUTO: &[&str] = &["on", "off", "auto"];
/// Types admis par `--cache-type-k/v` et leurs variantes « draft ».
pub const CACHE_TYPES: &[&str] = &[
    "f32", "f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl",
];
const BATCH: &[&str] = &["64", "128", "256", "512", "1024", "2048", "4096", "8192"];
const UBATCH: &[&str] = &["16", "32", "64", "128", "256", "512", "1024", "2048"];
const TEMP: &[&str] = &[
    "0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0", "1.2", "1.5", "2.0",
];
const TOP_P: &[&str] = &["0.5", "0.6", "0.7", "0.8", "0.85", "0.9", "0.95", "0.98", "1.0"];
const MIN_P: &[&str] = &["0", "0.01", "0.02", "0.05", "0.1", "0.15", "0.2"];
const TOP_K: &[&str] = &["0", "1", "10", "20", "40", "50", "64", "80", "100"];
const PENALTY: &[&str] = &[
    "0", "0.1", "0.2", "0.3", "0.5", "0.7", "1.0", "1.1", "1.2", "1.5",
];
const N_PREDICT: &[&str] = &[
    "-1", "256", "512", "1024", "2048", "4096", "8192", "16384", "32768", "49152", "65536",
    "98304", "131072",
];
const SPEC_TYPE: &[&str] = &[
    "none",
    "draft-simple",
    "draft-eagle3",
    "draft-mtp",
    "draft-dflash",
    "draft-dspark",
    "ngram-simple",
    "ngram-map-k",
    "ngram-map-k4v",
    "ngram-mod",
    "ngram-cache",
];
const SPEC_P_MIN: &[&str] = &[
    "0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.75", "0.8", "0.85", "0.9", "0.95",
];
const SPEC_N: &[&str] =&["1", "2", "3", "4", "5", "6", "8", "10", "16"];
const THREADS: &[&str] = &["1", "2", "4", "6", "8", "10", "12", "16", "24", "32"];
const SPLIT_MODE: &[&str] = &["none", "layer", "row", "tensor"];
const NUMA: &[&str] = &["distribute", "isolate", "numactl"];
const POOLING: &[&str] = &["none", "mean", "cls", "last", "rank"];
const KWARGS: &[&str] = &[
    "{\"preserve_thinking\": true}",
    "{\"preserve_thinking\": false}",
    "{\"enable_thinking\": true}",
    "{\"enable_thinking\": false}",
    "{\"thinking\": \"high\"}",
    "{\"thinking\": \"medium\"}",
    "{\"thinking\": \"low\"}",
];
const REASONING_FORMAT: &[&str] = &["auto", "none", "deepseek", "deepseek-legacy"];
const REASONING_EFFORT: &[&str] = &[
    "default", "minimal", "low", "medium", "high", "xhigh", "max",
];
const SLEEP_IDLE: &[&str] = &["60", "120", "300", "600", "900", "1800", "3600"];

/// Listes préenregistrées pour les champs de connexion (ex-inputs libres).
pub const HOSTS: &[&str] = &["127.0.0.1", "0.0.0.0", "localhost", "::1"];
pub const PORTS: &[&str] = &[
    "8080", "8081", "8000", "8888", "5000", "1234", "11434", "3000",
];

pub static PARAMS: &[ParamDef] = &[
    // ---------------- Performance ----------------
    ParamDef {
        name: "-ngl",
        label: "GPU Layers",
        presets: NGL,
        default: "999",
        help_fr: "Nombre de couches déchargées sur le GPU (999 = tout).",
        help_en: "Layers offloaded to the GPU (999 = all).",
        aliases: &["--gpu-layers", "--n-gpu-layers"],
        ..BASE
    },
    ParamDef {
        name: "--threads",
        label: "Threads",
        presets: THREADS,
        help_fr: "Threads CPU utilisés pour la génération.",
        help_en: "CPU threads used for generation.",
        aliases: &["-t"],
        ..BASE
    },
    ParamDef {
        name: "--parallel",
        label: "Parallel",
        presets: PARALLEL,
        default: "1",
        help_fr: "Nombre de requêtes traitées simultanément (-1 = automatique).",
        help_en: "Number of concurrent requests (-1 = automatic).",
        aliases: &["-np"],
        ..BASE
    },
    ParamDef {
        name: "--batch-size",
        label: "Batch Size",
        presets: BATCH,
        default: "2048",
        help_fr: "Taille du lot logique (prompt processing).",
        help_en: "Logical batch size (prompt processing).",
        aliases: &["-b"],
        ..BASE
    },
    ParamDef {
        name: "--ubatch-size",
        label: "Ubatch Size",
        presets: UBATCH,
        default: "512",
        help_fr: "Taille du micro-lot physique.",
        help_en: "Physical micro-batch size.",
        aliases: &["-ub"],
        ..BASE
    },
    ParamDef {
        name: "--split-mode",
        label: "Split Mode",
        presets: SPLIT_MODE,
        help_fr: "Répartition du modèle sur plusieurs GPU (tensor : expérimental).",
        help_en: "How to split the model across GPUs (tensor: experimental).",
        aliases: &["-sm"],
        ..BASE
    },
    ParamDef {
        name: "--numa",
        label: "NUMA",
        presets: NUMA,
        help_fr: "Stratégie d'allocation NUMA.",
        help_en: "NUMA allocation strategy.",
        ..BASE
    },
    // ---------------- Mémoire / cache ----------------
    ParamDef {
        name: "--ctx-size",
        label: "Context Size",
        presets: CTX,
        // Désactivé : llama-server prend alors le contexte prévu par le
        // modèle (sa valeur par défaut, 0).
        help_fr: "Taille de la fenêtre de contexte, en tokens. Désactivé ou 0 : celle prévue par le modèle.",
        help_en: "Context window size, in tokens. Disabled or 0: the one the model was built for.",
        group: Group::Memory,
        aliases: &["-c"],
        ..BASE
    },
    ParamDef {
        name: "--flash-attn",
        label: "Flash Attention",
        presets: ON_OFF_AUTO,
        default: "on",
        help_fr: "Noyaux d'attention optimisés (obligatoire pour un cache quantisé).",
        help_en: "Optimized attention kernels (required for a quantized cache).",
        group: Group::Memory,
        aliases: &["-fa"],
        ..BASE
    },
    ParamDef {
        name: "--cache-type-k",
        label: "Cache Type K",
        presets: CACHE_TYPES,
        default: "q8_0",
        help_fr: "Quantisation du cache des clés (KV).",
        help_en: "Quantization of the K cache.",
        group: Group::Memory,
        aliases: &["-ctk"],
        ..BASE
    },
    ParamDef {
        name: "--cache-type-v",
        label: "Cache Type V",
        presets: CACHE_TYPES,
        default: "q8_0",
        help_fr: "Quantisation du cache des valeurs (KV).",
        help_en: "Quantization of the V cache.",
        group: Group::Memory,
        aliases: &["-ctv"],
        ..BASE
    },
    ParamDef {
        name: "--no-mmap",
        label: "No mmap",
        kind: Kind::Flag,
        help_fr: "Charge le modèle en RAM au lieu de le mapper.",
        help_en: "Load the model into RAM instead of memory-mapping it.",
        group: Group::Memory,
        off_aliases: &["--mmap"],
        ..BASE
    },
    ParamDef {
        name: "--mlock",
        label: "mlock",
        kind: Kind::Flag,
        help_fr: "Verrouille le modèle en RAM (empêche le swap).",
        help_en: "Lock the model in RAM (prevents swapping).",
        group: Group::Memory,
        ..BASE
    },
    // ---------------- Échantillonnage ----------------
    ParamDef {
        name: "--temp",
        label: "Temperature",
        presets: TEMP,
        default: "0.8",
        help_fr: "0 = déterministe, plus haut = plus créatif.",
        help_en: "0 = deterministic, higher = more creative.",
        group: Group::Sampling,
        aliases: &["--temperature"],
        ..BASE
    },
    ParamDef {
        name: "--top-p",
        label: "Top-p",
        presets: TOP_P,
        default: "0.95",
        help_fr: "Échantillonnage par noyau (nucleus).",
        help_en: "Nucleus sampling.",
        group: Group::Sampling,
        ..BASE
    },
    ParamDef {
        name: "--top-k",
        label: "Top-k",
        presets: TOP_K,
        default: "40",
        help_fr: "Limite aux k tokens les plus probables (0 = désactivé).",
        help_en: "Limit to the k most likely tokens (0 = off).",
        group: Group::Sampling,
        ..BASE
    },
    ParamDef {
        name: "--min-p",
        label: "Min-p",
        presets: MIN_P,
        help_fr: "Probabilité minimale relative au meilleur token.",
        help_en: "Minimum probability relative to the top token.",
        group: Group::Sampling,
        ..BASE
    },
    ParamDef {
        name: "--presence-penalty",
        label: "Presence Penalty",
        presets: PENALTY,
        default: "0",
        help_fr: "Pénalise les tokens déjà présents.",
        help_en: "Penalizes tokens already present.",
        group: Group::Sampling,
        ..BASE
    },
    ParamDef {
        name: "--frequency-penalty",
        label: "Frequency Penalty",
        presets: PENALTY,
        help_fr: "Pénalise les tokens fréquents.",
        help_en: "Penalizes frequent tokens.",
        group: Group::Sampling,
        ..BASE
    },
    ParamDef {
        name: "--repeat-penalty",
        label: "Repeat Penalty",
        presets: PENALTY,
        help_fr: "Pénalise les répétitions.",
        help_en: "Penalizes repetitions.",
        group: Group::Sampling,
        ..BASE
    },
    ParamDef {
        name: "--n-predict",
        label: "Max Tokens",
        presets: N_PREDICT,
        help_fr: "Nombre maximum de tokens générés par réponse (-1 = illimité).",
        help_en: "Maximum tokens generated per answer (-1 = unlimited).",
        group: Group::Sampling,
        aliases: &["-n", "--predict"],
        ..BASE
    },
    ParamDef {
        name: "--pooling",
        label: "Pooling",
        presets: POOLING,
        help_fr: "Type de pooling pour les embeddings.",
        help_en: "Pooling type for embeddings.",
        group: Group::Sampling,
        ..BASE
    },
    // ---------------- Spéculation ----------------
    ParamDef {
        name: "--spec-type",
        label: "Spec Type",
        presets: SPEC_TYPE,
        default: "draft-mtp",
        help_fr: "Méthode de décodage spéculatif (draft-mtp : têtes MTP intégrées au modèle).",
        help_en: "Speculative decoding method (draft-mtp: MTP heads built into the model).",
        group: Group::Speculation,
        ..BASE
    },
    ParamDef {
        name: "--spec-draft-n-max",
        label: "Spec Draft N Max",
        presets: SPEC_N,
        default: "2",
        help_fr: "Nombre maximum de tokens proposés par passe.",
        help_en: "Maximum drafted tokens per pass.",
        group: Group::Speculation,
        aliases: &["--draft", "--draft-n", "--draft-max"],
        ..BASE
    },
    ParamDef {
        name: "--spec-draft-p-min",
        label: "Spec Draft P Min",
        presets: SPEC_P_MIN,
        help_fr: "Probabilité minimale pour qu'un token du brouillon soit proposé : le \
                  brouillon s'arrête dès qu'il est moins sûr que ce seuil. Plus haut = moins \
                  de tokens proposés mais davantage acceptés.",
        help_en: "Minimum probability for a draft token to be proposed: drafting stops as soon \
                  as it is less confident than this threshold. Higher = fewer tokens proposed \
                  but more of them accepted.",
        group: Group::Speculation,
        aliases: &["--draft-p-min"],
        ..BASE
    },
    ParamDef {
        name: "--cache-type-k-draft",
        label: "Draft Cache K",
        presets: CACHE_TYPES,
        help_fr: "Quantisation du cache K du modèle brouillon.",
        help_en: "K cache quantization of the draft model.",
        group: Group::Speculation,
        aliases: &["-ctkd", "--spec-draft-type-k"],
        ..BASE
    },
    ParamDef {
        name: "--cache-type-v-draft",
        label: "Draft Cache V",
        presets: CACHE_TYPES,
        help_fr: "Quantisation du cache V du modèle brouillon.",
        help_en: "V cache quantization of the draft model.",
        group: Group::Speculation,
        aliases: &["-ctvd", "--spec-draft-type-v"],
        ..BASE
    },
    // ---------------- Template ----------------
    ParamDef {
        name: "--jinja",
        label: "Jinja",
        kind: Kind::Flag,
        help_fr: "Utilise le template Jinja embarqué dans le GGUF.",
        help_en: "Use the Jinja template embedded in the GGUF.",
        group: Group::Template,
        off_aliases: &["--no-jinja"],
        ..BASE
    },
    ParamDef {
        name: "--chat-template-kwargs",
        label: "Chat Template Kwargs",
        presets: KWARGS,
        default: "{\"preserve_thinking\": true}",
        help_fr: "Arguments JSON passés au template de chat.",
        help_en: "JSON arguments forwarded to the chat template.",
        group: Group::Template,
        ..BASE
    },
    ParamDef {
        name: "--reasoning-format",
        label: "Reasoning Format",
        presets: REASONING_FORMAT,
        help_fr: "Où l'API place le raisonnement : deepseek le sépare dans reasoning_content.",
        help_en: "Where the API puts the reasoning: deepseek moves it to reasoning_content.",
        group: Group::Template,
        ..BASE
    },
    ParamDef {
        name: "--reasoning-effort",
        label: "Reasoning Effort",
        presets: REASONING_EFFORT,
        help_fr: "Niveau de réflexion transmis au template (default = celui du modèle).",
        help_en: "Reasoning level handed to the template (default = the model's own).",
        group: Group::Template,
        ..BASE
    },
    // ---------------- Multimodal ----------------
    ParamDef {
        name: "--mmproj",
        label: "Modèle vision",
        help_fr: "Fichier projecteur multimodal qui donne la vision au modèle. Il doit \
                  correspondre au modèle chargé.",
        help_en: "Multimodal projector file that gives the model vision. It must match the \
                  loaded model.",
        group: Group::Multimodal,
        aliases: &["-mm"],
        ..BASE
    },
    ParamDef {
        name: "--no-mmproj-offload",
        label: "Modèle vision sur CPU",
        kind: Kind::Flag,
        help_fr: "Garde le projecteur sur le CPU pour économiser de la VRAM (images plus \
                  lentes).",
        help_en: "Keeps the projector on the CPU to save VRAM (slower images).",
        group: Group::Multimodal,
        off_aliases: &["--mmproj-offload"],
        ..BASE
    },
    // ---------------- Serveur ----------------
    ParamDef {
        name: "--api-key",
        label: "Clé API",
        help_fr: "Exige cette clé dans l'en-tête Authorization des requêtes. Indispensable \
                  si le serveur écoute sur 0.0.0.0. Saisie dans un champ masqué ; le bouton \
                  👁 l'affiche. Plusieurs clés : séparez-les par des virgules.",
        help_en: "Requires this key in the requests' Authorization header. Essential when \
                  the server listens on 0.0.0.0. Typed into a masked field; the 👁 button \
                  shows it. Several keys: separate them with commas.",
        group: Group::Server,
        secret: true,
        ..BASE
    },
    ParamDef {
        name: "--sleep-idle-seconds",
        label: "Veille (s)",
        presets: SLEEP_IDLE,
        help_fr: "Décharge le modèle après ce nombre de secondes d'inactivité, pour libérer \
                  la VRAM ; il est rechargé à la requête suivante.",
        help_en: "Unloads the model after this many idle seconds to free VRAM; it is \
                  reloaded on the next request.",
        group: Group::Server,
        ..BASE
    },
];

pub fn def(index: usize) -> &'static ParamDef {
    &PARAMS[index]
}

pub fn index_of(name: &str) -> Option<usize> {
    PARAMS.iter().position(|p| p.name == name)
}

/// Ce qu'une option de ligne de commande désigne dans le catalogue.
#[derive(Debug, PartialEq, Eq)]
pub enum Resolved {
    /// Le paramètre `index`, sous son nom ou un alias.
    Param(usize),
    /// La forme qui désactive le drapeau `index` (`--no-jinja`).
    Off(usize),
}

pub fn resolve(option: &str) -> Option<Resolved> {
    PARAMS.iter().enumerate().find_map(|(i, p)| {
        if p.name == option || p.aliases.contains(&option) {
            Some(Resolved::Param(i))
        } else if p.off_aliases.contains(&option) {
            Some(Resolved::Off(i))
        } else {
            None
        }
    })
}

/// État d'un paramètre côté UI : la valeur choisie plus la liste
/// proposée (les presets, enrichis par les valeurs vues à l'import et par
/// les fichiers découverts).
#[derive(Clone)]
pub struct ParamState {
    pub value: String,
    pub options: Vec<String>,
}

impl ParamState {
    pub fn new(d: &ParamDef) -> Self {
        Self {
            value: d.default.to_string(),
            options: d.presets.iter().map(|s| s.to_string()).collect(),
        }
    }

    pub fn enabled(&self) -> bool {
        !self.value.trim().is_empty()
    }

    /// Ajoute une valeur inconnue à la liste (import de commande / profil)
    /// et la sélectionne, sans qu'elle devienne une saisie libre.
    pub fn adopt(&mut self, value: &str) {
        let v = value.trim();
        if v.is_empty() {
            self.value.clear();
            return;
        }
        self.offer(v);
        self.value = v.to_string();
    }

    /// Propose une valeur dans la liste sans la sélectionner.
    pub fn offer(&mut self, value: &str) {
        let v = value.trim();
        if !v.is_empty() && !self.options.iter().any(|o| o == v) {
            self.options.push(v.to_string());
        }
    }
}

pub fn default_states() -> Vec<ParamState> {
    PARAMS.iter().map(ParamState::new).collect()
}

/// Options actives dans l'ordre du catalogue, avec leur valeur éventuelle.
fn active(states: &[ParamState]) -> impl Iterator<Item = (&'static ParamDef, Option<&str>)> {
    states
        .iter()
        .enumerate()
        .filter(|(_, st)| st.enabled())
        .map(|(i, st)| {
            let d = def(i);
            match d.kind {
                Kind::Flag => (d, None),
                // Rogne les espaces d'une valeur saisie (clé API collée).
                Kind::Choice => (d, Some(st.value.trim())),
            }
        })
}

/// Arguments passés à `llama-server` : le catalogue, puis les options
/// importées hors catalogue, transmises telles quelles.
pub fn build_args(states: &[ParamState], extra: &[Vec<String>]) -> Vec<String> {
    let mut args = Vec::new();
    for (d, value) in active(states) {
        args.push(d.name.to_string());
        if let Some(v) = value {
            args.push(v.to_string());
        }
    }
    args.extend(extra.iter().flatten().cloned());
    args
}

/// `••••42` : assez pour reconnaître une clé sans la révéler.
pub fn mask(value: &str) -> String {
    let chars: Vec<char> = value.chars().collect();
    if chars.len() <= 4 {
        "••••".to_string()
    } else {
        let tail: String = chars[chars.len() - 2..].iter().collect();
        format!("••••{tail}")
    }
}

/// La commande complète, pour l'affichage et la copie.
pub struct Command<'a> {
    pub exe: &'a str,
    pub model: &'a str,
    pub host: &'a str,
    pub port: &'a str,
    pub states: &'a [ParamState],
    pub extra: &'a [Vec<String>],
}

impl Command<'_> {
    /// Une option par ligne, avec les continuations `^` de cmd.exe : la
    /// commande est directement collable dans un terminal.
    pub fn multiline(&self, masked: bool) -> String {
        self.parts(masked).join(" ^\n  ")
    }

    /// La même commande sur une seule ligne.
    pub fn one_line(&self, masked: bool) -> String {
        self.parts(masked).join(" ")
    }

    fn parts(&self, masked: bool) -> Vec<String> {
        let mut out = vec![
            quote_if_needed(self.exe),
            format!("--host {}", self.host),
            format!("--port {}", self.port),
            format!("-m {}", quote_if_needed(self.model)),
        ];
        for (d, value) in active(self.states) {
            out.push(match value {
                Some(v) if d.secret && masked => format!("{} {}", d.name, mask(v)),
                Some(v) => format!("{} {}", d.name, quote_if_needed(v)),
                None => d.name.to_string(),
            });
        }
        for tokens in self.extra {
            let line: Vec<String> = tokens.iter().map(|t| quote_if_needed(t)).collect();
            out.push(line.join(" "));
        }
        out
    }
}

fn quote_if_needed(v: &str) -> String {
    if v.contains(' ') || v.contains('"') || v.contains('{') {
        format!("\"{}\"", v.replace('"', "\\\""))
    } else {
        v.to_string()
    }
}

/// Sérialisation d'un profil : uniquement les paramètres actifs.
pub fn to_map(states: &[ParamState]) -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    for (i, st) in states.iter().enumerate() {
        if st.enabled() {
            m.insert(def(i).name.to_string(), st.value.clone());
        }
    }
    m
}

/// Applique un profil : tout ce qui n'y figure pas est désactivé.
pub fn from_map(map: &BTreeMap<String, String>) -> Vec<ParamState> {
    let mut states = default_states();
    for st in states.iter_mut() {
        st.value.clear();
    }
    for (k, v) in map {
        if let Some(i) = index_of(k) {
            states[i].adopt(v);
        }
    }
    states
}

#[cfg(test)]
mod tests {
    use super::*;

    fn only(name: &str, value: &str) -> Vec<ParamState> {
        let mut states = default_states();
        for st in states.iter_mut() {
            st.value.clear();
        }
        states[index_of(name).unwrap()].adopt(value);
        states
    }

    fn command<'a>(states: &'a [ParamState], extra: &'a [Vec<String>]) -> Command<'a> {
        Command {
            exe: r"C:\llama\llama-server.exe",
            model: r"D:\models\a.gguf",
            host: "127.0.0.1",
            port: "8080",
            states,
            extra,
        }
    }

    #[test]
    fn preview_is_pastable_every_line_continued() {
        let states = only("-ngl", "999");
        let cmd = command(&states, &[]).multiline(false);
        let lines: Vec<&str> = cmd.lines().collect();
        assert_eq!(lines.len(), 5);
        // Toutes les lignes sauf la dernière doivent porter la continuation.
        for line in &lines[..lines.len() - 1] {
            assert!(line.ends_with('^'), "ligne sans continuation : {line}");
        }
        assert!(lines.last().unwrap().ends_with("-ngl 999"));
        assert_eq!(lines[1].trim(), "--host 127.0.0.1 ^");
    }

    #[test]
    fn preview_quotes_paths_with_spaces() {
        let states = default_states();
        let cmd = Command {
            exe: r"C:\Program Files\llama\llama-server.exe",
            model: r"D:\my models\a.gguf",
            ..command(&states, &[])
        }
        .one_line(false);
        assert!(cmd.starts_with("\"C:\\Program Files\\llama\\llama-server.exe\""));
        assert!(cmd.contains("-m \"D:\\my models\\a.gguf\""));
        assert!(!cmd.contains('\n'));
    }

    #[test]
    fn negative_values_stay_on_their_option_line() {
        // Avant, une valeur commençant par « - » était prise pour une option
        // et rejetée sur une ligne à part dans l'aperçu.
        let states = only("--n-predict", "-1");
        let cmd = command(&states, &[]).multiline(false);
        assert!(cmd.lines().any(|l| l.trim() == "--n-predict -1"), "{cmd}");
    }

    #[test]
    fn secret_is_masked_on_screen_but_not_when_copied_or_launched() {
        let states = only("--api-key", "secret-key-42");
        assert!(command(&states, &[]).multiline(true).contains("--api-key ••••42"));
        assert!(command(&states, &[]).one_line(false).contains("--api-key secret-key-42"));
        assert_eq!(build_args(&states, &[]), ["--api-key", "secret-key-42"]);
    }

    #[test]
    fn extra_options_are_appended_verbatim() {
        let states = only("-ngl", "999");
        let extra = vec![vec!["--foo".to_string(), "a b".to_string()]];
        assert_eq!(build_args(&states, &extra), ["-ngl", "999", "--foo", "a b"]);
        assert!(command(&states, &extra).multiline(false).ends_with("--foo \"a b\""));
    }

    #[test]
    fn flags_take_no_value_on_the_command_line() {
        assert_eq!(build_args(&only("--jinja", "on"), &[]), ["--jinja"]);
    }

    #[test]
    fn disabled_parameters_are_absent() {
        let mut states = default_states();
        for st in states.iter_mut() {
            st.value.clear();
        }
        assert!(build_args(&states, &[]).is_empty());
    }

    #[test]
    fn names_and_aliases_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for p in PARAMS {
            for n in std::iter::once(&p.name)
                .chain(p.aliases)
                .chain(p.off_aliases)
            {
                assert!(seen.insert(*n), "« {n} » apparaît deux fois dans le catalogue");
            }
            assert!(
                p.kind == Kind::Flag || p.off_aliases.is_empty(),
                "{} : seules les cases à cocher ont des formes négatives",
                p.name
            );
            assert!(
                p.default.is_empty() || p.kind == Kind::Flag || p.presets.contains(&p.default),
                "{} : la valeur par défaut doit figurer dans sa liste",
                p.name
            );
        }
    }

    #[test]
    fn resolve_finds_aliases_and_negations() {
        assert_eq!(resolve("-ctkd"), index_of("--cache-type-k-draft").map(Resolved::Param));
        assert_eq!(resolve("--no-jinja"), index_of("--jinja").map(Resolved::Off));
        assert_eq!(resolve("--inconnue"), None);
    }

    /// Confronte le catalogue à l'aide d'une vraie version de llama-server :
    /// chaque nom et alias doit exister, et chaque valeur proposée doit être
    /// admise quand l'aide énumère les valeurs possibles.
    ///
    /// ```text
    /// llama-server --help > help.txt
    /// set LLAMA_SERVER_HELP=help.txt
    /// cargo test catalog_matches -- --ignored
    /// ```
    #[test]
    #[ignore = "nécessite la sortie de `llama-server --help` (LLAMA_SERVER_HELP)"]
    fn catalog_matches_llama_server_help() {
        let path = std::env::var("LLAMA_SERVER_HELP").expect("LLAMA_SERVER_HELP non défini");
        let help = std::fs::read_to_string(path).unwrap().replace('\r', "");
        let blocks = help_blocks(&help);

        let mut errors = Vec::new();
        for p in PARAMS {
            let names: Vec<&str> = std::iter::once(p.name)
                .chain(p.aliases.iter().copied())
                .chain(p.off_aliases.iter().copied())
                .collect();
            let known = |n: &str| blocks.iter().find(|(opts, _)| opts.iter().any(|o| o == n));
            let Some((_, text)) = known(p.name) else {
                errors.push(format!("{} : option inconnue de cette version", p.name));
                continue;
            };
            for n in &names {
                if known(n).is_none() {
                    errors.push(format!("{} : alias « {n} » inconnu", p.name));
                }
            }
            let allowed = allowed_values(text);
            if !allowed.is_empty() {
                // Les nombres sont validés par leur type, pas par une liste.
                for v in p.presets.iter().filter(|v| v.parse::<f64>().is_err()) {
                    if !allowed.iter().any(|a| a == v) {
                        errors.push(format!("{} : valeur « {v} » non admise {allowed:?}", p.name));
                    }
                }
            }
        }
        assert!(errors.is_empty(), "\n{}", errors.join("\n"));
    }

    /// Colonne des options d'une ligne d'aide : les jetons de tête qui sont
    /// des options (`-t,`) ou des arguments (`N`, `TYPE`, `{a,b}`,
    /// `[on|off]`, `a,b,c`), jusqu'au premier mot de la description. Les
    /// colonnes sont alignées par des espaces multiples, on ne peut donc pas
    /// couper au premier double espace.
    fn option_column(line: &str) -> Vec<&str> {
        line.split_whitespace()
            .take_while(|t| {
                let t = t.trim_end_matches(',');
                t.starts_with('-')
                    || t.starts_with('{')
                    || t.starts_with('[')
                    || t.contains(',')
                    || t.contains('|')
                    || (!t.is_empty()
                        && t.chars().all(|c| c.is_ascii_uppercase() || c == '_' || c.is_ascii_digit()))
            })
            .collect()
    }

    /// Découpe l'aide en blocs : (options de la première colonne, texte).
    fn help_blocks(help: &str) -> Vec<(Vec<String>, String)> {
        let mut blocks: Vec<(Vec<String>, String)> = Vec::new();
        for line in help.lines() {
            if line.starts_with('-') {
                let opts = option_column(line)
                    .into_iter()
                    .map(|t| t.trim_end_matches(','))
                    .filter(|t| t.starts_with('-'))
                    .map(str::to_string)
                    .collect();
                blocks.push((opts, line.to_string()));
            } else if let Some(last) = blocks.last_mut() {
                last.1.push('\n');
                last.1.push_str(line);
            }
        }
        blocks
    }

    /// Valeurs énumérées dans un bloc d'aide, sous toutes les formes
    /// utilisées par llama.cpp ; vide si l'option accepte n'importe quoi.
    fn allowed_values(text: &str) -> Vec<String> {
        fn push_list(out: &mut Vec<String>, s: &str, seps: &[char]) {
            for v in s.split(seps).map(str::trim).filter(|v| !v.is_empty()) {
                out.push(v.to_string());
            }
        }
        let mut out = Vec::new();
        let first = text.lines().next().unwrap_or("");
        let column = option_column(first).join(" ");
        let column = column.as_str();
        // `{a,b}` ou `[a|b]` dans la colonne, ou `a,b,c` en guise d'argument.
        for (open, close, seps) in [('{', '}', &[','][..]), ('[', ']', &['|'][..])] {
            if let (Some(a), Some(b)) = (column.find(open), column.rfind(close)) {
                push_list(&mut out, &column[a + 1..b], seps);
            }
        }
        if let Some(arg) = column.split(' ').next_back().filter(|a| a.contains(',') && !a.starts_with('-')) {
            push_list(&mut out, arg, &[',']);
        }
        for line in text.lines() {
            let l = line.trim();
            if let Some(rest) = l.strip_prefix("allowed values:") {
                push_list(&mut out, rest, &[',']);
            }
            // `- nom: description`
            if let Some(rest) = l.strip_prefix("- ") {
                if let Some((name, _)) = rest.split_once(':') {
                    if !name.contains(' ') {
                        out.push(name.split_whitespace().next().unwrap_or("").to_string());
                    }
                }
            }
            // `'minimal', 'low'`
            let mut parts = l.split('\'');
            parts.next();
            while let (Some(v), Some(_)) = (parts.next(), parts.next()) {
                if !v.contains(' ') && !v.is_empty() {
                    out.push(v.to_string());
                }
            }
            // `(default: auto)`
            if let Some(i) = l.find("default: ") {
                let v: String = l[i + 9..]
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
                    .collect();
                out.push(v);
            }
        }
        // Une option purement numérique n'énumère rien : seules des listes
        // d'au moins deux mots valent énumération.
        if out.iter().filter(|v| v.parse::<f64>().is_err()).count() < 2 {
            out.clear();
        }
        out
    }
}
