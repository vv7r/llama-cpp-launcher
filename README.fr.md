# llama.cpp launcher

*[English version](README.md)*

Lanceur graphique pour `llama-server` et `llama-bench` — réécriture en **Rust**
du projet [llamapilot](https://github.com/Hamrounmh/llamapilot) (WPF / .NET 8).

Deux différences assumées avec l'original :

1. **Aucune saisie libre pour les paramètres.** Chaque option est choisie dans
   une liste préenregistrée ; « désactivé » retire simplement l'option de la
   ligne de commande. Plus de faute de frappe qui fait échouer le démarrage.
2. **Un seul `.exe`, sans runtime.** Pas de .NET, pas de WebView2, pas de
   redistribuable Visual C++ : la CRT est liée statiquement.

![sans dépendance](https://img.shields.io/badge/runtime-aucun-2EA043)

## Installation

Téléchargez `llama-cpp-launcher.exe` depuis la page
[Releases](https://github.com/vv7r/llama-cpp-launcher/releases/latest), placez-le
dans un dossier à lui (il y écrit ses réglages) et lancez-le. L'exécutable
n'est pas signé : au premier lancement, Windows SmartScreen peut demander
*Informations complémentaires* → *Exécuter quand même*.

Au premier démarrage, indiquez :

- le **répertoire llama.cpp** — celui qui contient vos versions (chaque
  sous-dossier doit contenir `llama-server.exe`) ;
- le **répertoire des modèles** — scanné récursivement pour les `.gguf`.

Les fichiers `config.json`, `profiles/` et `benchmark.md` sont écrits à côté de
l'exécutable, ou dans le répertoire courant si cet emplacement est en lecture
seule.

## Fonctionnalités

### Configuration
Répertoires via sélecteur natif, listes déroulantes pour la version llama.cpp,
le modèle GGUF (les `mmproj` et les parties 2..N des modèles scindés sont
filtrées ; un modèle scindé affiche la taille de toutes ses parties), l'hôte
et le port.

### Paramètres de lancement
35 options réparties en sept groupes, chacune avec sa liste de valeurs et une
infobulle. Noms, alias et valeurs sont vérifiés contre `llama-server --help`
(voir *Compilation*). Le compteur d'options actives est affiché par groupe.

| Groupe | Options |
|---|---|
| Performance | `-ngl`, `--threads`, `--parallel`, `--batch-size`, `--ubatch-size`, `--split-mode`, `--numa` |
| Mémoire / Cache | `--ctx-size`, `--flash-attn`, `--cache-type-k`, `--cache-type-v`, `--no-mmap`, `--mlock` |
| Échantillonnage | `--temp`, `--top-p`, `--top-k`, `--min-p`, `--presence-penalty`, `--frequency-penalty`, `--repeat-penalty`, `--n-predict`, `--pooling` |
| Spéculation | `--spec-type`, `--spec-draft-n-max`, `--spec-draft-p-min`, `--cache-type-k-draft`, `--cache-type-v-draft` |
| Template | `--jinja`, `--chat-template-kwargs`, `--reasoning-format`, `--reasoning-effort` |
| Multimodal | `--mmproj` (choisi parmi les fichiers mmproj du répertoire des modèles), `--no-mmproj-offload` |
| Serveur | `--api-key`, `--sleep-idle-seconds` |

Dans la colonne de gauche, le **modèle vision** (`--mmproj`) se choisit juste
sous le modèle GGUF, présenté de la même façon, avec le nombre de projecteurs
détectés ; la section **Serveur** réunit l'hôte, le port, la clé API et la
mise en veille. Le port se choisit dans la liste ou, avec **Personnalisé…**,
dans un champ numérique limité à 1–65535. La **clé API**
est la seule valeur de paramètre qui se saisit : dans un champ masqué, avec un bouton 👁 pour la vérifier. Une alerte
signale un serveur ouvert au réseau (hôte autre que local) sans clé.

Chaque libellé, chaque liste et chaque bouton de l'application porte une
infobulle expliquant son rôle ; un bouton désactivé explique au survol
*pourquoi* il l'est.

### Commande générée
Le panneau du bas montre la commande exacte qui sera exécutée, **une option
par ligne** avec les continuations `^` de cmd.exe — elle est donc collable
telle quelle dans un terminal. Le panneau se redimensionne par son bord
supérieur, le texte est sélectionnable, et deux boutons copient la version
multi-lignes ou la version sur une seule ligne.

### Profils
Affichés comme une liste de conversations, le plus récent en haut. Chaque
carte montre le nom, le modèle et la version mémorisés, des tags résumant les
réglages clés — le **contexte** (`ctx 64k`) mis en avant, puis `ngl`, cache KV,
flash attention, parallélisme, spéculation (et son cache), effort et format de
raisonnement, vision, température, top-p / top-k / min-p, jinja, clé API — et un pied
avec le nombre d'options, l'adresse et une date relative (« il y a 5 min »,
« hier, 18:10 »).

- **Cliquer une carte** charge le profil : paramètres, hôte, port, et modèle et
  version s'ils existent encore sur le disque.
- **✏ Éditer** : renommer (la date est conservée), ou remplacer les réglages du
  profil par ceux affichés.
- **🗑 Supprimer**, avec confirmation dans la carte.
- Si les réglages affichés divergent du profil actif, la carte signale
  « modifié » et propose **Mettre à jour**.

Les profils écrits par les versions précédentes restent lisibles ; il leur
manque seulement le modèle mémorisé.

### Import d'une commande
Bouton **📥 Importer une commande**, à côté de **Défauts** et **Tout
désactiver** en tête des paramètres de lancement. Collez une commande `llama-server` existante : les continuations `^`, `` ` ``
et `\` sont gérées, ainsi que les guillemets, les formes `--option=valeur`,
tous les alias de llama.cpp (`-fa`, `-c`, `-mm`, `-ctkd`, `-n`…) et les formes
négatives (`--no-jinja`, `--mmap`).

**L'import ne perd rien, sans introduire de saisie libre :**

- une valeur absente d'une liste y est ajoutée et devient sélectionnable ;
- une option absente du catalogue est conservée dans la section
  **Hors catalogue** et transmise telle quelle au serveur. On peut l'y retirer,
  pas la modifier.

Ces ajouts sont conservés dans `config.json` et dans les profils.

### Installation de llama.cpp depuis zéro
Sans llama.cpp installé, l'application s'ouvre sur **Installer llama.cpp** :

- choix du dossier d'installation (les sources y sont clonées, les releases
  publiées à côté ; environ 2 Go avec CUDA) ;
- vérification des prérequis — Git, CMake, compilateur C++ (Visual Studio
  Build Tools), et selon le backend CUDA Toolkit ou Vulkan SDK. Pour chaque
  outil manquant, la commande `winget` qui l'installe est prête à copier ;
  **Revérifier** suffit ensuite, sans relancer l'application ;
- choix du backend (CUDA, Vulkan ou CPU), le plus adapté à la machine étant
  proposé ;
- **Télécharger et compiler** : clone de llama.cpp depuis GitHub, compilation,
  première release créée et sélectionnée.

Préférez un dossier au chemin court (par exemple `C:\llama`) : Windows limite
un chemin à 260 caractères et la compilation crée des fichiers très profonds.
Un clone existant peut aussi être choisi directement.

### Mise à jour de llama.cpp
Le menu **⚙ Options**, sous la liste des versions, regroupe l'installation, la
mise à jour et la suppression des versions. **Mettre à jour llama.cpp…** compile
llama.cpp depuis ses sources et publie le résultat dans une **nouvelle
release** (`llama-b11205-cuda`), placée dans un dossier à côté du dépôt. Les
versions existantes, y compris celle d'un serveur en cours, ne sont jamais
modifiées ; la nouvelle release est sélectionnée à la fin et les anciennes
restent disponibles, par exemple pour les comparer au benchmark.

- Dépôt source et dossier des releases sont détectés automatiquement.
- **Vérifier les mises à jour** (`git fetch`) affiche le nombre de commits en
  attente et les derniers messages, sans rien modifier.
- La compilation reprend les options de votre build existant (backend, GPU,
  générateur…), relues dans son `CMakeCache.txt`. Elle se fait dans
  `build-uillamacpp`, jamais dans votre dossier `build`.
- La première compilation part de zéro (20 à 60 min avec CUDA) ; les suivantes
  sont incrémentales. Progression, durée et annulation dans la fenêtre et dans
  la barre du haut ; le détail est dans la console.

Chaque mise à jour **conserve l'ancienne version**. Pour la supprimer :

- cocher **Supprimer l'ancienne version après la mise à jour** (décochée par
  défaut) : elle n'est supprimée qu'une fois la nouvelle compilée et vérifiée ;
- ou, à tout moment, **⚙ Options › Supprimer cette version…**,
  avec confirmation.

Seules les releases du dossier des releases peuvent être supprimées, jamais un
dossier de build situé ailleurs. Une version depuis laquelle un serveur tourne
est refusée et reste intacte.

La liste des versions affiche le **numéro de build** et le backend de chaque
version (`b11205 · CUDA · llama-b11205-cuda`), la plus récente en tête.

Prérequis : git, CMake et la chaîne de compilation de votre build actuel
(Visual Studio Build Tools, CUDA Toolkit…).

**FlashAttention et types de cache KV (CUDA).** Par défaut, llama.cpp ne
compile les noyaux FlashAttention que pour les paires K/V f16-f16,
q4_0-q4_0, q8_0-q8_0 et bf16-bf16 ; une autre paire fait tourner l'attention
sur le CPU, bien plus lentement. La case **FlashAttention pour tous les
types de cache KV** (mise à jour et installation) compile toutes les paires de
f16, bf16, q8_0, q5_1, q5_0, q4_1 et q4_0, y compris mixtes (K q8_0 + V q4_0) ;
la compilation est plus longue et la release porte le suffixe `-fa-all`.
iq4_nl et f32 n'ont pas de noyau FlashAttention CUDA. Dans les listes Cache
Type K / V et Draft Cache K / V, un **⚠ orange** signale les valeurs dont la
paire n'est pas compilée dans la release sélectionnée (versions de 2026 et
plus récentes, qui l'indiquent dans leur DLL).

### Console
Sortie `stdout`/`stderr` en temps réel, colorisée, défilement automatique,
copie et effacement. Les débits `PP`/`TG` sont extraits des logs et affichés
dans la barre du haut.

### Statistiques
Deux sous-onglets.

**Tokens** — pour chaque requête servie depuis le démarrage du serveur, les
débits de lecture du prompt (PP) et de génération (TG) sont relevés dans les
logs :

- minimum, **moyenne pondérée** (total des tokens divisé par le temps total :
  une petite requête dont le prompt est en cache ne fait plus chuter la
  moyenne), médiane, maximum et dernière valeur ; nombre de requêtes et de
  tokens, milliers séparés par une espace ;
- taux d'acceptation du brouillon avec le décodage spéculatif ;
- deux graphiques des débits selon la taille du contexte ; le survol d'un
  point affiche la taille du prompt et le détail de la requête.

Les statistiques repartent de zéro à chaque démarrage du serveur, ou avec
**Réinitialiser**.

**Consommation GPU** — courbe de la puissance de la carte (NVIDIA), énergie
consommée depuis le démarrage du serveur (avec les Wh pour 1000 tokens
générés), depuis l'ouverture de l'application, et en cumul conservé d'une
session à l'autre, en Wh ou kWh. Le coût se calcule avec le prix du kWh
indiqué (€, $, £ ou CHF).

### Benchmark
Les réglages de `llama-bench` (bouton **Défauts** à côté de leur titre) se
choisissent dans des listes : couches GPU
(`-ngl`), type du cache K/V, flash attention, tokens de prompt (`-p`) et
générés (`-n`), **profondeur de contexte** (`-d`, pour mesurer le débit sur
une conversation déjà longue) et nombre de répétitions. **Défauts** revient à
`NGL 999`, cache `f16`, flash attention `auto`, `pp512`, `tg128`, 3
répétitions.

On coche les versions et les modèles à mesurer, sans tout mesurer :

- **Mesurer la sélection** mesure chaque version cochée avec chaque modèle
  coché ;
- **Seulement les manquants** se limite aux combinaisons absentes de
  `benchmark.md` avec ces réglages.

Chaque ligne se supprime avec sa corbeille, ou toutes avec **Tout supprimer**
(après confirmation). Chaque résultat garde ses réglages (colonne **Config**) ; **Réglages actuels
seulement** n'affiche que les lignes comparables entre elles. Barre de
progression, interruption possible, tri par débit de prompt décroissant,
écriture de `benchmark.md` au format Markdown. Un `benchmark.md` écrit par une
version précédente reste lisible.

### Modèles
Lecture de l'en-tête GGUF (architecture, contexte maximum, nombre de couches,
nombre de tenseurs, présence d'un template de chat) — seul l'en-tête est lu,
c'est instantané même sur 40 Gio. Il est lu automatiquement, en tâche de fond,
pour chaque nouveau modèle, téléchargé ou copié, et mémorisé d'une session à
l'autre : au premier lancement sur un gros dossier, l'interface reste fluide
et la progression s'affiche dans l'onglet Modèles. Ouverture dans l'explorateur, suppression
avec confirmation, et téléchargement depuis Hugging Face : la liste des
dépôts GGUF de modèles de langage (texte et vision) est lue en direct, triée
par tendance ou par téléchargements, puis on choisit le fichier exact —
quantisation, variante (`mtp`…) et taille affichées. **Télécharger** le range
dans le répertoire des modèles (`<auteur>\<dépôt>\`), parties comprises, avec
progression, débit, temps restant et annulation ; il rejoint ensuite la liste
des modèles GGUF.

Un téléchargement coupé **reprend là où il s'est arrêté** : automatiquement
après une coupure réseau, ou au prochain clic sur **Télécharger** si les
tentatives se sont épuisées. Si la reprise est impossible (fichier modifié sur
Hugging Face, fichier partiel incohérent), le fichier partiel est supprimé et
le téléchargement recommence de zéro. La console détaille chaque reprise.

### Divers
- Barre du haut : état du serveur, jauge d'occupation de la VRAM et
  consommation électrique de la carte graphique en watts (cartes NVIDIA, via
  `nvidia-smi`), alignés sur les boutons Démarrer / Arrêter / Redémarrer.
- Six thèmes : Sombre, Clair, Nord, Dracula, Gruvbox et Solarized clair. En
  thème clair, les boutons d'action sont teintés plutôt que pleins.
- Interface bilingue FR/EN ; au premier lancement, la langue suit celle de
  Windows (français si Windows est en français, anglais sinon).
- Icône de l'application dessinée par le code, identique pour la fenêtre et
  l'exécutable.
- Sections de la colonne de gauche repliables, sauvegarde automatique de la
  configuration toutes les deux secondes.

## Compilation

```bash
cargo test
cargo build --release
# -> target/release/uillamacpp.exe
```

`.cargo/config.toml` active `+crt-static` sur la cible MSVC. Le workflow
[`release.yml`](.github/workflows/release.yml) produit l'exécutable sur chaque
tag `v*`.

Après une mise à jour de llama.cpp, vérifiez que le catalogue correspond
toujours à ce qu'accepte `llama-server` (noms, alias, valeurs proposées) :

```bash
llama-server --help > help.txt
LLAMA_SERVER_HELP=help.txt cargo test catalog_matches -- --ignored
```

## Architecture

```
src/
├── main.rs        # point d'entrée, fenêtre eframe
├── app.rs         # état et interface (panneaux, onglets, fenêtres modales)
├── params.rs      # catalogue des paramètres et de leurs listes de valeurs
├── config.rs      # config.json
├── profiles.rs    # profiles/*.json
├── discovery.rs   # scan des versions llama.cpp et des modèles GGUF
├── gguf.rs        # lecture de l'en-tête GGUF
├── cmdparse.rs    # import d'une ligne de commande
├── process.rs     # processus llama-server, flux de sortie, débits
├── stats.rs       # statistiques par requête (débits, contexte)
├── power.rs       # consommation du GPU, énergie et coût
├── fattn.rs       # paires K/V FlashAttention compilées d'une release
├── bench.rs       # llama-bench et benchmark.md
├── gpu.rs         # VRAM et consommation (nvidia-smi)
├── hub.rs         # téléchargement depuis Hugging Face
├── updater.rs     # installation et mise à jour de llama.cpp
├── icon.rs        # icône (fenêtre et exe, via build.rs)
└── theme.rs       # thèmes et boutons d'action
```

L'UI ne mute jamais l'état pendant le rendu : les interactions produisent des
`Action` appliquées après coup, ce qui évite les emprunts croisés et garde la
logique testable. Les processus enfants sont lus par des threads dédiés qui
poussent leurs lignes dans un canal, drainé à chaque image.

## Licence

MIT.
