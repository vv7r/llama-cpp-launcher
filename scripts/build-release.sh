#!/usr/bin/env bash
# Compile l'exécutable Linux de release sans chemins locaux.
#
# Équivalent Linux de build-release.ps1 : Rust inscrit dans le binaire le
# chemin des sources de chaque dépendance (messages de panique) ;
# --remap-path-prefix les remplace par des noms neutres, calculés ici à
# l'exécution : rien de personnel n'est écrit dans le dépôt.
#
#   scripts/build-release.sh
#
# Résultat : dist/llama-cpp-launcher (exécutable) et
# dist/llama-cpp-launcher-linux-x86_64.tar.gz (archive publiée, qui garde le
# droit d'exécution).

set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

cargo_home="${CARGO_HOME:-$HOME/.cargo}"
rustup_home="${RUSTUP_HOME:-$HOME/.rustup}"
target_dir="${CARGO_TARGET_DIR:-$root/target}"

# Séparateur 0x1F : un chemin peut contenir des espaces.
sep=$'\x1f'
export CARGO_ENCODED_RUSTFLAGS="--remap-path-prefix=$cargo_home=cargo${sep}--remap-path-prefix=$rustup_home=rustup${sep}--remap-path-prefix=$root=llama-cpp-launcher"

cargo build --release

mkdir -p dist
cp "$target_dir/release/uillamacpp" dist/llama-cpp-launcher
chmod 755 dist/llama-cpp-launcher
tar -C dist -czf dist/llama-cpp-launcher-linux-x86_64.tar.gz llama-cpp-launcher

# Vérification : aucun chemin du dossier personnel ni du projet ne doit rester.
for leak in "$HOME" "$root"; do
  if grep -qaF "$leak" dist/llama-cpp-launcher; then
    echo "Chemin local encore présent dans l'exécutable : $leak" >&2
    exit 1
  fi
done
echo "OK : dist/llama-cpp-launcher-linux-x86_64.tar.gz, sans chemin local."
