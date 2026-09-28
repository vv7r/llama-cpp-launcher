# Compile l'exécutable de release sans chemins locaux.
#
# Rust inscrit dans le binaire le chemin des sources de chaque dépendance
# (messages de panique) : C:\Users\<nom>\.cargo\registry\..., le dossier du
# projet, la toolchain. --remap-path-prefix les remplace par des noms neutres.
# Les chemins réels sont calculés ici, à l'exécution : rien de personnel
# n'est écrit dans le dépôt.
#
#   powershell -ExecutionPolicy Bypass -File scripts\build-release.ps1
#
# Résultat : target\release\uillamacpp.exe, et une copie
# dist\llama-cpp-launcher.exe (nom publié dans les releases GitHub).

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$cargoHome = if ($env:CARGO_HOME) { $env:CARGO_HOME } else { Join-Path $env:USERPROFILE ".cargo" }
$rustupHome = if ($env:RUSTUP_HOME) { $env:RUSTUP_HOME } else { Join-Path $env:USERPROFILE ".rustup" }

# RUSTFLAGS remplace les rustflags de .cargo/config.toml : la CRT statique
# (exécutable sans redistribuable Visual C++) est donc répétée ici.
$flags = @(
    "-C", "target-feature=+crt-static",
    "--remap-path-prefix=$cargoHome=cargo",
    "--remap-path-prefix=$rustupHome=rustup",
    "--remap-path-prefix=$root=llama-cpp-launcher"
)
# Séparateur 0x1F : un chemin peut contenir des espaces.
$env:CARGO_ENCODED_RUSTFLAGS = $flags -join [char]0x1F

cargo build --release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

New-Item -ItemType Directory -Force (Join-Path $root "dist") | Out-Null
$exe = Join-Path $root "target\release\uillamacpp.exe"
Copy-Item $exe (Join-Path $root "dist\llama-cpp-launcher.exe") -Force

# Vérification : aucun chemin du profil utilisateur ne doit rester.
$text = [System.Text.Encoding]::ASCII.GetString([System.IO.File]::ReadAllBytes($exe))
$leaks = @($env:USERPROFILE, $root) | Where-Object { $text.Contains($_) }
if ($leaks) {
    Write-Error "Chemins locaux encore présents dans l'exécutable : $($leaks -join ', ')"
}
Write-Host "OK : dist\llama-cpp-launcher.exe, sans chemin local."
