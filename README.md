# llama.cpp launcher

**A single-file app for Windows and Linux to run, tune and benchmark [llama.cpp](https://github.com/ggml-org/llama.cpp) — without ever typing a command-line flag.**

[![Release](https://img.shields.io/github/v/release/vv7r/llama-cpp-launcher)](https://github.com/vv7r/llama-cpp-launcher/releases/latest)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20x64-0078D4)
![Runtime](https://img.shields.io/badge/runtime-none-2EA043)
![Rust](https://img.shields.io/badge/built%20with-Rust%20%2B%20egui-B7410E)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

![llama.cpp launcher: launch parameters, profiles and live token statistics](docs/screenshot.png)

*[Version française](README.fr.md)*

---

## Why

`llama-server` has dozens of options, and a single typo — `--cache-type-k q8`
instead of `q8_0`, a value your build doesn't support — stops the server from
starting. This launcher puts every option behind a **closed list of valid
values**: you pick, you never type. "Disabled" simply removes the option from
the command line.

It also ships as **one executable with no runtime**: on Windows a single `.exe`
with no .NET, WebView or Visual C++ redistributable (the CRT is linked
statically); on Linux a single binary that only needs the C library. Download
it, run it.

It is a Rust rewrite of [llamapilot](https://github.com/Hamrounmh/llamapilot)
(WPF / .NET 8), extended with llama.cpp installation and updates, Hugging Face
downloads, live statistics and GPU power tracking.

## Contents

- [Features](#features)
- [Getting started](#getting-started)
- [Using the app](#using-the-app)
- [Where your data lives](#where-your-data-lives)
- [Building from source](#building-from-source)
- [Project layout](#project-layout)
- [Credits and license](#credits-and-license)

## Features

**Launch llama-server**
- 35 options in 7 groups (performance, memory/cache, sampling, speculative
  decoding, template, multimodal, server), each with a tooltip explaining what
  it does. Names, aliases and values are checked against real
  `llama-server --help` output.
- The exact command is shown live, one option per line, ready to paste in a
  terminal or copy as a single line.
- Start / Stop / Restart, colored console with auto-scroll, server status
  and live prompt/generation speeds in the top bar.
- **Import an existing command**: paste any `llama-server` command line. Aliases
  (`-c`, `-fa`, `-ngl`…), `--opt=value`, negative flags (`--no-jinja`) and
  cmd / PowerShell / bash line continuations are understood. Nothing is lost:
  an unknown value is added to the option's list, an unknown option is kept
  and passed through as-is.
- API key is the only typed value, in a masked field; it is masked in the
  console too.

**Profiles**
- Save settings as profiles shown like a chat list, with tags summarising the
  key choices (context size, KV cache, flash attention, speculative decoding,
  reasoning effort, vision…), rename, update or delete.

**Install and update llama.cpp from source**
- No llama.cpp yet? The app clones and builds it for your machine (CUDA,
  Vulkan or CPU), checking prerequisites (Git, CMake, a C++ compiler, CUDA
  Toolkit / Vulkan SDK). A missing tool installs in one click: its command
  (`winget` on Windows, `apt` on Linux) runs in a terminal window, then the
  tools are detected again. A graphics card the build cannot use yet is
  pointed out (NVIDIA without the CUDA Toolkit, AMD or Intel Arc without the
  Vulkan SDK).
- Updates build incrementally and publish a **new release folder** each time
  (`llama-b11235-cuda`): a running server is never overwritten, and older
  versions stay available for comparison. Deleting one is an explicit choice.
- **Change backend** at update time (CPU → CUDA or Vulkan, and back): the
  build starts over with the same options, the existing releases are kept.
- Optional **FlashAttention for every KV cache type**
  (`GGML_CUDA_FA_QUANTS=all`): by default llama.cpp only compiles the CUDA
  kernels for f16-f16, q4_0-q4_0, q8_0-q8_0 and bf16-bf16. The cache-type lists
  flag with ⚠ any K/V pair the selected build does not support.
- Values the selected model cannot use are flagged the same way, in orange:
  `--spec-type draft-mtp` on a model without MTP layers (llama-server would
  refuse to start), a context size beyond the model's training context.

**Models**
- Recursive scan of your models folder; split models are grouped, `mmproj`
  vision projectors are offered separately.
- GGUF header reading (architecture, max context, layers, chat template) in the
  background, cached between sessions — instant even on 40 GB files.
- **Hugging Face downloads**: browse trending or most-downloaded GGUF
  repositories (text and vision models only), pick the exact file (quant,
  variant, size) and download it. Interrupted downloads **resume** where they
  stopped.

**Statistics**
- Per-request prompt (PP) and generation (TG) speeds read from the server
  logs: min, **token-weighted average**, median, max, last, draft acceptance
  rate, and charts of speed versus context size (hover a point for the prompt
  size).
- **Power**: live wattage curves of the graphics card (NVIDIA) and of the
  processor, each one can be hidden (Ctrl + wheel zooms in on a time window,
  dragging moves through time); energy the card and the processor used
  (hover a value for the split) since the server
  started (with Wh per 1000 generated tokens) and since the app opened, each
  with its elapsed time, and in a running total — and its cost at your electricity
  price.

**Benchmark**
- Runs `llama-bench` on the models and versions you tick, with editable
  settings (GPU layers, KV cache, flash attention, prompt / generation tokens,
  **context depth**, repetitions). Results are kept in `benchmark.md` with the
  settings of each measurement; "missing only" skips what is already measured.

**Everything else**
- Top bar gauges: RAM, VRAM and GPU power, processor load and processor
  power. Graphics cards: NVIDIA through `nvidia-smi`; AMD and Intel through
  the Windows performance counters (VRAM and load, no power: Windows does not
  give it for those cards) or, on Linux, their driver (AMD: VRAM, load and
  power; Intel dedicated cards: power). Integrated graphics are left out.
  Processor power comes from the RAPL counters: Windows exposes them without
  a driver; Linux usually reserves them to root.
- Six themes: Dark, Light, Nord, Dracula, Gruvbox, Solarized Light.
- English and French; follows the system language on first launch.
- The UI renders on the integrated GPU when there is one, leaving the whole
  VRAM of the dedicated card to your models.

## Getting started

### Requirements

- **Windows** 10 or 11, 64-bit, or **Linux** x86_64 with a graphical desktop
  (X11 or Wayland) and glibc 2.35 or newer (Ubuntu 22.04, Debian 12, Fedora 36
  and later). Also runs in WSL2 on Windows 11, through WSLg.
- Optional: a graphics card for the VRAM gauge and power tracking (NVIDIA
  everywhere; AMD and Intel with fewer readings, see above).
- Only to build llama.cpp from source: Git, CMake, a C++ compiler (Visual Studio
  Build Tools on Windows, `build-essential` on Linux) and, for GPU builds, the
  CUDA Toolkit or the Vulkan SDK. The app tells you what is missing.

### Install

**Windows**

1. Download **`llama-cpp-launcher.exe`** from the
   [latest release](https://github.com/vv7r/llama-cpp-launcher/releases/latest).
2. Put it in a folder of its own (it stores its settings next to itself), for
   example `C:\llama\launcher\`.
3. Run it. The executable is not code-signed, so Windows SmartScreen may warn
   you the first time: click *More info* → *Run anyway*.

**Linux**

1. Download **`llama-cpp-launcher-linux-x86_64.tar.gz`** from the
   [latest release](https://github.com/vv7r/llama-cpp-launcher/releases/latest).
2. Extract it in a folder of its own (it stores its settings next to itself):

   ```bash
   mkdir -p ~/llama/launcher && tar -xzf llama-cpp-launcher-linux-x86_64.tar.gz -C ~/llama/launcher
   ~/llama/launcher/llama-cpp-launcher
   ```

The folder picker uses your desktop's file dialog (XDG portal), or `zenity`
when no portal is available. On launch, the app adds itself to the
applications menu with its icon (`~/.local/share/applications` and
`~/.local/share/icons`), updated if the executable moves. Under WSL, WSLg
only reads the system folders: for the icon in the Windows taskbar, run once
the `sudo` command the app shows in its console and in ⚙ Options.

### First launch

- **You already have llama.cpp**: pick the **llama.cpp directory** — the folder
  that contains your builds, each subfolder holding a `llama-server` executable
  (`bin/` and `build/bin/` are found too).
- **You don't**: the app opens on **Install llama.cpp**. Choose a folder (on
  Windows a short one such as `C:\llama`: Windows limits path length and the
  build goes deep), a backend, then **Download and build**. A CPU build takes a
  few minutes, a CUDA build 20 to 60 minutes.

Then pick your **models directory** (scanned recursively for `.gguf` files),
or download a model from the **Models** tab.

## Using the app

1. **Left column** — llama.cpp version, GGUF model, optional vision model,
   host and port, profiles.
2. **Middle column** — launch parameters by group. *Defaults* restores the
   defaults, *Disable all* starts from a bare command, *Import a command*
   reads an existing one.
3. **Top bar** — *Start*. The status turns into the server address: open it in
   a browser for the llama.cpp web UI, or point any OpenAI-compatible client at
   it.
4. **Tabs** — *Console* (live output), *Statistics*, *Benchmark*, *Models*.

Every label, list and button has a tooltip; a disabled button explains why it
is disabled.

**Updating llama.cpp**: *⚙ Options → Update llama.cpp…* checks for new commits,
builds them and selects the new release. Build options are taken from your
existing build (`CMakeCache.txt`), and the build happens in a dedicated
`build-uillamacpp` folder — your own `build` folder is never touched.

## Where your data lives

Everything is stored next to the executable, so the app stays portable:

| File | Content |
|---|---|
| `config.json` | folders, selections, parameters, theme, language, energy total |
| `profiles/*.json` | saved profiles |
| `benchmark.md` | benchmark results, as a Markdown table |
| `gguf-cache.json` | cached GGUF headers |

If `config.json` ever becomes unreadable, it is backed up to `config.json.bak`
instead of being overwritten.

## Building from source

Requires [Rust](https://rustup.rs) 1.95 or newer: on Windows with the MSVC
toolchain (Visual Studio Build Tools), on Linux with a C compiler
(`build-essential`).

```bash
git clone https://github.com/vv7r/llama-cpp-launcher.git
cd llama-cpp-launcher
cargo test
cargo build --release
# -> target/release/uillamacpp(.exe)
```

To build the executable that gets published, use the release script instead:

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File scripts\build-release.ps1
# -> dist\llama-cpp-launcher.exe
```

```bash
# Linux
scripts/build-release.sh
# -> dist/llama-cpp-launcher-linux-x86_64.tar.gz
```

Each script strips local paths from the binary (Rust embeds the source path of every
dependency, e.g. `C:\Users\<you>\.cargo\registry\…`) with
`--remap-path-prefix`, then checks that none is left.

On Windows, `.cargo/config.toml` links the C runtime statically, and `build.rs`
embeds the icon (drawn by `src/icon.rs`) into the executable. The
[release workflow](.github/workflows/release.yml) builds and publishes both
executables for every `v*` tag; the Linux one is built on Ubuntu 22.04 so that
it runs on older distributions too.

After updating llama.cpp, check that the option catalog still matches what
`llama-server` accepts:

```bash
llama-server --help > help.txt
LLAMA_SERVER_HELP=help.txt cargo test catalog_matches -- --ignored
```

## Project layout

```
src/
├── main.rs        entry point, window, GPU selection for rendering
├── app.rs         application state and UI (panels, tabs, dialogs)
├── params.rs      option catalog and the value lists of each option
├── cmdparse.rs    command-line import
├── config.rs      config.json
├── profiles.rs    profiles/*.json
├── discovery.rs   llama.cpp versions and GGUF models on disk
├── gguf.rs        GGUF header reader
├── process.rs     llama-server process and its output
├── stats.rs       per-request statistics from the server logs
├── power.rs       GPU and CPU power curves, energy and cost
├── gpu.rs         VRAM, load and power of the graphics cards
├── sysmon.rs      RAM, processor load and power
├── bench.rs       llama-bench and benchmark.md
├── hub.rs         Hugging Face listing and resumable downloads
├── updater.rs     llama.cpp install, update and release management
├── fattn.rs       FlashAttention K/V pairs compiled in a CUDA build
├── icon.rs        application icon (window, .exe on Windows, PNG on Linux)
├── desktop.rs     Linux menu entry and icon (~/.local/share)
└── theme.rs       themes and colored action buttons
```

The UI never mutates state while drawing: interactions produce `Action`s
applied after the frame. Child processes are read by dedicated threads that
push their lines into channels and wake the UI.

## Credits and license

- [llama.cpp](https://github.com/ggml-org/llama.cpp) by Georgi Gerganov and
  contributors — this app only drives its `llama-server` and `llama-bench`.
- [llamapilot](https://github.com/Hamrounmh/llamapilot), the original WPF
  launcher this project started from.
- Built with [egui / eframe](https://github.com/emilk/egui).

Released under the [MIT license](LICENSE).
