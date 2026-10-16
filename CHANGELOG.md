# Changelog

All notable changes to llama.cpp launcher. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [1.1.0] - 2026-10-16

### Added

- **Linux version** (x86_64, X11 or Wayland, glibc 2.35+; also runs in WSL2
  through WSLg), published as `llama-cpp-launcher-linux-x86_64.tar.gz`. The
  app adds itself to the applications menu with its icon; under WSL it shows
  the one `sudo` command that puts the icon in the Windows taskbar.
- **RAM and processor** in the top bar, next to the VRAM (order: RAM, VRAM,
  CPU): memory used, processor load and processor power in watts, from the
  RAPL counters (Windows exposes them without a driver; Linux usually keeps
  them for root).
- **AMD and Intel graphics cards**: VRAM and load on Windows for any vendor
  (Windows performance counters), VRAM, load and power for AMD on Linux, power
  for Intel dedicated cards on Linux. Integrated graphics are left out.
- **Power tab**: the processor curve next to the graphics card's, each one can
  be hidden; energy and cost now count the processor too (hover a value for
  the GPU / CPU split); elapsed time of the "since the server started" and
  "since the app opened" rows, in their own column.
- **Options the selected model cannot use** are flagged in orange, like the
  K/V cache pairs: `--spec-type draft-mtp` on a model without MTP layers
  (llama-server would refuse to start) and a context size beyond the model's
  training context.
- **Change the backend of an existing llama.cpp install** (for example from
  CPU to CUDA or Vulkan) in the update window; the build starts over with the
  same options, existing releases are kept.
- **Install a missing build tool from the app**: its `winget` (Windows) or
  `apt` (Linux) command runs in a terminal window, then the tools are detected
  again. A graphics card the build cannot use yet is pointed out (NVIDIA
  without the CUDA Toolkit, AMD or Intel Arc without the Vulkan SDK).
- `CHANGELOG.md`, and a compatibility table in the README.

### Changed

- Power chart: only time moves (Ctrl + wheel zooms on a time window, dragging
  pans within the recorded range); the watt axis stays fixed from 0. Time
  ticks fall on round times and show seconds when zoomed in.
- The "GPU power" statistics tab is now "Power".
- The Wh per 1000 tokens moved under the energy table.
- GGUF headers are read again once, to pick up the MTP information.

## [1.0.1] - 2026-09-29

### Changed

- Context size presets: a gap-free grid up to 1M tokens.
- CI: actions moved to their Node 24 releases.

### Fixed

- Console: no more empty band at the bottom.

## [1.0.0] - 2026-09-29

First release: a single Windows executable to run `llama-server` and
`llama-bench` with every option picked from a list, launch profiles, command
import, live token statistics and GPU power, benchmarks, Hugging Face
downloads with resume, and llama.cpp installation and updates from source.

[1.1.0]: https://github.com/vv7r/llama-cpp-launcher/compare/v1.0.1...v1.1.0
[1.0.1]: https://github.com/vv7r/llama-cpp-launcher/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/vv7r/llama-cpp-launcher/releases/tag/v1.0.0
