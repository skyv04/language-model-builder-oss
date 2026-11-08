# Language Model Builder: open-source edition

Build a small language model from scratch, step by step, on your own computer. A textbook
explains how language models work, and next to it a workbench lets you build one: pick a
tokenizer and a dataset, watch pre-training lower the loss, sample from the model, teach it to
answer questions with supervised fine-tuning, steer it with your own preferences (DPO), and chat
with the result.

This is an unofficial, open-source edition of [Language Model Builder](https://languagemodelbuilder.com)
by Felix Rieseberg, a macOS app for Apple silicon. It keeps the same look, chapters and workflow,
with an engine written in Python that runs on Linux, on Windows through WSL2, on Android phones
(Debian in PRoot) and on Macs, including Intel Macs. It trains on the GPU whenever it finds one it
can use.

> **Clean-room implementation.** The original app's source code is not public. Nothing here comes
> from its code, binaries, text or artwork. The layout follows the app's public screenshots, and
> every line of code, chapter text, example and icon in this repository was written for this
> edition or comes from the openly licensed projects listed under [Credits](#credits). This
> edition is not affiliated with or endorsed by the original author.

**Install on:** [Ubuntu and Debian](#ubuntu-and-debian) · [Windows (WSL2)](#windows-wsl2) ·
[Android (Debian in PRoot)](#android-debian-in-proot) · [macOS](#macos)

## Supported systems

| System | App window | Trains on |
|---|---|---|
| [Ubuntu 22.04+ and Debian 12+](#ubuntu-and-debian), x86-64 or ARM64; other Linux distributions too | its own window (WebKitGTK); otherwise a browser's app mode or the browser | NVIDIA GPUs (CUDA); AMD, Intel, Arm and Qualcomm GPUs (Vulkan); CPU |
| [Windows 10 or 11, through WSL2](#windows-wsl2) | a window of its own from Microsoft Edge, or a Linux window through WSLg | NVIDIA GPUs (CUDA); CPU |
| [Android: Debian in PRoot](#android-debian-in-proot) (Termux, ARM64) | its own window on a Termux:X11 desktop, or the phone's browser | Adreno GPUs (Vulkan through Mesa Turnip); CPU |
| [macOS 11+ on Apple silicon](#macos) | its own window (the Mac's WebKit) | the Mac's GPU (Metal); CPU |
| [macOS 11+ on Intel](#macos) | its own window (the Mac's WebKit) | CPU, or the GPU (Metal) when it measures faster |

Every system needs git and Python 3.10 or newer (3.10 to 3.12 on Intel Macs). The installer
needs no root access. It puts everything in a `.venv` folder inside the clone, adds the app to
your application menu (or to `~/Applications` on a Mac), and adds the `language-model-builder`
command to `~/.local/bin`. Only the system packages in the steps below need `sudo`.

## Ubuntu and Debian

```sh
sudo apt update
sudo apt install git python3 python3-venv python3-gi gir1.2-webkit2-4.1
git clone https://github.com/skyv04/language-model-builder-oss.git
cd language-model-builder-oss
bash install.sh
```

Open **Language Model Builder** from your application menu, or run `language-model-builder`.

- `python3-gi` and `gir1.2-webkit2-4.1` give the app its own window. Without them it opens in
  the app mode of Chromium, Chrome, Brave or Edge, and otherwise in your default browser.
- **NVIDIA GPU**: install the NVIDIA driver before running `install.sh`, so that `nvidia-smi`
  lists the GPU. The installer then installs PyTorch with CUDA.
- **AMD, Intel or Arm GPU**: install a Vulkan driver (`sudo apt install mesa-vulkan-drivers`)
  before running `install.sh`. The installer then adds the Vulkan backend.
- **Other distributions**: on Fedora run `sudo dnf install git python3 python3-gobject webkit2gtk4.1`,
  on Arch `sudo pacman -S git python python-gobject webkit2gtk-4.1`. Then clone the repository and
  run `bash install.sh` as above.

## Windows (WSL2)

You need Windows 11, or Windows 10 version 2004 or later.

1. Open **PowerShell** as administrator, run `wsl --install`, and restart Windows. Then open
   **Ubuntu** from the Start menu and create your Linux user.
2. In the Ubuntu terminal:

   ```sh
   sudo apt update
   sudo apt install git python3 python3-venv
   cd ~
   git clone https://github.com/skyv04/language-model-builder-oss.git
   cd language-model-builder-oss
   bash install.sh
   language-model-builder
   ```

- Keep the clone in your Linux home folder (`~`), not under `/mnt/c`. Windows drives are much
  slower from WSL.
- The app opens in a window of its own from Microsoft Edge on Windows (or from Chrome). Edge
  reaches the app at `127.0.0.1` through WSL's localhost forwarding, which is on by default.
  For a Linux window through WSLg instead, run `sudo apt install python3-gi gir1.2-webkit2-4.1`
  and start the app again.
- **NVIDIA GPU**: install the normal NVIDIA driver on Windows, and don't install a Linux driver
  inside WSL. When `nvidia-smi` works in Ubuntu, the installer installs PyTorch with CUDA.
- **Share model…** saves to your Windows **Downloads** folder, and *Open project folder* opens
  folders in Windows' File Explorer.

## Android (Debian in PRoot)

This runs in [Termux](https://termux.dev) on ARM64 phones and tablets, inside Debian started by
`proot-distro`. No Android root is needed.

1. Install **Termux** from [F-Droid](https://f-droid.org/packages/com.termux/) or its
   [GitHub releases](https://github.com/termux/termux-app/releases). For a desktop, also install
   **Termux:X11** from its [GitHub releases](https://github.com/termux/termux-x11/releases).
2. In Termux, install Debian and open a shell in it:

   ```sh
   pkg update
   pkg install proot-distro
   proot-distro install debian
   proot-distro login debian
   ```

3. Inside Debian:

   ```sh
   apt update
   apt install git python3 python3-venv
   git clone https://github.com/skyv04/language-model-builder-oss.git
   cd language-model-builder-oss
   bash install.sh
   ```

4. Open the app:
   - **Without a desktop**: run `language-model-builder serve --persist`, then open the
     `http://127.0.0.1:…` address it prints in Chrome on the same phone. Stop it with Ctrl+C.
   - **On a Termux:X11 desktop**: open it from the application menu, or run
     `language-model-builder`. For the app's own window, first run
     `apt install python3-gi gir1.2-webkit2-4.1`.
5. **Adreno GPU (optional)**: build Mesa's Turnip Vulkan driver with KGSL support into
   `/opt/mesa-adreno`, following [Unlocking phone hardware in Termux](https://skyel.net/blog/unlocking-phone-hardware-termux)
   and [step 1 of the Qwen Chat setup guide](https://github.com/skyv04/qwen-chat/blob/main/docs/termux-setup.md#1-start-with-a-working-gpu-desktop).
   The app finds the driver there by itself; set `VK_DRIVER_FILES` if you put it elsewhere. Then
   run `bash install.sh` again to add the Vulkan backend, and check the GPU with
   `language-model-builder probe-gpu --train`.

The app handles these PRoot details by itself:

- Host libraries preloaded through `LD_PRELOAD`, such as the camera and microphone shims of the
  [SELinux Hardware Bridge](https://github.com/skyv04/selinux-hardware-bridge), are removed from
  the engine's processes. Set `LMB_KEEP_LD_PRELOAD=1` to keep them.
- The WebKitGTK window uses software rendering and runs without its sandbox, because PRoot
  cannot provide one. The page is served only on 127.0.0.1, behind a random session token.
- Chromium, when used, starts with `--no-sandbox --disable-gpu`.

## macOS

You need macOS 11 (Big Sur) or later, on Apple silicon or Intel.

1. Install [Homebrew](https://brew.sh), then Python and git:

   ```sh
   brew install python@3.12 git
   ```

   A Python 3.10–3.12 installer from [python.org](https://www.python.org/downloads/macos/) works
   too. The `python3` that comes with Apple's developer tools is too old.
2. In Terminal:

   ```sh
   git clone https://github.com/skyv04/language-model-builder-oss.git
   cd language-model-builder-oss
   bash install.sh
   ```

3. Open **Language Model Builder** from Launchpad, Spotlight or `~/Applications`.

- The app opens in a native window drawn by the Mac's own WebKit (through
  [pywebview](https://pywebview.flowrl.com)), and trains on the GPU through PyTorch's Metal (MPS)
  backend.
- **Intel Macs**: PyTorch 2.2 is the last release for Intel Macs, so the installer uses it, with
  Python 3.10–3.12. Models train on the CPU until **Calibrate** (on *Project setup*) measures the
  GPU as faster.
- To use the `language-model-builder` command in Terminal, add `~/.local/bin` to your `PATH`.
  The installer prints the line to add.
- Opened from Launchpad, Spotlight or `~/Applications`, the app writes its log to
  `~/Library/Logs/language-model-builder.log`.

## Update, uninstall and installer options

Update:

```sh
cd language-model-builder-oss
git pull
bash install.sh
```

Uninstall:

```sh
bash install.sh --uninstall
```

This removes the menu entry (or the app in `~/Applications`), the icons and the command. Your
projects stay. To remove everything, also delete the data folder (see
[Where your data lives](#where-your-data-lives)) and the clone.

| Option | Effect |
|---|---|
| `--gpu` | Linux: install the Vulkan GPU backend even when no Vulkan driver was detected |
| `--cpu-only` | skip the CUDA and Vulkan GPU backends |
| `--no-menu` | skip the menu entry or Mac app, the icons and the command |
| `--uninstall` | remove the menu entry or Mac app, the icons and the command |
| `LMB_PYTHON=python3.12 bash install.sh` | build the environment with a specific Python |

## Using it

The sidebar has the same sections as the original:

- **Foundations** and **How models work**: interactive chapters on tokens, embeddings,
  transformers, training data, gradient descent and post-training.
- **Build your model**: the workbench.
  - *Project setup*: model shape and blueprint, tokenizer, compute device and calibration.
  - *Pre-training data* and *Fine-tuning data*: dataset catalogues, or bring your own file.
  - *Pre-training*, *Fine-tuning* and *Preference tuning*: live loss charts, samples,
    pause and resume.
  - *Sampling*: decoding controls, with token probabilities and alternatives.
  - *Chat*: an X-ray view of the conversation and **Share model…** export.

Training runs in a separate background process. By default it keeps going when you close the
window, and reopening the app reconnects to the run. Turn off *Keep training after closing the
window* in Settings to pause at a checkpoint instead.

### Datasets

| Pre-training | Licence | Fine-tuning | Licence |
|---|---|---|---|
| TinyStories | CDLA-Sharing-1.0 | Everyday Conversations | Apache-2.0 |
| Tiny Shakespeare | Public domain | No Robots | CC-BY-NC-4.0 |
| WikiText-2 | CC-BY-SA-3.0 | Dolly 15k | CC-BY-SA-3.0 |
| Simple English Wikipedia | CC-BY-SA-3.0 | OpenAssistant Conversations | Apache-2.0 |
| GoodWiki | CC-BY-SA-4.0 | SmolTalk | Apache-2.0 |
| arXiv Abstracts | CC0-1.0 | GSM8K | MIT |
| Children's Stories | Apache-2.0 | CoEdIT | Apache-2.0 |
| Classic Novels (Project Gutenberg) | Public domain (US) | Code Alpaca 20k | CC-BY-4.0 |
| Cosmopedia 100k | Apache-2.0 | SQuAD | CC-BY-SA-4.0 |
| | | UltraChat 200k | MIT |

No Robots may only be used for non-commercial purposes.

Each dataset is downloaded from its public source (mostly Hugging Face) the first time you use it,
and capped at about 30 MB. You can also add your own files:

- pre-training data: `.txt` or `.md` files;
- fine-tuning data: `.jsonl`, `.json` or `.csv` files with chat messages, or with
  instruction/response or question/answer columns.

### Exported models

**Share model…** writes a `.zip` file to your Downloads folder (on WSL, the Windows one). It
contains:

- `model.safetensors` (float32 weights) and `config.json`, which holds the shape and the chat
  template;
- a Hugging Face `tokenizer.json`;
- the PyTorch architecture (`model.py`, `config.py`);
- a README with loading instructions.

The model loads with only `torch`, `safetensors` and `tokenizers` installed.

## GPUs

| GPU | Systems | How it trains |
|---|---|---|
| NVIDIA | Linux, WSL2 | PyTorch with CUDA. The installer picks this build when `nvidia-smi` works. |
| Apple silicon (and Intel Macs' GPUs) | macOS | PyTorch with Metal (MPS). Nothing extra to install. |
| AMD, Intel, Arm Mali, Qualcomm Adreno | Linux, including PRoot on Android | tinygrad generates WGSL kernels, and wgpu-native runs them on Vulkan. |

`language-model-builder doctor` shows what the app found: the window type, CPU, GPUs and backends.
`language-model-builder probe-gpu --train` runs a short GPU training check.

- **Vulkan** needs a hardware driver. Software renderers such as lavapipe and SwiftShader are
  ignored. On Android in PRoot, Adreno GPUs work through Mesa's Turnip driver built with KGSL
  support (`-Dvulkan-drivers=freedreno -Dfreedreno-kmds=kgsl`); see
  [Android](#android-debian-in-proot).
- **Device choice**: the *Compute device* setting has Auto, CPU and GPU options. **Calibrate** on
  *Project setup* times training steps on every device for the current model shape. Auto then
  uses the fastest measured device. Until you calibrate, it uses CUDA or an Apple silicon GPU when
  there is one, and otherwise the CPU. Calibrate times only a few steps while the device is still
  cool, so on machines that slow down as they heat up, a long run can take about twice the
  Project setup estimate (see below). The time left shown during a run follows the measured speed.
- **Measured speed**: pre-training tokens per second on a Snapdragon phone with an Adreno 840 GPU,
  running Debian 12 in PRoot:

  | Model shape | Vocabulary | Batch | CPU (4 threads) | GPU (Vulkan) |
  |---|---|---|---|---|
  | Pocket (128 × 128, 4 blocks) | 4,096 | 16 | 5,740 | 16,064 (2.8×) |
  | Starter (256 × 256, 6 blocks) | 10,000 | 8 | 1,433 | 4,354 (3.0×) |
  | 256 × 256, 4 blocks | 50,261 | 4 | 737 | 2,932 (4.0×) |

  Each Vulkan job first spends 15–25 seconds compiling kernels, so very short runs can still
  finish sooner on the CPU.

  These are short benchmarks. Phones and fanless boards slow down under sustained load. On the
  phone above, a 2,000-step Pocket run held about 23,400 tokens/s on the GPU for its first 200
  steps and 9,000–11,000 after that, and took 7 minutes 17 seconds. Logging the GPU during the next
  run showed why: under full load, Android lowered its clock cap from 1,100 to 578 MHz within about
  35 seconds. Even throttled, the GPU stayed well ahead of the CPU.
- **Preference tuning (DPO)** runs on CUDA, Metal or the CPU. Sampling and chat run on the CPU,
  which is quick at these model sizes and leaves the GPU free for training.
- CPU training leaves one or two cores free (set `LMB_THREADS` to choose the thread count) and runs
  at a lower priority, so the desktop stays responsive. The *Project setup* page warns when a shape
  and batch size would need more memory than the machine has free.

## Commands

| Command | What it does |
|---|---|
| `language-model-builder` | open the app, and start its engine if needed |
| `language-model-builder app --window webkit` | open a given window type: `webkit` (the app's own window), `chromium` or `browser` |
| `language-model-builder serve [--port N] [--persist]` | run only the engine, and print the address to open in a browser. Without `--persist` it stops once no page has been open for 90 seconds and nothing is running. |
| `language-model-builder url` | print the running engine's address, with its access token |
| `language-model-builder doctor` | show the system, Python, window type, CPU and GPUs |
| `language-model-builder probe-gpu [--train]` | look for a GPU that can train; `--train` also checks a training step |

## Where your data lives

| | Linux, WSL and PRoot | macOS |
|---|---|---|
| Projects and datasets | `~/.local/share/language-model-builder` | `~/Library/Application Support/language-model-builder` |
| Downloads and caches | `~/.cache/language-model-builder` | `~/Library/Caches/language-model-builder` |
| Exported models | your Downloads folder (on WSL, the Windows one) | `~/Downloads` |
| App entry | `~/.local/share/applications/language-model-builder.desktop` | `~/Applications/Language Model Builder.app` |
| Command | `~/.local/bin/language-model-builder` | `~/.local/bin/language-model-builder` |

Every project is a plain folder: settings, tokenizer, token caches, checkpoints, metrics and
samples.

| Variable | Effect |
|---|---|
| `LMB_DATA_DIR`, `LMB_CACHE_DIR` | use other data and cache folders (`XDG_DATA_HOME` and `XDG_CACHE_HOME` work too) |
| `LMB_THREADS` | the number of CPU training threads |
| `LMB_DEBUG=1` | enable the web inspector in the app window |
| `LMB_CHROMIUM_FLAGS` | extra flags for a Chromium app window |
| `VK_DRIVER_FILES` | the Vulkan driver manifest to use |
| `LMB_KEEP_LD_PRELOAD=1`, `LMB_DROP_LD_PRELOAD=1` | keep or drop `LD_PRELOAD` in the engine's processes (dropped in PRoot by default) |
| `LMB_WEBKIT_SOFTWARE=1` | software rendering in the WebKitGTK window (the default in PRoot) |

## Troubleshooting

Start with `language-model-builder doctor`. It shows the system, Python, window type, CPU and GPUs
the app found.

- **"Python 3.10 or newer is required"**: Debian 11 and Ubuntu 20.04 ship older versions. Install a
  newer Python and point the installer at it: `LMB_PYTHON=/path/to/python3.12 bash install.sh`.
- **"Couldn't install PyTorch"**: PyTorch may not have builds yet for a brand-new Python. Use
  Python 3.12: `rm -rf .venv && LMB_PYTHON=python3.12 bash install.sh`.
- **The GPU isn't used**: run `language-model-builder probe-gpu`. If you installed the GPU driver
  after the app, run `rm -rf .venv && bash install.sh` so the installer chooses again. On Linux,
  `vulkaninfo --summary` must list your GPU (not `llvmpipe`), and your user needs access to
  `/dev/dri/renderD*` (usually through the `render` group).
- **No window opens, or it stays blank**: `language-model-builder app --window browser` opens the
  app in your default browser instead.
- **WSL: Edge says it can't reach the page**: WSL's localhost forwarding is off. In
  `%UserProfile%\.wslconfig`, set `localhostForwarding=true` under `[wsl2]` (or use
  `networkingMode=mirrored`), run `wsl --shutdown` in PowerShell, and start the app again.
- **macOS: nothing happens when you open the app**: read `~/Library/Logs/language-model-builder.log`.
- **You moved the clone**: run `bash install.sh` again in the new place. The menu entry, Mac app
  and command still point to the old folder until you do.

## Tested on

- **On real hardware**: Debian 12 ARM64 in PRoot (Termux) on a Snapdragon phone with an Adreno 840
  GPU: the installer, every chapter, CPU and Vulkan GPU training, and the WebKitGTK window on a
  Termux:X11 desktop.
- **With unit tests that imitate the system**: macOS (folders, memory, GPU choice, window, Finder)
  and WSL (the Windows Downloads folder, File Explorer, the Edge window).
- **Not yet on real hardware**: macOS (Apple silicon and Intel), Windows with WSL2, x86-64 Ubuntu
  and Debian, NVIDIA GPUs.

If you try one of these, please [open an issue](https://github.com/skyv04/language-model-builder-oss/issues)
with the output of `language-model-builder doctor`, whether it worked or not.

## Under the hood

```
model_builder/
  __main__.py        command line: app (default), serve, url, doctor, probe-gpu, worker
  app.py             starts or reuses the engine server and picks the window
  window.py          WebKitGTK window (Linux)
  webview_window.py  native window through pywebview (macOS)
  system.py          Linux, WSL or macOS: opening links and folders, Windows browsers
  server.py          local HTTP + server-sent events API on 127.0.0.1 (token, Host and Origin checks)
  worker.py          background training process (one job at a time)
  training.py        pre-training, SFT and DPO loops; checkpoints, resume, samples, calibration
  model.py           GPT-style decoder (pre-LayerNorm, tied embeddings) in PyTorch
  config.py          model shapes, kept free of PyTorch so the server starts quickly
  gpu_model.py, gpu_runtime.py   the same model and AdamW in tinygrad on wgpu/Vulkan
  tokenizer.py       byte-level BPE (4K/10K/16K) trained on your data, or GPT-2's BPE
  data.py            corpus building and the chat transcript format
  sampling.py        temperature, top-k, top-p and min-p decoding; attention maps
  datasets.py        dataset catalogue, resumable downloads, previews and imports
  devices.py         CPU and GPU discovery (CUDA, Metal, Vulkan)
  project.py         projects and their settings
  paths.py           data, cache and download folders on each system
  ui/                the web interface: HTML, CSS and ES modules, no build step
```

## Development

```sh
.venv/bin/python -m unittest discover -s tests -t . -v           # under a minute on a phone
LMB_TEST_GPU=1 .venv/bin/python -m unittest tests.test_gpu -v     # needs a hardware Vulkan GPU
.venv/bin/python -m model_builder serve --port 8765 --persist     # engine only
.venv/bin/python -m model_builder url                             # browser address of the running engine
```

`tests/test_platform.py` imitates macOS and WSL, so their code paths are tested on any Linux
machine. Set `LMB_DEBUG=1` to enable the web inspector. The UI uses plain ES modules and has no
build step: edit the files under `model_builder/ui/` and reload the window.

`tools/` holds helper scripts:

- `mock_server.py` serves the UI against a fake engine;
- `benchmark_devices.py` measures CPU and GPU training speed;
- `build_icons.py` regenerates the bundled icons, and `build_icns.py` the macOS icon;
- `build_embeddings.py` regenerates the *Embeddings* chapter's word map;
- `patch_tinygrad.py` is run by the installer.

## Credits

- Original concept, structure and design: [Language Model Builder](https://languagemodelbuilder.com)
  by Felix Rieseberg.
- [PyTorch](https://pytorch.org) (BSD-3-Clause), [tinygrad](https://github.com/tinygrad/tinygrad)
  (MIT), [wgpu-py](https://github.com/pygfx/wgpu-py) (BSD-2-Clause),
  [Hugging Face tokenizers](https://github.com/huggingface/tokenizers) (Apache-2.0),
  [safetensors](https://github.com/huggingface/safetensors) (Apache-2.0),
  [NumPy](https://numpy.org) (BSD-3-Clause), [PyArrow](https://arrow.apache.org) (Apache-2.0).
- On macOS: [pywebview](https://github.com/r0x0r/pywebview) (BSD-3-Clause) and
  [PyObjC](https://github.com/ronaldoussoren/pyobjc) (MIT).
- Interface icons from [Lucide](https://lucide.dev) (ISC); fonts
  [Inter](https://github.com/rsms/inter) and [JetBrains Mono](https://github.com/JetBrains/JetBrainsMono)
  (SIL OFL 1.1). Licence texts are in `model_builder/ui/licenses/`.
- Word vectors in the *Embeddings* chapter: a hand-picked subset of
  [GloVe 6B](https://nlp.stanford.edu/projects/glove/) by Pennington, Socher and Manning
  (PDDL 1.0). The GPT-2 tokenizer (modified MIT) is not bundled: it is downloaded from Hugging Face
  only if you choose it.
- Welcome artwork: Caspar David Friedrich, *Wanderer above the Sea of Fog* (about 1818), public
  domain.
- Datasets: see the table above; each dataset's card in the app links to its source and licence.

## License

The code and text in this repository are under the [MIT License](LICENSE). Bundled fonts, icons,
artwork and data keep their own licences, listed above and on the app's *Acknowledgements* page.
