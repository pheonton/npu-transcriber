# NPU Transcriber

Transcribe videos and audio files to subtitles (`.srt`) and plain text (`.txt`) with OpenAI's **Whisper** speech recognition model, running locally on the **Intel NPU** of Intel Core Ultra processors.

Everything runs on your own computer. Your recordings never leave it — the internet is only used once to install the program's Python packages and to download the Whisper model.

Because the work runs on the NPU, transcription barely uses the CPU or GPU, so you can keep working normally while a batch of videos is processed in the background.

---

## Contents

1. [Requirements](#requirements)
2. [Install the NPU driver](#install-the-npu-driver)
3. [Install NPU Transcriber](#install-npu-transcriber)
4. [Using the program](#using-the-program)
5. [Where files are stored](#where-files-are-stored)
6. [Updating, repairing and removing](#updating-repairing-and-removing)
7. [Troubleshooting](#troubleshooting)

---

## Requirements

| What | Notes |
|---|---|
| Intel Core Ultra processor with NPU | Meteor Lake, Lunar Lake, Arrow Lake or newer |
| Linux | Written and tested for **Debian 13 (trixie)**; other recent distributions work similarly |
| Linux kernel with the `intel_vpu` driver | Included in Debian's kernel (6.12). Lunar Lake needs 6.11 or newer |
| Intel NPU user-space driver | Not packaged by Debian — see [below](#install-the-npu-driver) |
| Python 3 with Tkinter and venv | `sudo apt install python3-tk python3-venv` |
| ffmpeg | `sudo apt install ffmpeg` — reads the audio from your videos |
| ~2 GB free disk space | Python packages (~0.5 GB) plus the model (0.5–1.6 GB) |

The program also runs on the **CPU** if no NPU is available — just slower.

---

## Install the NPU driver

The NPU needs two pieces: a **kernel driver** (already in Debian) and Intel's **user-space driver** (installed by hand, because Debian doesn't package it).

### 1. Check that the kernel sees the NPU

```bash
lspci | grep -iE "npu|vpu|processing accel"
lsmod | grep intel_vpu
ls -l /dev/accel/accel0
```

You should see an Intel NPU in `lspci`, the `intel_vpu` module loaded, and a device `/dev/accel/accel0`.
If `lspci` shows nothing, check that the NPU is enabled in your BIOS/UEFI.

### 2. Allow your user to access the NPU

The NPU device belongs to the `render` group (the `video` group is not enough):

```bash
sudo usermod -aG render $USER
```

Then **log out and back in** (or reboot). Check with `groups` that `render` is listed.

### 3. Install Intel's NPU user-space driver

Intel publishes ready-made packages for Ubuntu on the [linux-npu-driver releases page](https://github.com/intel/linux-npu-driver/releases). The **Ubuntu 24.04** build works on Debian trixie.

> Use the **`ubuntu2404`** archive, not `ubuntu2604` — the newer build may need newer system libraries than Debian trixie has.
> Check the releases page for the newest version; the commands below use v1.38.0 as an example.

```bash
mkdir ~/npu-driver && cd ~/npu-driver
wget https://github.com/intel/linux-npu-driver/releases/download/v1.38.0/linux-npu-driver-v1.38.0.20260910-34487311128-ubuntu2404.tar.gz
tar -xf linux-npu-driver-v1.38.0.20260910-34487311128-ubuntu2404.tar.gz

sudo apt update
sudo apt install libtbb12 libze1
sudo dpkg -i *.deb
```

If `apt` cannot find `libze1` (the Level Zero loader), the release notes link a `libze1` `.deb` that you can install the same way with `sudo dpkg -i`.

**Reboot** afterwards.

> These packages are installed from downloaded files, so `apt upgrade` will **not** update them. To get a newer driver, repeat this step with the newer release.

### 4. Check that it works

Start NPU Transcriber (next section). Next to the **Device** selection it shows **✓ NPU driver ready** — or, if something is missing, what it is and how to fix it.

To watch the NPU working during a transcription, this counter increases while the NPU is busy:

```bash
watch -n 2 cat $(find /sys/devices -name npu_busy_time_us | head -1)
```

---

## Install NPU Transcriber

The whole program is a single file, `npu-transcriber.py`.

1. Install the system pieces it needs:

   ```bash
   sudo apt install python3-tk python3-venv ffmpeg
   ```

2. Save `npu-transcriber.py`, for example in your home folder, and start it:

   ```bash
   python3 ~/npu-transcriber.py
   ```

   No virtual environment needs to be activated — the program finds and switches to its own environment by itself.

3. **First start:** a setup window explains that the program uses its own Python environment and offers to create it. Click **Set up**. It downloads a few hundred MB of Python packages (OpenVINO GenAI and helpers), which takes a few minutes. The program then restarts into its new environment.

4. **Download the model:** the **Advanced** section opens by itself and shows the model as *not downloaded yet*. Click **Download…**, keep the recommended **int8** version and click **Download** (~0.8 GB).

5. **Compile the model:** the model has to be compiled once for your NPU, which takes about a minute. Click **Compile now** — or simply start your first transcription, which compiles it automatically. The compiled model is saved, so later starts take seconds.

### Optional: add it to your applications menu

```bash
cat > ~/.local/share/applications/npu-transcriber.desktop <<EOF
[Desktop Entry]
Type=Application
Name=NPU Transcriber
Comment=Transcribe videos with Whisper on the Intel NPU
Exec=python3 $HOME/npu-transcriber.py %F
Icon=audio-input-microphone
Terminal=false
Categories=AudioVideo;Utility;
EOF
```

"NPU Transcriber" then appears in your applications menu, and you can open videos with it from your file manager's *Open with* menu.

You can also pass files on the command line: `python3 ~/npu-transcriber.py video1.mp4 video2.mkv`

---

## Using the program

1. **Add files** — click **Add files…** and pick one or more videos or audio files.
2. **Settings**
   - **Language** — *Auto-detect*, or set the language of your recordings for the most reliable results.
   - **Device** — **NPU** (recommended) or **CPU**. The NPU status is shown right next to it.
   - **Save to** — leave empty to save the results next to each video, or pick a folder.
   - **Create** — subtitles (`.srt`), plain text (`.txt`), or both.
3. Click **Start transcription.**

Below the progress bar you see the elapsed time and the **estimated time left**, and below that the file currently being processed. The estimate is based on how fast your past transcriptions ran. The transcript of each finished file also appears in the log.

**Stop after current file** finishes the file that is being transcribed and then stops. **Open output folder** opens the folder with the results.

### The Advanced section

Advanced is collapsed by default and **opens by itself when something there needs your attention** (model missing or not compiled, environment problem). Once the problem is solved it closes again.

- **Model** — the model folder (**Browse…** to use a model stored elsewhere, **Download…** to download one), whether it is downloaded, and whether it is compiled for the selected device, with **Compile now** or **Clear compiled models**.
- **Python environment** — the environment the program runs in, with **Update packages** and **Rebuild environment**.

### Model versions

| Version | Size | Notes |
|---|---|---|
| **int8** (recommended) | ~0.8 GB | Practically the same quality as fp16; the NPU handles 8-bit best |
| fp16 | ~1.6 GB | Reference quality, but slower on the NPU |
| int4 | ~0.5 GB | Smallest, but makes more mistakes on names and quiet speech |

---

## Where files are stored

| Location | Contents |
|---|---|
| `~/.config/npu-transcriber/settings.json` | Your settings and the measured transcription speeds |
| `~/.local/share/npu-transcriber/venv/` | The program's private Python environment |
| `~/.local/share/npu-transcriber/requirements.lock` | The exact package versions that were installed |
| `~/.local/share/npu-transcriber/models/` | Downloaded models (one folder per model) |
| `~/.cache/npu-transcriber/` | Compiled models — safe to delete, they are recreated when needed |

`~/.local`, `~/.config` and `~/.cache` are hidden folders; press **Ctrl+H** in your file manager to see them.

**Downloaded vs. compiled models:** the downloaded model is the general, device-independent version (the valuable part — deleting it means downloading again). The compiled model is a translation of it for your specific NPU, driver and OpenVINO version; the program can always recreate it, it just takes about a minute.

---

## Updating, repairing and removing

**Update the Python packages:** *Advanced → Update packages* installs the latest OpenVINO GenAI and restarts the program. The new versions are recorded, so a later rebuild installs exactly those again.

**Repair the environment:** if the environment breaks — for example after a Debian upgrade brings a new Python version — the program notices on start and offers to rebuild it. You can also do it any time with *Advanced → Rebuild environment*. Models and settings are kept.

**Update the model:** *Advanced → Download…* always fetches the latest version of the chosen model. Afterwards, use **Clear compiled models** so the status correctly shows that it needs compiling again.

**Remove the program completely:**

```bash
rm -rf ~/.config/npu-transcriber ~/.local/share/npu-transcriber ~/.cache/npu-transcriber
rm -f ~/npu-transcriber.py ~/.local/share/applications/npu-transcriber.desktop
```

To also remove the NPU driver:

```bash
sudo dpkg --purge intel-driver-compiler-npu intel-fw-npu intel-level-zero-npu
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| *"needs Tkinter for its window"* in the terminal | `sudo apt install python3-tk` |
| *"Python's venv module is missing"* in the setup window | `sudo apt install python3-venv` |
| *"ffmpeg is missing"* | `sudo apt install ffmpeg` |
| *"No NPU device found"* | Enable the NPU in the BIOS/UEFI; check that `lsmod \| grep intel_vpu` shows the kernel driver |
| *"No permission to use the NPU"* | `sudo usermod -aG render $USER`, then log out and back in |
| *"The NPU user-space driver is missing"* | Install Intel's driver packages — see [Install the NPU driver](#install-the-npu-driver) |
| Model download fails | Check your internet connection; Hugging Face must be reachable |
| The first start of a transcription takes about a minute | Normal: the model is being compiled for the NPU. Use **Compile now** to do this in advance |
| A warning about *hwloc … invalid information from the operating system* | Harmless (it concerns the hybrid CPU layout of Core Ultra); the program hides it |
| Window, text or check boxes are too small or too large | The program follows your desktop's scaling setting (`Xft.dpi`). To override it, add `"ui_scale": 1.5,` (1.0 = 100 %, 2.0 = 200 %) to `~/.config/npu-transcriber/settings.json` |
