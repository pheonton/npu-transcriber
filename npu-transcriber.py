#!/usr/bin/env python3
"""NPU Transcriber - transcribe videos with Whisper on the Intel NPU.

Start it with the system Python - no venv activation needed:
    python3 npu-transcriber.py [video files...]

The program manages its own Python environment:
  * First start: it offers to create a private environment in
    ~/.local/share/npu-transcriber/venv and install OpenVINO GenAI into it.
  * Every start: it checks that it runs inside that environment and restarts
    itself there if not. A broken environment (e.g. after a Python upgrade)
    can be rebuilt with one click.
  * Installed package versions are recorded in requirements.lock so a rebuild
    reproduces the exact working setup.

It never installs system packages. Missing system pieces (python3-tk,
python3-venv, ffmpeg, the NPU driver) are detected and the command to fix
them is shown.

Files:
  ~/.config/npu-transcriber/settings.json        settings
  ~/.local/share/npu-transcriber/venv/           private Python environment
  ~/.local/share/npu-transcriber/requirements.lock
  ~/.local/share/npu-transcriber/models/         default place for downloaded models
  ~/.cache/npu-transcriber/                      compiled-model cache (safe to delete)
"""
import hashlib
import importlib.util
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time

APP = "npu-transcriber"
_XDG_DATA = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
_XDG_CONFIG = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
_XDG_CACHE = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")

DATA_DIR = os.path.join(_XDG_DATA, APP)
MANAGED_ENV = os.path.join(DATA_DIR, "venv")
LOCK_FILE = os.path.join(DATA_DIR, "requirements.lock")
MODELS_DIR = os.path.join(DATA_DIR, "models")
CONFIG_FILE = os.path.join(_XDG_CONFIG, APP, "settings.json")
CACHE_DIR = os.path.join(_XDG_CACHE, APP)

PACKAGES = ["openvino-genai", "huggingface_hub"]
MODEL_CHOICES = [
    ("int8 - recommended, ~0.8 GB", "OpenVINO/whisper-large-v3-turbo-int8-ov"),
    ("fp16 - reference quality, ~1.6 GB, slower on the NPU", "OpenVINO/whisper-large-v3-turbo-fp16-ov"),
    ("int4 - smallest, ~0.5 GB, more mistakes", "OpenVINO/whisper-large-v3-turbo-int4-ov"),
]
DEFAULT_MODEL = os.path.join(MODELS_DIR, MODEL_CHOICES[0][1].split("/")[1])
NPU_DEVICE = "/dev/accel/accel0"

# Hide a harmless hwloc warning about Meteor Lake's hybrid CPU layout.
os.environ.setdefault("HWLOC_HIDE_ERRORS", "2")
os.environ.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk
    import tkinter.font as tkfont
except ImportError:
    sys.exit("NPU Transcriber needs Tkinter for its window.\n"
             "Install it with:  sudo apt install python3-tk")


# ================================================================ environment helpers

def load_config():
    try:
        with open(CONFIG_FILE, encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_config(cfg):
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def update_config(**changes):
    cfg = load_config()
    cfg.update(changes)
    save_config(cfg)


def configured_env():
    return load_config().get("env") or MANAGED_ENV


def env_python(env_dir):
    return os.path.join(env_dir, "bin", "python3")


def is_venv(env_dir):
    """True if the folder is a usable Python virtual environment (its python3 exists)."""
    return (os.path.isfile(os.path.join(env_dir, "pyvenv.cfg"))
            and os.path.isfile(env_python(env_dir)))


def running_in(env_dir):
    return os.path.realpath(sys.prefix) == os.path.realpath(env_dir)


def base_python():
    """The system Python this one is based on (also correct when running inside a venv)."""
    for candidate in (getattr(sys, "_base_executable", None), sys.executable, "/usr/bin/python3"):
        if candidate and os.path.isfile(candidate):
            return candidate
    return "python3"


def safe_to_create(path):
    """Only create/clear an environment where that can't destroy unrelated files."""
    if not os.path.exists(path):
        return True
    if os.path.realpath(path) == os.path.realpath(MANAGED_ENV):
        return True
    if os.path.isfile(os.path.join(path, "pyvenv.cfg")):
        return True
    return os.path.isdir(path) and not os.listdir(path)


def has_openvino_genai():
    return importlib.util.find_spec("openvino_genai") is not None


def venv_module_available():
    return importlib.util.find_spec("ensurepip") is not None


def script_path():
    return os.path.abspath(__file__)


def relaunch_in(env_dir, args):
    """Replace this process with the same script running in env_dir's Python."""
    py = env_python(env_dir)
    env = dict(os.environ, NPU_TRANSCRIBER_RELAUNCHED="1")
    sys.stdout.flush()
    os.execve(py, [py, script_path()] + list(args), env)


def restart_with_system_python(args):
    """Restart with the system Python (used to rebuild the environment we run in)."""
    py = base_python()
    env = dict(os.environ)
    env.pop("NPU_TRANSCRIBER_RELAUNCHED", None)
    sys.stdout.flush()
    os.execve(py, [py, script_path()] + list(args), env)


def stream(cmd, send):
    """Run a command, sending each output line to send(); returns the exit code."""
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
    except OSError as e:
        send(f"Could not run {cmd[0]}: {e}")
        return 1
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            send(line)
    return proc.wait()


def install_packages(env_dir, send, upgrade=False):
    """Install (or upgrade) the app's packages into env_dir and record the versions."""
    py = env_python(env_dir)
    stream([py, "-m", "pip", "install", "--upgrade", "pip"], send)

    if upgrade:
        send("Updating packages to the latest versions…")
        rc = stream([py, "-m", "pip", "install", "--upgrade", *PACKAGES], send)
    elif os.path.isfile(LOCK_FILE):
        send("Installing the recorded package versions…")
        rc = stream([py, "-m", "pip", "install", "-r", LOCK_FILE], send)
        if rc != 0:
            send("Recorded versions failed to install - falling back to the latest versions.")
            rc = stream([py, "-m", "pip", "install", *PACKAGES], send)
    else:
        send("Installing OpenVINO GenAI and helpers (a few hundred MB)…")
        rc = stream([py, "-m", "pip", "install", *PACKAGES], send)
    if rc != 0:
        return False

    check = subprocess.run([py, "-c", "import openvino, openvino_genai; print(openvino.__version__)"],
                           capture_output=True, text=True)
    if check.returncode != 0:
        send("OpenVINO GenAI was installed but cannot be imported:\n" + check.stderr.strip())
        return False
    send(f"OpenVINO {check.stdout.strip()} is ready.")

    freeze = subprocess.run([py, "-m", "pip", "freeze"], capture_output=True, text=True)
    if freeze.returncode == 0 and freeze.stdout.strip():
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(LOCK_FILE, "w", encoding="utf-8") as f:
            f.write(freeze.stdout)
        send(f"Recorded package versions in {LOCK_FILE}")
    return True


# ================================================================ system checks (detect only)

def ffmpeg_problem():
    if shutil.which("ffmpeg"):
        return None
    return "ffmpeg is missing (needed to read videos):  sudo apt install ffmpeg"


def npu_problem(devices):
    """Explain why the NPU is not usable, or None if it is."""
    if "NPU" in devices:
        return None
    if not os.path.exists(NPU_DEVICE):
        return ("No NPU device found (/dev/accel/accel0). Check that the NPU is enabled in "
                "the BIOS and that the intel_vpu kernel module is loaded.")
    if not os.access(NPU_DEVICE, os.R_OK | os.W_OK):
        return ("No permission to use the NPU. Run:  sudo usermod -aG render $USER  "
                "and then log out and back in.")
    return ("The NPU user-space driver is missing. Install Intel's linux-npu-driver "
            "packages (github.com/intel/linux-npu-driver/releases) and reboot.")


# ================================================================ shared UI bits

FONT_SIZE = 11  # points, like GNOME's default text size
UI_SCALE = 1.0  # set by apply_theme() from the desktop's DPI setting


def px(value):
    """Convert a size in 'normal' (96 DPI) pixels to pixels on this screen."""
    return int(round(value * UI_SCALE))


def _xft_dpi():
    """The DPI the desktop tells X applications to use (GNOME, KDE, Xfce set Xft.dpi)."""
    try:
        out = subprocess.run(["xrdb", "-query"], capture_output=True, text=True, timeout=2).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        if line.startswith("Xft.dpi:"):
            try:
                return float(line.split(":", 1)[1])
            except ValueError:
                return None
    return None


def _gnome_text_scale():
    try:
        out = subprocess.run(["gsettings", "get", "org.gnome.desktop.interface", "text-scaling-factor"],
                             capture_output=True, text=True, timeout=2).stdout.strip()
        return float(out) if out else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def detect_ui_scale(root):
    """Find the desktop's scaling factor (1.0 = 96 DPI).

    Order: "ui_scale" in settings.json (manual override) -> Xft.dpi ->
    GDK_SCALE / GDK_DPI_SCALE -> GNOME text scaling -> what Tk detected itself.
    """
    override = load_config().get("ui_scale")
    if isinstance(override, (int, float)) and override > 0:
        return float(override)
    dpi = _xft_dpi()
    if dpi:
        return dpi / 96
    try:
        env_scale = float(os.environ.get("GDK_SCALE", 1)) * float(os.environ.get("GDK_DPI_SCALE", 1))
    except ValueError:
        env_scale = 1.0
    if env_scale != 1.0:
        return env_scale
    text_scale = _gnome_text_scale()
    if text_scale and text_scale != 1.0:
        return text_scale
    return max(1.0, root.winfo_fpixels("1i") / 96)


def apply_theme(root):
    """Theme + DPI scaling, so text, check/radio boxes, arrows and window sizes all match."""
    global UI_SCALE
    UI_SCALE = min(4.0, max(0.75, detect_ui_scale(root)))
    # Fonts are sized in points; tk scaling = screen pixels per point.
    root.tk.call("tk", "scaling", UI_SCALE * 96 / 72)
    # Tk's standard fonts are 10 pt; match GNOME's default text size instead.
    for name in ("TkDefaultFont", "TkTextFont", "TkFixedFont", "TkMenuFont", "TkHeadingFont",
                 "TkCaptionFont", "TkTooltipFont", "TkIconFont"):
        tkfont.nametofont(name).configure(size=FONT_SIZE)
    family = tkfont.nametofont("TkDefaultFont").actual("family")
    # Created in Tk directly: a tkfont.Font object deletes its font when garbage-collected.
    root.tk.call("font", "create", "AppHeading", "-family", family, "-size", FONT_SIZE + 3, "-weight", "bold")
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    # The clam theme draws these in fixed pixels - scale them like the text.
    style.configure("TButton", padding=px(5))
    style.configure("TEntry", padding=px(2))
    style.configure("TCombobox", padding=px(2))
    style.configure("TCheckbutton", padding=px(2), indicatorsize=px(13),
                    indicatormargin=(px(1), px(1), px(5), px(1)))
    style.configure("TRadiobutton", padding=px(2), indicatorsize=px(13),
                    indicatormargin=(px(1), px(1), px(5), px(1)))
    for name in ("TScrollbar", "Vertical.TScrollbar", "Horizontal.TScrollbar", "TCombobox"):
        style.configure(name, arrowsize=px(14))
    style.configure("Horizontal.TProgressbar", thickness=px(14), arrowsize=px(14))
    root.option_add("*Scrollbar.width", px(14))  # classic Tk scrollbar of the log box
    # Tk's file dialog lists dot-files by default. Its code is loaded on first use, so
    # trigger loading with an invalid call, then hide them and add a "Show hidden" toggle.
    # Its file list is also a fixed 400x120 pixels - wrap the builder to scale it.
    try:
        root.tk.call("catch", "tk_getOpenFile -no-such-option")
        root.tk.call("set", "::tk::dialog::file::showHiddenBtn", "1")
        root.tk.call("set", "::tk::dialog::file::showHiddenVar", "0")
        root.tk.eval("""
            rename ::tk::dialog::file::Create ::tk::dialog::file::CreateUnscaled
            proc ::tk::dialog::file::Create {w class} {
                ::tk::dialog::file::CreateUnscaled $w $class
                $w.contents.icons.cHull.canvas configure -width %d -height %d
            }""" % (px(400), px(120)))
    except tk.TclError:
        pass


# ================================================================ file choosers
# GNOME's file chooser via zenity when installed, otherwise Tk's own.

def run_chooser(parent, cmd):
    """Run a chooser tool while the window keeps redrawing but ignores clicks.

    Returns the chosen paths, [] if cancelled, or None if the tool failed (then use Tk's)."""
    win = parent.winfo_toplevel()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                text=True, encoding="utf-8", errors="surrogateescape")
    except OSError:
        return None
    try:
        win.tk.call("tk", "busy", "hold", win)
    except tk.TclError:
        pass
    try:
        while proc.poll() is None:
            win.update()
            time.sleep(0.03)
    finally:
        try:
            win.tk.call("tk", "busy", "forget", win)
        except tk.TclError:
            pass
    out = proc.stdout.read()
    if proc.returncode == 1:
        return []
    if proc.returncode != 0:
        return None
    return [line for line in out.splitlines() if line]


def ask_files(parent, title, filetypes, initialdir=None):
    """Let the user pick one or more files; returns a list of paths (empty if cancelled)."""
    start = initialdir or os.path.expanduser("~")
    paths = None
    if shutil.which("zenity"):
        cmd = ["zenity", "--file-selection", "--multiple", "--separator=\n",
               "--title", title, "--filename", os.path.join(start, "")]
        for name, patterns in filetypes:
            cmd.append(f"--file-filter={name} | {patterns}")
        paths = run_chooser(parent, cmd)
    if paths is None:
        paths = filedialog.askopenfilenames(parent=parent, title=title, filetypes=filetypes,
                                            initialdir=start)
    return list(paths)


def ask_folder(parent, title, initialdir=None):
    """Let the user pick a folder; returns its path, or "" if cancelled."""
    start = initialdir or os.path.expanduser("~")
    paths = None
    if shutil.which("zenity"):
        paths = run_chooser(parent, ["zenity", "--file-selection", "--directory",
                                     "--title", title, "--filename", os.path.join(start, "")])
    if paths is None:
        return filedialog.askdirectory(parent=parent, title=title, initialdir=start) or ""
    return paths[0] if paths else ""


class LogBox(scrolledtext.ScrolledText):
    def __init__(self, parent, **kw):
        super().__init__(parent, wrap="word", state="disabled", **kw)

    def add(self, message):
        self.config(state="normal")
        self.insert("end", message + "\n")
        self.see("end")
        self.config(state="disabled")


# ================================================================ setup window

class SetupWindow(tk.Tk):
    """Creates, repairs or rebuilds the program's Python environment."""

    def __init__(self, env_dir, files, rebuild=False):
        super().__init__()
        apply_theme(self)
        self.title("NPU Transcriber - setup")
        self.geometry(f"{px(700)}x{px(560)}")
        self.minsize(px(560), px(460))
        self.files = files
        self.rebuild = rebuild
        self.events = queue.Queue()
        self.worker = None

        pad = {"padx": px(14), "pady": px(6)}
        ttk.Label(self, text="Python environment setup", font="AppHeading").pack(anchor="w", **pad)
        self.intro = ttk.Label(self, wraplength=px(660), justify="left")
        self.intro.pack(anchor="w", fill="x", **pad)
        self.checks = ttk.Label(self, wraplength=px(660), justify="left")
        self.checks.pack(anchor="w", fill="x", **pad)

        buttons = ttk.Frame(self)
        buttons.pack(fill="x", **pad)
        self.go_btn = ttk.Button(buttons, command=self._go)
        self.other_btn = ttk.Button(buttons, text="Use another environment…", command=self._choose_other)
        self.quit_btn = ttk.Button(buttons, text="Quit", command=self.destroy)
        self.go_btn.pack(side="left")
        self.other_btn.pack(side="left", padx=px(8))
        self.quit_btn.pack(side="right")

        self.progress = ttk.Progressbar(self, mode="indeterminate")
        self.progress.pack(fill="x", padx=px(14))
        self.log = LogBox(self, height=12)
        self.log.pack(fill="both", expand=True, **pad)

        self._set_target(env_dir)
        self.after(100, self._poll)
        if rebuild:
            self.after(300, self._go)

    def _set_target(self, env_dir):
        self.env_dir = env_dir
        if self.rebuild:
            self.mode = "rebuild"
            text = (f"Rebuilding the Python environment at:\n{env_dir}\n\n"
                    "It is deleted and created again; packages are reinstalled in the recorded versions. "
                    "Your models and settings are not affected.")
            label = "Rebuild"
        elif not os.path.exists(env_dir):
            self.mode = "create"
            text = ("NPU Transcriber uses its own Python environment with OpenVINO GenAI, "
                    f"so it doesn't touch the rest of your system. It will be created at:\n{env_dir}\n\n"
                    "This downloads a few hundred MB of Python packages and takes a few minutes.")
            label = "Set up"
        elif is_venv(env_dir):
            self.mode = "install"
            text = (f"The Python environment at\n{env_dir}\nis missing OpenVINO GenAI. "
                    "Install the required packages into it now?")
            label = "Install packages"
        else:
            self.mode = "rebuild"
            text = (f"The Python environment at\n{env_dir}\nis broken or incomplete "
                    "(this happens e.g. after a system Python upgrade). Rebuild it now? "
                    "Your models and settings are not affected.")
            label = "Rebuild"
        self.intro.config(text=text)
        self.go_btn.config(text=label)
        self._show_checks()

    def _show_checks(self):
        lines = []
        blocking = False
        if self.mode in ("create", "rebuild") and not venv_module_available():
            lines.append("✗ Python's venv module is missing:  sudo apt install python3-venv")
            blocking = True
        if self.mode in ("create", "rebuild") and not safe_to_create(self.env_dir):
            lines.append("✗ That folder already contains other files, so it won't be used. "
                         "Choose an empty folder or an existing environment.")
            blocking = True
        problem = ffmpeg_problem()
        if problem:
            lines.append("⚠ " + problem)
        self.checks.config(text="\n".join(lines), foreground="#b91c1c" if blocking else "#b45309")
        self.go_btn.config(state="disabled" if blocking else "normal")

    def _choose_other(self):
        folder = ask_folder(self, "Choose a Python environment, or an empty folder for a new one")
        if not folder:
            return
        update_config(env=folder)
        if is_venv(folder) and not self.rebuild:
            self.destroy()
            relaunch_in(folder, self.files)  # the restarted program checks its packages
        self.rebuild = False
        self._set_target(folder)

    def _go(self):
        self._show_checks()
        if str(self.go_btn.cget("state")) == "disabled":
            return
        for b in (self.go_btn, self.other_btn):
            b.config(state="disabled")
        self.progress.start(12)
        self.worker = threading.Thread(target=self._work, daemon=True)
        self.worker.start()

    def _work(self):
        send = lambda m: self.events.put(("log", m))  # noqa: E731
        ok = False
        try:
            env_dir = self.env_dir
            if self.mode in ("create", "rebuild"):
                os.makedirs(os.path.dirname(os.path.abspath(env_dir)), exist_ok=True)
                cmd = [base_python(), "-m", "venv"]
                if os.path.exists(env_dir):
                    cmd.append("--clear")
                send(("Rebuilding" if self.mode == "rebuild" else "Creating") + f" the environment in {env_dir}…")
                if stream(cmd + [env_dir], send) != 0 or not is_venv(env_dir):
                    send("Creating the environment failed. If the messages mention ensurepip, run:\n"
                         "    sudo apt install python3-venv")
                    return
            ok = install_packages(env_dir, send)
            if not ok:
                send("Installing the packages failed - see the messages above. "
                     "Check your internet connection and try again.")
        finally:
            self.events.put(("done", ok))

    def _poll(self):
        try:
            while True:
                kind, data = self.events.get_nowait()
                if kind == "log":
                    self.log.add(data)
                elif kind == "done":
                    self.progress.stop()
                    if data:
                        update_config(env=self.env_dir)
                        self.log.add("Setup complete - starting the transcriber…")
                        self.update()
                        self.after(800, self._launch)
                    else:
                        self.rebuild = False
                        self._set_target(self.env_dir)
                        self.other_btn.config(state="normal")
        except queue.Empty:
            pass
        self.after(150, self._poll)

    def _launch(self):
        self.destroy()
        relaunch_in(self.env_dir, self.files)


# ================================================================ transcription

def load_audio(path):
    """Decode any audio/video file to 16 kHz mono float32 using ffmpeg."""
    import numpy as np  # installed together with OpenVINO

    # aresample=async=1 fills gaps in the audio (damaged files, dropped packets) and a late
    # start with silence, so positions in the samples match the video's timeline.
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path,
           "-vn", "-af", "aresample=async=1:first_pts=0",
           "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
    try:
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found - install it with: sudo apt install ffmpeg")
    except subprocess.CalledProcessError as e:
        raise RuntimeError("ffmpeg could not read the file: " + e.stderr.decode(errors="replace").strip())
    audio = np.frombuffer(raw, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError("the file contains no audio track")
    return audio


def srt_time(seconds):
    ms = int(round(max(seconds, 0) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


SAMPLE_RATE = 16000
SAMPLE_SECONDS = 120         # length of the one-time speed measurement…
MIN_SAMPLE_FILE_SECONDS = 240  # …only taken from files at least this long


def media_duration(path):
    """Length of a video/audio file in seconds (via ffprobe), or None if unknown."""
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "default=noprint_wrappers=1:nokey=1", path],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return float(out) if out else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def result_entries(result, duration):
    """Whisper's result as [(start_s, end_s, text), ...]."""
    chunks = [c for c in (result.chunks or []) if c.text.strip()]
    if chunks:
        entries = []
        for c in chunks:
            begin = max(c.start_ts, 0)
            finish = c.end_ts if c.end_ts and c.end_ts > 0 else duration
            entries.append((begin, max(finish, begin), c.text.strip()))
        return entries
    texts = getattr(result, "texts", None)
    text = (texts[0] if texts else str(result)).strip()
    return [(0.0, duration, text)] if text else []


def speed_key(model, device):
    """Speed is remembered per device and model (int8 on the NPU ≠ fp16 on the CPU)."""
    return f"{device} · {os.path.basename(os.path.normpath(model))}"


def load_speeds():
    speeds = load_config().get("speed")
    return speeds if isinstance(speeds, dict) else {}


def save_speed(key, measured):
    """Store a measured speed, blending it with the earlier value so estimates improve over time."""
    speeds = load_speeds()
    old = speeds.get(key)
    speeds[key] = round(0.6 * old + 0.4 * measured if isinstance(old, (int, float)) else measured, 2)
    update_config(speed=speeds)
    return speeds[key]


def write_outputs(entries, src, out_dir, want_srt, want_txt):
    folder = out_dir or os.path.dirname(os.path.abspath(src))
    base = os.path.join(folder, os.path.splitext(os.path.basename(src))[0])
    text = " ".join(t for _, _, t in entries)

    written = []
    if want_srt:
        with open(base + ".srt", "w", encoding="utf-8") as srt:
            for i, (begin, finish, line) in enumerate(entries, 1):
                srt.write(f"{i}\n{srt_time(begin)} --> {srt_time(finish)}\n{line}\n\n")
        written.append(base + ".srt")
    if want_txt:
        with open(base + ".txt", "w", encoding="utf-8") as txt:
            txt.write(text + "\n")
        written.append(base + ".txt")
    return written, text


def roughly(seconds):
    """Human-friendly duration for estimates: 'less than a minute', 'about 4 min', 'about 1 h 05 min'."""
    seconds = max(0, seconds)
    if seconds < 60:
        return "less than a minute"
    minutes = int(round(seconds / 60))
    if minutes < 60:
        return f"about {minutes} min"
    return f"about {minutes // 60} h {minutes % 60:02} min"


def clock(seconds):
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}:{seconds % 3600 // 60:02}:{seconds % 60:02}"
    return f"{seconds // 60}:{seconds % 60:02}"


def model_cache_dir(model, device):
    """Compiled-model cache for one model folder on one device.

    Each model/device pair gets its own subfolder, so the program can tell
    whether that exact combination has been compiled before.
    """
    real = os.path.realpath(model)
    tag = hashlib.sha1(real.encode()).hexdigest()[:8]
    return os.path.join(CACHE_DIR, f"{os.path.basename(real)}-{tag}", device.lower())


def is_compiled(model, device):
    path = model_cache_dir(model, device)
    try:
        return any(entry.is_file() for entry in os.scandir(path))
    except OSError:
        return False


class Transcriber:
    """Loads the Whisper pipeline once and reuses it for every file."""

    def __init__(self):
        self.pipe = None
        self.key = None

    def is_loaded(self, model, device):
        return self.pipe is not None and self.key == (model, device)

    def get(self, model, device, log):
        if not self.is_loaded(model, device):
            import openvino_genai as ov_genai

            cached = is_compiled(model, device)
            log(f"Loading the model on {device}…" if cached else
                f"Compiling the model for {device} - this takes about a minute the first time…")
            cache = model_cache_dir(model, device)
            os.makedirs(cache, exist_ok=True)
            t0 = time.time()
            try:
                self.pipe = ov_genai.WhisperPipeline(model, device, CACHE_DIR=cache)
            except Exception:
                self.pipe = ov_genai.WhisperPipeline(model, device)
            self.key = (model, device)
            log(f"Model ready on {device} after {time.time() - t0:.0f} s.")
            if not is_compiled(model, device):
                log("Note: the compiled model could not be saved, so it is compiled again at every start.")
        return self.pipe


def folder_size(path):
    total = 0
    for root, _, names in os.walk(path):
        for n in names:
            try:
                total += os.path.getsize(os.path.join(root, n))
            except OSError:
                pass
    return total


def looks_like_model(path):
    return os.path.isfile(os.path.join(path, "openvino_encoder_model.xml"))


# ================================================================ model download dialog

class ModelDialog(tk.Toplevel):
    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.title("Download a Whisper model")
        self.geometry(f"{px(560)}x{px(330)}")
        self.transient(app)
        self.after(150, self._grab)
        self.downloading = False
        self.result = queue.Queue()

        pad = {"padx": px(12), "pady": px(4)}
        ttk.Label(self, text="Whisper large-v3-turbo for OpenVINO - choose a version:").pack(anchor="w", **pad)
        self.choice = tk.StringVar(value=MODEL_CHOICES[0][1])
        for label, repo in MODEL_CHOICES:
            ttk.Radiobutton(self, text=label, value=repo, variable=self.choice).pack(anchor="w", padx=px(24))

        dest = ttk.Frame(self)
        dest.pack(fill="x", padx=px(12), pady=(px(12), px(4)))
        ttk.Label(dest, text="Save in").pack(side="left")
        self.dest_var = tk.StringVar(value=MODELS_DIR)
        ttk.Entry(dest, textvariable=self.dest_var).pack(side="left", fill="x", expand=True, padx=px(6))
        ttk.Button(dest, text="Browse…", command=self._browse).pack(side="left")

        self.progress = ttk.Progressbar(self, mode="determinate", maximum=100)
        self.progress.pack(fill="x", **pad)
        self.status = ttk.Label(self, text="")
        self.status.pack(anchor="w", **pad)

        buttons = ttk.Frame(self)
        buttons.pack(fill="x", padx=px(12), pady=px(8))
        self.go_btn = ttk.Button(buttons, text="Download", command=self._start)
        self.close_btn = ttk.Button(buttons, text="Close", command=self._close)
        self.go_btn.pack(side="left")
        self.close_btn.pack(side="right")
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _grab(self):
        try:
            self.grab_set()  # keep the main window inactive while this dialog is open
        except tk.TclError:
            pass

    def _browse(self):
        folder = ask_folder(self, "Where to save models")
        if folder:
            self.dest_var.set(folder)

    def _close(self):
        if self.downloading:
            messagebox.showinfo("Download running", "Please wait until the download has finished.", parent=self)
            return
        self.destroy()

    def _start(self):
        repo = self.choice.get()
        self.target = os.path.join(self.dest_var.get().strip(), repo.split("/")[1])
        self.downloading = True
        self.go_btn.config(state="disabled")
        self.status.config(text="Checking download size…")
        self.total = 0
        threading.Thread(target=self._download, args=(repo,), daemon=True).start()
        self.after(500, self._watch)

    def _download(self, repo):
        try:
            from huggingface_hub import HfApi, snapshot_download
            try:
                info = HfApi().model_info(repo, files_metadata=True)
            except Exception as e:
                # Fails fast when offline; the download itself would only hang retrying.
                self.result.put(("error", f"Could not reach Hugging Face to download {repo}.\n"
                                          f"Check your internet connection.\n\n({type(e).__name__}: {e})"))
                return
            self.total = sum((s.size or 0) for s in info.siblings)
            os.makedirs(self.target, exist_ok=True)
            snapshot_download(repo_id=repo, local_dir=self.target)
            self.result.put(("ok", None))
        except Exception as e:
            self.result.put(("error", str(e)))

    def _watch(self):
        try:
            kind, data = self.result.get_nowait()
        except queue.Empty:
            done = folder_size(self.target) if os.path.isdir(self.target) else 0
            if self.total:
                self.progress.config(value=min(99, done * 100 / self.total))
                self.status.config(text=f"Downloading… {done / 1e9:.2f} of {self.total / 1e9:.2f} GB")
            else:
                self.status.config(text=f"Downloading… {done / 1e9:.2f} GB")
            self.after(500, self._watch)
            return
        self.downloading = False
        if kind == "ok" and looks_like_model(self.target):
            self.progress.config(value=100)
            self.app.model_var.set(self.target)
            self.app._save_settings()
            self.app._log(f"Model downloaded to {self.target}")
            self.destroy()
        else:
            self.go_btn.config(state="normal")
            message = data or "the download does not look like an OpenVINO Whisper model"
            self.status.config(text="Download failed.")
            messagebox.showerror("Download failed", message, parent=self)


# ================================================================ main window

class App(tk.Tk):
    def __init__(self, initial_files=()):
        super().__init__()
        apply_theme(self)
        self.title("NPU Transcriber")
        height = min(px(760), self.winfo_screenheight() - px(80))
        self.geometry(f"{px(820)}x{height}")
        self.minsize(px(700), min(px(560), height))

        self.cfg = load_config()
        self.auto_device = None
        self.compiling = False
        self.ready = False            # Advanced opens/closes by itself only once the window is shown
        self.model_problem = False
        self.env_problem = False
        self.adv_problem = False
        self.adv_auto_opened = False
        self.adv_dismissed = False
        self.events = queue.Queue()
        self.transcriber = Transcriber()
        self.stop_event = threading.Event()
        self.worker = None
        self.busy = False
        self.run = None               # progress of the current transcription batch
        self.last_output_dir = None

        self._build()
        for path in initial_files:
            if os.path.isfile(path) and path not in self._files():
                self.listbox.insert("end", path)
        self._update_env_status()
        self._check_system()
        self._update_model_status()
        self.after(400, self._became_ready)
        self.after(100, self._poll)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------------------------------------------------------- layout
    def _build(self):
        cfg = self.cfg
        pad = {"padx": px(10), "pady": px(6)}
        row = {"padx": px(6), "pady": px(4)}

        files = ttk.LabelFrame(self, text="Files to transcribe")
        files.pack(fill="x", **pad)
        list_frame = ttk.Frame(files)
        list_frame.pack(side="left", fill="both", expand=True, padx=px(6), pady=px(6))
        self.listbox = tk.Listbox(list_frame, height=6, selectmode="extended", activestyle="none")
        scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        self.listbox.config(yscrollcommand=scroll.set)
        self.listbox.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        buttons = ttk.Frame(files)
        buttons.pack(side="right", fill="y", padx=px(6), pady=px(6))
        self.add_btn = ttk.Button(buttons, text="Add files…", command=self._add_files)
        self.remove_btn = ttk.Button(buttons, text="Remove selected", command=self._remove_selected)
        self.clear_btn = ttk.Button(buttons, text="Clear list", command=self._clear)
        for b in (self.add_btn, self.remove_btn, self.clear_btn):
            b.pack(fill="x", pady=px(2))

        settings = ttk.LabelFrame(self, text="Settings")
        settings.pack(fill="x", **pad)
        settings.columnconfigure(1, weight=1)

        ttk.Label(settings, text="Language").grid(row=0, column=0, sticky="w", **row)
        names = [n for n, _ in LANGUAGES]
        self.lang_var = tk.StringVar(value=cfg.get("language") if cfg.get("language") in names else names[0])
        ttk.Combobox(settings, textvariable=self.lang_var, values=names,
                     state="readonly", width=18).grid(row=0, column=1, sticky="w", **row)

        ttk.Label(settings, text="Device").grid(row=1, column=0, sticky="nw", **row)
        self.device_var = tk.StringVar(value=cfg.get("device") or "NPU")
        device_row = ttk.Frame(settings)
        device_row.grid(row=1, column=1, columnspan=2, sticky="ew", **row)
        self.device_box = ttk.Combobox(device_row, textvariable=self.device_var, values=["NPU", "CPU"],
                                       state="readonly", width=18)
        self.device_box.pack(side="left", anchor="n")
        self.npu_label = ttk.Label(device_row, text="Checking the NPU…", foreground="#666",
                                   wraplength=px(460), justify="left")
        self.npu_label.pack(side="left", anchor="w", padx=(px(12), px(0)), fill="x", expand=True)
        # Wrap the NPU message to whatever width is left next to the device box.
        device_row.bind("<Configure>", lambda e: self.npu_label.config(
            wraplength=max(px(200), e.width - self.device_box.winfo_width() - px(24))))

        ttk.Label(settings, text="Save to").grid(row=2, column=0, sticky="w", **row)
        self.out_var = tk.StringVar(value=cfg.get("out_dir", ""))
        ttk.Entry(settings, textvariable=self.out_var).grid(row=2, column=1, sticky="ew", **row)
        ttk.Button(settings, text="Browse…", command=self._pick_output).grid(row=2, column=2, sticky="w", **row)
        ttk.Label(settings, text="Leave empty to save next to each video.",
                  foreground="#666").grid(row=3, column=1, sticky="w", padx=px(6))

        ttk.Label(settings, text="Create").grid(row=4, column=0, sticky="w", **row)
        formats = ttk.Frame(settings)
        formats.grid(row=4, column=1, sticky="w", **row)
        self.srt_var = tk.BooleanVar(value=cfg.get("srt", True))
        self.txt_var = tk.BooleanVar(value=cfg.get("txt", True))
        ttk.Checkbutton(formats, text="Subtitles (.srt)", variable=self.srt_var).pack(side="left", padx=(px(0), px(16)))
        ttk.Checkbutton(formats, text="Plain text (.txt)", variable=self.txt_var).pack(side="left")

        # Shown only when ffmpeg is missing.
        self.ffmpeg_label = ttk.Label(self, text="", wraplength=px(780), justify="left", foreground="#b45309")

        # Advanced section (collapsed by default)
        self.adv_toggle = ttk.Button(self, text="▸ Advanced", command=self._toggle_advanced)
        self.adv_toggle.pack(anchor="w", padx=px(10), pady=(px(6), px(0)))
        self.adv = ttk.Frame(self)

        model_box = ttk.LabelFrame(self.adv, text="Model")
        model_box.pack(fill="x", pady=(px(0), px(6)))
        model_box.columnconfigure(1, weight=1)
        self.model_var = tk.StringVar(value=cfg.get("model") or DEFAULT_MODEL)
        ttk.Label(model_box, text="Folder").grid(row=0, column=0, sticky="w", **row)
        ttk.Entry(model_box, textvariable=self.model_var, state="readonly").grid(row=0, column=1, sticky="ew", **row)
        model_btns = ttk.Frame(model_box)
        model_btns.grid(row=0, column=2, sticky="w", **row)
        self.browse_model_btn = ttk.Button(model_btns, text="Browse…", command=self._pick_model)
        self.download_btn = ttk.Button(model_btns, text="Download…", command=self._download_model)
        self.browse_model_btn.pack(side="left")
        self.download_btn.pack(side="left", padx=(px(4), px(0)))
        # Line 1: is the model downloaded?  Line 2: is it compiled? (+ matching button)
        self.download_status = ttk.Label(model_box, text="", justify="left")
        self.download_status.grid(row=1, column=1, sticky="w", padx=px(6), pady=(px(0), px(2)))
        self.compile_status = ttk.Label(model_box, text="", justify="left", wraplength=px(600))
        self.compile_status.grid(row=2, column=1, columnspan=2, sticky="w", padx=px(6), pady=(px(0), px(4)))
        self.model_actions = ttk.Frame(model_box)
        self.model_actions.grid(row=3, column=1, sticky="w", padx=px(6), pady=(px(0), px(6)))
        self.compile_btn = ttk.Button(self.model_actions, text="Compile now", command=self._compile_now)
        self.clear_cache_btn = ttk.Button(self.model_actions, text="Clear compiled models",
                                          command=self._clear_compiled)

        env_box = ttk.LabelFrame(self.adv, text="Python environment")
        env_box.pack(fill="x")
        env_box.columnconfigure(1, weight=1)
        self.env_var = tk.StringVar(value=cfg.get("env") or MANAGED_ENV)
        ttk.Label(env_box, text="Folder").grid(row=0, column=0, sticky="w", **row)
        ttk.Entry(env_box, textvariable=self.env_var, state="readonly").grid(row=0, column=1, sticky="ew", **row)
        env_btns = ttk.Frame(env_box)
        env_btns.grid(row=0, column=2, **row)
        self.env_change_btn = ttk.Button(env_btns, text="Change…", command=self._pick_env)
        self.env_default_btn = ttk.Button(env_btns, text="Use default", command=self._use_default_env)
        self.env_change_btn.pack(side="left")
        self.env_default_btn.pack(side="left", padx=(px(4), px(0)))
        self.env_status = ttk.Label(env_box, text="")
        self.env_status.grid(row=1, column=1, columnspan=2, sticky="w", padx=px(6))
        maint = ttk.Frame(env_box)
        maint.grid(row=2, column=1, columnspan=2, sticky="w", **row)
        self.update_btn = ttk.Button(maint, text="Update packages", command=self._update_packages)
        self.rebuild_btn = ttk.Button(maint, text="Rebuild environment", command=self._rebuild_env)
        self.update_btn.pack(side="left")
        self.rebuild_btn.pack(side="left", padx=px(8))
        self.adv_visible = False

        # Keep the model status current when the model or device changes.
        self.model_var.trace_add("write", lambda *_: self._update_model_status())
        self.device_var.trace_add("write", lambda *_: self._update_model_status())

        self.controls = ttk.Frame(self)
        self.controls.pack(fill="x", **pad)
        self.start_btn = ttk.Button(self.controls, text="Start transcription", command=self._start)
        self.stop_btn = ttk.Button(self.controls, text="Stop after current file", command=self._stop, state="disabled")
        self.open_btn = ttk.Button(self.controls, text="Open output folder", command=self._open_output, state="disabled")
        self.start_btn.pack(side="left")
        self.stop_btn.pack(side="left", padx=px(8))
        self.open_btn.pack(side="right")

        self.progress = ttk.Progressbar(self, mode="determinate", maximum=100)
        self.progress.pack(fill="x", padx=px(10))
        self.status_var = tk.StringVar(value="Ready. Add one or more files to begin.")
        ttk.Label(self, textvariable=self.status_var).pack(anchor="w", padx=px(10), pady=(px(4), px(0)))

        log_frame = ttk.LabelFrame(self, text="Log and transcript")
        log_frame.pack(fill="both", expand=True, **pad)
        self.log_box = LogBox(log_frame, height=10)
        self.log_box.pack(fill="both", expand=True, padx=px(6), pady=px(6))

    def _toggle_advanced(self, auto=False):
        if auto:
            self.adv_auto_opened = not self.adv_visible
        else:
            self.adv_auto_opened = False
            if self.adv_visible and self.adv_problem:
                self.adv_dismissed = True  # closed by hand: don't reopen for the same problem
        self.adv_visible = not self.adv_visible
        self.update_idletasks()
        width, height = self.winfo_width(), self.winfo_height()
        if self.adv_visible:
            self.adv.pack(fill="x", padx=px(10), pady=(px(4), px(0)), before=self.controls)
            self.adv_toggle.config(text="▾ Advanced")
            self.update_idletasks()
            # Grow the window so the buttons and log below stay visible.
            height = min(height + self.adv.winfo_reqheight() + px(4), self.winfo_screenheight() - px(60))
        else:
            shrink = self.adv.winfo_height() + px(4)
            self.adv.pack_forget()
            self.adv_toggle.config(text="▸ Advanced")
            height = max(height - shrink, self.minsize()[1])
        self.geometry(f"{width}x{height}")

    def _sync_advanced(self):
        """Open Advanced when something there needs attention; close it again once fixed."""
        if not self.ready:
            return
        problem = self.model_problem or self.env_problem
        if not problem:
            self.adv_dismissed = False
        self.adv_problem = problem
        if problem and not self.adv_visible and not self.adv_dismissed:
            self._toggle_advanced(auto=True)
        elif not problem and self.adv_visible and self.adv_auto_opened:
            self._toggle_advanced(auto=True)

    def _became_ready(self):
        self.ready = True
        self._sync_advanced()

    # ---------------------------------------------------------------- settings
    def _save_settings(self):
        device = self.device_var.get()
        if self.auto_device and device == self.auto_device:
            device = self.cfg.get("device") or device  # don't overwrite the saved choice with a fallback
        try:
            update_config(
                env=self.env_var.get(),
                language=self.lang_var.get(),
                device=device,
                model=self.model_var.get().strip(),
                out_dir=self.out_var.get().strip(),
                srt=self.srt_var.get(),
                txt=self.txt_var.get(),
            )
        except OSError as e:
            self._log(f"Could not save settings: {e}")

    # ---------------------------------------------------------------- checks
    def _check_system(self):
        def work():
            try:
                import openvino as ov
                devices = list(ov.Core().available_devices)
            except Exception as e:
                devices = []
                self.events.put(("log", f"OpenVINO could not list devices: {e}"))
            self.events.put(("devices", devices))
        threading.Thread(target=work, daemon=True).start()

    def _show_devices(self, devices):
        usable = [d for d in ("NPU", "CPU", "GPU") if d in devices] or ["CPU"]
        self.device_box.config(values=usable)
        if self.device_var.get() not in usable:
            # Temporary fallback: shown in the window, but the saved preference is kept.
            self.auto_device = usable[0]
            self.device_var.set(usable[0])
        problem = npu_problem(devices)
        if problem:
            self.npu_label.config(text="⚠ " + problem, foreground="#b45309")
        else:
            self.npu_label.config(text="✓ NPU driver ready", foreground="#15803d")
        missing = ffmpeg_problem()
        if missing:
            self.ffmpeg_label.config(text="⚠ " + missing)
            self.ffmpeg_label.pack(anchor="w", padx=px(12), before=self.adv_toggle)
        else:
            self.ffmpeg_label.pack_forget()

    def _update_env_status(self):
        env_dir = self.env_var.get()
        managed = os.path.realpath(env_dir) == os.path.realpath(MANAGED_ENV)
        where = "program's own environment" if managed else "custom environment"
        self.env_problem = not (running_in(env_dir) and has_openvino_genai())
        if not self.env_problem:
            self.env_status.config(text=f"✓ Running in the {where}, OpenVINO GenAI installed", foreground="#15803d")
        else:
            self.env_status.config(text=f"⚠ Not running in this environment ({sys.prefix})", foreground="#b45309")
        self._sync_advanced()

    # ---------------------------------------------------------------- environment actions
    def _restart_into(self, env_dir):
        self._save_settings()
        files = self._files()
        self.destroy()
        if is_venv(env_dir):
            relaunch_in(env_dir, files)       # restarted program installs packages if missing
        restart_with_system_python(files)     # no environment there yet -> setup window

    def _pick_env(self):
        folder = ask_folder(self, "Choose a Python environment (venv folder)",
                            initialdir=os.path.dirname(self.env_var.get()))
        if not folder:
            return
        if not is_venv(folder) and not safe_to_create(folder):
            messagebox.showerror("Not usable",
                                 f"{folder}\n\nThis is not a Python environment and it contains other files. "
                                 "Choose an existing environment or an empty folder.")
            return
        self.env_var.set(folder)
        if not running_in(folder):
            self._restart_into(folder)
        self._save_settings()
        self._update_env_status()

    def _use_default_env(self):
        if running_in(MANAGED_ENV):
            self.env_var.set(MANAGED_ENV)
            self._save_settings()
            self._update_env_status()
            return
        self.env_var.set(MANAGED_ENV)
        self._restart_into(MANAGED_ENV)

    def _update_packages(self):
        if not running_in(self.env_var.get()):
            messagebox.showinfo("Not available", "Restart the program in this environment first.")
            return
        if not messagebox.askyesno("Update packages",
                                   "Update OpenVINO GenAI and helpers to the latest versions?\n\n"
                                   "The program restarts afterwards. The new versions are recorded, "
                                   "so a later rebuild installs them again."):
            return
        self._set_busy(True)
        self.status_var.set("Updating packages…")

        def work():
            ok = install_packages(self.env_var.get(), lambda m: self.events.put(("log", m)), upgrade=True)
            self.events.put(("updated", ok))
        threading.Thread(target=work, daemon=True).start()

    def _rebuild_env(self):
        env_dir = self.env_var.get()
        if not messagebox.askyesno("Rebuild environment",
                                   f"Delete and recreate the Python environment at\n{env_dir}?\n\n"
                                   "Packages are reinstalled in the recorded versions. "
                                   "Models and settings are kept."):
            return
        self._save_settings()
        files = self._files()
        self.destroy()
        restart_with_system_python(["--rebuild"] + files)

    # ---------------------------------------------------------------- file list
    def _files(self):
        return list(self.listbox.get(0, "end"))

    def _add_files(self):
        paths = ask_files(self, "Choose videos or audio files", MEDIA_TYPES)
        existing = set(self._files())
        for p in paths:
            if p not in existing:
                self.listbox.insert("end", p)

    def _remove_selected(self):
        for index in reversed(self.listbox.curselection()):
            self.listbox.delete(index)

    def _clear(self):
        self.listbox.delete(0, "end")

    def _pick_model(self):
        folder = ask_folder(self, "Choose an OpenVINO Whisper model folder",
                            initialdir=MODELS_DIR if os.path.isdir(MODELS_DIR) else None)
        if not folder:
            return
        if not looks_like_model(folder):
            messagebox.showwarning("Not a Whisper model",
                                   f"{folder}\n\nThis folder has no openvino_encoder_model.xml, so it doesn't "
                                   "look like an OpenVINO Whisper model.")
        self.model_var.set(folder)

    def _download_model(self):
        ModelDialog(self)

    # ---------------------------------------------------------------- model status
    def _update_model_status(self):
        model = self.model_var.get()
        device = self.device_var.get()
        for w in self.model_actions.winfo_children():
            w.pack_forget()

        green, orange, red = "#15803d", "#b45309", "#b91c1c"
        self.model_problem = True

        # Line 1 - downloaded?
        downloaded = looks_like_model(model)
        if downloaded:
            name = os.path.basename(os.path.normpath(model))
            size = folder_size(model) / 1e9
            self.download_status.config(text=f"✓ downloaded  ·  {name}  ·  {size:.2f} GB", foreground=green)
        else:
            self.download_status.config(text="✗ not downloaded yet - use Download…", foreground=red)

        # Line 2 - compiled? (with the button that fits)
        if self.compiling:
            text, color = f"Compiling for {device}… (about a minute the first time)", orange
        elif not downloaded:
            text, color = "", orange
        elif is_compiled(model, device):
            text, color = f"✓ compiled for {device} - starts in seconds", green
            self.model_problem = False
            self.clear_cache_btn.pack(side="left")
        elif self.transcriber.is_loaded(model, device):
            text, color = f"✓ compiled for {device} - kept in memory until you close the program", green
            self.model_problem = False
            self.clear_cache_btn.pack(side="left")
        else:
            text, color = f"⚠ not compiled for {device} yet - the first start takes about a minute", orange
            self.compile_btn.pack(side="left")
        self.compile_status.config(text=text, foreground=color)
        if text:
            self.compile_status.grid()
        else:
            self.compile_status.grid_remove()  # nothing to compile without a model
        self._sync_advanced()

    def _compile_now(self):
        model, device = self.model_var.get(), self.device_var.get()
        self.compiling = True
        self._set_busy(True)
        self._update_model_status()
        self.status_var.set(f"Compiling the model for {device}…")

        def work():
            try:
                self.transcriber.get(model, device, lambda m: self.events.put(("log", m)))
                ok = True
            except Exception as e:
                self.events.put(("log", f"Could not compile the model for {device}: {e}"))
                ok = False
            self.events.put(("compiled", ok))
        threading.Thread(target=work, daemon=True).start()

    def _clear_compiled(self):
        if not messagebox.askyesno("Clear compiled models",
                                   "Delete all compiled models?\n\nThe downloaded models are kept. "
                                   "The next start compiles again, which takes about a minute."):
            return
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
        self.transcriber = Transcriber()  # also drop the copy held in memory
        self._log("Compiled models cleared.")
        self._update_model_status()

    def _pick_output(self):
        folder = ask_folder(self, "Choose where to save transcripts")
        if folder:
            self.out_var.set(folder)

    # ---------------------------------------------------------------- running
    def _set_busy(self, busy):
        self.busy = busy
        state = "disabled" if busy else "normal"
        for w in (self.start_btn, self.add_btn, self.remove_btn, self.clear_btn,
                  self.download_btn, self.browse_model_btn, self.compile_btn,
                  self.clear_cache_btn, self.device_box,
                  self.env_change_btn, self.env_default_btn, self.update_btn, self.rebuild_btn):
            w.config(state=state)
        if not busy:
            self.device_box.config(state="readonly")
        if busy:
            self._progress_waiting()
        else:
            self._progress_value(0)

    def _progress_waiting(self):
        """Bouncing bar: something is happening, but there's no percentage yet (e.g. compiling)."""
        self.progress.config(mode="indeterminate")
        self.progress.start(12)

    def _progress_value(self, percent):
        if str(self.progress.cget("mode")) != "determinate":
            self.progress.stop()
            self.progress.config(mode="determinate", maximum=100)
        self.progress["value"] = max(0.0, min(100.0, percent))

    def _start(self):
        files = self._files()
        model = self.model_var.get().strip()
        out_dir = self.out_var.get().strip()

        if not files:
            messagebox.showinfo("No files", "Add at least one video or audio file first.")
            return
        if not looks_like_model(model):
            if messagebox.askyesno("No model yet",
                                   f"There is no Whisper model in\n{model}\n\nDownload one now?"):
                self._download_model()
            return
        if ffmpeg_problem():
            messagebox.showerror("ffmpeg missing", ffmpeg_problem())
            return
        if out_dir and not os.path.isdir(out_dir):
            messagebox.showerror("Folder not found", f"The output folder does not exist:\n{out_dir}")
            return
        if not (self.srt_var.get() or self.txt_var.get()):
            messagebox.showinfo("Nothing to create", "Tick subtitles, plain text, or both.")
            return

        self._save_settings()
        settings = {
            "files": files,
            "model": model,
            "device": self.device_var.get(),
            "language": dict(LANGUAGES)[self.lang_var.get()],
            "out_dir": out_dir,
            "srt": self.srt_var.get(),
            "txt": self.txt_var.get(),
        }
        self.run = {
            "started": time.time(), "phase": "Preparing…", "n": len(files), "durations": [None] * len(files),
            "index": 0, "name": "", "done_audio": 0.0,
            "file_started": None,   # when Whisper started on the current file
            "speed": load_speeds().get(speed_key(model, settings["device"])),
        }
        self.stop_event.clear()
        self._set_busy(True)
        self.stop_btn.config(state="normal")
        self.worker = threading.Thread(target=self._work, args=(settings,), daemon=True)
        self.worker.start()

    def _stop(self):
        self.stop_event.set()
        self.stop_btn.config(state="disabled")
        self._log("Stopping after the current file…")

    def _work(self, s):
        send = self.events.put
        files, done = s["files"], 0
        device = s["device"]
        try:
            # Lengths of all files up front, so progress and remaining time cover the whole batch.
            durations = [media_duration(p) for p in files]
            send(("durations", durations))

            compiled = is_compiled(s["model"], device) or self.transcriber.is_loaded(s["model"], device)
            send(("phase", f"Loading the model on {device}…" if compiled else
                           f"Compiling the model for {device} - about a minute the first time…"))
            try:
                pipe = self.transcriber.get(s["model"], device, lambda m: send(("log", m)))
            except Exception as e:
                send(("log", f"Could not load the model on {device}: {e}"))
                return

            options = {"task": "transcribe", "return_timestamps": True}
            if s["language"]:
                options["language"] = f"<|{s['language']}|>"

            key = speed_key(s["model"], device)
            speed = load_speeds().get(key)   # audio seconds per second, from earlier runs
            for i, path in enumerate(files, 1):
                if self.stop_event.is_set():
                    send(("log", "Stopped."))
                    break
                name = os.path.basename(path)
                try:
                    audio = load_audio(path)
                    duration = len(audio) / SAMPLE_RATE
                    send(("file_start", (i - 1, len(files), name, duration)))

                    # No speed known yet for this device + model: measure it once on 2 minutes
                    # from the middle of the file (the start is often intro music or silence).
                    if not speed and duration >= MIN_SAMPLE_FILE_SECONDS:
                        send(("phase", f"Measuring the speed on {device} (only needed once)…"))
                        mid = len(audio) // 2
                        half = SAMPLE_SECONDS * SAMPLE_RATE // 2
                        t0 = time.time()
                        pipe.generate(audio[mid - half:mid + half].tolist(), **options)
                        speed = save_speed(key, SAMPLE_SECONDS / max(time.time() - t0, 0.01))
                        send(("log", f"Measured speed on {device}: {speed:.0f}× real time (saved for next time)."))

                    send(("log", f"[{i}/{len(files)}] {name}: transcribing {duration / 60:.1f} min of audio…"))
                    send(("transcribing", (time.time(), speed)))
                    t0 = time.time()
                    result = pipe.generate(audio.tolist(), **options)
                    took = time.time() - t0

                    written, text = write_outputs(result_entries(result, duration), path,
                                                  s["out_dir"], s["srt"], s["txt"])
                    measured = duration / max(took, 0.01)
                    if duration >= 30:  # very short files say little about the speed
                        speed = save_speed(key, measured)
                    send(("log", f"[{i}/{len(files)}] {name}: done in {clock(took)} "
                                 f"({measured:.0f}× real time) → " + ", ".join(os.path.basename(w) for w in written)))
                    send(("transcript", (name, text)))
                    send(("output_dir", os.path.dirname(written[0])))
                    send(("file_done", i - 1))
                    done += 1
                except Exception as e:
                    send(("log", f"[{i}/{len(files)}] {name}: ERROR - {e}"))
                    send(("file_done", i - 1))
        finally:
            send(("done", (done, len(files))))

    # ---------------------------------------------------------------- events from worker threads
    def _poll(self):
        try:
            while True:
                kind, data = self.events.get_nowait()
                if kind == "log":
                    self._log(data)
                elif kind == "devices":
                    self._show_devices(data)
                elif kind == "durations" and self.run:
                    self.run["durations"] = list(data)
                elif kind == "phase" and self.run:
                    self.run["phase"] = data
                elif kind == "file_start" and self.run:
                    index, _, name, duration = data
                    self.run["durations"][index] = duration  # exact length now that it's decoded
                    self.run.update(index=index, name=name, file_started=None, phase="Reading the audio…")
                elif kind == "transcribing" and self.run:
                    started, speed = data
                    self.run.update(file_started=started, speed=speed, phase=None)
                elif kind == "file_done" and self.run:
                    self.run["done_audio"] += self.run["durations"][data] or 0.0
                    self.run["file_started"] = None
                    if data + 1 < self.run["n"]:
                        self.run["phase"] = "Reading the next file…"
                elif kind == "transcript":
                    name, text = data
                    self._log(f"\n--- {name} ---\n{text}\n")
                elif kind == "output_dir":
                    self.last_output_dir = data
                    self.open_btn.config(state="normal")
                elif kind == "compiled":
                    self.compiling = False
                    self._set_busy(False)
                    self.status_var.set("Model ready." if data else "Compiling failed - see the log.")
                    self._update_model_status()
                elif kind == "done":
                    ok, n = data
                    took = time.time() - self.run["started"] if self.run else 0
                    self.run = None
                    self._set_busy(False)
                    self._progress_value(100 if ok else 0)
                    self.stop_btn.config(state="disabled")
                    self._update_model_status()  # the run compiled the model if it wasn't yet
                    self.status_var.set(f"Finished: {ok} of {n} file(s) transcribed in {clock(took)}.")
                    self.bell()
                elif kind == "updated":
                    self._set_busy(False)
                    if data:
                        self._log("Packages updated - restarting…")
                        self.after(800, lambda: self._restart_into(self.env_var.get()))
                    else:
                        self.status_var.set("Updating failed - see the log.")
        except queue.Empty:
            pass

        if self.run:
            self._show_progress()
        self.after(250, self._poll)

    def _show_progress(self):
        r, now = self.run, time.time()
        elapsed = clock(now - r["started"])
        if r["phase"]:
            self.status_var.set(f"{r['phase']}  ·  {elapsed}")
            return

        label = f"File {r['index'] + 1} of {r['n']}: {r['name']}  ·  {elapsed} elapsed"
        speed = r["speed"]  # audio seconds per second, measured on earlier files/runs
        if not speed:
            # First file on this device/model and too short to measure first: no estimate yet.
            self.status_var.set(f"{label}  ·  estimated time left: not known yet (measuring the speed)")
            return
        if not r["file_started"]:
            self.status_var.set(label)
            return

        # Whisper gives no progress while working, so the bar is an estimate from the known speed.
        duration = r["durations"][r["index"]] or 0.0
        file_pos = (now - r["file_started"]) * speed
        overdue = file_pos >= duration * 0.99
        file_pos = min(file_pos, duration * 0.99)  # never claim "done" before Whisper is
        total = sum(d for d in r["durations"] if d)
        done = r["done_audio"] + file_pos
        self._progress_value(100 * done / total if total else 0)

        if overdue:
            left = "taking a little longer than estimated - finishing up…"
        else:
            left = f"estimated time left: {roughly((total - done) / speed)}"
        self.status_var.set(f"{label}  ·  {left}")

    def _log(self, message):
        self.log_box.add(message)

    def _open_output(self):
        if self.last_output_dir:
            subprocess.Popen(["xdg-open", self.last_output_dir])

    def _on_close(self):
        if self.busy:
            if not messagebox.askyesno("Still working",
                                       "The program is still working. Quit anyway?\n"
                                       "Unfinished work will be lost."):
                return
        self._save_settings()
        self.destroy()


LANGUAGES = [
    ("Auto-detect", None),
    ("German", "de"),
    ("English", "en"),
    ("French", "fr"),
    ("Spanish", "es"),
    ("Italian", "it"),
    ("Dutch", "nl"),
    ("Polish", "pl"),
    ("Portuguese", "pt"),
]
MEDIA_TYPES = [
    ("Video and audio", "*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.mp3 *.wav *.m4a *.flac *.ogg *.opus"),
    ("All files", "*"),
]


# ================================================================ start-up

def main():
    args = sys.argv[1:]
    rebuild = "--rebuild" in args
    files = [a for a in args if a != "--rebuild"]
    env_dir = configured_env()

    if not rebuild:
        if running_in(env_dir):
            if has_openvino_genai():
                App(files).mainloop()
                return
        elif is_venv(env_dir) and not os.environ.get("NPU_TRANSCRIBER_RELAUNCHED"):
            relaunch_in(env_dir, files)   # the restarted program re-checks its packages

    SetupWindow(env_dir, files, rebuild=rebuild).mainloop()


if __name__ == "__main__":
    main()
