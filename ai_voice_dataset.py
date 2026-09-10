"""
VOICEGUARD — AI Voice & Deepfake Detection
Frontend demo UI. The detection verdict itself is still simulated
(there's no real ML model wired up — swap out `_finish_analysis` for a
real backend call when you have one), but the waveform panel now reads
and renders the *actual* audio you load, instead of a fake sine wave.

Optional dependencies for real waveform decoding:
    pip install numpy soundfile
    (pydub + ffmpeg on your PATH will also let it decode mp3/m4a)
If none of those are available, the app still runs fine — it just
shows a "no preview available" placeholder instead of a real waveform.
"""

import math
import random
import struct
import time
import tkinter as tk
import tkinter.font as tkfont
import wave
from datetime import datetime
from tkinter import filedialog

import customtkinter as ctk

# ----------------------------------------------------------------------
# THEME
# A small, named palette instead of scattering hex codes through the
# UI code. Colors are assigned by MEANING (brand / success / danger /
# warning) rather than just "the accent color", so the terminal log,
# the result panel, and the incident tracker all speak the same
# visual language. Palette: obsidian + champagne gold, aiming for a
# "premium security console" feel rather than a generic dark theme.
# ----------------------------------------------------------------------
COLORS = {
    "bg":            "#08090c",   # app background — near-black obsidian
    "panel":         "#131319",   # panel background
    "panel_alt":     "#0c0c11",   # slightly darker panel (inputs, canvas)
    "border":        "#232430",   # panel / input borders
    "border_light":  "#34323f",

    "text":          "#f2efe8",   # primary text — warm ivory, not stark white
    "text_dim":      "#9b95a3",   # secondary / muted text
    "text_faint":    "#57536e",   # placeholder-ish text

    "brand":         "#d4af37",   # champagne gold — brand / system / active
    "brand_soft":    "#8a6d23",   # dimmer gold for button fills
    "brand_glow":    "#4a3c17",   # very dim gold, used for glow/halo strokes

    "ok":            "#3ddc97",   # emerald — authentic / online / safe
    "ok_dim":        "#155c40",

    "danger":        "#ef5d6f",   # rose red — clone detected / incident
    "danger_dim":    "#7a1f2b",

    "warn":          "#f2a541",   # amber — analyzing / model activity
}


def pick_font(root, preferred, fallback="TkFixedFont"):
    """Return the first available font family from `preferred`."""
    available = set(tkfont.families(root))
    for name in preferred:
        if name in available:
            return name
    return fallback


# ----------------------------------------------------------------------
# AUDIO DECODING
# Best-effort waveform extraction. Tries a few backends in order of
# how little they depend on external binaries, and degrades cleanly
# to "no preview" if none are available or the file can't be read.
# ----------------------------------------------------------------------
def load_waveform_envelope(path, target_points=2000):
    """
    Decode an audio file into a normalized amplitude envelope:
    a list of (min, max) pairs in [-1, 1], `target_points` long,
    suitable for drawing a classic min/max waveform view.
    Returns None if the file couldn't be decoded by any available
    backend (caller should fall back to a placeholder UI state).
    """
    try:
        import numpy as np
    except ImportError:
        return None

    samples = None

    # Backend 1: soundfile (handles wav/flac/ogg well, no ffmpeg needed)
    if samples is None:
        try:
            import soundfile as sf
            data, _sr = sf.read(path, dtype="float32", always_2d=False)
            samples = data.mean(axis=1) if getattr(data, "ndim", 1) > 1 else data
        except Exception:
            samples = None

    # Backend 2: pydub (handles mp3/m4a, needs ffmpeg on PATH)
    if samples is None:
        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_file(path).set_channels(1)
            raw = np.array(seg.get_array_of_samples()).astype(np.float32)
            maxval = float(1 << (8 * seg.sample_width - 1))
            samples = raw / maxval
        except Exception:
            samples = None

    # Backend 3: stdlib `wave` (no dependencies, .wav only)
    if samples is None and path.lower().endswith(".wav"):
        try:
            with wave.open(path, "rb") as wf:
                n_frames = wf.getnframes()
                sample_width = wf.getsampwidth()
                n_channels = wf.getnchannels()
                raw = wf.readframes(n_frames)
            fmt_map = {1: "b", 2: "h", 4: "i"}
            if sample_width in fmt_map:
                fmt = "<" + fmt_map[sample_width] * (len(raw) // sample_width)
                ints = struct.unpack(fmt, raw)
                arr = np.array(ints, dtype=np.float32)
                if n_channels > 1:
                    arr = arr.reshape(-1, n_channels).mean(axis=1)
                samples = arr / float(2 ** (8 * sample_width - 1))
        except Exception:
            samples = None

    if samples is None or len(samples) == 0:
        return None

    samples = np.nan_to_num(np.asarray(samples, dtype=np.float32))
    peak = float(np.max(np.abs(samples))) or 1.0
    samples = samples / peak

    n = len(samples)
    points = min(target_points, n)
    chunk = max(1, n // points)
    envelope = []
    for i in range(0, n, chunk):
        seg = samples[i:i + chunk]
        if len(seg):
            envelope.append((float(seg.min()), float(seg.max())))
    return envelope or None


def resample_envelope(envelope, num_columns):
    """Regroup an envelope into exactly `num_columns` (min, max) pairs
    so the waveform redraws crisply at any canvas width."""
    if not envelope or num_columns <= 0:
        return []
    n = len(envelope)
    if n == num_columns:
        return envelope
    out = []
    for c in range(num_columns):
        start = (c * n) // num_columns
        end = max(start + 1, ((c + 1) * n) // num_columns)
        seg = envelope[start:end]
        lo = min(v[0] for v in seg)
        hi = max(v[1] for v in seg)
        out.append((lo, hi))
    return out


# ----------------------------------------------------------------------
# REUSABLE "PANEL" WIDGET
# Every box in the mockup (AUDIO INPUT, DETECTION RESULT, etc.) is the
# same shape: a bordered card with an uppercase header. Building it
# once keeps the six panels visually consistent.
# ----------------------------------------------------------------------
class Panel(ctk.CTkFrame):
    def __init__(self, master, title, mono_font, **kwargs):
        super().__init__(
            master,
            fg_color=COLORS["panel"],
            border_width=1,
            border_color=COLORS["border"],
            corner_radius=12,
            **kwargs,
        )
        self.grid_columnconfigure(0, weight=1)

        header = ctk.CTkLabel(
            self,
            text=title,
            font=(mono_font, 11, "bold"),
            text_color=COLORS["text_dim"],
            anchor="w",
        )
        header.grid(row=0, column=0, sticky="ew", padx=18, pady=(15, 6))

        divider = ctk.CTkFrame(self, height=1, fg_color=COLORS["border"])
        divider.grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 10))

        # Subclasses/callers add their own content starting at row=2.
        self.body_row = 2


# ----------------------------------------------------------------------
# MAIN APPLICATION
# ----------------------------------------------------------------------
class VoiceGuardApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("VoiceGuard — AI Voice & Deepfake Detection")
        self.geometry("1180x720")
        self.minsize(980, 640)
        self.configure(fg_color=COLORS["bg"])

        ctk.set_appearance_mode("dark")

        self.mono = pick_font(self, ["Cascadia Mono", "JetBrains Mono", "Consolas",
                                      "DejaVu Sans Mono", "Courier New", "Courier"])
        self.serif = pick_font(self, ["Didot", "Big Caslon", "Optima", "Georgia",
                                       "Garamond", "Times New Roman"], fallback=self.mono)

        # ---- state ----
        self.audio_path = None
        self.waveform_envelope = None       # decoded (min, max) pairs, or None
        self.previous_flags = 0
        self.analysis_running = False
        self.analysis_start_time = None
        self.analysis_duration = 0.0
        self._pending_after_ids = []        # so RESET can cancel in-flight steps
        self._idle_pulse_phase = 0.0
        self._status_on = True

        self._build_layout()
        self._log("SYSTEM", "VoiceGuard core initialized.")
        self._log("SYSTEM", "Detection model loaded (simulated).")
        self._log("SYSTEM", "Awaiting audio input.")

        self.waveform.bind("<Configure>", lambda e: self._draw_waveform())
        self._animate_waveform()
        self._blink_status()
        self._update_start_button_state()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------
    # LAYOUT
    # ------------------------------------------------------------
    def _build_layout(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_topbar()

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 16))
        body.grid_columnconfigure(0, weight=45)
        body.grid_columnconfigure(1, weight=55)
        body.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(body, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        right = ctk.CTkFrame(body, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        right.grid_rowconfigure(2, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self._build_audio_input_panel(left)
        self._build_waveform_panel(left)
        self._build_controls(left)

        self._build_result_panel(right)
        self._build_incident_panel(right)
        self._build_terminal_panel(right)

    def _build_topbar(self):
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 14))
        bar.grid_columnconfigure(1, weight=1)

        title_box = ctk.CTkFrame(bar, fg_color="transparent")
        title_box.grid(row=0, column=0, sticky="w")

        # Letter-spaced wordmark for a logotype feel.
        wordmark = " ".join("VOICEGUARD")
        ctk.CTkLabel(
            title_box, text=wordmark,
            font=(self.serif, 22, "bold"), text_color=COLORS["brand"],
        ).pack(side="left")

        ctk.CTkLabel(
            title_box, text="   AI Voice & Deepfake Detection",
            font=(self.mono, 12), text_color=COLORS["text_dim"],
        ).pack(side="left", padx=(10, 0), pady=(6, 0))

        status_box = ctk.CTkFrame(bar, fg_color="transparent")
        status_box.grid(row=0, column=2, sticky="e")

        self.status_dot = ctk.CTkLabel(
            status_box, text="●", font=(self.mono, 14),
            text_color=COLORS["ok"],
        )
        self.status_dot.pack(side="left", padx=(0, 6))

        ctk.CTkLabel(
            status_box, text="SYSTEM ONLINE",
            font=(self.mono, 12, "bold"), text_color=COLORS["text"],
        ).pack(side="left")

        # Thin gold accent line under the topbar — a small "premium" touch.
        accent = ctk.CTkFrame(bar, height=2, fg_color=COLORS["brand_glow"])
        accent.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(14, 0))

    # ------------------------------------------------------------
    # LEFT COLUMN
    # ------------------------------------------------------------
    def _build_audio_input_panel(self, master):
        panel = Panel(master, "AUDIO INPUT", self.mono)
        panel.grid(row=0, column=0, sticky="ew", pady=(14, 12))
        panel.grid_columnconfigure(0, weight=1)

        # -- file picker row --
        file_row = ctk.CTkFrame(panel, fg_color="transparent")
        file_row.grid(row=panel.body_row, column=0, sticky="ew", padx=18)
        file_row.grid_columnconfigure(0, weight=1)

        self.file_display = ctk.CTkEntry(
            file_row, placeholder_text="No file selected",
            font=(self.mono, 11), fg_color=COLORS["panel_alt"],
            border_color=COLORS["border_light"], text_color=COLORS["text"],
            state="disabled",
        )
        self.file_display.grid(row=0, column=0, sticky="ew", padx=(0, 8), ipady=4)

        ctk.CTkButton(
            file_row, text="BROWSE", width=90, font=(self.mono, 11, "bold"),
            fg_color=COLORS["brand_soft"], hover_color=COLORS["brand"],
            text_color="#1a1305", corner_radius=8, command=self._browse_audio,
        ).grid(row=0, column=1)

        # -- phone number row --
        ctk.CTkLabel(
            panel, text="PHONE NUMBER", font=(self.mono, 10, "bold"),
            text_color=COLORS["text_dim"],
        ).grid(row=panel.body_row + 1, column=0, sticky="w", padx=18, pady=(14, 4))

        self.phone_row = ctk.CTkFrame(panel, fg_color=COLORS["panel_alt"],
                                       border_width=1, border_color=COLORS["border_light"],
                                       corner_radius=8)
        self.phone_row.grid(row=panel.body_row + 2, column=0, sticky="ew",
                             padx=18, pady=(0, 16))
        self.phone_row.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            self.phone_row, text="+91", font=(self.mono, 12, "bold"),
            text_color=COLORS["text_dim"],
        ).grid(row=0, column=0, padx=(10, 4), pady=6)

        vcmd = (self.register(self._validate_phone), "%P")
        self.phone_entry = ctk.CTkEntry(
            self.phone_row, placeholder_text="XXXXXXXXXX", font=(self.mono, 12),
            fg_color="transparent", border_width=0, text_color=COLORS["text"],
            validate="key", validatecommand=vcmd,
        )
        self.phone_entry.grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=6)
        self.phone_entry.bind("<KeyRelease>", lambda e: self._update_start_button_state())

    def _validate_phone(self, proposed):
        ok = proposed == "" or (proposed.isdigit() and len(proposed) <= 10)
        return ok

    def _build_waveform_panel(self, master):
        panel = Panel(master, "LIVE AUDIO SIGNAL", self.mono)
        panel.grid(row=1, column=0, sticky="nsew", pady=(0, 12))
        panel.grid_rowconfigure(panel.body_row, weight=1)

        self.waveform = tk.Canvas(
            panel, bg=COLORS["panel_alt"], highlightthickness=1,
            highlightbackground=COLORS["border_light"], height=140,
        )
        self.waveform.grid(row=panel.body_row, column=0, sticky="nsew",
                            padx=18, pady=(0, 18))

        self.waveform_caption = ctk.CTkLabel(
            panel, text="No audio loaded", font=(self.mono, 10),
            text_color=COLORS["text_faint"],
        )
        self.waveform_caption.grid(row=panel.body_row + 1, column=0,
                                    sticky="w", padx=18, pady=(0, 14))

    def _build_controls(self, master):
        row = ctk.CTkFrame(master, fg_color="transparent")
        row.grid(row=2, column=0, sticky="ew")
        row.grid_columnconfigure(0, weight=3)
        row.grid_columnconfigure(1, weight=1)

        self.start_button = ctk.CTkButton(
            row, text="▶  START ANALYSIS", height=46, font=(self.mono, 13, "bold"),
            fg_color=COLORS["brand_soft"], hover_color=COLORS["brand"],
            text_color="#1a1305", corner_radius=9, command=self._start_analysis,
        )
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        ctk.CTkButton(
            row, text="RESET", height=46, font=(self.mono, 13, "bold"),
            fg_color="transparent", hover_color=COLORS["panel_alt"],
            border_width=1, border_color=COLORS["border_light"], corner_radius=9,
            text_color=COLORS["text_dim"], command=self._reset_all,
        ).grid(row=0, column=1, sticky="ew")

    # ------------------------------------------------------------
    # RIGHT COLUMN
    # ------------------------------------------------------------
    def _build_result_panel(self, master):
        panel = Panel(master, "DETECTION RESULT", self.mono)
        panel.grid(row=0, column=0, sticky="ew", pady=(14, 12))
        panel.grid_columnconfigure(0, weight=1)

        self.result_label = ctk.CTkLabel(
            panel, text="WAITING", font=(self.serif, 23, "bold"),
            text_color=COLORS["text_dim"],
        )
        self.result_label.grid(row=panel.body_row, column=0, pady=(6, 4))

        self.confidence_label = ctk.CTkLabel(
            panel, text="Confidence: —", font=(self.mono, 12),
            text_color=COLORS["text_dim"],
        )
        self.confidence_label.grid(row=panel.body_row + 1, column=0)

        self.confidence_bar = ctk.CTkProgressBar(
            panel, height=6, progress_color=COLORS["brand"],
            fg_color=COLORS["panel_alt"],
        )
        self.confidence_bar.set(0)
        self.confidence_bar.grid(row=panel.body_row + 2, column=0,
                                  sticky="ew", padx=40, pady=(10, 18))

    def _build_incident_panel(self, master):
        self.incident_panel = Panel(master, "INCIDENT TRACKING", self.mono)
        panel = self.incident_panel
        panel.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        panel.grid_columnconfigure(1, weight=1)

        def stat_row(r, label):
            ctk.CTkLabel(
                panel, text=label, font=(self.mono, 11),
                text_color=COLORS["text_dim"], anchor="w",
            ).grid(row=r, column=0, sticky="w", padx=18, pady=3)
            value = ctk.CTkLabel(
                panel, text="—", font=(self.mono, 11, "bold"),
                text_color=COLORS["text"], anchor="e",
            )
            value.grid(row=r, column=1, sticky="e", padx=18, pady=3)
            return value

        self.incident_phone = stat_row(panel.body_row, "Phone:")
        self.incident_status = stat_row(panel.body_row + 1, "Status:")
        self.incident_status.configure(text="NO INCIDENT", text_color=COLORS["text_dim"])
        self.incident_flags = stat_row(panel.body_row + 2, "Flags this session:")
        self.incident_flags.configure(text="0")

        # small bottom padding under the last row
        ctk.CTkLabel(panel, text="", height=1).grid(row=panel.body_row + 3, column=0)

    def _build_terminal_panel(self, master):
        panel = Panel(master, "BACKEND TERMINAL", self.mono)
        panel.grid(row=2, column=0, sticky="nsew")
        panel.grid_rowconfigure(panel.body_row, weight=1)
        panel.grid_columnconfigure(0, weight=1)

        self.terminal = ctk.CTkTextbox(
            panel, font=(self.mono, 11), fg_color=COLORS["panel_alt"],
            border_width=1, border_color=COLORS["border_light"], corner_radius=8,
            text_color=COLORS["text"], wrap="word", activate_scrollbars=True,
        )
        self.terminal.grid(row=panel.body_row, column=0, sticky="nsew",
                            padx=18, pady=(0, 18))
        self.terminal.configure(state="disabled")

        # color tags for the different log sources / message types
        self.terminal._textbox.tag_config("SYSTEM", foreground=COLORS["brand"])
        self.terminal._textbox.tag_config("MODEL", foreground=COLORS["warn"])
        self.terminal._textbox.tag_config("WARNING", foreground=COLORS["danger"])
        self.terminal._textbox.tag_config("MSG_OK", foreground=COLORS["ok"])
        self.terminal._textbox.tag_config("MSG_BAD", foreground=COLORS["danger"])
        self.terminal._textbox.tag_config("TIME", foreground=COLORS["text_faint"])
        # message-tags render on top of source-tags since they're added later
        self.terminal._textbox.tag_raise("MSG_OK")
        self.terminal._textbox.tag_raise("MSG_BAD")

    # ------------------------------------------------------------
    # TERMINAL LOGGING
    # message_tag (optional) colors ONLY the message text (not the
    # timestamp/source prefix) — used to highlight a final verdict
    # line without recoloring its whole row.
    # ------------------------------------------------------------
    def _log(self, source, message, message_tag=None):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.terminal.configure(state="normal")
        self.terminal.insert("end", f"[{timestamp}] ", "TIME")
        self.terminal.insert("end", f"{source}: ", source)
        msg_start = self.terminal.index("end-1c")
        self.terminal.insert("end", f"{message}\n")
        if message_tag:
            msg_end = self.terminal.index("end-2c")
            self.terminal._textbox.tag_add(message_tag, msg_start, msg_end)
        self.terminal.configure(state="disabled")
        self.terminal.see("end")

    # ------------------------------------------------------------
    # WAVEFORM RENDERING
    # Idle (no file): a faint centered line with a slow breathing dot.
    # Loaded, not analyzing: the real decoded waveform of the file.
    # Analyzing: the same waveform, with a sweeping "scan" highlight
    # and playhead tracking how far the (simulated) analysis has got.
    # ------------------------------------------------------------
    def _draw_waveform(self):
        c = self.waveform
        c.delete("all")
        w = c.winfo_width() or 400
        h = c.winfo_height() or 140
        mid = h / 2

        c.create_line(0, mid, w, mid, fill=COLORS["border"], dash=(2, 3))

        if self.analysis_running and self.waveform_envelope:
            self._draw_scanning_waveform(c, w, h, mid)
        elif self.waveform_envelope:
            self._draw_static_waveform(c, w, h, mid)
        else:
            self._draw_idle_state(c, w, h, mid)

    def _draw_idle_state(self, c, w, h, mid):
        pulse = (math.sin(self._idle_pulse_phase) + 1) / 2  # 0..1
        # faint flat line
        c.create_line(w * 0.08, mid, w * 0.92, mid,
                       fill=COLORS["border_light"], width=1)
        # slow breathing dot in the middle, standing in for "ready / no signal"
        r = 3 + pulse * 2.5
        color = COLORS["text_faint"] if self.audio_path is None else COLORS["warn"]
        c.create_oval(w / 2 - r, mid - r, w / 2 + r, mid + r,
                       outline="", fill=color)

    def _draw_static_waveform(self, c, w, h, mid):
        cols = max(1, int(w // 3))
        env = resample_envelope(self.waveform_envelope, cols)
        amp = h * 0.42
        col_w = w / max(1, len(env))
        for i, (lo, hi) in enumerate(env):
            x = i * col_w + col_w / 2
            y1 = mid - hi * amp
            y2 = mid - lo * amp
            if y2 - y1 < 1.4:
                y1 -= 0.7
                y2 += 0.7
            # soft halo behind the bar, then the crisp bar on top
            c.create_line(x, y1, x, y2, fill=COLORS["brand_glow"], width=3)
            c.create_line(x, y1, x, y2, fill=COLORS["brand"], width=1)

    def _draw_scanning_waveform(self, c, w, h, mid):
        progress = 0.0
        if self.analysis_start_time is not None and self.analysis_duration > 0:
            elapsed = time.monotonic() - self.analysis_start_time
            progress = max(0.0, min(1.0, elapsed / self.analysis_duration))

        cols = max(1, int(w // 3))
        env = resample_envelope(self.waveform_envelope, cols)
        amp = h * 0.42
        col_w = w / max(1, len(env))
        scanned_cols = int(progress * len(env))

        for i, (lo, hi) in enumerate(env):
            x = i * col_w + col_w / 2
            jitter = random.uniform(-1.5, 1.5) if i < scanned_cols else 0
            y1 = mid - hi * amp + jitter
            y2 = mid - lo * amp + jitter
            if y2 - y1 < 1.4:
                y1 -= 0.7
                y2 += 0.7
            if i < scanned_cols:
                # already-processed portion: bright, glowing amber/gold
                c.create_line(x, y1, x, y2, fill=COLORS["brand_glow"], width=4)
                c.create_line(x, y1, x, y2, fill=COLORS["warn"], width=1.4)
            else:
                # not-yet-processed portion: dim, muted
                c.create_line(x, y1, x, y2, fill=COLORS["border_light"], width=1)

        # playhead
        px = progress * w
        c.create_line(px, 4, px, h - 4, fill=COLORS["warn"], width=2)
        c.create_oval(px - 3, 2, px + 3, 8, outline="", fill=COLORS["warn"])

    def _animate_waveform(self):
        self._idle_pulse_phase += 0.12
        self._draw_waveform()
        interval = 33 if self.analysis_running else 90
        after_id = self.after(interval, self._animate_waveform)
        self._pending_after_ids.append(after_id)

    def _blink_status(self):
        self._status_on = not self._status_on
        color = COLORS["ok"] if self._status_on else COLORS["ok_dim"]
        self.status_dot.configure(text_color=color)
        after_id = self.after(700, self._blink_status)
        self._pending_after_ids.append(after_id)

    # ------------------------------------------------------------
    # ACTIONS
    # ------------------------------------------------------------
    def _browse_audio(self):
        path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[("Audio files", "*.wav *.mp3 *.m4a *.flac *.ogg"),
                       ("All files", "*.*")],
        )
        if not path:
            return
        self.audio_path = path
        filename = path.split("/")[-1].split("\\")[-1]
        self.file_display.configure(state="normal")
        self.file_display.delete(0, "end")
        self.file_display.insert(0, filename)
        self.file_display.configure(state="disabled")
        self._log("SYSTEM", f"Audio file loaded: {filename}")

        envelope = load_waveform_envelope(path)
        self.waveform_envelope = envelope
        if envelope:
            self._log("SYSTEM", f"Waveform decoded ({len(envelope)} samples).")
            self.waveform_caption.configure(
                text=f"{filename} — waveform preview", text_color=COLORS["text_faint"])
        else:
            self._log("WARNING",
                       "Couldn't decode a waveform preview for this file "
                       "(install numpy + soundfile, or ffmpeg for mp3/m4a).")
            self.waveform_caption.configure(
                text=f"{filename} — no waveform preview available",
                text_color=COLORS["text_faint"])

        self._update_start_button_state()

    def _update_start_button_state(self):
        has_file = self.audio_path is not None
        has_phone = len(self.phone_entry.get()) == 10
        ready = has_file and has_phone and not self.analysis_running
        if not self.analysis_running:
            self.start_button.configure(
                state="normal" if ready else "disabled",
                fg_color=COLORS["brand_soft"] if ready else COLORS["panel_alt"],
                text_color="#1a1305" if ready else COLORS["text_faint"],
            )

    def _start_analysis(self):
        if self.analysis_running:
            return
        if not self.audio_path:
            self._log("WARNING", "No audio file selected. Aborting analysis.")
            return
        phone = self.phone_entry.get()
        if len(phone) != 10:
            self._log("WARNING", "Enter a valid 10-digit phone number.")
            return

        self.analysis_running = True
        self.start_button.configure(state="disabled", text="ANALYZING...",
                                     fg_color=COLORS["panel_alt"],
                                     text_color=COLORS["text_faint"])
        self.result_label.configure(text="ANALYZING...", text_color=COLORS["warn"])
        self.confidence_label.configure(text="Confidence: —")
        self.confidence_bar.configure(progress_color=COLORS["brand"])
        self.confidence_bar.set(0)

        steps = [
            ("SYSTEM", "Loading audio buffer..."),
            ("MODEL", "Extracting MFCC features..."),
            ("MODEL", "Running spectrogram analysis..."),
            ("MODEL", "Computing voice embedding vector..."),
            ("MODEL", "Comparing against clone signature database..."),
        ]
        step_delay_ms = 500
        self.analysis_start_time = time.monotonic()
        self.analysis_duration = (len(steps) + 1) * (step_delay_ms / 1000.0)
        self._run_steps(steps, 0, phone, step_delay_ms)

    def _run_steps(self, steps, index, phone, step_delay_ms):
        if index < len(steps):
            source, message = steps[index]
            self._log(source, message)
            self.confidence_bar.set((index + 1) / (len(steps) + 1))
            after_id = self.after(
                step_delay_ms,
                lambda: self._run_steps(steps, index + 1, phone, step_delay_ms))
            self._pending_after_ids.append(after_id)
        else:
            after_id = self.after(step_delay_ms, lambda: self._finish_analysis(phone))
            self._pending_after_ids.append(after_id)

    def _finish_analysis(self, phone):
        is_clone = random.random() < 0.5
        confidence = random.uniform(78, 99)

        if is_clone:
            self.result_label.configure(text="SYNTHETIC VOICE DETECTED",
                                         text_color=COLORS["danger"])
            self._log("MODEL", f"Result: SYNTHETIC — confidence {confidence:.1f}%",
                       message_tag="MSG_BAD")
            self.incident_status.configure(text="INCIDENT DETECTED",
                                            text_color=COLORS["danger"])
            self.incident_panel.configure(border_color=COLORS["danger_dim"])
            self.confidence_bar.configure(progress_color=COLORS["danger"])
            self.previous_flags += 1
        else:
            self.result_label.configure(text="AUTHENTIC VOICE",
                                         text_color=COLORS["ok"])
            self._log("MODEL", f"Result: AUTHENTIC — confidence {confidence:.1f}%",
                       message_tag="MSG_OK")
            self.incident_status.configure(text="NO INCIDENT",
                                            text_color=COLORS["text_dim"])
            self.incident_panel.configure(border_color=COLORS["border"])
            self.confidence_bar.configure(progress_color=COLORS["ok"])

        self.confidence_label.configure(text=f"Confidence: {confidence:.1f}%")
        self.confidence_bar.set(confidence / 100)
        self.incident_phone.configure(text=f"+91 {phone}")
        self.incident_flags.configure(text=str(self.previous_flags))

        self.analysis_running = False
        self.analysis_start_time = None
        self.start_button.configure(text="▶  START ANALYSIS")
        self._update_start_button_state()

    def _reset_all(self):
        # cancel any in-flight analysis steps / animations before wiping state
        for after_id in self._pending_after_ids:
            try:
                self.after_cancel(after_id)
            except ValueError:
                pass
        self._pending_after_ids.clear()

        self.audio_path = None
        self.waveform_envelope = None
        self.previous_flags = 0
        self.analysis_running = False
        self.analysis_start_time = None

        self.file_display.configure(state="normal")
        self.file_display.delete(0, "end")
        self.file_display.configure(state="disabled")
        self.phone_entry.delete(0, "end")
        self.waveform_caption.configure(text="No audio loaded",
                                         text_color=COLORS["text_faint"])

        self.result_label.configure(text="WAITING", text_color=COLORS["text_dim"])
        self.confidence_label.configure(text="Confidence: —")
        self.confidence_bar.configure(progress_color=COLORS["brand"])
        self.confidence_bar.set(0)

        self.incident_phone.configure(text="—")
        self.incident_status.configure(text="NO INCIDENT", text_color=COLORS["text_dim"])
        self.incident_flags.configure(text="0")
        self.incident_panel.configure(border_color=COLORS["border"])

        self.start_button.configure(text="▶  START ANALYSIS")

        self.terminal.configure(state="normal")
        self.terminal.delete("1.0", "end")
        self.terminal.configure(state="disabled")
        self._log("SYSTEM", "System reset.")

        self._update_start_button_state()
        self._animate_waveform()
        self._blink_status()

    def _on_close(self):
        for after_id in self._pending_after_ids:
            try:
                self.after_cancel(after_id)
            except ValueError:
                pass
        self.destroy()


if __name__ == "__main__":
    app = VoiceGuardApp()
    app.mainloop()