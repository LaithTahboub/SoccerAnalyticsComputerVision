#!/usr/bin/env python3
import sys
import threading
from pathlib import Path

root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

import tkinter as tk
from tkinter import ttk

from src.analytics import make_analytics
from src.pipeline import compute_minimap

# ── Palette ───────────────────────────────────────────────────────────────────
BG        = "#0d1117"
SIDEBAR   = "#161b22"
CARD_BG   = "#21262d"
BORDER    = "#30363d"
TEXT      = "#e6edf3"
MUTED     = "#8b949e"
LEFT_COL  = "#1f6feb"
RIGHT_COL = "#d29922"
GREEN_BG  = "#238636"
GREEN_HOV = "#2ea043"

FONT_TITLE = ("Segoe UI", 16, "bold")
FONT_HEAD  = ("Segoe UI", 11, "bold")
FONT_BODY  = ("Segoe UI", 10)
FONT_MONO  = ("Consolas", 10)
FONT_SM    = ("Segoe UI", 9)

BAR_W = 320
BAR_H = 18


def get_clips():
    d = Path("data")
    return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.exists() else []


# ── Widgets ───────────────────────────────────────────────────────────────────

class BarCard(tk.Frame):
    """Analytics card with two horizontal bars (left/right team)."""

    def __init__(self, parent, label, unit="", possession=False, fmt=".2f"):
        super().__init__(parent, bg=CARD_BG, padx=18, pady=14,
                         highlightbackground=BORDER, highlightthickness=1)
        self.unit = unit
        self.possession = possession
        self.fmt = fmt

        tk.Label(self, text=label, font=FONT_HEAD, bg=CARD_BG,
                 fg=TEXT).grid(row=0, column=0, columnspan=3,
                               sticky="w", pady=(0, 10))

        self._left_canvas,  self._left_lbl  = self._row(1, "Left",  LEFT_COL)
        self._right_canvas, self._right_lbl = self._row(2, "Right", RIGHT_COL)

    def _row(self, row, name, color):
        tk.Label(self, text=name, font=FONT_BODY, bg=CARD_BG, fg=MUTED,
                 width=5, anchor="w").grid(row=row, column=0, sticky="w")
        c = tk.Canvas(self, width=BAR_W, height=BAR_H,
                      bg=CARD_BG, highlightthickness=0)
        c.grid(row=row, column=1, padx=(6, 10))
        lbl = tk.Label(self, text="—", font=FONT_MONO, bg=CARD_BG,
                       fg=TEXT, width=14, anchor="w")
        lbl.grid(row=row, column=2, sticky="w")
        self._draw_bar(c, 0, color)
        return c, lbl

    def update_values(self, left, right):
        if self.possession:
            total = left + right or 1
            lw = int(BAR_W * left / total)
            rw = BAR_W - lw
            self._draw_bar(self._left_canvas,  lw, LEFT_COL)
            self._draw_bar(self._right_canvas, rw, RIGHT_COL)
            self._left_lbl.config( text=f"{left:.0f} {self.unit}".strip())
            self._right_lbl.config(text=f"{right:.0f} {self.unit}".strip())
        else:
            mx = max(left, right, 1e-9)
            self._draw_bar(self._left_canvas,  int(BAR_W * left  / mx), LEFT_COL)
            self._draw_bar(self._right_canvas, int(BAR_W * right / mx), RIGHT_COL)
            u = f" {self.unit}" if self.unit else ""
            self._left_lbl.config( text=f"{left:{self.fmt}}{u}")
            self._right_lbl.config(text=f"{right:{self.fmt}}{u}")

    @staticmethod
    def _draw_bar(canvas, width, color):
        canvas.delete("all")
        canvas.create_rectangle(0, 4, BAR_W, BAR_H - 4,
                                 fill=BORDER, outline="")
        if width > 0:
            canvas.create_rectangle(0, 4, width, BAR_H - 4,
                                     fill=color, outline="")


# ── Main App ──────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Soccer Analytics Dashboard")
        self.configure(bg=BG)
        self.geometry("980x700")
        self.minsize(800, 500)
        self._build_ui()

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_sidebar()
        self._build_main()

    def _build_sidebar(self):
        sb = tk.Frame(self, bg=SIDEBAR, width=230)
        sb.pack(side="left", fill="y")
        sb.pack_propagate(False)

        tk.Label(sb, text="Soccer Analytics", font=FONT_TITLE,
                 bg=SIDEBAR, fg=TEXT).pack(pady=(28, 2), padx=18, anchor="w")
        tk.Label(sb, text="CMSC 426  ·  Computer Vision",
                 font=FONT_SM, bg=SIDEBAR, fg=MUTED).pack(padx=18, anchor="w")

        tk.Frame(sb, bg=BORDER, height=1).pack(fill="x", pady=18)

        # clip selector
        tk.Label(sb, text="CLIP", font=FONT_SM, bg=SIDEBAR,
                 fg=MUTED).pack(padx=18, anchor="w")
        clips = get_clips()
        self._clip_var = tk.StringVar(value=clips[0] if clips else "")
        self._clip_box = ttk.Combobox(sb, textvariable=self._clip_var,
                                       values=clips, state="readonly", width=20)
        self._clip_box.pack(padx=18, pady=(4, 14))

        # run button
        self._run_btn = tk.Button(
            sb, text="▶  Run Analysis", font=FONT_HEAD,
            bg=GREEN_BG, fg=BG, activebackground=GREEN_HOV,
            activeforeground=TEXT, relief="flat", pady=9,
            cursor="hand2", command=self._run)
        self._run_btn.pack(padx=18, fill="x")

        # progress bar
        self._progress = ttk.Progressbar(sb, mode="indeterminate")
        self._progress.pack(padx=18, pady=(10, 0), fill="x")

        tk.Frame(sb, bg=BORDER, height=1).pack(fill="x", pady=18)

        self._status_lbl = tk.Label(sb, text="Ready.", font=FONT_SM,
                                     bg=SIDEBAR, fg=MUTED,
                                     wraplength=194, justify="left")
        self._status_lbl.pack(padx=18, anchor="w")

        # legend
        tk.Frame(sb, bg=BORDER, height=1).pack(fill="x", pady=18)
        leg = tk.Frame(sb, bg=SIDEBAR)
        leg.pack(padx=18, anchor="w")
        for row, (col, label) in enumerate([(LEFT_COL, "Left team"),
                                             (RIGHT_COL, "Right team")]):
            tk.Frame(leg, bg=col, width=12, height=12).grid(
                row=row, column=0, pady=(0 if row == 0 else 5, 0))
            tk.Label(leg, text=f"  {label}", font=FONT_SM,
                     bg=SIDEBAR, fg=TEXT).grid(row=row, column=1, sticky="w")

    def _build_main(self):
        outer = tk.Frame(self, bg=BG)
        outer.pack(side="left", fill="both", expand=True)

        # scrollable canvas
        cvs = tk.Canvas(outer, bg=BG, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=cvs.yview)
        self._inner = tk.Frame(cvs, bg=BG)

        self._inner.bind("<Configure>",
            lambda e: cvs.configure(scrollregion=cvs.bbox("all")))
        cvs.create_window((0, 0), window=self._inner, anchor="nw")
        cvs.configure(yscrollcommand=vsb.set)

        cvs.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        cvs.bind_all("<MouseWheel>",
            lambda e: cvs.yview_scroll(-1 if e.delta > 0 else 1, "units"))

        self._build_cards()

    def _build_cards(self):
        p = dict(padx=22, pady=8, fill="x")
        self._cards = {}

        specs = [
            # key              label                              unit   possession  fmt
            ("possession",    "Ball Possession",                 "%",   True,  ".0f"),
            ("team_speed",    "Average Team Speed",              "m/s", False, ".2f"),
            ("distance",      "Total Distance Covered",          "m",   False, ".1f"),
            ("passes",        "Passes",                          "",    False, ".0f"),
            ("compactness",   "Team Compactness (avg spread)",   "m",   False, ".1f"),
            ("avg_poss",      "Avg Possession Duration",         "s",   False, ".2f"),
            ("ball_speed",    "Avg Ball Speed",                  "m/s", False, ".2f"),
        ]
        for key, label, unit, poss, fmt in specs:
            card = BarCard(self._inner, label, unit, poss, fmt)
            card.pack(**p)
            self._cards[key] = card

    # ── pipeline thread ───────────────────────────────────────────────────────

    def _run(self):
        clip = self._clip_var.get()
        if not clip:
            return
        self._run_btn.config(state="disabled")
        self._progress.start(12)
        self._status("Processing  " + clip + " …")
        threading.Thread(target=self._worker, args=(clip,), daemon=True).start()

    def _worker(self, clip):
        try:
            _, _, _, _, frames = compute_minimap(clip)
            analytics = make_analytics()
            for dets in frames:
                for a in analytics:
                    a.update(dets)
            self.after(0, self._show_results, analytics)
        except Exception as exc:
            self.after(0, self._show_error, str(exc))

    # ── results ───────────────────────────────────────────────────────────────

    def _show_results(self, analytics):
        self._progress.stop()
        self._run_btn.config(state="normal")
        self._status("Done.")

        by = {type(a).__name__: a for a in analytics}

        # Possession
        a = by["Possession"]
        tot = a.counts["left"] + a.counts["right"]
        if tot:
            lp = 100.0 * a.counts["left"] / tot
            self._cards["possession"].update_values(lp, 100.0 - lp)

        # Team speed
        a = by["TeamSpeed"]
        self._cards["team_speed"].update_values(
            *self._mean2(a.speeds["left"], a.speeds["right"]))

        # Distance
        a = by["DistanceTravelled"]
        self._cards["distance"].update_values(
            a.distances["left"], a.distances["right"])

        # Passes
        a = by["Passes"]
        self._cards["passes"].update_values(
            a.counts["left"], a.counts["right"])

        # Compactness
        a = by["Compactness"]
        self._cards["compactness"].update_values(
            *self._mean2(a.spread["left"], a.spread["right"]))

        # Avg possession duration
        a = by["AveragePossessionRate"]
        lv = (sum(a.pos_duration["left"])  / len(a.pos_duration["left"])  / a.fps
              if a.pos_duration["left"] else 0.0)
        rv = (sum(a.pos_duration["right"]) / len(a.pos_duration["right"]) / a.fps
              if a.pos_duration["right"] else 0.0)
        self._cards["avg_poss"].update_values(lv, rv)

        # Ball speed
        a = by["AverageBallSpeed"]
        self._cards["ball_speed"].update_values(
            *self._mean2(a.speeds["left"], a.speeds["right"]))

    def _show_error(self, msg):
        self._progress.stop()
        self._run_btn.config(state="normal")
        self._status("Error: " + msg[:200])

    # ── helpers ───────────────────────────────────────────────────────────────

    def _status(self, msg):
        self._status_lbl.config(text=msg)

    @staticmethod
    def _mean2(left_list, right_list):
        lv = sum(left_list)  / len(left_list)  if left_list  else 0.0
        rv = sum(right_list) / len(right_list) if right_list else 0.0
        return lv, rv


if __name__ == "__main__":
    App().mainloop()