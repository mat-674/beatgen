"""Tqdm-based progress helpers used by every terminal-facing script in this repo.

Conventions used by every helper in this module:

- Bars write to **stderr** by default (keeps stdout clean for piping — the
  ``app_train.py`` regex parsers at lines 46-71 read stdout and must keep seeing
  the byte-identical ``[stage1]/[stage2]/[ok]/[skip]/[fail]/done:`` lines).
- ``should_use_tqdm(explicit)`` resolves the ``--bar {auto,on,off}`` CLI flag
  to a boolean. ``auto`` returns ``sys.stderr.isatty()``, so piped runs degrade
  to a line-logger instead of trying (and failing) to draw CR-based bars.
- Nested bars use ``tqdm(position=0/1, leave=...)`` so they stack cleanly. The
  outer ``epoch_progress`` is sticky (``leave=True``); inner step bars are
  ephemeral (``leave=False``).
- All helpers accept ``*, explicit="auto"`` and pass it to ``should_use_tqdm``
  so the user can force-disable via ``--bar off`` even on a TTY.
- ``UiProgress`` is the GUI-friendly counterpart: same ``update`` /
  ``set_postfix`` / ``set_phase`` surface as ``tqdm``/``NullBar`` but instead
  of drawing to a terminal it forwards every tick to a callback the Gradio
  front-end hands us. Pass it as ``bar=`` into ``generate_notes`` /
  ``generate.run`` and the long-running inference loop shows in the UI without
  the user staring at a frozen spinner.
"""
from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Optional

from tqdm import tqdm


# ---------------------------------------------------------------------------
# Public helpers — bar policy / ETA / JSONL log
# ---------------------------------------------------------------------------

def should_use_tqdm(explicit: str = "auto") -> bool:
    """Resolve ``--bar {auto,on,off}`` to a boolean.

    - ``"on"``  → always draw the bar (even when piped; lines will look messy).
    - ``"off"`` → never draw the bar; callers should fall back to a line logger.
    - ``"auto"`` → draw the bar iff ``sys.stderr.isatty()``.
    """
    if explicit == "on":
        return True
    if explicit == "off":
        return False
    return sys.stderr.isatty()


def format_eta(done: int, total: int, elapsed: float) -> str:
    """Return ``"ETA mm:ss"`` for in-progress, ``"mm:ss"`` for done==total.

    Returns ``"ETA --:--"`` when we have no progress yet — avoids a divide-
    by-zero blowup on the first heartbeat when ``n_done == 0``.
    """
    if total <= 0:
        return ""
    if done >= total:
        m, s = divmod(int(elapsed), 60)
        return f"{m:02d}:{s:02d}"
    if done <= 0 or elapsed < 1.0:
        return "ETA --:--"
    rate = done / max(elapsed, 1e-9)
    remain = (total - done) / max(rate, 1e-9)
    m, s = divmod(int(remain), 60)
    return f"ETA {m:02d}:{s:02d}"


def log_jsonl(path: Optional[Path], event: dict) -> None:
    """Append ``event`` as one JSON object per line to ``path``.

    ``path`` may be None — the call is a no-op in that case. The file is opened
    in append mode every time so the writer survives crashes / Ctrl+C without
    losing buffered data.
    """
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


@contextmanager
def ui_progress(total: int, desc: str, *, callback=None, explicit: str = "auto"):
    """Yield a bar that is wired to both the terminal (tqdm) AND a GUI callback.

    The yielded object satisfies the same ``update``/``set_postfix`` interface
    used everywhere else in this repo, so call-sites don't have to branch on
    "am I in a TTY or in Gradio?". Internally it forwards every tick to:

    - the underlying tqdm/NullBar (so a CLI user still sees a nice bar)
    - the ``callback`` (so a GUI front-end can render the same data — Gradio
      ``gr.Progress`` plus a Markdown status block)

    Pass ``callback=None`` and you get exactly the legacy tqdm/NullBar
    behaviour, so existing CLI scripts keep working untouched.
    """
    if callback is None:
        # Fast path — same as the legacy helpers.
        with _song_bar(total=total, desc=desc, explicit=explicit) as bar:
            yield bar
            return
    ui = UiProgress(total=total, desc=desc, callback=callback)
    base = _tqdm_or_null(total=total, desc=desc, explicit=explicit)
    try:
        yield _CompositeBar(ui, base)
    finally:
        ui.close()
        base.close()


# ---------------------------------------------------------------------------
# UiProgress — GUI-friendly counterpart to tqdm/NullBar
# ---------------------------------------------------------------------------

class UiProgress:
    """Forwards progress ticks to a GUI callback (e.g. Gradio ``gr.Progress``).

    The class is intentionally small: it has the same surface every other bar
    in this repo has (``update``, ``set_postfix``, ``set_phase``, ``close``)
    so ``generate_notes`` and friends don't need a special code path when
    called from ``app.py`` vs from a CLI.

    Throttling is enforced here, not in the callers, because every existing
    caller already pushes a tick per frame (~2-4k per song) and we don't want
    to throttle on the callback side where Gradio already deduplicates.

    Parameters
    ----------
    total : int
        Number of ticks this bar will receive. ``<= 0`` is allowed and means
        "indeterminate" — ``update`` accepts anything but ``progress(...)`` is
        never called with an absolute fraction.
    desc : str
        Human label; forwarded to the callback so the UI can show a heading.
    callback : Callable
        ``callback(fraction: float, phase: str, postfix: dict) -> None``
        called on each throttled tick. ``fraction`` is ``done / total`` in
        ``[0, 1]`` (or ``None`` for indeterminate). The callback may be any
        function; the Gradio side passes a closure that updates
        ``gr.Progress`` + a status Markdown.
    min_interval : float, default 0.1
        Seconds between callbacks. ``tqdm`` itself refreshes at ~4-10 Hz and
        Gradio collapses to one repaint per network round-trip, so 10 Hz here
        is plenty fast for the user but never starves the main thread.
    """

    __slots__ = ("total", "n", "desc", "_phase", "_postfix",
                 "_callback", "_min_interval", "_last_t", "_closed")

    def __init__(self, total: int, desc: str, callback, *,
                 min_interval: float = 0.1) -> None:
        self.total = max(int(total), 0)
        self.n = 0
        self.desc = desc
        self._phase = desc
        self._postfix: dict[str, Any] = {}
        self._callback = callback
        self._min_interval = float(min_interval)
        self._last_t = 0.0
        self._closed = False

    def _emit(self, force: bool = False) -> None:
        if self._closed:
            return
        now = time.monotonic()
        if not force and (now - self._last_t) < self._min_interval:
            return
        self._last_t = now
        frac = (self.n / self.total) if self.total > 0 else None
        try:
            self._callback(frac, self._phase, dict(self._postfix))
        except Exception:
            # A misbehaving callback (e.g. Gradio queue closed mid-run) must
            # never abort the inference loop — swallow and keep ticking.
            pass

    # tqdm-compatible surface — every existing caller uses one of these.

    def update(self, n: int = 1) -> None:
        self.n += n
        self._emit()

    def set_postfix(self, ordered: bool = False, refresh: bool = True,
                    **kw: Any) -> None:
        self._postfix.update({k: v for k, v in kw.items() if v is not None})
        if refresh:
            self._emit()

    def set_phase(self, phase: str, *, total: int | None = None) -> None:
        """Reset progress for a new sub-phase (e.g. analyze → decode → pack).

        ``total`` defaults to keeping the current total; pass a new int to
        resize the bar (useful when the second phase has a different tick
        count, like decode where ``total = number of onsets``).
        """
        self._phase = str(phase)
        if total is not None and total >= 0:
            self.total = int(total)
        self.n = 0
        self._postfix.clear()
        self._emit(force=True)

    def set_description(self, desc: str, refresh: bool = True) -> None:
        self._phase = str(desc)
        if refresh:
            self._emit(force=True)

    def write(self, msg: str, *, flush: bool = True) -> None:
        # ``write`` is tqdm-only; mirror it so call-sites don't crash if they
        # also pass a UiProgress to a helper that logs through ``bar.write``.
        # We re-emit so the GUI status line picks the message up.
        self._postfix["msg"] = msg
        if flush:
            self._emit(force=True)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Always push a final 100% (or last seen fraction) so the UI doesn't
        # get stuck mid-bar when the work item exits cleanly.
        try:
            frac = (self.n / self.total) if self.total > 0 else None
            self._callback(frac, self._phase, dict(self._postfix))
        except Exception:
            pass

    def __enter__(self) -> "UiProgress":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class _CompositeBar:
    """Fan-out wrapper: one underlying UiProgress + one legacy tqdm/NullBar.

    Lets ``generate_notes`` accept ``bar=`` as either an UiProgress (Gradio
    path) or a tqdm (CLI path) without each call-site having to branch. Both
    inner bars receive every ``update`` / ``set_postfix`` / ``set_phase``.
    """

    __slots__ = ("ui", "base")

    def __init__(self, ui: UiProgress, base: Any) -> None:
        self.ui = ui
        self.base = base

    def update(self, n: int = 1) -> None:
        self.ui.update(n)
        self.base.update(n)

    def set_postfix(self, ordered: bool = False, refresh: bool = True,
                    **kw: Any) -> None:
        self.ui.set_postfix(ordered=ordered, refresh=refresh, **kw)
        self.base.set_postfix(ordered=ordered, refresh=refresh, **kw)

    def set_phase(self, phase: str, *, total: int | None = None) -> None:
        self.ui.set_phase(phase, total=total)
        self.base.set_postfix(ordered=False, phase=phase)
        if total is not None and total >= 0 and hasattr(self.base, "total"):
            self.base.total = total    # tqdm exposes .total; NullBar mirrors it

    def write(self, msg: str, *, flush: bool = True) -> None:
        self.ui.write(msg, flush=flush)
        self.base.write(msg, flush=flush)

    def set_description(self, desc: str, refresh: bool = True) -> None:
        self.ui.set_description(desc, refresh=refresh)
        if hasattr(self.base, "set_description"):
            self.base.set_description(desc, refresh=refresh)

    def close(self) -> None:
        self.ui.close()
        self.base.close()

    def __enter__(self) -> "_CompositeBar":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _tqdm_or_null(total: int, desc: str, *, explicit: str):
    """Build the legacy tqdm-or-NullBar depending on the policy flag.

    Splits out from ``song_progress`` so ``ui_progress`` can reuse it
    without dragging a ``contextmanager`` along.
    """
    if should_use_tqdm(explicit):
        return tqdm(total=total, desc=desc, dynamic_ncols=True,
                    mininterval=0.2, file=sys.stderr, leave=False)
    return NullBar(total=total, desc=desc)


@contextmanager
def _song_bar(total: int, desc: str, *, explicit: str = "auto"):
    bar = _tqdm_or_null(total=total, desc=desc, explicit=explicit)
    try:
        yield bar
    finally:
        bar.close()


# ---------------------------------------------------------------------------
# NullBar — pipe/``--bar off`` fallback. Keeps the same call shape as tqdm.
# ---------------------------------------------------------------------------

class NullBar:
    """No-op progress object used when ``should_use_tqdm`` returns False.

    Mirrors the tqdm API surface used by the rest of this repo
    (``update``, ``set_postfix``, ``write``, ``close``). When ``flush=True`` is
    passed to ``write`` (it always is in our callers), each line lands on its
    own row of stdout/stderr and can be grep'd / piped downstream.
    """

    __slots__ = ("total", "n", "desc", "_postfix", "_last_flush_t")

    def __init__(self, total: int, desc: str = "") -> None:
        self.total = total
        self.n = 0
        self.desc = desc
        self._postfix: dict[str, Any] = {}
        self._last_flush_t = 0.0

    def update(self, n: int = 1) -> None:
        self.n += n
        # Throttle postfix prints to ~2 Hz so pipe logs don't drown in noise.
        now = time.monotonic()
        if self._postfix and (now - self._last_flush_t) >= 0.5:
            self._flush_postfix()

    def set_postfix(self, ordered: bool = False, refresh: bool = True, **kw: Any) -> None:
        self._postfix.update({k: v for k, v in kw.items() if v is not None})
        if refresh:
            now = time.monotonic()
            if (now - self._last_flush_t) >= 0.5:
                self._flush_postfix()

    def _flush_postfix(self) -> None:
        self._last_flush_t = time.monotonic()
        bits = " ".join(f"{k}={v}" for k, v in self._postfix.items())
        print(f"[{self.desc}] {self.n}/{self.total} {bits}".rstrip(), file=sys.stderr, flush=True)

    def write(self, msg: str, *, flush: bool = True) -> None:
        print(msg, file=sys.stderr, flush=flush)

    def close(self) -> None:
        if self._postfix:
            self._flush_postfix()

    # tqdm uses these names; keep them as no-ops so callers can pass either.
    def set_description(self, desc: str, refresh: bool = True) -> None:
        self.desc = desc

    def __enter__(self) -> "NullBar":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Context managers — one per "shape" of progress the codebase needs
# ---------------------------------------------------------------------------

@contextmanager
def song_progress(total: int, desc: str = "build_dataset", *, explicit: str = "auto"):
    """Coarse-grained bar for multi-phase work (dataset build, generate phases).

    ``total`` is the total number of "ticks" (songs for build_dataset, named
    sub-phases for generate). The bar is sticky — it stays on screen after the
    ``with`` block exits so users can read the final postfix.
    """
    if should_use_tqdm(explicit):
        bar: Any = tqdm(total=total, desc=desc, dynamic_ncols=True,
                        mininterval=0.2, file=sys.stderr, leave=True)
    else:
        bar = NullBar(total=total, desc=desc)
    try:
        yield bar
    finally:
        bar.close()


@contextmanager
def epoch_progress(total_epochs: int, stage: str, *, explicit: str = "auto"):
    """Outer epoch bar — one tick per epoch.

    Pinned to ``position=0`` so it sits above the inner step bar
    (``position=1``, ``leave=False``) when the training scripts nest them.
    """
    if should_use_tqdm(explicit):
        bar = tqdm(total=total_epochs, desc=f"[{stage}] epoch",
                   dynamic_ncols=True, mininterval=0.5, file=sys.stderr,
                   position=0, leave=True)
    else:
        bar = NullBar(total=total_epochs, desc=f"[{stage}] epoch")
    try:
        yield bar
    finally:
        bar.close()


@contextmanager
def decode_progress(total_frames: int, song_name: str, *, explicit: str = "auto"):
    """Bar over the Stage 2 autoregressive decode loop in ``generate.py``.

    Refresh is throttled (``mininterval=0.5``) because 2-4k frame songs refresh
    would otherwise dominate wall-time on slow terminals.
    """
    if should_use_tqdm(explicit):
        bar = tqdm(total=total_frames, desc=f"decode:{song_name}",
                   dynamic_ncols=True, mininterval=0.5, file=sys.stderr,
                   leave=True)
    else:
        bar = NullBar(total=total_frames, desc=f"decode:{song_name}")
    try:
        yield bar
    finally:
        bar.close()


@contextmanager
def inner_step_bar(total: int, *, desc: str = "step", explicit: str = "auto",
                   position: int = 1):
    """Inner ephemeral step bar — one tick per optimizer step inside an epoch.

    ``leave=False`` so the bar disappears after the epoch ends and the outer
    epoch bar can render cleanly.
    """
    if should_use_tqdm(explicit):
        bar = tqdm(total=total, desc=desc, dynamic_ncols=True,
                   mininterval=1.0, file=sys.stderr, position=position, leave=False)
    else:
        bar = NullBar(total=total, desc=desc)
    try:
        yield bar
    finally:
        bar.close()