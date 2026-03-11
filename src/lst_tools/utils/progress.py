"""Thin progress-bar abstraction (rich > tqdm > no-op fallback)."""

from __future__ import annotations
import logging
from typing import Callable

logger = logging.getLogger(__name__)

try:
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
    from rich.table import Column

    _HAVE_RICH = True
except ImportError:
    _HAVE_RICH = False
    Progress = None  # type: ignore

try:
    from tqdm import tqdm  # type: ignore

    _HAVE_TQDM = True
except ImportError:
    _HAVE_TQDM = False
    tqdm = None  # type: ignore


class _AdvanceCtx:
    """Base class for progress-bar context managers."""
    def __enter__(self) -> Callable[[int], None]:  # returns advance(delta)
        raise NotImplementedError

    def __exit__(self, exc_type, exc, tb) -> None:
        raise NotImplementedError


# --------------------------------------------------
# adapter for progress bar using rich
# --------------------------------------------------

class _RichCtx(_AdvanceCtx):
    """Progress-bar adapter using *rich*."""
    def __init__(self, total: int, desc: str | None, *, persist: bool) -> None:
        self.total = int(total)
        self.desc = desc or ""
        self.persist = persist
        self.prog = None
        self.task = None

    def __enter__(self):

        # get width of description to align bars us at least 8 characters

        desc_width = max(len(self.desc), 8)
        self.prog = Progress(
            TextColumn(
                "{task.description}",
                markup=False,
                table_column=Column(no_wrap=True, min_width=desc_width),
            ),
            BarColumn(bar_width=None, table_column=Column(ratio=4)),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            # disable transient if we want to keep the bar after completion -> if self.persist is True then transient is False -> we keep the bar
            transient=not self.persist,
            expand=True,
        )
        self.prog.start()
        self.task = self.prog.add_task(self.desc, total=self.total)

        def advance(n: int = 1) -> None:
            self.prog.update(self.task, advance=int(n))  # type: ignore

        return advance

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.prog is not None:
            self.prog.stop()



# --------------------------------------------------
# adapter for progress bar using tqdm
# --------------------------------------------------

class _TqdmCtx(_AdvanceCtx):
    """Progress-bar adapter using *tqdm*."""
    def __init__(self, total: int, desc: str | None, *, persist: bool) -> None:
        self.total = int(total)
        self.desc = desc or ""
        self.persist = persist
        self.bar = None

    def __enter__(self):
        self.bar = tqdm(total=self.total, desc=self.desc, leave=self.persist)  # type: ignore

        def advance(n: int = 1) -> None:
            if self.bar is not None:
                self.bar.update(int(n))

        return advance

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.bar is not None:
            self.bar.close()


class _NoopCtx(_AdvanceCtx):
    """Silent no-op fallback when no progress library is available."""
    def __init__(
        self, total: int, desc: str | None, *, persist: bool | None = None
    ) -> None:
        self._ = (total, desc, persist)  # quiet linters

    def __enter__(self):
        def advance(n: int = 1) -> None:
            pass

        return advance

    def __exit__(self, exc_type, exc, tb) -> None:
        pass


def progress(
    *, total: int, desc: str | None = None, persist: bool = True, **kwargs
) -> _AdvanceCtx:
    """
    Create a progress context manager.

    Accepts `desc=` or `description=`; returns a context manager whose __enter__
    gives an `advance(delta)` function.
    """
    # alias: description -> desc
    if desc is None and "description" in kwargs:
        desc = kwargs.pop("description")
    # ignore any extra kwargs to stay lenient

    if _HAVE_RICH:
        return _RichCtx(total, desc, persist=persist)
    if _HAVE_TQDM:
        return _TqdmCtx(total, desc, persist=persist)
    return _NoopCtx(total, desc)
