"""Progress-bar wrapper around rich."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Column


# --------------------------------------------------
# ricch progress context manager
# --------------------------------------------------
class _RichCtx:
    """Progress-bar context manager using *rich*."""

    def __init__(self, total: int, desc: str | None, *, persist: bool) -> None:
        self.total = int(total)
        self.desc = desc or ""
        self.persist = persist
        self.prog = None
        self.task = None

    def __enter__(self):

        # get width of description to align bars at least 8 characters
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
            # disable transient if we want to keep the bar after completion
            # if self.persist is True then transient is False -> we keep the bar
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
# public API
# --------------------------------------------------
def progress(
    *, total: int, desc: str | None = None, persist: bool = True, **kwargs
) -> _RichCtx:
    """Create a progress context manager.

    Accepts ``desc=`` or ``description=``; returns a context manager whose
    ``__enter__`` gives an ``advance(delta)`` function.
    """
    # alias: description -> desc
    if desc is None and "description" in kwargs:
        desc = kwargs.pop("description")

    return _RichCtx(total, desc, persist=persist)
