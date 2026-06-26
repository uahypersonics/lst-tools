"""Typed configuration dataclasses for lst_tools.

Each section of the TOML config maps to a dataclass with sensible defaults.
The top-level ``Config`` object can be built from a raw dict
(``Config.from_dict(d)``) or directly from a TOML file
(``Config.from_toml("lst.cfg")``).

Access config fields as attributes::

    cfg.lst.solver.type
    cfg.flow_conditions.mach
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
import re
import dataclasses
from pathlib import Path
from typing import Any


# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# toml loader guards for Python <3.11 (tomli) and 3.11+ (tomllib)
# --------------------------------------------------
try:
    import tomllib as _toml
except ModuleNotFoundError:
    try:
        import tomli as _toml  # type: ignore[no-redef]
    except ModuleNotFoundError as e:
        raise ImportError(
            "lst_tools.config requires 'tomli' on Python <3.11. "
            "Install with: pip install tomli"
        ) from e


# --------------------------------------------------
# Boolean normalisation (True/False -> true/false before TOML parsing)
# --------------------------------------------------

# regex group patterns for matching Python-style booleans (True/False) in TOML config files

# group 1: key and equals sign (e.g. "is_body_fitted = ")
key_equals = r"(\s*[A-Za-z0-9_.\-]+\s*=\s*)"
# group 2: the boolean value (True or False)
bool_value = r"(True|False)"
# group 3: optional trailing comment (e.g. "# some comment")
comment    = r"(#.*)?"

# compiled regex search pattern (note: \s* allows for optional whitespace between groups 2 and 3)
search_pattern_bool = re.compile(
    rf"^{key_equals}{bool_value}\s*{comment}$",
    re.MULTILINE,
)


def _normalize_toml_bool(text: str) -> tuple[str, int]:
    """Replace Python-style ``True``/``False`` with TOML ``true``/``false``.

    Scans the entire config text for lines where a key is set to
    ``True`` or ``False`` (Python-style) and replaces them with
    lowercase ``true``/``false`` (TOML-style).

    Args:
        text (str): Raw TOML config file contents.

    Returns:
        tuple[str, int]: A tuple of (fixed_text, count) where
            *fixed_text* is the corrected config string and
            *count* is the number of booleans that were replaced.
    """

    # count how many matches exist
    count = len(search_pattern_bool.findall(text))

    # replace True/False with true/false
    def _lower(m: re.Match) -> str:
        # capture the comment or set to empty string if group(3) is None
        comment = m.group(3) or ""
        # reconstruct the line with the original key and equals sign, the lowercased boolean, and the comment
        return f"{m.group(1)}{m.group(2).lower()} {comment}".rstrip()

    # updated text with replacements
    fixed = search_pattern_bool.sub(_lower, text)

    return fixed, count


# --------------------------------------------------
# Complex-number coercion  "(a, b)" -> complex
# --------------------------------------------------
def _to_complex(v: Any) -> complex:
    """Coerce a value to a Python complex number.

    Handles multiple input formats commonly found in config files:
    ``complex``, ``int``/``float``, ``(real, imag)`` strings/tuples,
    and ``i``/``j`` notation (e.g. ``"1+2j"``).

    Args:
        v (Any): Value to convert. Can be a complex, numeric, list/tuple
            of two numbers, or a string like ``"(0.1, -0.05)"``.

    Returns:
        complex: The converted complex number.

    Raises:
        ValueError: If *v* cannot be interpreted as a complex number.
    """
    if isinstance(v, complex):
        return v
    if isinstance(v, (int, float)):
        return complex(float(v), 0.0)
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return complex(float(v[0]), float(v[1]))
    s = str(v).strip()
    if s.startswith("(") and s.endswith(")"):
        parts = s[1:-1].split(",", 1)
        if len(parts) == 2:
            return complex(float(parts[0]), float(parts[1]))

    # i/j notation fallback (e.g. "1+2j" or "3-4i")
    lo = s.lower()
    if "j" in lo or "i" in lo:
        return complex(lo.replace("i", "j"))
    raise ValueError(f"cannot convert {v!r} to complex")


# --------------------------------------------------
# Scalar coercion helpers
# --------------------------------------------------
def _opt_float(v: Any) -> float | None:
    """Convert *v* to float, or return ``None`` if empty/missing."""
    if v is None:
        return None
    if isinstance(v, str) and v.strip() in ("", "none", "null"):
        return None
    return float(v)


def _opt_int(v: Any) -> int | None:
    """Convert *v* to int, or return ``None`` if empty/missing."""
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, str) and v.strip() in ("", "none", "null"):
        return None
    return int(float(str(v)))


def _opt_str(v: Any) -> str | None:
    """Convert *v* to str, or return ``None`` if missing or empty."""
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _coerce_bool(v: Any) -> bool:
    """Convert *v* to bool. Accepts common truthy/falsy strings like ``"yes"``, ``"no"``, ``"1"``, etc."""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "t", "1", "yes", "y", "on"):
        return True
    if s in ("false", "f", "0", "no", "n", "off"):
        return False
    raise ValueError(f"cannot convert {v!r} to bool")


# --------------------------------------------------
# Base mixin for serialisation helpers
#
# _ConfigBase is a "mixin" class, it is NOT meant to be used on its own.
# All config dataclasses (FlowConditions, Geometry, Config, etc.) inherit
# from it, which gives them .to_dict(), .to_toml_dict(), and a custom __eq__.
# Write shared logic here once, all subclasses get it automatically.
# --------------------------------------------------
class _ConfigBase:
    """Shared helpers for all config dataclasses."""

    def __eq__(self, other: Any) -> bool:
        """Custom equality check — allows comparing a dataclass to a plain dict.

        This is needed because @dataclasses.dataclass(eq=False) disables the
        auto-generated __eq__.  We disable it so we can support dict comparison
        (useful in tests, e.g. ``assert cfg.geometry == {"type": 1, ...}``).
        """
        # case 1: comparing against a plain dict -> convert self to dict first
        if isinstance(other, dict):
            return dataclasses.asdict(self) == other  # type: ignore[arg-type]
        # case 2: comparing against another dataclass -> convert both to dict
        if hasattr(other, "__dataclass_fields__"):
            return dataclasses.asdict(self) == dataclasses.asdict(other)  # type: ignore[arg-type]
        # case 3: unknown type -> let Python handle it
        return NotImplemented

    def to_dict(self) -> dict[str, Any]:
        """Return a plain nested dict (preserves Python types)."""
        return dataclasses.asdict(self)  # type: ignore[arg-type]

    def to_toml_dict(self) -> dict[str, Any]:
        """Return a dict suitable for ``tomli_w.dumps`` (complex -> string).

        Same as to_dict() but converts complex numbers to "(re,im)" strings,
        because TOML has no native complex type.
        """
        return _clean_dict(dataclasses.asdict(self))  # type: ignore[arg-type]


# --------------------------------------------------
# helper function to recursively convert config values to toml-serialisable types
# --------------------------------------------------
def _clean_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Convert complex values to ``"(re,im)"`` strings for TOML serialisation."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _clean_dict(v)
        elif isinstance(v, complex):
            out[k] = f"({v.real},{v.imag})"
        else:
            out[k] = v
    return out


# --------------------------------------------------
# FlowConditions dataclass
# --------------------------------------------------
@dataclasses.dataclass(eq=False)
# inherit from _ConfigBase to get .to_dict(), .to_toml_dict(), and custom __eq__
class FlowConditions(_ConfigBase):
    """Free-stream / stagnation flow conditions."""
    mach: float | None = None
    re1: float | None = None
    gamma: float = 1.4
    cp: float = 1005.025
    cv: float = 717.875
    rgas: float = 287.15
    pres_0: float | None = None
    temp_0: float | None = None
    pres_inf: float | None = None
    temp_inf: float | None = None
    dens_inf: float | None = None
    uvel_inf: float | None = None
    visc_law: int = 0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FlowConditions:
        """Build a ``FlowConditions`` from a plain dict."""
        kw: dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            if f.name in d:
                v = d[f.name]
                if f.name == "visc_law":
                    kw[f.name] = int(v) if v is not None else 0
                else:
                    kw[f.name] = _opt_float(v) if v is not None else None
        return cls(**kw)


# --------------------------------------------------
# Geometry dataclass
# --------------------------------------------------
@dataclasses.dataclass(eq=False)
# inherit from _ConfigBase to get .to_dict(), .to_toml_dict(), and custom __eq__
class Geometry(_ConfigBase):
    """Body geometry specification."""
    type: int | None = None
    theta_deg: float | None = None
    r_nose: float | None = None
    l_ref: float = 1.0
    is_body_fitted: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Geometry:
        """Build a ``Geometry`` from a plain dict."""
        kw: dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            if f.name in d:
                v = d[f.name]
                if f.name == "type":
                    kw[f.name] = _opt_int(v)
                elif f.name == "is_body_fitted":
                    kw[f.name] = _coerce_bool(v) if v is not None else False
                elif f.name == "l_ref":
                    kw[f.name] = float(v) if v is not None else 1.0
                else:
                    kw[f.name] = _opt_float(v)
        return cls(**kw)


# --------------------------------------------------
# MeanflowConversion dataclass
# --------------------------------------------------
@dataclasses.dataclass(eq=False)
# inherit from _ConfigBase to get .to_dict(), .to_toml_dict(), and custom __eq__
class MeanflowConversion(_ConfigBase):
    """Index range and options for meanflow conversion."""
    i_s: int = 0
    i_e: int | None = None
    d_i: int = 1
    set_v_zero: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MeanflowConversion:
        """Build a ``MeanflowConversion`` from a plain dict."""
        _svz = d.get("set_v_zero")
        return cls(
            i_s=int(d.get("i_s", 0)),
            i_e=_opt_int(d.get("i_e")),
            d_i=int(d.get("d_i", 1)),
            set_v_zero=_coerce_bool(_svz) if _svz is not None else True,
        )


# --------------------------------------------------
# LstSolver settings dataclass
# --------------------------------------------------
@dataclasses.dataclass(eq=False)
# inherit from _ConfigBase to get .to_dict(), .to_toml_dict(), and custom __eq__
class LstSolver(_ConfigBase):
    """LST solver settings."""
    type: int = 1
    is_simplified: bool = True
    alpha_i_threshold: float = -100.0
    generalized: int = 0
    spatial_temporal: int = 1
    energy_formulation: int = 1

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LstSolver:
        """Build a ``LstSolver`` from a plain dict.

        Explicit ``None`` values are treated as "not provided" and fall
        back to the field default.
        """
        _is = d.get("is_simplified")
        _type = d.get("type")
        _gen = d.get("generalized")
        _st = d.get("spatial_temporal")
        _ef = d.get("energy_formulation")
        _ait = d.get("alpha_i_threshold")
        return cls(
            type=int(_type) if _type is not None else 1,
            is_simplified=_coerce_bool(_is) if _is is not None else True,
            alpha_i_threshold=float(_ait) if _ait is not None else -100.0,
            generalized=int(_gen) if _gen is not None else 0,
            spatial_temporal=int(_st) if _st is not None else 1,
            energy_formulation=int(_ef) if _ef is not None else 1,
        )


# --------------------------------------------------
# LstOptions dataclass
# --------------------------------------------------
@dataclasses.dataclass(eq=False)
# inherit from _ConfigBase to get .to_dict(), .to_toml_dict(), and custom __eq__
class LstOptions(_ConfigBase):
    """LST analysis options."""
    geometry_switch: int | None = None
    longitudinal_curvature: int = 0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LstOptions:
        """Build a ``LstOptions`` from a plain dict.

        Explicit ``None`` values are treated as "not provided" and fall
        back to the field default.
        """
        _lc = d.get("longitudinal_curvature")
        return cls(
            geometry_switch=_opt_int(d.get("geometry_switch")),
            longitudinal_curvature=int(_lc) if _lc is not None else 0,
        )


# --------------------------------------------------
# LstParams dataclass
# --------------------------------------------------
@dataclasses.dataclass(eq=False)
# inherit from _ConfigBase to get .to_dict(), .to_toml_dict(), and custom __eq__
class LstParams(_ConfigBase):
    """LST solver parameters."""
    ny: int = 150
    yl_in: float = 0.0
    tol_lst: float = 1e-5
    max_iter: int = 15
    x_s: float | None = None
    x_e: float | None = None
    i_step: int | None = None
    tracking_dir: int = 1  # 0 = downstream, 1 = upstream
    f_min: float | None = None
    f_max: float | None = None
    d_f: float | None = None
    f_init: float = 0.0
    beta_s: float | None = None
    beta_e: float | None = None
    d_beta: float | None = None
    beta_init: float = 0.0
    alpha_0: complex = 0 + 0j

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LstParams:
        """Build a ``LstParams`` from a plain dict."""
        kw: dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            if f.name not in d:
                continue
            v = d[f.name]
            if f.name == "alpha_0":
                kw[f.name] = _to_complex(v) if v is not None else 0 + 0j
            elif f.name in ("ny", "max_iter"):
                kw[f.name] = int(v) if v is not None else f.default
            elif f.name in ("i_step",):
                kw[f.name] = _opt_int(v)
            elif f.name == "tracking_dir":
                kw[f.name] = int(v) if v is not None else 1
            else:
                kw[f.name] = _opt_float(v)
        return cls(**kw)


# --------------------------------------------------
# LstIO dataclass
# --------------------------------------------------
@dataclasses.dataclass(eq=False)
# inherit from _ConfigBase to get .to_dict(), .to_toml_dict(), and custom __eq__
class LstIO(_ConfigBase):
    """LST I/O paths."""
    baseflow_input: str = "meanflow.bin"
    solution_output: str = "growth_rate.dat"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LstIO:
        """Build a ``LstIO`` from a plain dict."""
        return cls(
            baseflow_input=str(d.get("baseflow_input", "meanflow.bin")),
            solution_output=str(d.get("solution_output", "growth_rate.dat")),
        )


# --------------------------------------------------
# LstConfig dataclass (groups solver + options + params + io)
# --------------------------------------------------
@dataclasses.dataclass(eq=False)
# inherit from _ConfigBase to get .to_dict(), .to_toml_dict(), and custom __eq__
class LstConfig(_ConfigBase):
    """All LST-related configuration (solver + options + params + io)."""
    solver: LstSolver = dataclasses.field(default_factory=LstSolver)
    options: LstOptions = dataclasses.field(default_factory=LstOptions)
    params: LstParams = dataclasses.field(default_factory=LstParams)
    io: LstIO = dataclasses.field(default_factory=LstIO)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LstConfig:
        """Build a ``LstConfig`` from a plain dict."""
        return cls(
            solver=LstSolver.from_dict(d.get("solver", {})),
            options=LstOptions.from_dict(d.get("options", {})),
            params=LstParams.from_dict(d.get("params", {})),
            io=LstIO.from_dict(d.get("io", {})),
        )


# --------------------------------------------------
# SeedTable dataclass
# --------------------------------------------------
@dataclasses.dataclass(eq=False)
# inherit from _ConfigBase to get .to_dict(), .to_toml_dict(), and custom __eq__
class SeedTable(_ConfigBase):
    """Tracking initial-guess seed table generation.

    When ``enabled`` is True, ``setup tracking`` runs the ridge tracker
    from :mod:`lst_tools.process.maxima` on the parsing solution, picks
    ``n_seeds`` evenly-spaced points along each detected ridge (one ridge
    per physical mode), and writes a ``seed_alpha.dat`` file into every
    ``kc_*`` case directory.  The Fortran tracking solver reads that file
    on startup and uses the nearest seed (within a normalized
    ``threshold`` distance in (x, f) space) to override its
    marched/extrapolated initial guess at each station.

    File presence alone activates the seed table on the solver side,
    so disabling generation here is sufficient to fall back to the
    standard tracking initial-guess flow.
    """

    # master switch
    enabled: bool = False

    # source tecplot ASCII (parsing or refined tracking output); None -> auto
    source_file: str | None = None

    # number of seeds to emit per detected ridge (one ridge = one mode)
    n_seeds: int = 12

    # acceptance floor: only stations with alpi >= min_growth contribute seeds
    min_growth: float = 10.0

    # ridge tracker controls (forwarded to process.maxima._track_ridges)
    gate_tol: float = 0.10  # relative frequency gate for matching peaks across stations
    min_valid: int = 5      # minimum stations a ridge must span to be accepted

    # number of de-spike / prominence smoothing passes applied to the alpi
    # field BEFORE ridge detection. 0 disables smoothing entirely (default).
    #
    # Smoothing helps reject isolated outliers (e.g. off-mode lobes near the
    # leading edge) that would otherwise fragment the banana ridge into many
    # short pieces. With smooth_passes=0, the raw noisy field can cause the
    # ridge tracker to split a single mode into 20+ fragments, each shorter
    # than min_valid, leaving gaps in the seed coverage.
    #
    # The keep_mask used for seed gating is now built from the union of all
    # detected ridge bands (slope-agnostic, not DP-based), so it remains
    # correct regardless of smooth_passes. smooth_passes only affects the
    # DETECTION quality — whether the banana appears as 1 or 20 fragments.
    #
    # Recommended: smooth_passes=2 for steep bending ridges (mode-2 banana).
    # Use smooth_passes=0 only when you want to see every raw ridge fragment
    # or the field is already clean.
    smooth_passes: int = 0

    # If True, build a keep_mask from the union of all detected ridge bands
    # and discard any candidate seed that falls outside this mask.
    #
    # The mask covers ±3 freq-bins around every point of every ridge that
    # survived the min_valid filter. This rejects off-ridge outliers and
    # noise peaks the ridge tracker happened to latch onto, without
    # constraining valid seeds anywhere along the detected ridge path.
    # Unlike the old DP-based mask, this approach is slope-agnostic and
    # correctly covers steep banana ridges all the way to their tail.
    #
    # Disable only for diagnostic runs where you want to see every candidate
    # the ridge tracker found.
    gate_by_keep_mask: bool = True

    # optional clipping windows (empty list -> no clipping)
    x_range: list[float] = dataclasses.field(default_factory=list)
    f_range: list[float] = dataclasses.field(default_factory=list)

    # solver-side override radius in normalized (x,f) space; written to file header
    threshold: float = 0.15

    # output filename (must match SEED_FILE constant in seed_table.f90)
    output_file: str = "seed_alpha.dat"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SeedTable:
        """Build a ``SeedTable`` from a plain dict."""
        # default empty list for optional ranges
        _x_range = d.get("x_range") or []
        _f_range = d.get("f_range") or []

        # validate ranges
        x_range = [float(v) for v in _x_range]
        f_range = [float(v) for v in _f_range]

        # build with explicit per-field coercion
        _enabled = d.get("enabled")
        _src = d.get("source_file")
        _n_seeds = d.get("n_seeds")
        _min_growth = d.get("min_growth")
        _gate_tol = d.get("gate_tol")
        _min_valid = d.get("min_valid")
        _smooth_passes = d.get("smooth_passes")
        _gate_by_keep_mask = d.get("gate_by_keep_mask")
        _threshold = d.get("threshold")
        _output = d.get("output_file")

        return cls(
            enabled=_coerce_bool(_enabled) if _enabled is not None else False,
            source_file=_opt_str(_src),
            n_seeds=int(_n_seeds) if _n_seeds is not None else 12,
            min_growth=float(_min_growth) if _min_growth is not None else 10.0,
            gate_tol=float(_gate_tol) if _gate_tol is not None else 0.10,
            min_valid=int(_min_valid) if _min_valid is not None else 5,
            smooth_passes=int(_smooth_passes) if _smooth_passes is not None else 0,
            gate_by_keep_mask=_coerce_bool(_gate_by_keep_mask)
                if _gate_by_keep_mask is not None else True,
            x_range=x_range,
            f_range=f_range,
            threshold=float(_threshold) if _threshold is not None else 0.15,
            output_file=str(_output) if _output is not None else "seed_alpha.dat",
        )


# --------------------------------------------------
# HpcConfig dataclass
# --------------------------------------------------
@dataclasses.dataclass(eq=False)
# inherit from _ConfigBase to get .to_dict(), .to_toml_dict(), and custom __eq__
class HpcConfig(_ConfigBase):
    """HPC job-scheduler settings."""
    account: str | None = None
    nodes: int | None = None
    time: str | None = None
    partition: str | None = None
    extra_env: dict[str, str] | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HpcConfig:
        """Build an ``HpcConfig`` from a plain dict."""
        # read extra_env as a plain dict of string key-value pairs
        raw_env = d.get("extra_env")
        extra_env = {str(k): str(v) for k, v in raw_env.items()} if raw_env else None

        return cls(
            account=_opt_str(d.get("account")),
            nodes=_opt_int(d.get("nodes")),
            time=_opt_str(d.get("time")),
            partition=_opt_str(d.get("partition")),
            extra_env=extra_env,
        )


# --------------------------------------------------
# Processing sub-sections
# --------------------------------------------------
@dataclasses.dataclass(eq=False)
class TrackingProcessing(_ConfigBase):
    """Post-processing settings for tracking results."""

    interpolate: bool = False
    gate_tol: float = 0.10
    min_valid: int = 40
    peak_order: int = 1

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrackingProcessing:
        """Build a ``TrackingProcessing`` from a plain dict."""
        _interp = d.get("interpolate")
        _gate = d.get("gate_tol")
        _minv = d.get("min_valid")
        _peak = d.get("peak_order")
        return cls(
            interpolate=_coerce_bool(_interp) if _interp is not None else False,
            gate_tol=float(_gate) if _gate is not None else 0.10,
            min_valid=int(_minv) if _minv is not None else 40,
            peak_order=int(_peak) if _peak is not None else 1,
        )


@dataclasses.dataclass(eq=False)
class SpectraProcessing(_ConfigBase):
    """Post-processing settings for spectra results."""

    alpr_min: float | None = None
    alpr_max: float | None = None
    alpi_min: float | None = None
    alpi_max: float | None = None
    branch_gate: float = 0.25
    branch_min_points: int = 2
    isolation_k: int = 3
    isolation_threshold: float | None = None
    classify_min_points: int = 3

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SpectraProcessing:
        """Build a ``SpectraProcessing`` from a plain dict."""
        _alpr_min = d.get("alpr_min")
        _alpr_max = d.get("alpr_max")
        _alpi_min = d.get("alpi_min")
        _alpi_max = d.get("alpi_max")
        _branch_gate = d.get("branch_gate")
        _branch_min = d.get("branch_min_points")
        _iso_k = d.get("isolation_k")
        _iso_thresh = d.get("isolation_threshold")
        _class_min = d.get("classify_min_points")
        return cls(
            alpr_min=_opt_float(_alpr_min),
            alpr_max=_opt_float(_alpr_max),
            alpi_min=_opt_float(_alpi_min),
            alpi_max=_opt_float(_alpi_max),
            branch_gate=float(_branch_gate) if _branch_gate is not None else 0.25,
            branch_min_points=int(_branch_min) if _branch_min is not None else 2,
            isolation_k=int(_iso_k) if _iso_k is not None else 3,
            isolation_threshold=_opt_float(_iso_thresh),
            classify_min_points=int(_class_min) if _class_min is not None else 3,
        )


@dataclasses.dataclass(eq=False)
class ParsingProcessing(_ConfigBase):
    """Reserved post-processing settings for parsing results."""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ParsingProcessing:
        """Build a ``ParsingProcessing`` from a plain dict."""
        return cls()


# --------------------------------------------------
# Processing dataclass
# --------------------------------------------------
@dataclasses.dataclass(eq=False)
# inherit from _ConfigBase to get .to_dict(), .to_toml_dict(), and custom __eq__
class Processing(_ConfigBase):
    """Grouped post-processing settings."""

    tracking: TrackingProcessing = dataclasses.field(default_factory=TrackingProcessing)
    spectra: SpectraProcessing = dataclasses.field(default_factory=SpectraProcessing)
    parsing: ParsingProcessing = dataclasses.field(default_factory=ParsingProcessing)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Processing:
        """Build a ``Processing`` from a plain dict."""
        return cls(
            tracking=TrackingProcessing.from_dict(d.get("tracking", {})),
            spectra=SpectraProcessing.from_dict(d.get("spectra", {})),
            parsing=ParsingProcessing.from_dict(d.get("parsing", {})),
        )


# --------------------------------------------------
# ExtractConfig — wall-normal profile extraction settings
# --------------------------------------------------
@dataclasses.dataclass(eq=False)
class ExtractConfig(_ConfigBase):
    """Settings for ``lst-tools extract`` (FE-quad profile extraction)."""

    # path to the Tecplot FE-quad input file (used when no CLI argument is given)
    input_file: str | None = None
    # output file paths (None means derive from the input file location)
    hdf5_out: str | None = None
    profiles_out: str | None = None
    wall_out: str | None = None
    # requested surface side for extraction (None means CLI/default behavior)
    surface: str | None = None
    # wall-normal points per extracted profile
    n_eta: int | None = None
    # wall-normal point distribution name
    eta_distribution: str | None = None
    # streamwise x-coordinates for profile extraction (None means use CLI default)
    stations: list[float] | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExtractConfig:
        """Build an ``ExtractConfig`` from a plain dict."""
        kw: dict[str, Any] = {}

        # validate input_file
        if "input_file" in d and d["input_file"] is not None:
            kw["input_file"] = str(d["input_file"])

        # validate optional output paths
        for key in ("hdf5_out", "profiles_out", "wall_out"):
            if key in d and d[key] is not None:
                kw[key] = str(d[key])

        # validate surface selection
        if "surface" in d and d["surface"] is not None:
            raw_surface = str(d["surface"]).strip().lower()
            if raw_surface:
                if raw_surface not in {"lower", "upper"}:
                    raise ValueError("extract.surface must be 'lower' or 'upper'")
                kw["surface"] = raw_surface

        # validate wall-normal point count
        if "n_eta" in d and d["n_eta"] not in (None, ""):
            raw_n_eta = int(d["n_eta"])
            if raw_n_eta < 2:
                raise ValueError("extract.n_eta must be at least 2")
            kw["n_eta"] = raw_n_eta

        # validate wall-normal point distribution
        if "eta_distribution" in d and d["eta_distribution"] is not None:
            raw_distribution = str(d["eta_distribution"]).strip().lower()
            if raw_distribution:
                if raw_distribution not in {"uniform", "cosine"}:
                    raise ValueError(
                        "extract.eta_distribution must be 'uniform' or 'cosine'"
                    )
                kw["eta_distribution"] = raw_distribution

        # validate stations list
        # treat empty string as unset — matches the pattern used by other optional fields
        if "stations" in d and d["stations"] is not None:
            raw_stations = d["stations"]
            if isinstance(raw_stations, str) and not raw_stations.strip():
                pass  # empty string → leave stations at its default
            elif not isinstance(raw_stations, (list, tuple)):
                raise ValueError("extract.stations must be a list of floats")
            else:
                kw["stations"] = [float(v) for v in raw_stations]

        return cls(**kw)


# --------------------------------------------------
# Assembled Config (includes all sections)
# --------------------------------------------------
@dataclasses.dataclass(eq=False)
# inherit from _ConfigBase to get .to_dict(), .to_toml_dict(), and custom __eq__
class Config(_ConfigBase):
    """Complete lst_tools configuration.

    Build from a raw dict::

        cfg = Config.from_dict(raw)

    Build directly from a TOML file::

        cfg = Config.from_toml("lst.cfg")

    Access fields as attributes::

        cfg.lst.solver.type
        cfg.flow_conditions.mach
    """

    # root config fields
    input_file: str = "base_flow.hdf5"
    lst_exe: str = "lst.x"
    # add nested dataclasses for each config section
    flow_conditions: FlowConditions = dataclasses.field(default_factory=FlowConditions)
    geometry: Geometry = dataclasses.field(default_factory=Geometry)
    meanflow_conversion: MeanflowConversion = dataclasses.field(default_factory=MeanflowConversion)
    lst: LstConfig = dataclasses.field(default_factory=LstConfig)
    hpc: HpcConfig = dataclasses.field(default_factory=HpcConfig)
    processing: Processing = dataclasses.field(default_factory=Processing)
    seed_table: SeedTable = dataclasses.field(default_factory=SeedTable)
    extract: ExtractConfig = dataclasses.field(default_factory=ExtractConfig)

    # validation method to check value constraints and raise ValueError if any violations are found
    def validate(self) -> Config:
        """Check value constraints; raise ``ValueError`` listing all violations."""
        errors: list[str] = []

        # meanflow_conversion
        mc = self.meanflow_conversion
        if mc.i_s < 0:
            errors.append("meanflow_conversion.i_s must be >= 0")
        if mc.i_e is not None and mc.i_e < 0:
            errors.append("meanflow_conversion.i_e must be >= 0 or None")
        if mc.d_i < 1:
            errors.append("meanflow_conversion.d_i must be >= 1")

        # geometry
        g = self.geometry
        if g.r_nose is not None and g.r_nose < 0:
            errors.append("geometry.r_nose must be >= 0")
        if g.theta_deg is not None and not (0 <= g.theta_deg <= 180):
            errors.append("geometry.theta_deg must be in [0, 180]")

        # lst.solver
        s = self.lst.solver
        if s.type < 0:
            errors.append("lst.solver.type must be >= 0")
        if s.spatial_temporal not in (0, 1):
            errors.append("lst.solver.spatial_temporal must be 0 or 1")

        # hpc
        h = self.hpc
        if h.nodes is not None and h.nodes <= 0:
            errors.append("hpc.nodes must be > 0")

        if errors:
            raise ValueError(
                "config validation errors:\n  - " + "\n  - ".join(errors)
            )
        return self

    # constructor from dict (parses nested sections and applies defaults)
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Config:
        """Build a validated ``Config`` from a plain nested dict.

        Each top-level key in *d* is dispatched to the corresponding
        section dataclass (e.g. ``d["flow_conditions"]`` is forwarded
        to ``FlowConditions.from_dict``).  Missing sections get their
        default values.  The resulting ``Config`` is validated before
        being returned.

        Args:
            d (dict[str, Any]): Raw config dictionary, typically the
                output of ``tomllib.loads()``.

        Returns:
            Config: Fully populated and validated configuration.

        Raises:
            ValueError: If validation finds constraint violations.
        """
        _input = d.get("input_file")
        _exe = d.get("lst_exe")
        cfg = cls(
            input_file=str(_input) if _input is not None else "base_flow.hdf5",
            lst_exe=str(_exe) if _exe is not None else "lst.x",
            flow_conditions=FlowConditions.from_dict(d.get("flow_conditions", {})),
            geometry=Geometry.from_dict(d.get("geometry", {})),
            meanflow_conversion=MeanflowConversion.from_dict(
                d.get("meanflow_conversion", {})
            ),
            lst=LstConfig.from_dict(d.get("lst", {})),
            hpc=HpcConfig.from_dict(d.get("hpc", {})),
            processing=Processing.from_dict(d.get("processing", {})),
            seed_table=SeedTable.from_dict(d.get("seed_table", {})),
            extract=ExtractConfig.from_dict(d.get("extract", {})),
        )
        return cfg.validate()

    # constructor from TOML file (reads file, normalizes booleans, parses TOML, then calls from_dict)
    @classmethod
    def from_toml(
        cls,
        path: str | Path,
    ) -> Config:
        """Parse a TOML file into a ``Config``.

        Args:
            path (str | Path): Path to a TOML configuration file.

        Returns:
            Config: Validated configuration object.
        """
        cfg_file = Path(path)

        logger.debug("loading %s", cfg_file)

        raw = cfg_file.read_text(encoding="utf-8")
        raw, n_fixed = _normalize_toml_bool(raw)
        if n_fixed:
            logger.warning(
                "auto-corrected %d Python-style boolean(s) in %s — use lowercase true/false in TOML",
                n_fixed,
                cfg_file,
            )

        data = _toml.loads(raw)
        cfg = cls.from_dict(data)

        logger.debug("loaded config from %s", cfg_file)

        return cfg
