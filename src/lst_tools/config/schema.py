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
    """Convert *v* to str, or return ``None`` if missing."""
    if v is None:
        return None
    return str(v)


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
    is_simplified: bool = False
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
            is_simplified=_coerce_bool(_is) if _is is not None else False,
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
        if s.type is not None and s.type < 0:
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
