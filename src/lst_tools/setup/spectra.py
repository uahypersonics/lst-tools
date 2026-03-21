"""Set up spectra (local LST) calculations.

Reads the configuration, splits the domain into per-station
subdirectories, generates input decks, and writes HPC job scripts.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from lst_tools.convert import generate_lst_input_deck
from lst_tools.hpc import detect, hpc_configure, script_build

from ._common import (
    read_baseflow_stations,
    resolve_config,
    scaffold_case_dir,
    write_launcher_script,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# private helpers
# --------------------------------------------------

def _resolve_x_locations(cfg) -> np.ndarray:
    """Read baseflow stations and return the x-locations selected by the config.

    Parameters
    ----------
    cfg : Config
        Validated configuration object.

    Returns
    -------
    np.ndarray
        Selected streamwise locations.

    Raises
    ------
    FileNotFoundError
        If the baseflow file does not exist.
    """
    baseflow_input = cfg.lst.io.baseflow_input
    if baseflow_input is not None:
        baseflow_input = Path(str(baseflow_input))

    if baseflow_input is None or not Path(baseflow_input).is_file():
        raise FileNotFoundError(
            "baseflow file %s not found; "
            "cannot set up spectra without baseflow data" % baseflow_input
        )

    x_baseflow = read_baseflow_stations(baseflow_input)
    lst_params = cfg.lst.params

    x_s = lst_params.x_s
    x_e = lst_params.x_e
    i_step = lst_params.i_step or 1

    if x_s is None or x_e is None:
        logger.warning("x_s or x_e not specified in config")
        logger.info(
            "using full x-range from baseflow: x_s=%s, x_e=%s",
            f"{x_baseflow[0]:.6f}", f"{x_baseflow[-1]:.6f}"
        )
        x_s = x_baseflow[0]
        x_e = x_baseflow[-1]

    idx_s = int(np.argmin(np.abs(x_baseflow - x_s)))
    idx_e = int(np.argmin(np.abs(x_baseflow - x_e)))
    x_locations = x_baseflow[idx_s : idx_e + 1 : i_step]

    logger.info("read %d stations from %s", len(x_baseflow), baseflow_input)
    logger.info(
        "x_s=%s (idx=%d), x_e=%s (idx=%d), i_step=%d",
        f"{x_s:.6f}", idx_s, f"{x_e:.6f}", idx_e, i_step
    )
    logger.info(
        "selected %d x-locations: [%s ... %s]",
        len(x_locations), f"{x_locations[0]:.3f}", f"{x_locations[-1]:.3f}"
    )

    return x_locations


def _resolve_frequencies(cfg) -> np.ndarray:
    """Build the frequency array from config ``[lst.params]``.

    Returns
    -------
    np.ndarray
        Frequency values in Hz.

    Raises
    ------
    ValueError
        If ``f_min``, ``f_max``, or ``d_f`` is not specified.
    """
    lst_params = cfg.lst.params
    if lst_params.f_min is None or lst_params.f_max is None or lst_params.d_f is None:
        raise ValueError(
            "frequency range (f_min, f_max, d_f) not fully specified in config [lst.params]; "
            "update configuration file and rerun setup"
        )

    f_min = lst_params.f_min
    f_max = lst_params.f_max
    d_f = lst_params.d_f
    frequencies = np.arange(f_min, f_max + d_f, d_f)

    logger.debug(
        "frequencies from config [lst.params]: f_min=%s, f_max=%s, d_f=%s",
        f_min, f_max, d_f
    )
    logger.debug(
        "generated %d frequencies: [%s ... %s]",
        len(frequencies), frequencies[0], frequencies[-1]
    )

    return frequencies


def _resolve_wavenumbers(cfg) -> np.ndarray:
    """Build the wavenumber array from config ``[lst.params]``.

    Returns
    -------
    np.ndarray
        Spanwise wavenumber values.

    Raises
    ------
    ValueError
        If ``beta_s``, ``beta_e``, or ``d_beta`` is not specified.
    """
    lst_params = cfg.lst.params
    if lst_params.beta_s is None or lst_params.beta_e is None or lst_params.d_beta is None:
        raise ValueError(
            "wavenumber range (beta_s, beta_e, d_beta) not fully specified in config [lst.params]; "
            "update configuration file and rerun setup"
        )

    beta_s = lst_params.beta_s
    beta_e = lst_params.beta_e
    d_beta = lst_params.d_beta
    wavenumbers = np.arange(beta_s, beta_e + d_beta, d_beta)

    logger.debug(
        "wavenumbers from config [lst.params]: beta_s=%s, beta_e=%s, d_beta=%s",
        beta_s, beta_e, d_beta
    )
    logger.debug(
        "generated %d wavenumbers: [%s ... %s]",
        len(wavenumbers), wavenumbers[0], wavenumbers[-1]
    )

    return wavenumbers


def _build_case_name(x_loc: float, freq: float, beta: float) -> str:
    """Format a spectra case directory name from physical parameters."""
    x_str = f"{x_loc:05.2f}".replace(".", "pt")
    f_str = f"{freq / 1000:007.2f}".replace(".", "pt")

    if beta >= 0:
        b_str = f"pos{abs(beta):007.2f}".replace(".", "pt")
    else:
        b_str = f"neg{abs(beta):007.2f}".replace(".", "pt")

    return f"x_{x_str}_m_f_{f_str}_khz_beta_{b_str}"


def _setup_single_case(
    cfg,
    cfg_spectra,
    case_dir: Path,
    input_path: Path,
    baseflow_input: Path,
) -> Path:
    """Scaffold one spectra case directory and generate its input deck.

    Returns the path to the written input deck.
    """
    lst_exe = cfg.lst_exe
    scaffold_case_dir(case_dir, baseflow_input, lst_exe)

    written_input_path = generate_lst_input_deck(
        cfg=cfg_spectra, out_path=input_path
    )

    hpc_cfg = hpc_configure(cfg, set_defaults=False)

    if lst_exe is None:
        lst_exe = "lst.x"

    if hpc_cfg.scheduler.lower() != "unknown":
        logger.info("generating hpc script for scheduler")
        script_build(
            hpc_cfg,
            case_dir,
            lst_exe=lst_exe,
            args=["lst_input.dat", ">run.log"],
            extra_env=cfg.hpc.extra_env,
        )

    return written_input_path


# --------------------------------------------------
# main function to set up spectra calculations
# --------------------------------------------------

def spectra_setup(
    *,
    cfg: Mapping[str, Any] | None = None,
) -> list[Path]:
    """Set up spectral analysis calculations for LST solver.

    Creates LST input decks configured for spectral analysis at streamwise
    locations specified in the configuration file with frequency and wavenumber
    ranges from ``[lst.params]``.

    Parameters
    ----------
    cfg : Mapping[str, Any], optional
        Configuration dictionary. If None, will search for and load lst.cfg.

    Returns
    -------
    list[Path]
        Paths to the written input decks.
    """
    cfg = resolve_config(cfg)
    logger.info("configuration loaded and validated")

    baseflow_input = cfg.lst.io.baseflow_input
    if baseflow_input is not None:
        baseflow_input = Path(str(baseflow_input))

    x_locations = _resolve_x_locations(cfg)
    frequencies = _resolve_frequencies(cfg)
    wavenumbers = _resolve_wavenumbers(cfg)

    out_dir = Path(".")
    logger.debug("output directory: %s", out_dir.absolute())

    cfg_spectra = copy.deepcopy(cfg)
    cfg_spectra.lst.solver.type = 2
    cfg_spectra.lst.solver.is_simplified = False

    # --------------------------------------------------
    # generate input files for each (x, frequency, wavenumber) combination
    # --------------------------------------------------

    generated_files: list[Path] = []

    for x_loc in x_locations:
        for freq in frequencies:
            for beta in wavenumbers:
                cfg_spectra.lst.params.x_s = float(x_loc)
                cfg_spectra.lst.params.x_e = float(x_loc)
                cfg_spectra.lst.params.i_step = 1

                cfg_spectra.lst.params.f_min = float(freq)
                cfg_spectra.lst.params.f_max = float(freq)
                cfg_spectra.lst.params.d_f = 0.0
                cfg_spectra.lst.params.f_init = float(freq)

                cfg_spectra.lst.params.beta_s = float(beta)
                cfg_spectra.lst.params.beta_e = float(beta)
                cfg_spectra.lst.params.d_beta = 0.0
                cfg_spectra.lst.params.beta_init = float(beta)

                case_name = _build_case_name(x_loc, freq, beta)
                case_dir = out_dir / case_name
                case_dir.mkdir(parents=True, exist_ok=True)

                input_path = case_dir / "lst_input.dat"

                logger.info("")
                logger.info("--------------------------------------------------")
                logger.info(
                    "generating case %d: %s", len(generated_files) + 1, case_name
                )
                logger.info("x_location: %s", x_loc)
                logger.info("frequency: %s", freq)
                logger.info("wavenumber: %s", beta)

                try:
                    written = _setup_single_case(
                        cfg, cfg_spectra, case_dir, input_path, baseflow_input
                    )
                    generated_files.append(written)
                except (OSError, ValueError, KeyError) as e:
                    logger.error("failed to set up %s: %s", case_name, e)
                    continue

    # --------------------------------------------------
    # launcher script
    # --------------------------------------------------

    if generated_files:
        case_dirs = [str(f.parent.name) for f in generated_files]
        env = detect()

        if env.scheduler.value == "pbs":
            submit_cmd = "qsub"
        elif env.scheduler.value == "slurm":
            submit_cmd = "sbatch"
        else:
            submit_cmd = None

        script_path = write_launcher_script(
            case_dirs,
            script_name="run_cases.sh",
            submit_cmd=submit_cmd,
        )
        logger.info("wrote launcher script %s", script_path)

    # --------------------------------------------------
    # summary
    # --------------------------------------------------

    total_cases = len(x_locations) * len(frequencies) * len(wavenumbers)

    logger.info("spectra computation setup complete")
    logger.info("generated %d cases", len(generated_files))
    logger.info("total combinations specified by user: %d", total_cases)

    if len(generated_files) != total_cases:
        logger.warning(
            "some cases failed: %d errors", total_cases - len(generated_files)
        )

    return generated_files
