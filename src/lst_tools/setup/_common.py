"""Shared helpers for setup modules (tracking, spectra, parsing)."""

from __future__ import annotations

import logging
import shutil
import stat
from pathlib import Path
from pprint import pformat
from typing import Any, Mapping

import numpy as np

from lst_tools.config import (
    check_consistency,
    find_config,
    format_report,
    read_config,
)
from lst_tools.data_io import LastracReader

logger = logging.getLogger(__name__)


# --------------------------------------------------
# resolve and validate configuration
# --------------------------------------------------
def resolve_config(cfg: Mapping[str, Any] | None) -> Any:
    """Load config from disk if not provided and run consistency checks.

    Parameters
    ----------
    cfg : Mapping or None
        Already-loaded config, or None to auto-detect from the working
        directory.

    Returns
    -------
    Config
        Validated configuration object.

    Raises
    ------
    FileNotFoundError
        If no config file can be found.
    ValueError
        If the consistency check finds errors.
    """
    if cfg is None:
        logger.info("configuration not provided, attempting to load...")
        cfg_file = find_config()
        if cfg_file is None:
            raise FileNotFoundError("no config file found in working directory")
        cfg = read_config(cfg_file)
        logger.debug(
            "no config file was explicitly provided"
            " -> use detected config file: %s",
            cfg_file,
        )

    logger.debug("validated config:")
    logger.debug(pformat(cfg))

    errors, warns = check_consistency(cfg)
    report = format_report(errors, warns)
    logger.info("consistency check report:")
    logger.info("")
    logger.info("%s", report)

    if errors:
        raise ValueError(
            f"found {len(errors)} consistency errors; fix and rerun\n\n{report}"
        )

    return cfg


# --------------------------------------------------
# read baseflow station x-coordinates
# --------------------------------------------------
def read_baseflow_stations(baseflow_input: str | Path) -> np.ndarray:
    """Read streamwise station locations from a LASTRAC meanflow binary.

    Parameters
    ----------
    baseflow_input : str or Path
        Path to the ``meanflow.bin`` file.

    Returns
    -------
    np.ndarray
        1-D array of station *s*-coordinates.
    """
    path = Path(str(baseflow_input))
    fio = LastracReader(path, endianness="<")

    header = fio.read_header()
    logger.debug("baseflow header:")
    for key, val in header.items():
        logger.debug("  %20s : %s", key, val)

    n_station = int(header["n_station"])
    x_baseflow = np.empty(n_station, dtype=float)

    for i in range(n_station):
        header = fio.read_station_header()
        x_baseflow[i] = float(header["s"])
        # skip the six station vectors (eta, u, v, w, temp, pres)
        fio.skip_records(6)

    fio.close()
    return x_baseflow


# --------------------------------------------------
# read baseflow profiles at a subset of stations
# --------------------------------------------------
def read_baseflow_profiles(
    baseflow_input: str | Path,
    n_samples: int = 50,
) -> dict:
    """Read eta, velocity, and temperature profiles at evenly spaced stations.

    The meanflow binary is sequential, so every station header is read,
    but profile vectors are only extracted at the sampled indices.
    All other stations have their six records skipped.

    Parameters
    ----------
    baseflow_input : str or Path
        Path to the ``meanflow.bin`` file.
    n_samples : int, optional
        Target number of stations to sample (default 50).
        If the file has fewer stations, all are read.

    Returns
    -------
    dict
        Keys: ``x`` (1-D array of sampled s-coordinates),
        ``eta`` (list of 1-D arrays), ``uvel`` (list of 1-D arrays),
        ``vvel`` (list of 1-D arrays), ``wvel`` (list of 1-D arrays),
        ``temp`` (list of 1-D arrays), ``stat_uvel`` (list of floats).
    """
    path = Path(str(baseflow_input))
    fio = LastracReader(path, endianness="<")

    file_header = fio.read_header()
    n_station = int(file_header["n_station"])

    # choose evenly spaced indices
    if n_station <= n_samples:
        sample_idx = set(range(n_station))
    else:
        sample_idx = set(
            int(round(i))
            for i in np.linspace(0, n_station - 1, n_samples)
        )

    x_list: list[float] = []
    eta_list: list[np.ndarray] = []
    uvel_list: list[np.ndarray] = []
    vvel_list: list[np.ndarray] = []
    wvel_list: list[np.ndarray] = []
    temp_list: list[np.ndarray] = []
    stat_uvel_list: list[float] = []

    for i in range(n_station):
        stn = fio.read_station_header()

        if i not in sample_idx:
            fio.skip_records(6)
            continue

        x_list.append(float(stn["s"]))
        stat_uvel_list.append(float(stn["stat_uvel"]))

        eta = fio.read_station_vector()   # eta
        uvel = fio.read_station_vector()  # u
        vvel = fio.read_station_vector()  # v
        wvel = fio.read_station_vector()  # w
        temp = fio.read_station_vector()  # temp
        fio.skip_records(1)               # pres (skip)

        eta_list.append(eta)
        uvel_list.append(uvel)
        vvel_list.append(vvel)
        wvel_list.append(wvel)
        temp_list.append(temp)

    fio.close()

    return {
        "x": np.array(x_list),
        "eta": eta_list,
        "uvel": uvel_list,
        "vvel": vvel_list,
        "wvel": wvel_list,
        "temp": temp_list,
        "stat_uvel": stat_uvel_list,
    }


# --------------------------------------------------
# scaffold a case directory (meanflow + executable)
# --------------------------------------------------
def scaffold_case_dir(
    case_dir: str | Path,
    meanflow_src: str | Path,
    lst_exe: str | None,
) -> None:
    """Create a case directory and populate it with meanflow and executable.

    Parameters
    ----------
    case_dir : str or Path
        Target directory (created if it does not exist).
    meanflow_src : str or Path
        Path to the meanflow binary to copy.
    lst_exe : str or None
        Path to the LST executable, or None to skip copying.
    """
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    # copy meanflow
    meanflow_src = Path(meanflow_src)
    shutil.copy2(meanflow_src, case_dir / meanflow_src.name)
    logger.debug("copied %s to %s", meanflow_src.name, case_dir)

    # copy LST executable (if configured)
    if lst_exe:
        lst_exe_src = Path(lst_exe)
        if lst_exe_src.exists() and lst_exe_src.is_file():
            lst_exe_dst = case_dir / lst_exe_src.name
            shutil.copy2(lst_exe_src, lst_exe_dst)
            mode = lst_exe_dst.stat().st_mode
            lst_exe_dst.chmod(
                mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
            )
            logger.info("copied lst executable to %s", lst_exe_dst)
        else:
            logger.warning(
                "lst executable path not found or not a file -> '%s'",
                lst_exe_src,
            )
    else:
        logger.info(
            "no lst executable specified; skipping copy"
        )


# --------------------------------------------------
# write a launcher script to submit / run all cases
# --------------------------------------------------
def write_launcher_script(
    dirs: list[str],
    *,
    script_name: str = "run_jobs.sh",
    submit_cmd: str | None = None,
    fname_run_script: str | None = None,
) -> Path:
    """Write an executable bash script that submits or runs each case.

    Parameters
    ----------
    dirs : list[str]
        Case directory names.
    script_name : str
        Output file name.
    submit_cmd : str or None
        HPC submit command (``qsub``, ``sbatch``). If None a local
        direct-execution line is emitted instead.
    fname_run_script : str or None
        Name of the HPC run script inside each directory.  Ignored when
        *submit_cmd* is None.

    Returns
    -------
    Path
        Path to the written launcher script.
    """
    out = Path(script_name)

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
    ]

    for d in dirs:
        lines.append(f'echo "Running {d}..."')
        if submit_cmd is None:
            # local execution
            lines.append(
                f"( cd {d} && ./lst.x lst_input.dat >run.log 2>&1 )"
            )
        elif fname_run_script:
            lines.append(f"( cd {d} && {submit_cmd} {fname_run_script} )")
        else:
            lines.append(f"( cd {d} && {submit_cmd} run.*.* )")
        lines.append("")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    mode = out.stat().st_mode
    out.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    logger.debug("wrote executable launcher script: %s", out)
    return out
