"""
Post-processing functionality for LST spectra calculations

This module provides functions to process the output data from LST spectra
calculations, including spectral analysis, frequency-wavenumber data collection,
and visualization preparation.
"""

# --------------------------------------------------
# load necessary libraries
# --------------------------------------------------

from __future__ import annotations
from pathlib import Path
from typing import Mapping, Any
import logging
import re
from lst_tools.data_io import read_tecplot_ascii

logger = logging.getLogger(__name__)


# --------------------------------------------------
# main function to process spectra results
# --------------------------------------------------

def spectra_process(
    *,
    cfg: Mapping[str, Any] | None = None,
) -> Path:
    """
    Process LST spectra calculation results

    This function processes the output data from LST spectra calculations,
    collecting results from multiple x-location directories, analyzing
    spectral data across frequencies and wavenumbers, and preparing
    data for visualization.

    Parameters
    ----------
    cfg : Optional[Mapping[str, Any]]
        Configuration dictionary or path to config file

    Returns
    -------
    Path
        Path to the processed results file
    """

    # --------------------------------------------------
    # output for user
    # --------------------------------------------------

    logger.info("processing spectra...")

    # --------------------------------------------------
    # Step 1: Find and parse case directories
    # --------------------------------------------------

    # output for user
    logger.info("scanning for case directories...")

    # pattern to match: x_00pt10_m_f_0006pt00_khz_beta_pos0500pt00
    case_pattern = re.compile(
        r"^x_(\d+pt\d+)_m_f_(\d+pt\d+)_khz_beta_(pos|neg)(\d+pt\d+)$"
    )

    # find all matching directories in current path
    current_dir = Path(".")

    # generate empty list for case directories
    case_dirs = []

    # generate empty list for case info
    case_info = []

    # loop over items in current directory iteratively
    for item in current_dir.iterdir():
        # check if item is a directory
        if item.is_dir():
            # check if directory name matches case pattern defined above
            match = case_pattern.match(item.name)

            # if there is a match collect all relevant info and data
            if match:
                # extract matched groups
                x_str, f_str, sign, beta_str = match.groups()

                # get x-location float
                x_val = float(x_str.replace("pt", "."))

                # get frequency (stored in kHz) as float
                f_val = float(f_str.replace("pt", ".")) * 1000

                # get wavenumber as float
                beta_val = float(beta_str.replace("pt", "."))

                # apply sign to beta
                if sign == "neg":
                    beta_val = -beta_val

                # add to case lists
                case_dirs.append(item)

                # store case info
                case_info.append(
                    {
                        "path": item,
                        "name": item.name,
                        "x": x_val,
                        "freq": f_val,
                        "beta": beta_val,
                    }
                )

    # output error if no case directories found
    if not case_dirs:

        logger.error("no case directories found matching pattern")
        logger.error(
            "expected pattern: x_##pt##_m_f_####pt##_khz_beta_pos####pt##"
        )
        logger.error("abort processing")
        return Path(".")

    # output for user
    logger.info("found %d case directories", len(case_dirs))

    # --------------------------------------------------
    # Step 2: Analyze parameter space for animation planning
    # --------------------------------------------------

    # collect unique values for each parameter (x, freq, beta)
    x_values = sorted(list(set(case["x"] for case in case_info)))
    freq_values = sorted(list(set(case["freq"] for case in case_info)))
    beta_values = sorted(list(set(case["beta"] for case in case_info)))

    # output for user
    logger.info("spectra parameter space:")
    logger.info(
        "  x-locations: %d values [%.3f ... %.3f]",
        len(x_values), x_values[0], x_values[-1],
    )
    logger.info(
        "  frequencies: %d values [%.0f ... %.0f Hz]",
        len(freq_values), freq_values[0], freq_values[-1],
    )
    logger.info(
        "  wavenumbers: %d values [%.2f ... %.2f]",
        len(beta_values), beta_values[0], beta_values[-1],
    )

    # determine animation variable
    if len(x_values) > 1:
        logger.info("-> animation over x-locations (%d frames)", len(x_values))
    elif len(freq_values) > 1:
        logger.info("-> animation over frequencies (%d frames)", len(freq_values))
    elif len(beta_values) > 1:
        logger.info("-> animation over wavenumbers (%d frames)", len(beta_values))
    else:
        logger.info("-> single case, no animation needed")

    # --------------------------------------------------
    # Step 3: load spectra data from each case
    # --------------------------------------------------

    loaded_cases = 0
    missing_cases: list[str] = []

    for case in case_info:
        # look for Eigenvalues_* files in case directory
        eigenvalue_files = list(case["path"].glob("Eigenvalues_*"))

        # take the first one if multiple found set to None if not found
        spectra_file = eigenvalue_files[0] if eigenvalue_files else None

        # if no spectra file found, log as missing case and continue to next case
        if spectra_file is None:
            missing_cases.append(case["name"])
            continue

        # load the data
        tecplot_data = read_tecplot_ascii(spectra_file)

        # add spectra_file to case
        case["spectra_file"] = spectra_file

        # add spectra data to case
        case["spectra_data"] = tecplot_data

        # increase loaded cases count
        loaded_cases += 1

    # output for user
    logger.info("loaded spectra data for %d/%d cases", loaded_cases, len(case_info))

    # report missing cases
    if missing_cases:
        preview = ", ".join(missing_cases[:5])

        if len(missing_cases) > 5:
            preview += ", ..."

        logger.warning("missing spectra files for cases: %s", preview)

    # --------------------------------------------------
    # Step 4: try to detect discrete modes
    # --------------------------------------------------

    # --------------------------------------------------
    # Step 5: tecplot file for animation
    # --------------------------------------------------


    # output for user
    logger.info("spectra processing complete (basic directory scan)")

    raise NotImplementedError(
        "spectra_process is not yet fully implemented (steps 4/5 missing)"
    )
