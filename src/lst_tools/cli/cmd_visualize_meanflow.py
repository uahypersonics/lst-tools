"""lst-tools visualize meanflow — plot profiles from a LASTRAC meanflow binary."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from lst_tools.setup._common import read_baseflow_profiles


# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# main CLI command
# --------------------------------------------------
def cmd_visualize_meanflow(
    fpath: Annotated[
        Path,
        typer.Argument(help="Path to a meanflow.bin file."),
    ] = Path("meanflow.bin"),
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out",
            "-o",
            help="Output directory for rendered plots.",
        ),
    ] = Path("meanflow_profiles"),
    dpi: Annotated[
        int,
        typer.Option("--dpi", min=72, help="PNG output DPI."),
    ] = 200,
) -> None:
    """Plot boundary-layer profiles from a LASTRAC meanflow binary."""

    # validate that file exists
    if not fpath.is_file():
        typer.echo(f"error: {fpath} not found", err=True)
        raise typer.Exit(1)

    try:
        _visualize_meanflow(fpath, out_dir, dpi)
    except Exception as exc:
        typer.echo(f"error: {exc}", err=True)
        # debug output for devs
        logger.debug("visualize meanflow failed", exc_info=True)
        raise typer.Exit(1)


# --------------------------------------------------
# plotting implementation
# --------------------------------------------------
def _visualize_meanflow(
    fpath: Path,
    out_dir: Path,
    dpi: int,
) -> None:
    """Read meanflow binary and produce shifted profile plots.

    Each profile is plotted at its streamwise station location,
    shifted horizontally so the boundary layer growth is visible.
    One figure per variable: u/u_e, w/w_e (if crossflow), T/T_e.
    """

    # import plotting dependency only when this command is used
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for 'lst-tools visualize meanflow'"
        ) from exc

    # read profiles from the meanflow binary (equidistant in x)
    n_profiles = 10
    profiles = read_baseflow_profiles(fpath, n_samples=n_profiles, spacing="x")

    # extract arrays
    x_vals = profiles["x"]
    eta_list = profiles["eta"]
    uvel_list = profiles["uvel"]
    wvel_list = profiles["wvel"]
    temp_list = profiles["temp"]

    n_stations = len(x_vals)

    # check if crossflow is present
    has_crossflow = any(np.any(w != 0.0) for w in wvel_list)

    # compute median gap between consecutive stations for profile scaling
    gaps = np.diff(x_vals)
    median_gap = float(np.median(gaps))

    # set absolute gradient threshold used for BL edge detection
    gradient_threshold = 0.1

    # compute gradients at the last station
    eta_local = eta_list[-1]
    uvel_local = uvel_list[-1]
    temp_local = temp_list[-1]

    duvel_deta = np.gradient(uvel_local, eta_local)
    dtemp_deta = np.gradient(temp_local, eta_local)

    # find the first eta where the gradient drops below the threshold
    # after the active gradient region has started
    def _find_first_below_threshold_eta(
        eta_values: np.ndarray,
        gradient_values: np.ndarray,
    ) -> float:
        active_mask = np.abs(gradient_values) > gradient_threshold
        if not active_mask.any():
            return float(eta_values[-1])

        first_active = int(np.argmax(active_mask))
        inactive_mask = np.abs(gradient_values[first_active:]) <= gradient_threshold
        if inactive_mask.any():
            first_inactive = int(np.argmax(inactive_mask))
            return float(eta_values[first_active + first_inactive])

        return float(eta_values[-1])

    # find the first eta where the u-gradient is no longer active
    delta_uvel = _find_first_below_threshold_eta(eta_local, duvel_deta)

    # find the first eta where the T-gradient is no longer active
    delta_temp = _find_first_below_threshold_eta(eta_local, dtemp_deta)

    if has_crossflow:
        wvel_local = wvel_list[-1]
        dwvel_deta = np.gradient(wvel_local, eta_local)

        # find the first eta where the w-gradient is no longer active
        delta_wvel = _find_first_below_threshold_eta(eta_local, dwvel_deta)
    else:
        delta_wvel = None

    # clip eta using the largest detected BL thickness
    delta_candidates = [delta_uvel, delta_temp]
    if delta_wvel is not None:
        delta_candidates.append(delta_wvel)
    delta_max = max(delta_candidates)
    eta_clip = 1.5 * delta_max

    # report detected BL thicknesses used for plotting
    delta_wvel_str = f"{delta_wvel:.6f}" if delta_wvel is not None else "n/a"
    typer.echo(
        "gradient-based deltas: "
        f"u={delta_uvel:.6f}, "
        f"w={delta_wvel_str}, "
        f"T={delta_temp:.6f}, "
        f"eta_clip={eta_clip:.6f}"
    )

    # create output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # build list of variables to plot
    variables = [
        ("uvel", uvel_list, r"$u / u_e$", "meanflow_uvel.png"),
        ("temp", temp_list, r"$T / T_e$", "meanflow_temp.png"),
    ]
    if has_crossflow:
        variables.insert(1, ("wvel", wvel_list, r"$w / w_e$", "meanflow_wvel.png"))

    # plot each variable as a separate figure
    for var_name, var_list, ylabel, fname in variables:
        fig, ax = plt.subplots(figsize=(8.5, 2))

        # scale factor: map the max profile value to 2x the median station gap
        # (allow some overlap at closely-spaced stations for visibility)
        var_max = max(np.max(np.abs(v)) for v in var_list)
        scale = 2.0 * median_gap / var_max if var_max > 0 else median_gap

        for i in range(n_stations):
            # shift profile horizontally to its station location
            x_shift = x_vals[i]
            profile = var_list[i]
            eta = eta_list[i]

            # scale profile so widest fills ~2x median_gap width
            x_plot = x_shift + profile * scale

            ax.plot(x_plot, eta, "b-", linewidth=0.8)

            # draw vertical baseline at station location
            ax.axvline(x_shift, color="0.7", linewidth=0.3, zorder=0)

        ax.set_xlabel(r"$x$ [m]")
        ax.set_ylabel(r"$\eta$")
        ax.set_ylim(0, eta_clip)
        ax.set_title(f"{ylabel} — {fpath.name}")
        fig.tight_layout()

        # save figure
        out_file = out_dir / fname
        fig.savefig(out_file, dpi=dpi)
        plt.close(fig)

        typer.echo(f"wrote {out_file}")
