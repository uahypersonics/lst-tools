"""HDF5 and Tecplot output writers for sampled profiles."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from pathlib import Path
import logging

import h5py
import numpy as np

from ._types import SampledProfiles


# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# HDF5 writer
# --------------------------------------------------
def write_profiles_hdf5(
    path: str | Path,
    profiles: SampledProfiles,
    attrs: dict[str, float],
) -> Path:
    """Write the extracted profiles to HDF5.

    All datasets have shape (N_ETA, n_stations) — rows are wall-normal points,
    columns are streamwise stations.

    Args:
        path: Output HDF5 file path.
        profiles: Sampled wall-normal profile data.
        attrs: Freestream attribute dictionary for root-level metadata.

    Returns:
        Resolved output path.
    """

    # convert to Path object
    out_path = Path(path)

    # write HDF5 file
    with h5py.File(out_path, "w") as hdf5_file:
        # write all root-level freestream attributes as scalar float64
        for attr_name, attr_value in attrs.items():
            hdf5_file.attrs[attr_name] = np.float64(attr_value)

        # write all datasets as 2D arrays with shape (N_ETA, n_stations)
        # the .T transposes from (n_stations, n_eta) to (n_eta, n_stations)
        hdf5_file.create_dataset("x",    data=profiles.sample_x.T, dtype=np.float64)
        hdf5_file.create_dataset("y",    data=profiles.sample_y.T, dtype=np.float64)
        hdf5_file.create_dataset("uvel", data=profiles.uvel.T,     dtype=np.float64)
        hdf5_file.create_dataset("vvel", data=profiles.vvel.T,     dtype=np.float64)
        hdf5_file.create_dataset("wvel", data=profiles.wvel.T,     dtype=np.float64)
        hdf5_file.create_dataset("temp", data=profiles.temp.T,     dtype=np.float64)
        hdf5_file.create_dataset("pres", data=profiles.pres.T,     dtype=np.float64)
        hdf5_file.create_dataset("dens", data=profiles.rho.T,      dtype=np.float64)

    return out_path


# --------------------------------------------------
# Tecplot writers
# --------------------------------------------------
def write_profiles_tecplot(
    path: str | Path,
    profiles: SampledProfiles,
) -> Path:
    """Write the extracted wall-normal profiles as a multi-zone Tecplot file.

    Args:
        path: Output Tecplot ASCII file path.
        profiles: Sampled wall-normal profile data.

    Returns:
        Resolved output path.
    """

    # convert to Path object
    out_path = Path(path)

    # write one ordered 1-D zone per station
    with out_path.open("w", encoding="utf-8") as stream:
        stream.write('TITLE = "extracted_profiles"\n')
        stream.write('VARIABLES = "x" "y" "eta" "u" "v" "w" "T" "p" "rho"\n')

        for station_index in range(profiles.station_x.size):
            station_x = profiles.station_x[station_index]
            station_y = profiles.station_y[station_index]
            zone_name = (
                f"extracted_profile_{station_index:03d} "
                f"x={station_x:.6e} "
                f"y={station_y:.6e}"
            )
            stream.write(
                f'ZONE T="{zone_name}", I={profiles.eta.size}, DATAPACKING=POINT\n'
            )

            for eta_index in range(profiles.eta.size):
                stream.write(
                    f"{profiles.sample_x[station_index, eta_index]:.8e} "
                    f"{profiles.sample_y[station_index, eta_index]:.8e} "
                    f"{profiles.eta[eta_index]:.8e} "
                    f"{profiles.uvel[station_index, eta_index]:.8e} "
                    f"{profiles.vvel[station_index, eta_index]:.8e} "
                    f"{profiles.wvel[station_index, eta_index]:.8e} "
                    f"{profiles.temp[station_index, eta_index]:.8e} "
                    f"{profiles.pres[station_index, eta_index]:.8e} "
                    f"{profiles.rho[station_index, eta_index]:.8e}\n"
                )

    return out_path


def write_wall_profile_tecplot(
    path: str | Path,
    wall_x: np.ndarray,
    wall_y: np.ndarray,
) -> Path:
    """Write the extracted wall curve as a 1-D Tecplot line zone.

    Args:
        path: Output Tecplot ASCII file path.
        wall_x: Wall x-coordinates.
        wall_y: Wall y-coordinates.

    Returns:
        Resolved output path.
    """

    # convert to Path object
    out_path = Path(path)

    # write a simple ordered line zone
    with out_path.open("w", encoding="utf-8") as stream:
        stream.write('TITLE = "wall_profile"\n')
        stream.write('VARIABLES = "x" "y"\n')
        stream.write(f'ZONE T="wall", I={wall_x.size}, DATAPACKING=POINT\n')

        for x_value, y_value in zip(wall_x, wall_y):
            stream.write(f"{x_value:.8e} {y_value:.8e}\n")

    return out_path
