import datetime as dt
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from icenet.data.sic.mask import Masks
from icenet.plotting.utils import get_obs_da
from icenet.process.predict import get_refcube, get_refsic

from icenet_sipn_south.cli import diagnostic_args

from ..process.icenet import IceNetForecastLoader
from .sea_ice_area import SeaIceArea


class SIPNSouthOutputs(
    IceNetForecastLoader,
    SeaIceArea,
):
    """SIPN Sea Ice Outlook Submission for IceNet (with daily averaging)"""

    def __init__(
        self,
        prediction_pipeline_path: str,
        prediction_name: str,
        forecast_init_date: dt.date,
        forecast_leadtime: int = None,
        hemisphere: str = "south",
        get_obs: bool = False,
        plot: bool = False,
        group_name: str = "BAS",
    ) -> None:
        r"""
        Initialise SIPNSouthOutputs instance for IceNet (with daily averaging).

        This class extends `IceNetForecastLoader` and `SeaIceArea` to provide a comprehensive
        set of functionalities for processing, analysing, and plotting Sea Ice Prediction Network
        (SIPN) outputs from the IceNet model. It is designed only for use against Antarctic
        (`hemisphere="south"`) forecasts.

        Args:
            prediction_pipeline_path: Path to the `icenet-pipeline` instance with predictions.
            prediction_name: Name of the prediction output file (without extension).
            forecast_init_date: Date when the forecast was initialised (format: YYYY-MM-DD).
            forecast_leadtime: Forecast lead time in days. If not provided, it will be determined
                from the maximum lead time available in the input data.
            hemisphere (optional): Hemisphere for which the forecast is generated ("north" or "south").
                                   This class is meant for use with Southern hemisphere.
                                   Defaults to "south".
            get_obs (optional): Whether to retrieve and include observational data from OSI-SAF.
                                Default is False.
            plot (optional): Whether to generate plots of Sea Ice Area (SIA) and Sea Ice Prediction Network
                             (SIPN) results.
                             Default is False.
            group_name (optional): Name of the submission group for SIPN.
                                   Default is "BAS".

        Raises:
            Exception: If an invalid hemisphere is specified.
        """
        self.north = False
        self.south = False
        self.hemisphere = hemisphere
        self.plot = plot
        self.group_name = group_name
        if hemisphere.casefold() == "north":
            self.north = True
        elif hemisphere.casefold() == "south":
            self.south = True
        else:
            raise Exception(
                "Incorrect hemisphere specified, should be `north` or `south`"
            )

        self.forecast_init_date = forecast_init_date
        self.forecast_init_date_dt = dt.date(
            *list(map(int, self.forecast_init_date.split("-")))
        )

        self.forecast_start_date = pd.to_datetime(forecast_init_date) + dt.timedelta(
            days=1
        )

        if forecast_leadtime:
            self.forecast_leadtime = forecast_leadtime
        else:
            self.forecast_leadtime = None

        super().__init__(prediction_pipeline_path, prediction_name)
        self.create_ensemble_dataset()

        if forecast_leadtime:
            self.forecast_end_date = pd.to_datetime(forecast_init_date) + dt.timedelta(
                days=int(forecast_leadtime)
            )
        else:
            self.forecast_end_date = pd.to_datetime(forecast_init_date) + dt.timedelta(
                days=int(self.xarr.leadtime.values.max())
            )

        if get_obs:
            try:
                # Get Observational data from osisaf, if it exists.
                sic = get_obs_da(
                    self.hemisphere,
                    self.forecast_start_date,
                    self.forecast_end_date,
                )
                sic = sic.rename({"time": "leadtime"})
                # Rearrange to have leadtime as last dimension like icenet.
                sic = sic.transpose("yc", "xc", "leadtime")
                # Add extra dimension to emulate having forecast time in icenet.
                self.obs = sic.expand_dims(new_dim=[0])
            except (KeyError, OSError) as e:
                logging.error(
                    f"{e}\nObservational data not available for given forecast init date"
                )

    def check_forecast_period(self):
        """Used to check that the forecast period is within June-November."""
        raise NotImplementedError

    def diagnostic_1(self, method: str = "mean", output_dir: str | None = None):
        r"""
        Process Diagnostic 1: Total Sea Ice Area.

        This method processes the total sea ice area data for Diagnostic 1 of the Sea Ice
        Prediction Network (SIPN) submission. It computes the daily sea ice area using either
        the mean, ensemble members, or observed data based on the specified `method`. If plotting
        is enabled (`plot=True` in `__init__`), it generates a time series plot of Sea Ice
        Area (SIA). The processed data is then saved as CSV files for each ensemble member or the
        mean value, along with observational data if available.

        Args:
            method: Method used to compute sea ice area. Can be "mean" (default) for the mean SIA,
                "ensemble" for individual ensemble members' SIA, or "observation" for observed SIA.
            output_dir (optional): Directory path where the CSV files for Diagnostic 1 will be saved.

        Raises:
            NotImplementedError: If an invalid `method` is provided.

        Notes:
          - The method "ensemble" saves a separate CSV file for each ensemble member, while
            "mean" and "observation" save a single CSV file with the respective data.
          - Observational data is included in the plot and saved if available (`get_obs=True` in
            `__init__`).
        """
        print("Processing Diagnostic 1")
        self.xarr = self.xarr_
        self.compute_daily_sea_ice_area(method=method)
        if hasattr(self, "obs"):
            self.compute_daily_sea_ice_area(method="observation")
        if self.plot:
            self.plot_sia()

        if method == "ensemble":
            sia = self.xarr.sea_ice_area_daily
            ens_members = sia.sizes["ensemble"]
            for ensemble in range(ens_members):
                self.save_diagnostic_1(
                    sia.sel(ensemble=ensemble),
                    column_name="sea_ice_area_daily",
                    descr="totalarea",
                    forecast_id=str(ensemble + 1),
                    output_dir=output_dir,
                )
        else:
            sia = self.xarr.sea_ice_area_daily_mean
            self.save_diagnostic_1(
                sia,
                column_name="sea_ice_area_daily_mean",
                descr="totalarea",
                forecast_id=str(1),
                output_dir=output_dir,
            )

    def diagnostic_2(self, method: str = "mean", output_dir: str | None = None):
        r"""
        Process Diagnostic 2: Binned Sea Ice Area.

        This method processes the binned sea ice area data for Diagnostic 2 of the Sea Ice
        Prediction Network (SIPN) submission. It computes the daily sea ice area in predefined
        bins using either the mean or individual ensemble members' values based on the specified
        `method`. The processed data is then saved as CSV files for each ensemble member or the
        mean value.

        Args:
            method: Method used to compute binned sea ice area. Can be "mean" (default) for the
                mean SIA in bins, or "ensemble" for individual ensemble members' SIA in bins.
            output_dir (optional): Directory path where the CSV files for Diagnostic 2 will be saved.

        Raises:
            NotImplementedError: If an invalid `method` is provided.

        Notes:
          - The method "ensemble" saves a separate CSV file for each ensemble member, while
            "mean" saves a single CSV file with the respective data.
        """
        print("Processing Diagnostic 2")
        self.xarr = self.xarr_
        self.compute_binned_daily_sea_ice_area(method=method)

        if method == "ensemble":
            sia_binned = self.xarr.sea_ice_area_binned_daily
            ens_members = sia_binned.sizes["ensemble"]
            for ensemble in range(ens_members):
                self.save_diagnostic_2(
                    sia_binned.sel(ensemble=ensemble),
                    descr="regional-area",
                    forecast_id=str(ensemble + 1),
                    output_dir=output_dir,
                )
        else:
            sia_binned = self.xarr.sea_ice_area_binned_daily_mean
            self.save_diagnostic_2(
                sia_binned,
                descr="regional-area",
                forecast_id=str(1),
                output_dir=output_dir,
            )

    def diagnostic_3(self, method: str = "mean", output_dir: str | None = None):
        r"""
        Process Diagnostic 3: Sea Ice Concentration (SIC) to CF Convention netCDF output.

        This method processes the sea ice concentration data for Diagnostic 3 of the Sea Ice
        Prediction Network (SIPN) submission. It computes the mean SIC or individual ensemble
        members' SIC based on the specified `method`. The processed data is then saved as a NetCDF
        file with the required metadata as per CF Convention.

        Args:
            method: Method used to compute sea ice concentration. Can be "mean" (default) for the
                mean SIC, or "ensemble" for individual ensemble members' SIC.
            output_dir (optional): Directory path where the NetCDF file for Diagnostic 3 will be saved.

        Raises:
            NotImplementedError: If an invalid `method` is provided.

        Notes:
          - The method "ensemble" saves a separate NetCDF file for each ensemble member, while
            "mean" saves a single NetCDF file with the respective data.
        """
        print("Processing Diagnostic 3")
        self.xarr = self.xarr_
        north, south = self.north, self.south
        xarr = self.xarr

        ref_sic = xr.open_dataset(get_refsic(north, south))
        ref_cube = get_refcube(north, south)
        land_mask = Masks(north=north, south=south).get_land_mask()
        longitude = xarr.lon.values
        longitude[longitude < 0] += 360

        # Sea ice concentration mean (in %)
        siconc = (
            xarr.sel(time=self.forecast_init_date)["sic_mean"].transpose(
                "leadtime", "yc", "xc"
            )
            * 100.0
        )

        # Grid cell area (in m^2)
        areacello = np.full_like(land_mask, 25 * 25, dtype=float) * 1e6

        # Land-sea mask, percentage of grid cell covered by ocean (in %)
        sftof = (~land_mask * 100).astype(float)

        diag3_ds_template = xr.Dataset(
            data_vars=dict(
                siconc=(
                    ("time", "yc", "xc"),
                    siconc.data,
                    {"units": "%", "long_name": "Sea-ice area fraction"},
                ),
                areacello=(
                    ("yc", "xc"),
                    areacello,
                    {"units": "m2", "long_name": "Ocean Grid-Cell Area"},
                ),
                sftof=(
                    ("yc", "xc"),
                    sftof,
                    {"units": "%", "long_name": "Sea Area Fraction"},
                ),
                # Lambert_Azimuthal_Grid=ref_sic.Lambert_Azimuthal_Grid,
            ),
            coords=dict(
                time=pd.date_range(
                    self.forecast_init_date,
                    end=self.forecast_end_date,
                    inclusive="right",
                ),
                xc=xarr.xc * 1000.0,
                yc=xarr.yc * 1000.0,
                latitude=(
                    ("yc", "xc"),
                    xarr.lat.data,
                    {"units": "degrees east", "long_name": "Latitude"},
                ),
                longitude=(
                    ("yc", "xc"),
                    longitude,
                    {"units": "degrees north", "long_name": "Longitude"},
                ),
            ),
        )

        diag3_ds_template["xc"].attrs = {
            "units": "m",
            "long_name": "x-coordinate in Lambert Azimuthal Equal Area projection",
        }

        diag3_ds_template["yc"].attrs = {
            "units": "m",
            "long_name": "y-coordinate in Lambert Azimuthal Equal Area projection",
        }

        diag3_ds_template.attrs["projection"] = "Lambert Azimuthal Equal Area"
        diag3_ds_template.attrs["proj4"] = ref_sic.Lambert_Azimuthal_Grid.attrs[
            "proj4_string"
        ]

        if method == "ensemble":
            ens_members = xarr.sizes["ensemble"]
            for ensemble in range(ens_members):
                siconc = (
                    xarr.sel(time=self.forecast_init_date, ensemble=ensemble)[
                        "sic"
                    ].transpose("leadtime", "yc", "xc")
                    * 100.0
                )
                xarr_out = diag3_ds_template.copy()
                xarr_out["siconc"].data = siconc.data
                self.save_diagnostic_3(
                    xarr=xarr_out,
                    descr="concentration",
                    forecast_id=str(ensemble + 1),
                    output_dir=output_dir,
                )
        else:
            siconc = (
                xarr.sel(time=self.forecast_init_date)["sic_mean"].transpose(
                    "leadtime", "yc", "xc"
                )
                * 100.0
            )
            xarr_out = diag3_ds_template.copy()
            xarr_out["siconc"].data = siconc.data
            self.save_diagnostic_3(
                xarr=xarr_out,
                descr="concentration",
                forecast_id=str(1),
                output_dir=output_dir,
            )

        # xarr_out.sftof.plot()
        # xarr_out.areacello.plot()

    def save_diagnostic_1(
        self,
        sia: xr.DataArray,
        column_name: str,
        descr: str,
        forecast_id: str = "001",
        output_dir: str | None = None,
    ):
        r"""
        Save Diagnostic 1 as a text file.

        This method saves the Sea Ice Area (SIA) data for Diagnostic 1 SIPN submission as a text file.
        The data is rounded to 4 decimal places and saved in the specified output directory
        or a default one if not provided.

        Args:
            sia: Sea Ice Area dataset containing the SIA data.
            column_name: Name of the column containing the SIA data in the `sia` dataset.
            descr: Description of the diagnostic being saved (e.g., "area").
            forecast_id (optional): Ensemble member identifier (default is "001"). Not used in this method.
            output_dir (optional): Directory path where the text file will be saved.
        """
        output_dir = self.get_output_dir(output_dir) / Path("txt")
        self.make_output_dir(output_dir)

        df = sia.to_dataframe()
        days = df.index.array
        start_date, end_date = days[0], days[-1]
        start_date, end_date = (
            pd.to_datetime(start_date).strftime("%Y%m%d"),
            pd.to_datetime(end_date).strftime("%Y%m%d"),
        )
        filepath = output_dir / Path(
            f"{self.group_name}_{forecast_id.zfill(3)}_{start_date}-{end_date}_{descr}.txt"
        )
        df_sia = df[column_name].to_frame().T
        df_sia_rounded = df_sia.map(lambda x: f"{x:.4f}").astype(float)
        df_sia_rounded.to_csv(filepath, index=False, header=False)

    def save_diagnostic_2(
        self,
        sia_binned: xr.DataArray,
        descr: str,
        forecast_id: str = "001",
        output_dir: str | None = None,
    ):
        r"""
        Save Diagnostic 2 as a text file.

        This method saves the binned Sea Ice Area (SIA) data for Diagnostic 2 of SIPN submission
        as a text file. The data is rounded to 4 decimal places and saved in the specified output
        directory or a default one if not provided.

        Args:
            sia_binned: Binned Sea Ice Area dataarray containing the SIA data.
            descr: Description of the diagnostic being saved (e.g., "area_binned").
            forecast_id: Ensemble member identifier.
                         Default is "001".
            output_dir: Optional directory path where the text file will be saved.
                        Default is None and the default directory will be used.
        """

        output_dir = self.get_output_dir(output_dir) / Path("txt")
        self.make_output_dir(output_dir)

        df = sia_binned.transpose("bins", "day").to_dataframe().reset_index()
        days = df.day
        start_date, end_date = days.min(), days.max()
        start_date, end_date = (
            pd.to_datetime(start_date).strftime("%Y%m%d"),
            pd.to_datetime(end_date).strftime("%Y%m%d"),
        )

        filepath = output_dir / Path(
            f"{self.group_name}_{forecast_id.zfill(3)}_{start_date}-{end_date}_{descr}.txt"
        )

        df_pivot = df.pivot(index="bins", columns="day", values=sia_binned.name)

        # Round values as required
        df_pivot = df_pivot.map(lambda x: f"{x:.4f}").astype(float)
        df_pivot.to_csv(filepath, index=False, header=False)

    def save_diagnostic_3(
        self,
        xarr: xr.Dataset,
        descr: str,
        forecast_id: str = "001",
        output_dir: str | None = None,
    ):
        r"""
        Output xarray DataSet to a NetCDF file.

        This method saves the Sea Ice Concentration (SIC) data as per Diagnostic 3 of SIPN submission
        as a compressed NetCDF file in the specified output directory or a default one if not provided.

        Args:
            xarr: Dataset containing the SIC data to be saved.
            descr: Description of the diagnostic being saved (e.g., "concentration").
            forecast_id (optional): Ensemble member identifier.
                                    Default is "001".
            output_dir (optional): Directory path where the NetCDF file will be saved.
                                   Default is None and the default directory will be used.

        Returns:
            None
        """
        output_dir = self.get_output_dir(output_dir) / Path("netcdf")
        self.make_output_dir(output_dir)

        file_path = output_dir / Path(f"bas_{forecast_id.zfill(3)}_{descr}.nc")

        compression = dict(zlib=True, complevel=9)
        vars_encoding = {var: compression for var in xarr.data_vars}
        coords_encoding = {coord: compression for coord in xarr.coords}

        xarr.to_netcdf(file_path, mode="w", encoding=vars_encoding | coords_encoding)

    def get_output_dir(self, output_dir: str | Path | None = None) -> Path:
        r"""
        Get the output directory path based on forecast dates or provided input.

        This method returns the output directory path where diagnostic files will be saved. If a
        `output_dir` is provided, it is returned as is. Otherwise, a default output directory based
        on the forecast start and end years is created.

        Args:
            output_dir: Optional Path object representing the directory path to use for saving diagnostics.
                Defaults to None, in which case the method generates a default output directory based
                on forecast dates.

        Returns:
            Path: The output directory path as a Path object.
        """
        if not output_dir:
            start_year = pd.to_datetime(self.forecast_start_date).strftime("%Y")
            end_year = pd.to_datetime(self.forecast_end_date).strftime("%Y")
            output_dir = Path(f"outputs/{start_year}-{end_year}")
        return output_dir

    def make_output_dir(self, output_dir: str):
        r"""
        Create the output directory if it doesn't exist.

        This method creates the specified output directory as a `Path` object if it does not already
        exist. If the directory already exists, no action is taken.

        Args:
            output_dir: The path to the directory that should be created, represented as a Path object.
        """
        if not os.path.exists(output_dir):
            output_dir.mkdir(parents=True, exist_ok=True)


def main():
    r"""
    Main entry point for running SIPN diagnostics and generating outputs.

    This is the main entry point for running Sea Ice Prediction Network (SIPN) diagnostics and
    generating outputs using the `icenet_sipn_south` module. It parses command-line arguments,
    initialises a `SIPNSouthOutputs` object with provided configuration, and runs the specified
    diagnostics.

    Args:
        None (command-line arguments are parsed using `argparse`).

    Returns:
        None

    Command-Line Arguments:
      - `pipeline_path`: Path to the IceNet pipeline root directory.
      - `predict_name`: Name of the prediction dataset (found under `pipeline/results/predict/`,
                        e.g., "fc.2024-11-30_south").
      - `forecast_init_date`: Forecast initialisation date (format: YYYY-MM-DD).
      - `--diagnostics` (optional): Comma-separated list of diagnostics to run (e.g., "1,2,3" for all three).
      - `--forecast_leadtime` (optional): Forecast lead time in days. Only used if hindcasting.
      - `--get_obs` (optional): Set to True if downloading observational data is required.
      - `--method` (optional): Method used for diagnostics (default is "mean").
      - `--plot` (optional): Set to True if generating diagnostic plots is desired.

    Notes:
      - The script expects command-line arguments to be provided when running the `main`
        function.
      - This function initialises a `SIPNSouthOutputs` object with the provided configuration
        and runs the specified diagnostics by calling the corresponding methods on this object.
    """
    args = diagnostic_args()

    pipeline_path = Path(args.pipeline_path)
    if not pipeline_path.exists():
        raise ValueError(
            "Please define valid path to IceNet pipeline root directory.\n{args.pipeline_path} does not exist"
        )

    prediction = SIPNSouthOutputs(
        prediction_pipeline_path=args.pipeline_path,
        prediction_name=args.predict_name,
        forecast_init_date=args.forecast_init_date,
        forecast_leadtime=args.forecast_leadtime,  # Optional - to only load subset of days
        hemisphere="south",
        get_obs=args.get_obs,  # Set to False if not hindcasting or OSI-SAF
        # observational data not already downloaded
        plot=args.plot,
    )

    if "1" in args.diagnostics:
        prediction.diagnostic_1(method=args.method)
    if "2" in args.diagnostics:
        prediction.diagnostic_2(method=args.method)
    if "3" in args.diagnostics:
        prediction.diagnostic_3(method=args.method)


if __name__ == "__main__":
    main()
