import datetime as dt
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from icenet.data.sic.mask import Masks
from icenet.plotting.utils import get_obs_da
from icenet.process.predict import get_refsic, get_refcube

from ..process.icenet import IceNetForecastLoader
from .sea_ice_area import SeaIceArea

class SIPNSouthOutputs(
    IceNetForecastLoader,
    SeaIceArea,
):
    """SIPN Sea Ice Outlook Submission for IceNet (with daily averaging)
    """

    def __init__(
        self,
        prediction_pipeline_path: str,
        prediction_name: str,
        forecast_init_date: dt.date,
        forecast_leadtime: int = None,
        hemisphere: str = "south",
        get_obs=False,
        group_name="BAS",
    ) -> None:
        """
        Args:
            prediction_path: Path to the numpy prediction outputs
        """
        self.north = False
        self.south = False
        self.hemisphere = hemisphere
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
            except KeyError as e:
                logging.error(
                    f"Observational data not available for given forecast init date\n{e}"
                )

    def check_forecast_period(self):
        """Used to check that the forecast period is within June-November."""
        raise NotImplementedError

    def diagnostic_1(self, method="mean", output_dir=None):

        self.xarr = self.xarr_
        self.compute_daily_sea_ice_area(method=method)
        self.compute_daily_sea_ice_area(method="observation")
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

    def diagnostic_2(self, method="mean", output_dir=None):
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

    def diagnostic_3(self, method="mean", output_dir=None):
        self.xarr = self.xarr_
        north, south = self.north, self.south
        xarr = self.xarr

        ref_sic = xr.open_dataset(get_refsic(north, south))
        ref_cube = get_refcube(north, south)
        land_mask = Masks(north=north, south=south).get_land_mask()
        longitude = xarr.lon.values
        longitude[longitude<0] += 360

        # Sea ice concentration mean (in %)
        siconc = xarr.sel(time=self.forecast_init_date)["sic_mean"].transpose("leadtime", "yc", "xc") * 100.

        # Grid cell area (in m^2)
        areacello = np.full_like(land_mask, 25*25, dtype=float) * 1E6

        # Land-sea mask, percentage of grid cell covered by ocean (in %)
        sftof = (~land_mask * 100).astype(float)

        diag3_ds_template = xr.Dataset(
            data_vars=dict(
                siconc=(("time", "yc", "xc"),
                        siconc.data,
                        {
                            "units": "%",
                            "long_name": "Sea-ice area fraction"
                        }
                    ),
                areacello=(("yc", "xc"),
                        areacello,
                        {
                            "units": "m2",
                            "long_name": "Ocean Grid-Cell Area"
                        }
                        ),
                sftof=(("yc", "xc"),
                        sftof,
                            {
                                "units": "%",
                                "long_name": "Sea Area Fraction"
                            }
                    ),
                # Lambert_Azimuthal_Grid=ref_sic.Lambert_Azimuthal_Grid,
            ),
            coords=dict(
                time=pd.date_range(self.forecast_init_date, end=self.forecast_end_date, inclusive="right"),
                xc=xarr.xc * 1000.,
                yc=xarr.yc * 1000.,
                latitude=(("yc", "xc"),
                        xarr.lat.data,
                        {
                            "units": "degrees east",
                            "long_name": "Latitude"
                        }
                        ),
                longitude=(("yc", "xc"),
                        longitude,
                        {
                            "units": "degrees north",
                            "long_name": "Longitude"
                        }
                        ),
            )
        )

        diag3_ds_template["xc"].attrs = {
                            "units": "m",
                            "long_name": "x-coordinate in Lambert Azimuthal Equal Area projection"
                    }

        diag3_ds_template["yc"].attrs = {
                            "units": "m",
                            "long_name": "y-coordinate in Lambert Azimuthal Equal Area projection"
                    }

        diag3_ds_template.attrs["projection"] = "Lambert Azimuthal Equal Area"
        diag3_ds_template.attrs["proj4"] = ref_sic.Lambert_Azimuthal_Grid.attrs["proj4_string"]

        if method == "ensemble":
            ens_members = xarr.sizes["ensemble"]
            for ensemble in range(ens_members):
                siconc = xarr.sel(time=self.forecast_init_date, ensemble=ensemble)["sic"].transpose("leadtime", "yc", "xc") * 100.
                xarr_out = diag3_ds_template.copy()
                xarr_out["siconc"].data = siconc.data
                self.save_diagnostic_3(
                    xarr=xarr_out,
                    descr="concentration",
                    forecast_id=str(ensemble + 1),
                    output_dir=output_dir,
                )
        else:
            siconc = xarr.sel(time=self.forecast_init_date)["sic_mean"].transpose("leadtime", "yc", "xc") * 100.
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
        self, sia, column_name, descr, forecast_id="001", output_dir=None
    ):
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
        self, sia_binned, descr, forecast_id="001", output_dir=None
    ):
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

    def save_diagnostic_3(self, xarr, descr, forecast_id="001", output_dir=None):
        """Output xarray DataSet to netCDF file.

        Compress and output to netCDF file as per Diagnostic 3.

        Args:
            xarr (xarr.DataSet): Dataset to output to file.
            forecast_id (int): Which ensemble member is being generated.

        Returns:
            None
        """
        output_dir = self.get_output_dir(output_dir) / Path("netcdf")
        self.make_output_dir(output_dir)

        file_path = output_dir / Path(f"bas_{forecast_id.zfill(3)}_{descr}.nc")

        compression = dict(zlib=True, complevel=9)
        vars_encoding = {var: compression for var in xarr.data_vars}
        coords_encoding = {coord: compression for coord in xarr.coords}

        xarr.to_netcdf(file_path,
                        mode="w",
                        encoding=vars_encoding | coords_encoding
                    )

    def get_output_dir(self, output_dir=None):
        if not output_dir:
            start_year = pd.to_datetime(self.forecast_start_date).strftime("%Y")
            end_year = pd.to_datetime(self.forecast_end_date).strftime("%Y")
            output_dir = Path(f"outputs/{start_year}-{end_year}")
        return output_dir

    def make_output_dir(self, output_dir):
        if not os.path.exists(output_dir):
            output_dir.mkdir(parents=True, exist_ok=True)
