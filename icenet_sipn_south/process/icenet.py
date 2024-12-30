import datetime as dt
import logging
import os
from abc import ABC
from datetime import timedelta
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from icenet.data.sic.mask import Masks
from icenet.plotting.video import xarray_to_video as xvid
from icenet.process.predict import get_prediction_data


class IceNetForecastLoader(ABC):
    def __init__(
        self,
        prediction_pipeline_path: str,
        prediction_name: str,
    ) -> None:
        """
        Args:
            forecast_init_date: Which forecast initialisation date to process (Only select one).
                E.g. "2024-11-30"
        """
        self.prediction_pipeline_path = prediction_pipeline_path
        self.prediction_name = prediction_name

    @property
    def get_hemisphere(self):
        """
        Return the hemisphere based on the attributes `north` and `south` of the IceNetOutputPostProcess class instance.
        If `north` is True, return string: "north".
        If `south` is True, return string: "south".
        Otherwise, raise an Exception indicating that the hemisphere is not specified.
        """
        if self.north:
            return "north"
        elif self.south:
            return "south"
        else:
            raise Exception("Hemisphere not specified!")

    @property
    def get_pole(self):
        """
        Return the pole value based on the attributes 'north' and 'south' of the IceNetOutputPostProcess class instance.
        If `north` is True, return 1.
        If `south` is True, return -1.
        Otherwise, raise an Exception indicating that the hemisphere is not specified.
        """
        if self.north:
            return 1
        elif self.south:
            return -1
        else:
            raise Exception("Hemisphere not specified!")

    def get_mask(self):
        """
        Generate the land mask using the Masks class instance with the specified parameters.
        Returns the Masks class instance and the generated land mask.
        """
        mask_gen = Masks(south=self.south, north=self.north)
        land_mask = mask_gen.get_land_mask()
        return mask_gen, land_mask

    def create_ensemble_dataset(self, date_index: bool = False):
        """
        Create an xarray Dataset including ensemble prediction data based on netCDF
        output by icenet.

        Args:
            date_index: If True, set forecast dates as index; otherwise, use
                forecast index integers.

        Returns:
            xarray.Dataset: Dataset containing ensemble prediction data.
        """

        icenet_output_netcdf_file = (
            os.path.join(
                self.prediction_pipeline_path,
                "results",
                "predict",
                self.prediction_name,
            )
            + ".nc"
        )

        ds = xr.open_dataset(icenet_output_netcdf_file).isel(
            leadtime=slice(None, self.forecast_leadtime)
        )
        try:
            ds_subset = ds.sel(time=[self.forecast_init_date])
        except KeyError as e:
            logging.error(f"Forecast {self.forecast_init_date} not in netCDF file")
            logging.error(
                "Available forecast init dates in netCDF file: ",
                f"{[pd.to_datetime(str(date)).strftime('%Y-%m-%d') for date in ds.time.data]}",
            )
            raise e

        # Dimensions for all data being inserted into xarray
        if date_index:
            data_dims_list = ["ensemble", "time", "yc", "xc", "forecast_date"]
        else:
            data_dims_list = ["ensemble", "time", "yc", "xc", "leadtime"]

        # Apply np.nan to land mask regions (emulating icenet output)
        mask_gen, land_mask = self.get_mask()
        land_mask_nan = land_mask.astype(float)
        land_mask_nan[land_mask] = np.nan
        land_mask_nan[~land_mask] = 1.0

        arr, data, ens_members = get_prediction_data(
            root=self.prediction_pipeline_path,
            name=self.prediction_name,
            date=self.forecast_init_date_dt,
            return_ensemble_data=True,
        )
        # dates = pd.to_datetime(ds.forecast_date.time.data)
        # dates = pd.to_datetime(ds.forecast_date.leadtime.data)
        # init_date = ds_subset.forecast_date.time.data
        init_date = self.forecast_init_date_dt
        days = ds_subset.forecast_date.leadtime.data.tolist()
        # print(init_date)
        dates = [init_date + timedelta(days=day) for day in days]

        # Apply 0.0 to inactive cell regions
        for i, forecast_lead_date in enumerate(dates):
            # Get active cell mask for month of this forecast lead date
            grid_cell_mask = mask_gen.get_active_cell_mask(forecast_lead_date.month)
            # Applying to SIC mean read from numpy to directly compare against
            # `icenet_output` = 0
            # Apply to SIC mean
            arr[~grid_cell_mask, i, 0] = 0.0
            # Apply to SIC std dev
            arr[~grid_cell_mask, i, 1] = 0.0

            # Applying to Ensemble prediction outputs
            for ensemble in range(ens_members):
                data[ensemble, :, ~grid_cell_mask, i] = 0.0

        # Select subset of leadtime to work with based on user specification
        data = data[..., : self.forecast_leadtime]
        xarr = ds_subset.copy()
        # print(ds_subset.coords)
        # print("___")
        # print(ds_subset.dims)
        # print("___")
        # return
        # print(xarr)

        # Add full ensemble prediction data to the original icenet DataSet
        sic = (
            data_dims_list,
            data * land_mask_nan[np.newaxis, np.newaxis, :, :, np.newaxis],
        )
        xarr["sic"] = sic

        self.ds = ds
        ds.close()

        if not date_index:
            self.xarr = xarr
            self.xarr_ = xarr
        else:
            return xarr

    def save_data(self, output_path, reference="BAS_icenet", drop_vars=None):
        output_path = os.path.join(
            output_path, "netcdf", f"{reference}_{self.get_hemisphere}.nc"
        )

        Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

        if drop_vars is None:
            xarr = self.xarr
        else:
            xarr = self.xarr.copy()
            xarr = xarr.drop_vars(drop_vars, errors="ignore")

        compression = dict(zlib=True, complevel=9)
        vars_encoding = {var: compression for var in xarr.data_vars}
        coords_encoding = {coord: compression for coord in xarr.coords}

        xarr.to_netcdf(output_path, encoding=vars_encoding | coords_encoding)

    def get_data_date_indexed(self):
        """Get forecast Xarray Dataset with forecast dates set as index
        instead of forecast index integers.
        """
        xarr = self.create_ensemble_dataset(date_index=True)
        return xarr

    def plot_ensemble_mean(self):
        """Plots the ensemble mean."""
        self.create_ensemble_dataset()
        mask_gen, land_mask = self.get_mask()

        forecast_date = self.xarr.time.values[0]

        fc = (
            self.xarr.sic_mean.isel(time=0)
            .drop_vars("time")
            .rename(dict(leadtime="time"))
        )
        fc["time"] = [
            pd.to_datetime(forecast_date) + dt.timedelta(days=int(e))
            for e in fc.time.values
        ]

        # Convert eastings and northings from kilometers to metres.
        # Needed if coastlines is enabled in following `xvid`` call.
        fc["xc"] = fc["xc"].data * 1000
        fc["yc"] = fc["yc"].data * 1000

        anim = xvid(
            fc,
            15,
            figsize=(8, 8),
            mask=land_mask,
            mask_type="contour",
            north=self.north,
            south=self.south,
            coastlines=False,
        )
        return anim
