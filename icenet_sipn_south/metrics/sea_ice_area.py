from abc import ABC
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import xarray as xr


class SeaIceArea(ABC):
    """Sea Ice Area computation."""

    @property
    def data(self):
        return self.xarr

    def _compute_sea_ice_area(
        self,
        sea_ice_concentration: xr.DataArray,
        grid_cell_area: float = 25 * 25,
        threshold: float = 0.15,
    ):
        r"""
        Compute Sea Ice Area for an image for a given day.

        Computes the total Sea Ice Area (SIC>=15%) for one day.
        It takes a sea ice concentration data array and optional parameters for grid cell area,
        concentration threshold.

        Args:
            sea_ice_concentration:
                The input sea ice concentration data with dimensions (time, xc, yc) and
                ensemble if multi-model ensemble is used, i.e. (ensemble, time, xc, yc).
            grid_cell_area (optional):
                Grid cell area in km². Default is 25km by 25km.
            threshold (optional):
                Sea Ice Concentration threshold for considering a grid cell as sea ice.
                Default is 15% (0.15).

        Returns:
            xr.DataArray
                Total Sea Ice Area in 10⁶ km² with dimensions (time) if ensemble dimension exists,
                otherwise (time, xc, yc).
        """
        # self.clear_sia()
        sic = sea_ice_concentration

        # Mask values less than the threshold
        sic = sic.where(sic >= threshold)

        # Prevent summation across "ensemble" dimension
        valid_dims = [dim for dim in sic.dims if dim != "ensemble"]

        # Sum across all axes except ensemble (Gives SIA in km^2)
        sea_ice_area = sic.sum(dim=valid_dims) * grid_cell_area

        # SIPN South requests SIA units to be in 10^6 km^2
        sea_ice_area /= 1e6

        return sea_ice_area

    def _compute_binned_sea_ice_area(
        self, sea_ice_concentration: xr.DataArray, *args, **kwargs
    ):
        r"""
        Compute binned Sea Ice Area for a singular day.

        Computes Sea Ice Area binned by longitude. Binning is per 10deg from 0 to 360.
        E.g. 0<=lon<10, 10<=lon<20, ..., 350<=lon<360

        Args:
            sea_ice_concentration: Sea Ice Concentration as a fraction, not %.
            grid_cell_area (optional): Area of each individual cell, default assumes EASE2 25km grid.  Default is 25km by 25km.
            threshold (optional): Threshold to apply masking when computing Sea Ice Area (SIA).

        Returns:
            sea_ice_area_binned (xarray.DataArray): Computed Sea Ice Area, excluding masked regions. Dims: [36, 90]
        """
        sic = sea_ice_concentration

        # Convert longitudes from (-180 to 180) to (0 to 360) as per Diagnostic 2 of SIPN South.
        lon_360 = sic.lon.values
        lon_360[lon_360 < 0] += 360

        sic = sic.assign_coords({"lon_360": (("yc", "xc"), lon_360)})
        # sic = sic.assign_coords(lon_360=((sic.lon + 360) % 360))

        longitude_bin = xr.DataArray(np.linspace(0, 360, 36 + 1))

        sea_ice_area_binned = sic.groupby_bins("lon_360", longitude_bin).map(
            self._compute_sea_ice_area, args=args, **kwargs
        )

        return sea_ice_area_binned

    def compute_daily_sea_ice_area(
        self,
        method: str = "mean",
        grid_cell_area: float = 25 * 25,
        threshold: float = 0.15,
    ):
        r"""
        Compute daily Sea Ice Area (SIC>15%) based on specified method.

        Computes the total Sea Ice Area (SIC>15%) for each day based on the specified method.
        The daily sea ice area is calculated by summing the sea ice concentration within each
        grid cell, considering only cells where the sea ice concentration exceeds the specified
        threshold. The result is a time series of daily total sea ice area in million square kilometers
        (10⁶ km²).  The result is added to the instance's `xarr` attribute as a new variable.

        Args:
            method (optional):
                Method to use for calculating the daily sea ice area.
                - "mean": Use mean sea ice concentration from all available ensemble members.
                - "ensemble": Calculate daily total sea ice area for each ensemble member separately.
                - "observation": Use observed sea ice concentrations to compute the daily total sea
                ice area.
                Default is "mean".
            grid_cell_area (optional):
                Area of a single grid cell in square kilometers (km²). Default is 625 km² (25km x 25km).
            threshold (optional):
                Minimum SIC required for a grid cell to be included in the total sea ice area
                calculation. Default is 0.15 (15%).

        Returns:
            SeaIceAreaCalculator
                Instance of `SeaIceAreaCalculator` for method chaining.

        """
        if method.casefold() == "mean":
            sic = self.xarr.sic_mean
        elif method.casefold() == "ensemble":
            sic = self.xarr.sic
        elif method.casefold() == "observation":
            sic = self.obs

        kwargs = {
            "grid_cell_area": grid_cell_area,
            "threshold": threshold,
        }
        sea_ice_area_daily = np.asarray(
            [
                sic.isel(leadtime=day - 1)
                .map_blocks(self._compute_sea_ice_area, kwargs=kwargs)
                .values
                for day in self.xarr.leadtime
            ]
        )
        forecast_dates = pd.to_datetime(self.xarr.forecast_date[0])

        if method == "mean":
            sea_ice_area_daily_ds = xr.Dataset(
                data_vars=dict(
                    sea_ice_area_daily_mean=(["day"], sea_ice_area_daily),
                ),
                coords=dict(
                    day=forecast_dates,
                ),
            )
        elif method == "observation":
            sea_ice_area_daily_osisaf_ds = xr.Dataset(
                data_vars=dict(
                    sea_ice_area_daily_osisaf=(["day"], sea_ice_area_daily),
                ),
                coords=dict(
                    day=forecast_dates,
                ),
            )
            sea_ice_area_daily_osisaf_ds["sea_ice_area_daily_osisaf"].attrs[
                "long_name"
            ] = "Total Sea-Ice Area for each day"
            sea_ice_area_daily_osisaf_ds["sea_ice_area_daily_osisaf"].attrs["units"] = (
                "10⁶ km²"
            )
        elif method == "ensemble":
            sea_ice_area_daily_mean = sea_ice_area_daily.mean(axis=1)
            sea_ice_area_daily_stddev = sea_ice_area_daily.std(axis=1)

            sea_ice_area_daily_ds = xr.Dataset(
                data_vars=dict(
                    sea_ice_area_daily=(["day", "ensemble"], sea_ice_area_daily),
                    sea_ice_area_daily_mean=(["day"], sea_ice_area_daily_mean),
                    sea_ice_area_daily_stddev=(["day"], sea_ice_area_daily_stddev),
                ),
                coords=dict(
                    day=forecast_dates,
                    ensemble=list(range(sea_ice_area_daily.shape[1])),
                ),
            )

        if method != "observation":
            sea_ice_area_daily_ds["sea_ice_area_daily_mean"].attrs["long_name"] = (
                "Total Sea-Ice Area for each day"
            )
            sea_ice_area_daily_ds["sea_ice_area_daily_mean"].attrs["units"] = "10⁶ km²"

            self.xarr = xr.merge([self.xarr, sea_ice_area_daily_ds], compat="override")
        else:
            self.xarr = xr.merge(
                [self.xarr, sea_ice_area_daily_osisaf_ds], compat="override"
            )

        return self

    def compute_monthly_sea_ice_area(
        self, method="mean", grid_cell_area: float = 25 * 25, threshold: float = 0.15
    ):
        r"""
        Monthly Sea Ice Area from daily.

        Computes the total Sea Ice Area (SIC>=15%) for each day, then, averages the result
        from each of the days of the month into a monthly average Sea Ice Area.
        """
        self.compute_daily_sea_ice_area(
            method=method, grid_cell_area=grid_cell_area, threshold=threshold
        )

        sea_ice_area_daily_ds = self.xarr.sea_ice_area_daily_mean

        # Aggregate daily Sea Ice Area to get monthly
        sea_ice_area_monthly_ds = (
            sea_ice_area_daily_ds.groupby(self.xarr.day.dt.month)
            .mean()
            .rename("sea_ice_area_monthly_mean")
        )

        self.xarr["sea_ice_area_monthly_mean"] = sea_ice_area_monthly_ds
        self.xarr["sea_ice_area_monthly_mean"].attrs["long_name"] = (
            "Total Sea-Ice Area for each month, averaged from daily"
        )
        self.xarr["sea_ice_area_monthly_mean"].attrs["units"] = "10⁶ km²"
        self.xarr.month.attrs = {
            "long_name": "months for which mean sea ice area are computed for"
        }

        return self

    def compute_binned_daily_sea_ice_area(
        self,
        method: str = "mean",
        grid_cell_area: float = 25 * 25,
        threshold: float = 0.15,
    ):
        r"""
        Compute binned (by 10° longitude) daily Sea Ice Area (SIC > 15%) for each day.

        This method calculates the total sea ice area in bins of 10° longitude for each day,
        based on the provided `method`. It sums the sea ice concentration within each grid cell
        falling within a specific bin and considers only cells where the SIC exceeds the specified
        threshold. The result is added to the instance's `xarr` attribute as a new variable.

        Args:
            method (optional):
                Method to use for calculating the daily sea ice area.
                - "mean": Use mean sea ice concentration from all available ensemble members.
                - "ensemble": Calculate daily total sea ice area for each ensemble member separately.
                - "observation": Use observed sea ice concentrations to compute the daily total sea
                ice area.
                Default is "mean".
            grid_cell_area (optional):
                Area of a single grid cell in square kilometers (km²). Default is 625 km² (25km x 25km).
            threshold (optional):
                Minimum SIC required for a grid cell to be included in the total sea ice area
                calculation. Default is 0.15 (15%).

        Returns:
            SeaIceAreaCalculator
                Instance of `SeaIceAreaCalculator` for method chaining.

        """
        if method.casefold() == "mean":
            sic = self.xarr.sic_mean
        elif method.casefold() == "ensemble":
            sic = self.xarr.sic
        elif method.casefold() == "observation":
            sic = self.obs

        kwargs = {
            "grid_cell_area": grid_cell_area,
            "threshold": threshold,
        }
        sea_ice_area_binned_daily = np.asarray(
            [
                sic.isel(leadtime=day - 1)
                .map_blocks(self._compute_binned_sea_ice_area, kwargs=kwargs)
                .values
                for day in self.xarr.leadtime
            ]
        )

        forecast_dates = pd.to_datetime(self.xarr.forecast_date[0])

        bins = 36
        if method == "mean":
            sea_ice_area_binned_daily_ds = xr.Dataset(
                data_vars=dict(
                    sea_ice_area_binned_daily_mean=(
                        ["day", "bins"],
                        sea_ice_area_binned_daily,
                    ),
                ),
                coords=dict(
                    day=forecast_dates,
                    # bins=bins,
                ),
            )
        elif method == "observation":
            sea_ice_area_binned_daily_osisaf_ds = xr.Dataset(
                data_vars=dict(
                    sea_ice_area_binned_daily_osisaf=(
                        ("day", "bins"),
                        sea_ice_area_binned_daily,
                    ),
                ),
                coords=dict(
                    day=forecast_dates,
                    # bins=bins,
                ),
            )
            sea_ice_area_binned_daily_osisaf_ds[
                "sea_ice_area_binned_daily_osisaf"
            ].attrs["long_name"] = "Total Sea-Ice Area for each day"
            sea_ice_area_binned_daily_osisaf_ds[
                "sea_ice_area_binned_daily_osisaf"
            ].attrs["units"] = "10⁶ km²"
        elif method == "ensemble":
            sea_ice_area_binned_daily_mean = sea_ice_area_binned_daily.mean(axis=1)
            sea_ice_area_binned_daily_stddev = sea_ice_area_binned_daily.std(axis=1)

            sea_ice_area_binned_daily_ds = xr.Dataset(
                data_vars=dict(
                    sea_ice_area_binned_daily=(
                        ("day", "ensemble", "bins"),
                        sea_ice_area_binned_daily,
                    ),
                    sea_ice_area_binned_daily_mean=(
                        ("day", "bins"),
                        sea_ice_area_binned_daily_mean,
                    ),
                    sea_ice_area_binned_daily_stddev=(
                        ("day", "bins"),
                        sea_ice_area_binned_daily_stddev,
                    ),
                ),
                coords=dict(
                    day=forecast_dates,
                    ensemble=list(range(sea_ice_area_binned_daily.shape[1])),
                    # bins=bins,
                ),
            )

        if method != "observation":
            sea_ice_area_binned_daily_ds["sea_ice_area_binned_daily_mean"].attrs[
                "long_name"
            ] = "Total Sea-Ice Area for each day per 10° longitude bin"
            sea_ice_area_binned_daily_ds["sea_ice_area_binned_daily_mean"].attrs[
                "units"
            ] = "10⁶ km²"

            self.xarr = xr.merge(
                [self.xarr, sea_ice_area_binned_daily_ds], compat="override"
            )
        else:
            self.xarr = xr.merge(
                [self.xarr, sea_ice_area_binned_daily_osisaf_ds], compat="override"
            )

        return self

    def plot_sia(self, show_plot: bool = False, output_dir: str | None = None):
        r"""
        Plot Sea Ice Area (SIA) time series.

        This method generates a time series plot of Sea Ice Area (SIA) using the SIA data available in
        the instance's `xarr` attribute. It includes the mean SIA from the IceNet model and optionally,
        individual ensemble run results, as well as observed SIA from OSI-SAF if available.

        Args:
            show_plot (optional): Whether to display the generated plot.
            output_dir (optional): Directory path where the plot will be saved.

        Notes:
        - The plotted SIA data depends on the available variables in the instance's `xarr` attribute.
            It always includes the mean SIA from IceNet (`sea_ice_area_daily_mean`).
            If individual ensemble run results are present (`sea_ice_area_daily`), they are also plotted,
            with uncertainty bands representing minimum and maximum values across ensembles (They are not
            standard deviations).
        - Observed SIA from OSI-SAF (`sea_ice_area_daily_osisaf`) is included in the plot if available.
        """
        data_plt = {
            "Time": pd.date_range(
                start=self.forecast_init_date,
                end=self.forecast_end_date,
                inclusive="right",
            ),
            "IceNet Mean": self.xarr.sea_ice_area_daily_mean,
        }

        # Add individual ensemble run results
        if hasattr(self.xarr, "sea_ice_area_daily"):
            method = "ensemble"
        else:
            method = "mean"

        y_values = ["IceNet Mean"]
        if method == "ensemble":
            sia = self.xarr.sea_ice_area_daily
            ens_members = sia.sizes["ensemble"]
            ensemble_values = []
            for ensemble in range(ens_members):
                key = f"IceNet Ensemble {ensemble}"
                data_plt[key] = sia.isel(ensemble=ensemble)
                y_values += [key]
                ensemble_values += [key]

            # Plus or minus min/max ensemble values
            y_upper = sia.max(dim="ensemble")
            y_lower = sia.min(dim="ensemble")

        # OSI-SAF Truth
        if hasattr(self.xarr, "sea_ice_area_daily_osisaf"):
            osisaf_key = "Observation (osisaf)"
            data_plt[osisaf_key] = self.xarr.sea_ice_area_daily_osisaf
            # Add observations to plot if they exist
            y_values += [osisaf_key]

        df_plt = pd.DataFrame(data_plt)

        x = data_plt["Time"].strftime("%Y-%m-%d")

        fig = go.Figure()

        # Styling
        month_range = "-".join(
            pd.date_range(
                start=self.forecast_start_date,
                end=self.forecast_end_date,
                inclusive="both",
                freq="MS",
            )
            .strftime("%b")
            .tolist()
        )
        start_year = pd.to_datetime(self.forecast_start_date).strftime("%Y")
        end_year = pd.to_datetime(self.forecast_end_date).strftime("%Y")
        hemisphere = self.get_hemisphere
        pole = "Arctic" if hemisphere.casefold() == "north" else "Antarctic"
        style_dict = {
            "layout.plot_bgcolor": "rgba(0, 0, 0, 0)",
            "layout.font.family": "Times New Roman",
            "layout.xaxis.linecolor": "black",
            "layout.xaxis.ticks": "inside",
            "layout.xaxis.mirror": True,
            "layout.xaxis.showline": True,
            "layout.yaxis.linecolor": "black",
            "layout.yaxis.ticks": "inside",
            "layout.yaxis.mirror": True,
            "layout.yaxis.showline": True,
            "layout.showlegend": True,
            "layout.legend.bgcolor": "rgba(0, 0, 0, 0)",
            "layout.legend.font.family": "monospace",
            "layout.yaxis.range": (0, 14),
            "layout.xaxis.title": "Time",
            "layout.yaxis.title": "10<sup>6</sup> km<sup>2</sup>",
            "layout.title": f"{month_range} {start_year}-{end_year} {pole} Sea Ice Area (SIA)",
        }

        fig.update(**style_dict)
        for y in reversed(y_values):
            fig.add_trace(
                go.Scatter(
                    name=y,
                    x=x,
                    y=data_plt[y],
                    mode="lines",
                    showlegend=True,
                )
            )

        # Add uncertainty bands to IceNet ensemble result
        if method == "ensemble":
            # Upper bound
            fig.add_trace(
                go.Scatter(
                    name="Upper Bound",
                    x=df_plt["Time"],
                    y=y_upper,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            # Lower bound
            fig.add_trace(
                go.Scatter(
                    name="Lower Bound",
                    x=df_plt["Time"],
                    y=y_lower,
                    mode="lines",
                    line=dict(width=0),
                    fillcolor="rgba(173, 216, 230, 0.5)",
                    fill="tonexty",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            for trace in fig.data:
                if trace.name in ensemble_values:
                    trace.visible = "legendonly"

        fig.update_xaxes(
            tickmode="linear",
            tick0=self.forecast_init_date,
            dtick=86400000 * 11,
            minor=dict(
                ticklen=6,
                tickcolor="black",
                showgrid=True,
                gridwidth=1,
                gridcolor="rgba(0, 0, 0, 1)",
            ),
        )

        fig.update_yaxes(
            minor=dict(showgrid=False),
            linecolor="black",
            gridcolor="rgba(0, 0, 0, 0.1)",
        )

        fig.update_layout(
            title=dict(font=dict(size=24)),
            xaxis_title=None,
            showlegend=True,
            legend_title_text="",
            font=dict(size=14),
        )

        # `get_output_dir` and `make_output_dir` are defined in the derived class
        output_dir = self.get_output_dir(output_dir) / Path("png")
        self.make_output_dir(output_dir)

        start_date = self.forecast_init_date
        end_date = self.forecast_end_date
        start_date = pd.to_datetime(self.forecast_start_date).strftime("%Y%m%d")
        end_date = pd.to_datetime(self.forecast_end_date).strftime("%Y%m%d")
        descr = "total_sea_ice_area"
        filepath = output_dir / Path(
            f"{self.group_name}_{start_date}-{end_date}_{descr}.png"
        )

        fig.write_image(filepath, scale=2, width=1920, height=1080)
        if show_plot:
            fig.show()
