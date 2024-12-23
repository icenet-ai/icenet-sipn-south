from abc import ABC

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
        sea_ice_concentration,
        ensemble_axis=0,
        grid_cell_area=25 * 25,
        threshold=0.15,
        plot=False,
    ):
        """Compute Sea Ice Area for an image for a given day.

        Computes the total Sea Ice Area (SIC>=15%) for one day.

        Args:
            ensemble_axis: Axis at which the ensemble members are stored.
                            Used to omit reduction operation across this axis.
        """
        # self.clear_sia()
        sic = sea_ice_concentration

        # Mask values less than the threshold
        sic = sic.where(sic >= threshold)

        if plot:
            xr.plot.imshow(sic.squeeze())
            return

        # Prevent summation across "ensemble" dimension
        valid_dims = [dim for dim in sic.dims if dim != "ensemble"]

        # Sum across all axes except ensemble
        sea_ice_area = sic.sum(dim=valid_dims) * (grid_cell_area / 1e6)

        return sea_ice_area


    def _compute_binned_sea_ice_area(self, sea_ice_concentration, *args, **kwargs):
        """Compute binned Sea Ice Area for a singular day.

        Computes Sea Ice Area binned by longitude. Binning is per 10deg from 0 to 360.
        E.g. 0<=lon<10, 10<=lon<20, ..., 350<=lon<360

        Args:
            sea_ice_concentration (xarray.DataArray): Sea Ice Concentration as a fraction, not %.
            grid_cell_area (float, optional): Area of each individual cell, default assumes EASE2 25km grid. Defaults to 25*25.
            threshold (float, optional): Threshold to apply masking when computing Sea Ice Area (SIA).

        Returns:
            sea_ice_area_binned (xarray.DataArray): Computed Sea Ice Area, excluding masked regions. Dims: [36, 90]
        """
        sic = sea_ice_concentration

        # Convert longitudes from (-180 to 180) to (0 to 360) as per Diagnostic 2 of SIPN South.
        lon_360 = sic.lon.values
        lon_360[lon_360<0] += 360
        lon_360

        sic = sic.assign_coords({"lon_360": (("xc", "yc"), lon_360)})

        longitude_bin = xr.DataArray(np.linspace(0, 360, 36+1))

        sea_ice_area_binned = sic.groupby_bins("lon_360", longitude_bin).map(self._compute_sea_ice_area, args=args, **kwargs)

        return sea_ice_area_binned


    def compute_daily_sea_ice_area(
        self, method="mean", grid_cell_area=25 * 25, threshold=0.15, plot=False
    ):
        """Daily Sea Ice Area.

        Computes the total Sea Ice Area (SIC>15%) for each day.
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
            "plot": plot,
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
        self, method="mean", grid_cell_area=25 * 25, threshold=0.15, plot=False
    ):
        """Monthly Sea Ice Area from daily.

        Computes the total Sea Ice Area (SIC>=15%) for each day, then, averages the result
        from each of the days of the month into a monthly average Sea Ice Area.
        """
        self.compute_daily_sea_ice_area(
            method=method, grid_cell_area=grid_cell_area, threshold=threshold, plot=plot
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
        self, method="mean", grid_cell_area=25 * 25, threshold=0.15, plot=False
    ):
        """Binned daily Sea Ice Area.

        Computes the binned Sea Ice Area (SIC>15%) for each day.
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
            "plot": plot,
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
                    sea_ice_area_binned_daily_mean=(["day", "bins"], sea_ice_area_binned_daily),
                ),
                coords=dict(
                    day=forecast_dates,
                    # bins=bins,
                ),
            )
        elif method == "observation":
            sea_ice_area_binned_daily_osisaf_ds = xr.Dataset(
                data_vars=dict(
                    sea_ice_area_binned_daily_osisaf=(("day", "bins"), sea_ice_area_binned_daily),
                ),
                coords=dict(
                    day=forecast_dates,
                    # bins=bins,
                ),
            )
            sea_ice_area_binned_daily_osisaf_ds["sea_ice_area_binned_daily_osisaf"].attrs[
                "long_name"
            ] = "Total Sea-Ice Area for each day"
            sea_ice_area_binned_daily_osisaf_ds["sea_ice_area_binned_daily_osisaf"].attrs["units"] = (
                "10⁶ km²"
            )
        elif method == "ensemble":
            sea_ice_area_binned_daily_mean = sea_ice_area_binned_daily.mean(axis=1)
            sea_ice_area_binned_daily_stddev = sea_ice_area_binned_daily.std(axis=1)

            sea_ice_area_binned_daily_ds = xr.Dataset(
                data_vars=dict(
                    sea_ice_area_binned_daily=(("day", "ensemble", "bins"), sea_ice_area_binned_daily),
                    sea_ice_area_binned_daily_mean=(("day", "bins"), sea_ice_area_binned_daily_mean),
                    sea_ice_area_binned_daily_stddev=(("day", "bins"), sea_ice_area_binned_daily_stddev),
                ),
                coords=dict(
                    day=forecast_dates,
                    ensemble=list(range(sea_ice_area_binned_daily.shape[1])),
                    # bins=bins,
                ),
            )

        if method != "observation":
            sea_ice_area_binned_daily_ds["sea_ice_area_binned_daily_mean"].attrs["long_name"] = (
                "Total Sea-Ice Area for each day per 10° longitude bin"
            )
            sea_ice_area_binned_daily_ds["sea_ice_area_binned_daily_mean"].attrs["units"] = "10⁶ km²"

            self.xarr = xr.merge([self.xarr, sea_ice_area_binned_daily_ds], compat="override")
        else:
            self.xarr = xr.merge(
                [self.xarr, sea_ice_area_binned_daily_osisaf_ds], compat="override"
            )

        return self

    def plot_sia(self):
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

        fig.show()
