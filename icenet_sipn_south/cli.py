import argparse
import re


def csv_arg(string: str) -> list:
    csv_items = []
    string = re.sub(r"^\'(.*)\'$", r"\1", string)

    for el in string.split(","):
        if len(el) == 0:
            csv_items.append(None)
        else:
            csv_items.append(el)
    return csv_items


def diagnostic_args() -> object:
    ap = argparse.ArgumentParser()
    ap.add_argument("pipeline_path", help="Path to the root IceNet-pipeline directory")
    ap.add_argument(
        "predict_name",
        help="Name of prediction (found under `pipeline/results/predict/`)",
    )
    ap.add_argument("forecast_init_date", help="Start date of the forecast to use")

    ap.add_argument(
        "-d",
        "--diagnostics",
        help="Comma separated list of diagnostics to run, Options: `1,2,3`",
        type=csv_arg,
        default=[1, 2, 3],
    )
    ap.add_argument(
        "-fl",
        "--forecast_leadtime",
        type=int,
        default=90,
        help="IceNet by default forecasts up to 93 days ahead, can instead specify how many days to process for this diagnostic, Default=90",
    )
    ap.add_argument(
        "-go",
        "--get-obs",
        default=False,
        action="store_true",
        help="Whether to include OSI-SAF observational data in plot/processing",
    )
    method_choices = ["ensemble", "mean"]
    ap.add_argument(
        "-m",
        "--method",
        default=method_choices[0],
        choices=method_choices,
        help="Whether to process ensemble of predictions or ensemble mean",
    )
    ap.add_argument(
        "-p",
        "--plot",
        default=False,
        action="store_true",
        help="Whether to display any plots being generated",
    )

    args = ap.parse_args()
    return args
