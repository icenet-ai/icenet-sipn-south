#!/bin/bash

# Start year to process
export YEAR_START=${1}
# End year to process
export YEAR_END=${2}
# Which SIPN South Diagnostics to generate (comma separated), Options: 1,2,3
export DIAGNOSTICS=${3:-"1,2,3"}
# Whether to use SIC mean or process individual ensemble SIC forecast predictions.
export METHOD=${4:-"ensemble"}
# SIPN South only requires 90 day leadtime, should we process a different leadtime?
# IceNet outputs a 93 day leadtime forecast by default
export FORECAST_LEADTIME=${5:-"90"}

# Loop through and generate diagnostics for specified years
for ((i=$YEAR_START;i<=$YEAR_END;i++))
do
    YEAR=$i
    echo Processing ${YEAR}
    icenet_sipnsouth_diagnostics ../icenet-pipeline fc.${YEAR}-11-30_south ${YEAR}-11-30 -d ${DIAGNOSTICS} -m ${METHOD} -fl ${FORECAST_LEADTIME}
    #icenet_sipnsouth_diagnostics ../icenet-pipeline fc.${YEAR}-11-30_south ${YEAR}-11-30 -d ${DIAGNOSTICS} -p -m ${METHOD} -fl ${FORECAST_LEADTIME}
done
