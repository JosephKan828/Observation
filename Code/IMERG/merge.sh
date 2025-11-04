#!/bin/sh

# Merge IMERG file from 2006 to 2017

PATH_INPUT="/work/DATA/Satellite/IMERG_daily/"
PATH_OUTPUT="/work/b11209013/2025_Research/IMERG/"

# Collect files
files=()
for year in $(seq 2006 2017); do
    files+=($(ls ${PATH_INPUT}*${year}.nc))
done

# Apply CDO to merge files
cdo -P 8 -mergetime ${files[@]} ${PATH_OUTPUT}IMERG_06_17.nc

cp ${PATH_OUTPUT}IMERG_06_17.nc outfile
cdo -P 8 ydaymean outfile clim.nc
cdo -P 8 lowpass,3 clim.nc clim_lowpass.nc
cdo -P 8 ydaysub outfile clim_lowpass.nc ${PATH_OUTPUT}IMERG_06_17_lowpass.nc

rm outfile clim.nc clim_lowpass.nc