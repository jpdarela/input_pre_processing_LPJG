# Preprocessing tool

Transform netCDF4 files (dimensions and attributes) from **ISIMIP** format (gridded climate time series in rectilinear latitude–longitude format) to a **station-based observational layout** to use with LPJ-GUESS.

It only works with ISIMIP input files: [ISIMIP Repository](https://data.isimip.org/search/tree/ISIMIP2a/InputData/climate/atmosphere/watch-wfdei/)

TODO: add functionality for other input datasets.

Look at the scripts [preprocess.bat](./preprocess.bat) and [preprocess.sh](./preprocess.sh)

USE:

`$ python ./prepocess_lpjginput.py <region> [optional <plot>]`

  `<region> : sa | af | as | eu`

  `<plot> : 0 | 1`

sa = South America

af = Central Africa

as = Southeast Asia

eu = Europe

The plot option uses matplolib imshow to plot one layer of the original data (To check the extent)

There is also the global option. It requires a lot of memory to run.

## Outputs

NetCD4 files are named as:
<variable_name>_`<region>`_1979_2016_watch-wfdei.nc

Gridlist files are named as:
gridlist_`<region>`.txt

WHERE:

* the attribute "standard_name" for the variables match LPJ-GUESS requirements
* use the variable names listed below to edit the insfile

<variable_name> can be one of:

    hurs - relative_humidity (%)

    pr - precipitation flux (kg m-2 s-1)

    ps - surface_pressure (Pa)

    rsds - shortwave downwelling radiation (W m-2)

    tas - air surface temperature (K)

    wind - (m s-1)

    vpd - (kPa)

`<region>` can be one of:

    sa - Northern South America

    af - Central Africa

    as - Southeastern Asia

    eu - Europe
