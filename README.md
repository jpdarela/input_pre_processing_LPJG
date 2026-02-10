# Preprocessing tool

Transform netCDF4 files (dimensions and attributes) from **ISIMIP** format (gridded climate time series in rectilinear latitude–longitude format) to a **station-based observational layout** to use with LPJ-GUESS.

Works with ISIMIP input files

Look at the scripts [preprocess.bat](./preprocess.bat) and [preprocess.sh](./preprocess.sh)

USE:

`$ python ./prepocess_lpjginput.py <region>`

  `<region> : sa | af | as | eu`

sa = South America

af = Central Africa

as = Southeast Asia

eu = Europe

There is also the global option. It requires a lot of memory to run.

## Outputs

NetCD4 files are named as:
<variable_name>_`<region>`_dataset_name.nc

Gridlist files are named as:
gridlist_`<region>`_dataset_name-gridlist_type.txt

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
