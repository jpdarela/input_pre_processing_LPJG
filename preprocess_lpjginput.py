from pathlib import Path
from sys import argv
import os
import platform


from copy import deepcopy
from netCDF4 import Dataset
from numpy import ma, arange, full, hstack, concatenate, meshgrid, exp, zeros, array
from numpy import float32 as flt
import numpy as np


"""
    Subsets ISIMIP netCDF4 input files & Generate netCDF files with
    station-based observational layout (time,gridcell) with chunking (time, 1) and gridlists
    for LPJ-GUESS experiments. For the bbox selection look at the definition of
    upper left corner and lower right corner: goto LINE 105

USE:

$ python ./prepocess_lpjginput.py <region> [optional <plot>]
  <region> : sa | af | as | eu
  <plot> : 0 | 1
  Look the README.txt
"""
# Folders containing the ISIMIP netCDF data files (all variables in the same folder)
# This script does not check for data consistency in the time dimension.
# It assumes that all files are both present and fit to merge in time without gaps or overlaps.

# ISIMIP_DATA_PATH = Path("../../CAETE-DVM/input/20CRv3-ERA5/spinclim_raw/")
# DATASET = "20CRv3-ERA5_spinclim"
# FILE_EXT = "nc"
# TSPAN = "1801-1900"

# ISIMIP_DATA_PATH = Path("../../CAETE-DVM/input/20CRv3-ERA5/obsclim_raw/")
# DATASET = "20CRv3-ERA5_obsclim"
# FILE_EXT = "nc"
# TSPAN = "1901-2024"

# ISIMIP_DATA_PATH = Path("../../CAETE-DVM/input/MPI-ESM1-2-HR/historical_raw/")
# DATASET = "MPI-ESM1-2-HR_historical"
# FILE_EXT = "nc"
# TSPAN = "1851-2014"

# ISIMIP_DATA_PATH = Path("../../CAETE-DVM/input/MPI-ESM1-2-HR/ssp370_raw/")
# DATASET = "MPI-ESM1-2-HR_ssp370"
# FILE_EXT = "nc"
# TSPAN = "2015-2100"

ISIMIP_DATA_PATH = Path("../../CAETE-DVM/input/MPI-ESM1-2-HR/ssp585_raw/")
DATASET = "MPI-ESM1-2-HR_ssp585"
FILE_EXT = "nc"
TSPAN = "2015-2100"

# dataset name to append to output files
REFERENCES = f"{DATASET} (isi-mip@pik-potsdam.de) - {TSPAN}. Adapted to LPJ-GUESS"
ROOT = Path(os.getcwd())

# Path to your land mask file (360x720 array)
LAND_MASK_FILE = Path("./mask_raisg-360-720.npy")  # Update this path
USE_EXTERNAL_MASK = True  # Set to True to use external mask



guess_var_mtdt= {'tas':  ["K", 'air_temperature'],
                 'rsds': ['W m-2', "surface_downwelling_shortwave_flux_in_air"],
                 'vpd':  ["kPa", "vpd"],
                 'ps': ["Pa", "surface_air_pressure"],
                 'pr': ["kg m-2 s-1", "precipitation_flux"],
                 'sfcwind': ["m s-1", "wind_speed"], # Aparently the same
                 'hurs': ["%", 'relative_humidity']}

try:
    reg = argv[1]
except:
    reg = "sa"

try:
    plot = argv[2]
except:
    plot = 0

def rename_rhs(lst):
    cp = deepcopy(lst)
    new_name = "hurs"
    idx = 0
    for i, var in enumerate(lst):
        if var == "rhs":
            idx = i
            break

    cp[idx] = new_name
    return cp

def find_coord(N:float, W:float, RES:float=0.5) -> tuple[int, int]:
    """

    :param N:float: latitude in decimal degrees
    :param W:float: Longitude in decimal degrees
    :param RES:float: Resolution in degrees (Default value = 0.5)

    """

    Yc = round(N, 2)
    Xc = round(W, 2)

    Ymax = 90 - RES/2
    Ymin = Ymax * (-1)
    Xmax = 180 - RES/2
    Xmin = Xmax * (-1)

    # snap --- hook invalid values to the borders
    if abs(Yc) > Ymax:
        if Yc < 0:
            Yc = Ymin
        else:
            Yc = Ymax

    if abs(Xc) > Xmax:
        if Xc < 0:
            Xc = Xmin
        else:
            Xc = Xmax

    Yind = 0
    Xind = 0

    lon = arange(Xmin, 180, RES)
    lat = arange(Ymax, -90, RES * (-1))

    while Yc < lat[Yind]:
        Yind += 1

    if Xc <= 0:
        while Xc > lon[Xind]:
            Xind += 1
    else:
        Xind += lon.size // 2
        while Xc > lon[Xind]:
            Xind += 1

    return Yind, Xind

# define the bounding box to extraction
def calc_dim(y0, y1, x0, x1):
    return len(range(y0, y1)), len(range(x0, x1))

# find_coord signature: (Latitude, Longitude) [in decimal degrees, GRS - Lat/Long] ---LOOK the README.txt file
if reg == "sa":
    y0, x0 = find_coord(12, -86) # upper left corner
    y1, x1 = find_coord(-20, -35) # lower right corner
    ys, xs = calc_dim(y0, y1, x0, x1)
elif reg == "af":
    y0, x0 = find_coord(12, -20) # upper left corner
    y1, x1 = find_coord(-20, 52) # lower right corner
    ys, xs = calc_dim(y0, y1, x0, x1)
elif reg == "as":
    y0, x0 = find_coord(12, 90) # upper left corner
    y1, x1 = find_coord(-20, 155) # lower right corner
    ys, xs = calc_dim(y0, y1, x0, x1)
elif reg == "eu":
    y0, x0 = find_coord(72, -12) # upper left corner
    y1, x1 = find_coord(33, 33) # lower right corner
    ys, xs = calc_dim(y0, y1, x0, x1)
elif reg == "global":
    y0, x0 = find_coord(90, -180) # upper left corner
    y1, x1 = find_coord(-90, 180) # lower right corner
    ys, xs = calc_dim(y0, y1, x0, x1)
else:
    # sa
    y0, x0 = find_coord(12, -86) # upper left corner
    y1, x1 = find_coord(-20, -35) # lower right corner
    ys, xs = calc_dim(y0, y1, x0, x1)
    print(f"WTF is {reg}?")
    reg = "sa"

# TODO: Test the match case structure (Python 3.10+)
# # find_coord signature: (Latitude, Longitude) [in decimal degrees, GRS - Lat/Long] ---LOOK the README.txt file
# match reg:
#     case "sa":
#         y0, x0 = find_coord(12, -86)  # upper left corner
#         y1, x1 = find_coord(-20, -35)  # lower right corner
#         ys, xs = calc_dim(y0, y1, x0, x1)
#     case "af":
#         y0, x0 = find_coord(12, -20)  # upper left corner
#         y1, x1 = find_coord(-20, 52)  # lower right corner
#         ys, xs = calc_dim(y0, y1, x0, x1)
#     case "as":
#         y0, x0 = find_coord(12, 90)  # upper left corner
#         y1, x1 = find_coord(-20, 155)  # lower right corner
#         ys, xs = calc_dim(y0, y1, x0, x1)
#     case "eu":
#         y0, x0 = find_coord(72, -12)  # upper left corner
#         y1, x1 = find_coord(33, 33)  # lower right corner
#         ys, xs = calc_dim(y0, y1, x0, x1)
#     case "global":
#         y0, x0 = find_coord(90, -180)  # upper left corner
#         y1, x1 = find_coord(-90, 180)  # lower right corner
#         ys, xs = calc_dim(y0, y1, x0, x1)
#     case _:
#         # Default case (equivalent to else)
#         y0, x0 = find_coord(12, -86)  # upper left corner
#         y1, x1 = find_coord(-20, -35)  # lower right corner
#         ys, xs = calc_dim(y0, y1, x0, x1)
#         print(f"WTF is {reg}?")
#         reg = "sa"

def Vsat_slope(Tair:array,method=3, esat=True)->array:
# Translated to python from the bigleaf R package

#' Saturation Vapor Pressure (Esat) and Slope of the Esat Curve
#'
#' @references Sonntag D. 1990: Important new values of the physical constants of 1986, vapor
#'             pressure formulations based on the ITS-90 and psychrometric formulae.
#'             Zeitschrift fuer Meteorologie 70, 340-344.
#'
#'             World Meteorological Organization 2008: Guide to Meteorological Instruments
#'             and Methods of Observation (WMO-No.8). World Meteorological Organization,
#'             Geneva. 7th Edition.
#'
#'             Alduchov, O. A. & Eskridge, R. E., 1996: Improved Magnus form approximation of
#'             saturation vapor pressure. Journal of Applied Meteorology, 35, 601-609
#'
#'             Allen, R.G., Pereira, L.S., Raes, D., Smith, M., 1998: Crop evapotranspiration -
#'             Guidelines for computing crop water requirements - FAO irrigation and drainage
#'             paper 56, FAO, Rome.

    """ RESULTS IN kPa """

    methods = ("Sonntag_1990","Alduchov_1996","Allen_1998")
    assert method <= 3 or method >= 1, "Methods:\n1 - Sonntag_1990\n2 - Alduchov_1996\n3 - Allen_1998"
    formula = methods[method - 1]
    # print(f"Saturation Vapor Pressure (Esat) and Slope of the Esat Curve using the formula: {formula}")

    if formula == "Sonntag_1990":
        a = 611.2
        b = 17.62
        c = 243.12
    elif (formula == "Alduchov_1996"):
        a = 610.94
        b = 17.625
        c = 243.04
    elif (formula == "Allen_1998"):
        a = 610.8
        b = 17.27
        c = 237.3

    tair_degc = Tair - 273.15
    Pa2kPa = 1e-3

  # saturation vapor pressure
    if esat:

        Esat = a * exp((b * tair_degc) / (c + tair_degc))
        return Esat * Pa2kPa

    # slope of the saturation vapor pressure curve
    else:
        Delta = a * (exp((b * tair_degc)/(c + tair_degc)) * (b/(c + tair_degc) - (b * tair_degc)/(c + tair_degc)**2))
        return Delta * Pa2kPa

def VPD(Tair:array, RH:array)->array:
    """Estimate vpd from Tair (K) and RH (%)"""
    svp = Vsat_slope(Tair) # (kPa)
    avp = svp * (RH/100.0)
    # invert the signal to fit LPJG purpose
    return -(svp - avp)

def get_nctime(dset:Path=None):
    # time management
    TIME = {}
    if dset is not None:
        # print(f"Set time for {dset}")
        with Dataset(dset, 'r') as fh:
            tm = fh.variables["time"]
            array = tm[...]
            array.data[array.mask] = array.fill_value
            TIME["time"] = array.data
            TIME["ndays"] = TIME["time"].size
            TIME["fill_value"] = array.fill_value
            TIME["calendar"] = tm.calendar
            TIME["units"] = tm.units
            TIME["axis"] = u"T"
    return TIME

def get_crs(dset:Path=None):
    crs = {}
    if dset is not None:
        # print(f"Get crs for {var}")
        with Dataset(dset, 'r') as fh:
            lat = fh.variables["lat"]
            lon = fh.variables["lon"]

            crs["lat"] = lat[:].data[y0: y1]
            crs["lon"] = lon[:].data[x0: x1]
            crs["lat_long_name"] = lat.long_name
            crs["lon_long_name"] = lon.long_name
            crs["lat_std_name"] = lat.standard_name
            crs["lon_std_name"] = lon.standard_name

            crs["lat_units"] = lat.units
            crs["lon_units"] = lon.units
            crs["lat_axis"] = u"Y"
            crs["lon_axis"] = u"X"
    return crs

def get_metadata(dset:Path, var:str):
    metadata = {}
    if dset is not None:
        print(f"Get metadata for {var}")
        with Dataset(dset, 'r') as fh:
            tm = fh.variables[var]
            metadata["std_name"] = tm.standard_name
            metadata["long_name"] = tm.long_name
            metadata["units"] = tm.units
            metadata["fill_value"] = tm._FillValue
            metadata["missing_value"] = tm.missing_value
    return metadata

def list_files(path:Path, extension:str, var:str|None = None)->list:
    """ """
    os.chdir(path)
    files = sorted([Path(f) for f in os.listdir('.') if f.split('.')[-1] == extension])
    os.chdir(ROOT)
    if var is not None:
        files = [Path(os.path.join(path, f)).resolve() for f in files if var in f.name]
        return files
    files = [Path(os.path.join(path, f)).resolve() for f in files]
    return files

def read_data(dset:Path, var:str)->ma.MaskedArray: # type: ignore
    with Dataset(dset, 'r') as fh:
        # yield data
        variable = fh.variables[var]
        tsize = variable.get_dims()[0].size
        for i in range(tsize):
            yield variable[i, y0:y1, x0:x1]

def load_mask(mask_file: Path, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    """
    Load external region mask and crop to the specified region

    Args:
        mask_file: Path to the .npy file containing mask (360x720)
        y0, y1, x0, x1: Bounding box indices for cropping

    Returns:
        Cropped mask where True = exclude, False = include (True for masked areas)
    """
    if not mask_file.exists():
        print(f"Warning: mask file {mask_file} not found. Using original masking method.")
        return None

    try:
        # Load the full mask (assuming 360x720: lat x lon)
        full_mask = np.load(mask_file)
        print(f"Loaded external mask with shape: {full_mask.shape}")

        # Crop to the specified region
        cropped_mask = full_mask[y0:y1, x0:x1]
        print(f"Cropped mask to region with shape: {cropped_mask.shape}")

        # Convert to boolean if needed
        if cropped_mask.dtype != bool:
            cropped_mask = cropped_mask.astype(bool)

        # Your mask: True = masked (exclude), False = unmasked (include Amazon)
        # Script expects: True = exclude, False = include
        # So we can use your mask directly without inversion
        return cropped_mask

    except Exception as e:
        print(f"Error loading Amazon mask: {e}")
        return None

def crop_rect(var_dir:Path, var:str):

    time_data = []
    variable_data = []
    dt = list_files(var_dir, FILE_EXT, "_" + var + "_")
    # return dt

    mtdt = get_metadata(dt[0], var)
    # return dt, mtdt
    crs = get_crs(dt[0])

    print(f"\n\tCollecting files at {dt[0].parent}", end="\n\n")
    for i, fpath in enumerate(dt):
        tm = get_nctime(fpath)
        time_data.append(tm["time"])
        data = zeros(shape=(tm["ndays"], ys, xs))
        print(f"File {i + 1}", fpath.name)
        ds = read_data(fpath, var)
        for j, layer in enumerate(ds):
            data[j,:,:] = layer
            print(f"\rExtracting layer {j + 1}", end="", flush=True)
        print("\r", end="", flush=True)
        variable_data.append(data)
    print("\rDONE", end=" "*100)
    print("")

    tm["time"] = hstack(tuple(time_data))
    tm["ndays"] = tm["time"].size

    return concatenate(tuple(variable_data), axis=0,), mtdt, tm, crs

def process_var(var_dir: Path, var: str):
    """From the set of input ISI-MIP files, create the elements for the final datasets.
    Arguments: var_dir -> path to the folder containing the nc4 files of the variable var)
    """
    # It means crop rectangle but do other things also
    dt, mt, tm, crs = crop_rect(var_dir, var)

    tdim = dt.shape[0]  # get time dimension length

    # Load external mask if available
    if USE_EXTERNAL_MASK and LAND_MASK_FILE.exists():
        valid_mask = load_mask(LAND_MASK_FILE, y0, y1, x0, x1)
        if valid_mask is not None:
            # Use mask directly (True = exclude, False = include areas)
            mask = valid_mask
            valid_grid_points = np.sum(~mask)  # Count of used areas (False values)
            excluded_points = np.sum(mask)  # Count of excluded areas (True values)
            print(f"Using external mask.  Points: {valid_grid_points}, Excluded points: {excluded_points}")
        else:
            # Fallback to original method
            mask = dt[0, :, :] > 9.9e19
            print("Using original fill-value based masking")
    else:
        # Original method: the standard fill value of the watch+wfdei from ISI-MIP2 is 1e+20
        mask = dt[0, :, :] > 9.9e19
        print("Using original fill-value based masking")

    # Note that the creation of this mask need to be correct for the lat lon merging process
    # station dimension for the input files (count points - where mask is False)
    station_dim = (mask == False).sum()
    print(f"Total stations: {station_dim}")

    # allocate some space
    lats_idx = zeros(shape=(station_dim), dtype=flt)
    lons_idx = zeros(shape=(station_dim), dtype=flt)
    station_names = full(shape=(station_dim), fill_value="station_y-x", dtype='<U15')
    # allocate an array for output
    out = zeros(shape=(station_dim, tdim), dtype=flt)

    # Merge lat and long dimensions to conform LJP input pattern
    yv, xv = meshgrid(crs["lat"], crs["lon"], indexing="ij")
    counter = 0
    for i in range(crs["lat"].size):
        for j in range(crs["lon"].size):

            # DO SOME PRINTING TO CHECK CORRECTNESS
            # lat = yv[i, j]
            # lon = xv[i, j]
            # ny, nx = find_coord(lat, lon)
            # print(i, j, ny - y0, nx - x0, mask[i, j], yv[i, j], xv[i, j])

            # Filter Amazon grid points (mask[i,j] = False means include Amazon area)
            if not mask[i, j]:
                lat = yv[i, j]
                lon = xv[i, j]
                ny, nx = find_coord(lat, lon)
                lats_idx[counter] = lat
                lons_idx[counter] = lon
                station_names[counter] = f"station_{ny}-{nx}"
                out[counter, :] = dt[:, i, j]
                counter += 1

    print(f"Processed {counter} stations")
    return lats_idx, lons_idx, station_names, out, mt, tm, crs

def write_gridlist(fname, lat, lon, name, cf=True):
    # When writing from windows the line ending is \n (Automatic trasform to \r\n)
    endline = "\r\n"
    if platform.system() == "Linux":
        pass
    elif platform.system() == "Windows":
        endline = "\n"

    if cf:
        fnm = "_".join(fname.split("_")[1:]) + "_CF"
    else:
        fnm = "_".join(fname.split("_")[1:]) + "_lonlat"

    with open(f"gridlist_{fnm}.txt", 'w', encoding="utf-8") as fh:
        for i in range(name.size):
            nm = name[i].strip()
            if not cf:
                fh.write(f"{str(round(lon[i], 2))}\t{str(round(lat[i], 2))}\t{nm}{endline}")
            else:
                fh.write(f"{i}\t{nm}{endline}")

def write_files(fname = None,
               var=None,
               arr_in=None,
               metadata=None,
               time=None,
               crs=None,
               lat=None,
               lon=None,
               names=None,
               reference=None):

    if fname is not None:
        dset = Dataset(os.path.join(Path('./'), fname + ".nc"), mode="w", format="NETCDF4")

    # Transpose array to have (time, station)
    # We want time as the first dimension to enable concatenation in time (The output files are not fit to CDO)
    arr = arr_in.T
    # Create netCDF dimensions
    dset.createDimension("station",size=arr.shape[1])
    # dset.createDimension("time",size=time["ndays"])
    dset.createDimension("time",size=None)

    # Data description
    long_name = metadata["long_name"]
    dset.description = f"ISIMIP3a {DATASET} for LPJ-GUESS - {long_name}"
    dset.reference = reference
    dset.featureType = "timeSeries"

    # Create netCDF variables

    S = dset.createVariable("station", 'i4', ("station",))
    X  = dset.createVariable("lon", 'f4', ("station",), fill_value=metadata["missing_value"])
    Y =  dset.createVariable("lat", 'f4', ("station",), fill_value=metadata["missing_value"])
    SN = dset.createVariable("station_name", '<U6', ("station", ),fill_value="station_y-x")
    T  = dset.createVariable("time", 'f4', ("time",), fill_value=metadata["missing_value"])
    D  = dset.createVariable(var, 'f4', ("time", "station"), fill_value=1e+20, chunksizes=(time["ndays"], 1))

    # transpose to time, station. We want time as the first dimension to enable concatenation in time
    # Assign data to variables & add attributes

    # station index
    S[...] = arange(arr.shape[1], dtype="i4")

    # time variable
    T[...] = time['time']
    T.units = time['units']
    T.calendar = time["calendar"]
    T.standard_name = "time"
    T.axis = time["axis"]

    # lon & lat variables
    X[...] = lon
    X.units = crs["lon_units"]
    X.long_name = crs["lon_long_name"]
    X.standard_name = crs["lon_std_name"]
    X.axis = crs["lon_axis"]

    Y[...] = lat
    Y.units = crs["lat_units"]
    Y.long_name = crs["lat_long_name"]
    Y.standard_name = crs["lat_std_name"]
    Y.axis = crs["lat_axis"]

    # station names
    SN[...] = names
    SN.long_name = "Station-name_ny-nx"
    SN.cf_role = "timeseries_id"

    D[...] = arr
    D.units = guess_var_mtdt[var][0]
    D.standard_name = guess_var_mtdt[var][1] #metadata["std_name"]
    D.long_name = metadata["long_name"]
    D.missing_value = metadata["missing_value"]
    D.coordinates = u"lon lat"

    dset.close()

    return fname, lat, lon, names

def make_vpd():
    """Calculate VPD from existing tas and hurs files"""

    ptas = Path(f"./tas_{reg}_{DATASET}.nc").resolve()
    phurs = Path(f"./hurs_{reg}_{DATASET}.nc").resolve()

    # Check if required files exist
    if not ptas.exists():
        print(f"Error: {ptas} not found. Need to process 'tas' first.")
        return None
    if not phurs.exists():
        print(f"Error: {phurs} not found. Need to process 'hurs' first.")
        return None

    print("Calculating VPD from existing tas and hurs files...")

    with Dataset(ptas, "r") as tasfh, Dataset(phurs, "r") as hursfh:
        # Get all required data from the existing files
        tas_arr = tasfh.variables["tas"][...]  # Shape: (time, station)
        hurs_arr = hursfh.variables["hurs"][...]  # Shape: (time, station)

        # Extract spatial metadata from tas file (these should be 1D arrays)
        lat = tasfh.variables["lat"][...]  # Shape: (station,)
        lon = tasfh.variables["lon"][...]  # Shape: (station,)
        names = tasfh.variables["station_name"][...]  # Shape: (station,)

        # Extract time info
        time_var = tasfh.variables["time"]
        tm = {
            "time": time_var[...],
            "ndays": time_var.size,
            "units": time_var.units,
            "calendar": time_var.calendar,
            "axis": "T"
        }

        # Extract CRS info from variable attributes
        lat_var = tasfh.variables["lat"]
        lon_var = tasfh.variables["lon"]
        crs = {
            "lat_units": lat_var.units,
            "lat_long_name": lat_var.long_name,
            "lat_std_name": lat_var.standard_name,
            "lat_axis": "Y",
            "lon_units": lon_var.units,
            "lon_long_name": lon_var.long_name,
            "lon_std_name": lon_var.standard_name,
            "lon_axis": "X"
        }

        # Create metadata for VPD with proper data types
        metadata = {
            "units": guess_var_mtdt["vpd"][0],
            "std_name": guess_var_mtdt["vpd"][1],
            "long_name": "Vapor Pressure Deficit",
            "fill_value": flt(1e+20),  # Use float32 to match variable type
            "missing_value": flt(1e+20)  # Use float32 to match variable type
        }

    # Calculate VPD - tas_arr and hurs_arr should both be (time, station)
    vpd = VPD(tas_arr, hurs_arr)  # This should return (time, station)

    # Write VPD file - need to transpose vpd to (station, time) for write_files
    write_files(fname=f"vpd_{reg}_{DATASET}", var="vpd", arr_in=vpd.T, metadata=metadata,
                time=tm, crs=crs, lat=lat, lon=lon,
                names=names, reference=REFERENCES)

    print("VPD calculation completed.")
    return None

def main():
    folder_data = ISIMIP_DATA_PATH.resolve()
    GRDFLAG = False
    vrs = ["hurs", "pr", "rsds", "ps", "tas", "sfcwind"]
    enable = ["hurs", "pr", "rsds", "ps", "tas", "sfcwind", "vpd"]

    # Process main variables
    for var in vrs:
        print(var, end=" ")
        if var in enable:
            print("enable")
            lat, lon, names, arr, metadata, tm, crs = process_var(folder_data, var)
            grdlst_data = write_files(f"{var}_{reg}_{DATASET}", var, arr, metadata,
                                        tm, crs, lat, lon, names, REFERENCES)

            if not GRDFLAG:
                GRDFLAG = True
                write_gridlist(*grdlst_data)
                write_gridlist(*grdlst_data, cf=False)
        else:
            pass

    # Calculate VPD if requested
    if "vpd" in enable:
        make_vpd()


if __name__ == "__main__":
    main()
    pass
