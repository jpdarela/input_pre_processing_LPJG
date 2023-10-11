from pathlib import Path
from sys import argv
import os
import platform


from copy import deepcopy
from netCDF4 import Dataset
from numpy import arange, ma, zeros, full, hstack, concatenate, meshgrid, exp, array
from numpy import float32 as flt


"""
    Subsets ISI-MIP2 input files & Generate netCDF files and gridlists
    for LPJ-GUESS experiments. For the bbox selection look at the definition of
    upper left corner and lower right corner: goto LINE 105

USE:

$ python ./prepocess_lpjginput.py <region> [optional <plot>]
  <region> : sa | af | as | eu
  <plot> : 0 | 1
  Look the README.txt
"""

FILE_EXT = "nc4"
ROOT = Path(os.getcwd())
WATCH_WFDEI_DATA = Path("ISIMIP2a")
REFERENCES = u"WFDEI Forcing Data converted for ISI-MIP2 (isi-mip@pik-potsdam.de) - 1979-2016. Adapted to LPJ-GUESS"

guess_var_mtdt= {'tas':  ["K", 'air_temperature'],
                 'rsds': ['W m-2', "surface_downwelling_shortwave_flux_in_air"],
                 'vpd':  ["kPa", "vpd"],
                 'ps': ["Pa", "surface_air_pressure"],
                 'pr': ["kg m-2 s-1", "precipitation_flux"],
                 'wind': ["m s-1", "wind_speed"], # Aparently the same
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

def Vsat_slope(Tair:array,method=3)->array:
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
    Esat = a * exp((b * tair_degc) / (c + tair_degc))
    Esat = Esat * Pa2kPa

  # slope of the saturation vapor pressure curve
    Delta = a * (exp((b * tair_degc)/(c + tair_degc)) * (b/(c + tair_degc) - (b * tair_degc)/(c + tair_degc)**2))
    Delta = Delta * Pa2kPa

    return Esat, Delta

def VPD(Tair:array, RH:array)->array:
    """Estimate vpd from Tair (K) and RH (%)"""
    svp = Vsat_slope(Tair)[0] # (kPa)
    avp = svp * (RH/100.0)

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
        # print(f"Get metadata for {var}")
        with Dataset(dset, 'r') as fh:
            tm = fh.variables[var]
            metadata["std_name"] = tm.standard_name
            metadata["long_name"] = tm.long_name
            metadata["units"] = tm.units
            metadata["fill_value"] = tm._FillValue
            metadata["missing_value"] = tm.missing_value
    return metadata

def list_files(path:Path, extension:str)->list:
    """ """
    os.chdir(path)
    files = sorted([Path(f) for f in os.listdir('.') if f.split('.')[-1] == extension])
    os.chdir(ROOT)
    files = [Path(os.path.join(path, f)).resolve() for f in files]
    return files

def read_data(dset:Path, var:str)->ma.MaskedArray:
    with Dataset(dset, 'r') as fh:
        # yield data
        variable = fh.variables[var]
        tsize = variable.get_dims()[0].size
        for i in range(tsize):
            yield variable[i, y0:y1, x0:x1]

def crop_rect(var_dir:Path, var:str):

    time_data = []
    variable_data = []
    dt = list_files(var_dir, FILE_EXT)

    mtdt = get_metadata(dt[0], var)
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

def process_var(var_dir:Path, var:str):
    """From the set of input ISI-MIP files, create the elements for the final datasets.
    Arguments: var_dir -> path to the folder containing the nc4 files of the variable var)
    """
    # It means crop rectangle but do other things also
    dt, mt, tm, crs = crop_rect(var_dir, var)
    tdim = dt.shape[0] # get time dimension length
    # the standard fill value of the watch+wfdei from ISI-MIP2 is 1e+20
    # Note that the creation of this mask need to be correct for the lat lon merging process
    mask = dt[0,:,:] > 9.9e19
    # station dimension for the input files
    station_dim = (mask == False).sum()

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

            # Filter land grid points
            if not mask[i, j]:
                lat = yv[i, j]
                lon = xv[i, j]
                ny, nx = find_coord(lat, lon)
                lats_idx[counter] = lat
                lons_idx[counter] = lon
                station_names[counter] = f"station_{ny}-{nx}"
                out[counter, :] = dt[:, i, j]
                counter += 1

    return lats_idx, lons_idx, station_names, out, mt, tm, crs

def write_gridlist(fname, lat, lon , name):
    # When writing from windows the line ending is \n (Automatic trasform to \r\n)
    endline = "\r\n"
    if platform.system() == "Linux":
        pass
    elif platform.system() == "Windows":
        endline = "\n"
    fnm = "_".join(fname.split("_")[1:])
    # print(platform.system())
    with open(f"gridlist_{fnm}.txt", 'w', encoding="utf-8") as fh:
        for i in range(name.size):
            nm = name[i].strip()
            fh.write(f"{str(round(lon[i], 2))}\t{str(round(lat[i], 2))}\t{nm}{endline}")

def write_files(fname = None,
               var=None,
               arr=None,
               metadata=None,
               time=None,
               crs=None,
               lat=None,
               lon=None,
               names=None,
               reference=None):

    if fname is not None:
        dset = Dataset(os.path.join(Path('./'), fname + "_1979_2016_watch-wfdei.nc"), mode="w", format="NETCDF4")


    # Create netCDF dimensions
    dset.createDimension("station",size=arr.shape[0])
    dset.createDimension("time",size=time["ndays"])

    # Data description
    long_name = metadata["long_name"]
    dset.description = f"WATCH+WFDEI for LPJ-GUESS - {long_name}"
    dset.reference = reference
    dset.featureType = "timeSeries"

    # Create netCDF variables
    S = dset.createVariable("station", 'i4', ("station",))
    X  = dset.createVariable("lon", 'f4', ("station",), fill_value=metadata["missing_value"])
    Y =  dset.createVariable("lat", 'f4', ("station",), fill_value=metadata["missing_value"])
    SN = dset.createVariable("station_name", '<U6', ("station", ),fill_value="station_y-x")
    T  = dset.createVariable("time", 'f4', ("time",), fill_value=metadata["missing_value"])
    D  = dset.createVariable(var, 'f4', ("station", "time"), fill_value=1e+20)

    S[...] = arange(arr.shape[0], dtype="i4")
    T[...] = time['time']
    T.units = time['units']
    T.calendar = time["calendar"]
    T.standard_name = "time"
    T.axis = time["axis"]

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

def make_vpd(lat, lon, names, metadata, tm, crs):

    ptas = Path(f"./tas_{reg}_1979_2016_watch-wfdei.nc").resolve()
    phurs  = Path(f"./hurs_{reg}_1979_2016_watch-wfdei.nc").resolve()


    with Dataset(ptas,  "r")  as tasfh, Dataset(phurs, "r") as hursfh:
        tas_arr = tasfh.variables["tas"][...].data # K
        hurs_arr = hursfh.variables["hurs"][...].data # %

    vpd = VPD(tas_arr, hurs_arr)

    mtdt = deepcopy(metadata)
    mtdt["units"] = guess_var_mtdt["vpd"][0]
    mtdt["std_name"] = guess_var_mtdt["vpd"][1]
    mtdt["long_name"] = "Vapor Pressure Deficit"

    # call write NC
    write_files(fname=f"vpd_{reg}", var="vpd", arr=vpd, metadata=mtdt,
                time=tm, crs=crs, lat=lat, lon=lon,
                names=names, reference=REFERENCES)
    return None

def main():
    folder_data = ROOT/WATCH_WFDEI_DATA
    GRDFLAG = False
    vrs = [Path(var) for var in os.listdir(path=folder_data) if (folder_data/Path(var)).is_dir()]
    vrstr = [str(v) for v in vrs]
    vrstr = rename_rhs(vrstr)

    enable = ["hurs"]#, "pr", "rsds", "ps", "tas", "wind", "vpd"]

    # print(vrstr)
    vrs = [folder_data/Path(v) for v in vrs]
    # print(vrs)
    if plot:
        import matplotlib.pyplot as plt
        dt = list_files(vrstr[0], FILE_EXT)
        ds = read_data(dt[0], vrstr[0])
        a = ds.__next__()

        plt.imshow(a)
        plt.show()

    for path, var in zip(vrs, vrstr):

        if var in enable:
            # print(var, path)
            lat, lon, names, arr, metadata, tm, crs = process_var(path, var)
            grdlst_data = write_files(f"{var}_{reg}", var, arr, metadata,
                                        tm, crs, lat, lon, names, REFERENCES)

            if not GRDFLAG:
                GRDFLAG = True
                write_gridlist(*grdlst_data)
        else: pass

    # Finally: write VPD. The order is important here as make_vpd uses items
    # from the preceding loop to process data.
    # The implementation here is dependent on the processing of at least one of the main variables.
    if "vpd" in enable and len(enable) > 1:
        make_vpd(lat, lon, names, metadata, tm, crs)

if __name__ == "__main__":
    main()
