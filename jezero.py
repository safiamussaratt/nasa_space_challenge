import json
import numpy as np
import rasterio

URL = "https://planetarymaps.usgs.gov/mosaic/Mars/HRSC_MOLA_Blend/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif"
R = 3396190.0  # Mars radius (m)

# Box around Jezero, in degrees (lon is -180..180, so 77E is just +77)
LAT_MIN, LAT_MAX = 17.2, 19.6
LON_MIN, LON_MAX = 76.3, 78.7

STEP = 2  

opts = {
    "GDAL_HTTP_VERSION": "1.1",
    "GDAL_HTTP_USERAGENT": "Mozilla/5.0",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "2",
}

with rasterio.Env(**opts):
    with rasterio.open("/vsicurl/" + URL) as ds:
        win = ds.window(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
        win = win.round_offsets().round_lengths()
        elev = ds.read(1, window=win).astype("float32")
        wt = ds.window_transform(win)
        nodata = ds.nodata
        deg = ds.res[0]

if nodata is not None:
    elev[elev == nodata] = np.nan

rows, cols = elev.shape
# true bounds of what we actually read
lon_min = wt.c
lat_max = wt.f
lon_max = lon_min + cols * deg
lat_min = lat_max - rows * deg

# cell size in meters (north-south is constant, east-west shrinks with latitude)
dy = np.radians(deg) * R
lats = lat_max - (np.arange(rows) + 0.5) * deg
dx = dy * np.cos(np.radians(lats))

print("shape:", elev.shape)
print("elevation min/max (m):", np.nanmin(elev), np.nanmax(elev))
print("cell size ~", round(dy), "m")

dz_dx = np.gradient(elev, axis=1) / (2 * dx[:, None]) * 2  # per-cell difference / spacing
dz_dy = -np.gradient(elev, axis=0) / dy                    # row 0 is north, so flip sign
slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
slope = np.nan_to_num(slope, nan=90.0)  # treat missing data as impassable
elev_out = np.nan_to_num(elev, nan=0.0)

elev_out = elev_out[::STEP, ::STEP]
slope = slope[::STEP, ::STEP]

out = {
    "bbox": {"lat_min": float(lat_min), "lat_max": float(lat_max),
             "lon_min": float(lon_min), "lon_max": float(lon_max)},
    "rows": int(elev_out.shape[0]),
    "cols": int(elev_out.shape[1]),
    "cell_size_m": float(dy * STEP),
    "row_order": "north_to_south",
    "elevation_m": np.round(elev_out).astype(int).tolist(),
    "slope_deg": np.round(slope, 1).tolist(),
}
with open("jezero_grid.json", "w") as f:
    json.dump(out, f)
print("saved jezero_grid.json")