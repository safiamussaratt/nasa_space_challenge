import rasterio

URL = "https://planetarymaps.usgs.gov/mosaic/Mars/HRSC_MOLA_Blend/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif"

opts = {
    "GDAL_HTTP_VERSION": "1.1",              # avoid HTTP/2 problems
    "GDAL_HTTP_USERAGENT": "Mozilla/5.0",    # some servers reject the default agent
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "2",
}

with rasterio.Env(**opts):
    with rasterio.open("/vsicurl/" + URL) as ds:
        print("size:", ds.width, ds.height)
        print("bounds:", ds.bounds)
        print("pixel size:", ds.res)
        print("dtype:", ds.dtypes[0], "nodata:", ds.nodata)
        print("crs:", ds.crs)