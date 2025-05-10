import os
import numpy as np
import rasterio
from rasterio.features import geometry_mask
import geopandas as gpd
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

# ========= CONFIGURATION =========
folder = r'D:\cal_LST\LC08_L1TP_120038_20150902_20200908_02_T1'
shp_path = r'D:\cal_LST\NJ_SHP_new\nanjing.shp'
use_clip = True

# ========= HELPER FUNCTIONS =========
def find_file(folder, key):
    files = os.listdir(folder)
    return os.path.join(folder, [f for f in files if key in f][0])

def read_band(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(float), src.profile

def get_mtl_value(lines, key):
    for line in lines:
        if key in line:
            return float(line.strip().split('=')[-1])
    return None

def depsilon(Pv):
    if Pv < 0.4:
        return 0.0038 * Pv
    elif Pv > 0.4:
        return 0.0038 * (1 - Pv)
    else:
        return 0.0019

def calculate_emissivity(Pv):
    epsilon = Pv * 0.9867 + (1 - Pv) * 0.9648 + depsilon(Pv)
    return epsilon

# ========= MAIN PROCESS =========
def calculate_LST(folder, shp_path, use_clip=True):
    # Locate all required files
    b10_path = find_file(folder, '_B10.TIF')
    b3_path  = find_file(folder, '_B3.TIF')
    b4_path  = find_file(folder, '_B4.TIF')
    b5_path  = find_file(folder, '_B5.TIF')
    b6_path  = find_file(folder, '_B6.TIF')
    mtl_path = find_file(folder, '_MTL.txt')

    # Read bands
    b10_dn, profile = read_band(b10_path)
    b3, _ = read_band(b3_path)
    b4, _ = read_band(b4_path)
    b5, _ = read_band(b5_path)
    b6, _ = read_band(b6_path)

    # Read metadata
    with open(mtl_path) as f:
        mtl_lines = f.readlines()
    ML = get_mtl_value(mtl_lines, 'RADIANCE_MULT_BAND_10')
    AL = get_mtl_value(mtl_lines, 'RADIANCE_ADD_BAND_10')
    K1 = get_mtl_value(mtl_lines, 'K1_CONSTANT_BAND_10')
    K2 = get_mtl_value(mtl_lines, 'K2_CONSTANT_BAND_10')

    # Step 1: Brightness Temperature
    L = ML * b10_dn + AL
    Tb = K2 / (np.log((K1 / L) + 1))  # Kelvin

    # NDVI and Pv
    ndvi = (b5 - b4) / (b5 + b4 + 1e-10)
    valid_ndvi = ndvi[(ndvi >= 0) & (ndvi <= 1)]
    ndvi_min, ndvi_max = np.percentile(valid_ndvi, [5, 95])
    Pv = (ndvi - ndvi_min) / (ndvi_max - ndvi_min)
    Pv = np.clip(Pv, 0, 1)

    # Emissivity calculation
    vectorized_emissivity = np.vectorize(calculate_emissivity)
    epsilon = vectorized_emissivity(Pv)

    # Water mask using MNDWI and Otsu threshold
    mndwi = (b3 - b6) / (b3 + b6 + 1e-10)
    mndwi_flat = mndwi[~np.isnan(mndwi)].ravel()
    otsu_thresh = threshold_otsu(mndwi_flat)
    water_mask = mndwi > otsu_thresh
    epsilon = np.where(water_mask, 0.99683, epsilon)

    # Step 2: Single Channel LST
    a = -62.7182
    b = 0.4339
    w = 2.580568  # water vapor
    t = -0.1134 * w + 1.0035  # atmospheric transmittance
    C = t * epsilon
    D = (1 - t) * (1 + (1 - epsilon) * t)
    T0 = 22.5 + 273.15
    Ta = 16.0110 + 0.92621 * T0
    Ts = ((a * (1 - C - D) + (b * (1 - C - D) + C + D) * Tb - D * Ta) / C) - 273.15

    # Optional: Clip to shapefile area
    if use_clip:
        gdf = gpd.read_file(shp_path).to_crs(profile['crs'])
        geometry = [geom.__geo_interface__ for geom in gdf.geometry]
        with rasterio.open(b10_path) as src:
            transform = src.transform
            mask_arr = geometry_mask(geometry, transform=transform, invert=True, out_shape=Ts.shape)
        Ts = np.where(mask_arr, Ts, np.nan)
        ndvi = np.where(mask_arr, ndvi, np.nan)

    # Save to GeoTIFF
    out_path = os.path.join(folder, 'LST_output.tif')
    nodata_val = -9999
    Ts_filled = np.where(np.isnan(Ts), nodata_val, Ts)
    profile.update(dtype=rasterio.float32, count=1, nodata=nodata_val)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(Ts_filled.astype(np.float32), 1)

    # Print stats
    print("NDVI min/max:", np.nanmin(ndvi), np.nanmax(ndvi))
    print("Otsu Threshold (MNDWI):", otsu_thresh)
    print("LST min/max/mean (°C):", np.nanmin(Ts), np.nanmax(Ts), np.nanmean(Ts))
    print("Vegetation Coverage Pv min/max/mean:", np.nanmin(Pv), np.nanmax(Pv), np.nanmean(Pv))
    print("Saved to:", out_path)

    # Optional plot
    plt.imshow(Ts, cmap='hot', origin='upper')
    plt.title('Land Surface Temperature (°C)')
    plt.colorbar(label='Ts (°C)')
    plt.show()

if __name__ == "__main__":
    calculate_LST(folder, shp_path, use_clip=True)
