import xarray as xr
import os
import pandas as pd
import glob

data_dir_vh=r'c:\Workspace\Data\GEE\test2_WUR_GEE_autoDownload\S1_VH_GEE'
data_dir_vv=r'c:\Workspace\Data\GEE\test2_WUR_GEE_autoDownload\S1_VV_GEE'
os.chdir(data_dir_vh)
os.getcwd()

# -----------------------------------------------------------------------
# import Hansen data (Global Forest Change - gfc) from GEE:
# -----------------------------------------------------------------------
gfc_filename = r'fcl_2018_Hensen_roi_tapajos_utm21S.tif'
da_gfc = xr.open_rasterio(gfc_filename)
da_gfc.coords["band"] = [label_1 for label_1 in da_gfc.descriptions]
da_loss = da_gfc.sel(band='lossyear', drop=True)

# -----------------------------------------------------------------------
# import S1-VH data:
# -----------------------------------------------------------------------
fileNames = glob.glob(os.path.join(data_dir_vh,"S1A_IW_*.tif"))
list_of_da_vh = list()
for filename in fileNames:
    aux_da = xr.open_rasterio(filename).rename({'band': 'time'})
    aux_da.coords['time'] = [pd.Timestamp(os.path.basename(filename).split("_")[4][0:8]).to_datetime64()]
    list_of_da_vh.append(aux_da)

da_vh = xr.concat(list_of_da_vh, dim='time')
ds_vh = da_vh.to_dataset(name="VH")
# -----------------------------------------------------------------------
# import S1-VV data:
# -----------------------------------------------------------------------
fileNames_vv = glob.glob(os.path.join(data_dir_vv,"S1A_IW_*.tif"))
list_of_da_vv = list()
for filename in fileNames_vv:
    aux_da = xr.open_rasterio(filename).rename({'band': 'time'})
    aux_da.coords['time'] = [pd.Timestamp(os.path.basename(filename).split("_")[4][0:8]).to_datetime64()]
    list_of_da_vv.append(aux_da)

da_vv = xr.concat(list_of_da_vv, dim='time')
ds_vv = da_vv.to_dataset(name="VV")
# -----------------------------------------------------------------------
# combine the datasets:
# -----------------------------------------------------------------------
# as the time dimension is different, the two ds-s are aligned to the intersection of ds-s time stamps (join='inner')
ds_align = xr.align(ds_vh, ds_vv, join='inner')
# convert tuple of aligned ds-s ino a single ds:
ds = xr.merge(ds_align)
# -----------------------------
# calculate the cross ratio (CR)
ds = ds.assign(CR=ds.VH / ds.VV)
# -----------------------------
# calculate the Radar Vegetation Index (RVI)
ds = ds.assign(RVI=(4*pow(10,ds.VH/10.))/(pow(10,ds.VV/10)+pow(10,ds.VV/10)))
# -----------------------------------------------------------------------
# add Hansen lossyear raster as a coordinate in the ds:
# -----------------------------------------------------------------------
ds.coords['LossYear'] = (('y', 'x'), da_loss.values)
# -----------------------------------------------------------------------
# save the ds as netCDF file
# -----------------------------------------------------------------------
ds.to_netcdf(os.path.join(data_dir_vh,"aoi1_S1_as_xarray_ds.nc"),mode='w')
