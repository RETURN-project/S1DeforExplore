import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import seaborn as sns
import os
import pandas as pd
from rasterio.plot import show
import seaborn as sns
import glob
import scipy
import math
from skimage import measure
import scipy.signal
import time

# ----------------------------------------------------------------------
# read the xarray data structure derived with 'collection2xarray_ds.py':
# ----------------------------------------------------------------------
data_dir=r'c:\Workspace\Data\GEE\test2_WUR_GEE_autoDownload\S1_VH_GEE'
os.chdir(data_dir)
os.getcwd()
#
ds = xr.open_dataset('aoi1_S1_as_xarray_ds.nc')
# --------------------------------------------------------
# plot a mean time series based on the selected loss year:
# --------------------------------------------------------
start_date = np.datetime64('2016-01-01')
end_date = np.datetime64('2020-01-01')
datelist = pd.date_range(start=start_date, end=end_date, freq='YS').tolist()
# plotting:
fig, axes = plt.subplots(1, 1,figsize=(14, 4),sharex=True, sharey=True)
ds.VH.where(ds.LossYear.values == 16).mean(dim=['x','y']).plot(ax=axes, color='red')
ds.VH.where(ds.LossYear.values == 17).mean(dim=['x','y']).plot(ax=axes, color='green')
ds.VH.where(ds.LossYear.values == 18).mean(dim=['x','y']).plot(ax=axes, color='blue')
axes.set_xticks(datelist)
axes.set_axisbelow(True)
axes.set_yticks(np.arange(-20, -9, 1))
axes.minorticks_on()
axes.grid(which='major', linestyle='-', linewidth=1)
axes.grid(which='minor', linestyle=':', linewidth=0.5)
axes.legend(['2016','2017','2018'],loc="lower right")
axes.set_title('Mean Time Series Derived from All Pixels of Particular "lossyear"')
plt.tight_layout()

# -----------------------------------------------------------------------
# the mean TS for several radar values (VV, VH, CR, RVI):
# -----------------------------------------------------------------------
#
vh_mean = ds.VH.where(ds.LossYear.values == 0).mean(dim=['x','y'])
vv_mean = ds.VV.where(ds.LossYear.values == 0).mean(dim=['x','y'])
#
cr_mean = ds.CR.where(ds.LossYear.values == 0).mean(dim=['x','y'])
rvi_mean = ds.RVI.where(ds.LossYear.values == 0).mean(dim=['x','y'])
#
# -----------------------------------------------------------------------
# other statistics:
# -----------------------------------------------------------------------
start = time.time()
vh_median = ds.VH.quantile(0.5, dim="time")
end = time.time()
print(end - start)
#
vh_q10 = ds.VH.quantile(0.1, dim="time")
vh_q90 = ds.VH.quantile(0.9, dim="time")
# --------------------------------------------------
# Get the reference TS from manually-defined polygon
# --------------------------------------------------
mean_all = ds.VH.mean(dim='time')

# -----------------------------------------------------------------------
# estimate the noise of the data:
# -----------------------------------------------------------------------
# get the da of non disturbed pixels
da_no_loss = ds.VH.where(ds.LossYear.values == 0)
# define the function to apply to ds:
def daily_res(x):
    return x - x.mean()

# calculate the residual
da_no_loss_res = da_no_loss.groupby('time').apply(daily_res)

pix_NO_loss = da_no_loss_res.values.flatten()
pix_NO_loss = pix_NO_loss[~np.isnan(pix_NO_loss)]

pix_NO_loss
pix_NO_loss.mean()
res_std = pix_NO_loss.std()


#  plotting histogram with seborn:
f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
# Add a graph in each part
sns.boxplot(pix_NO_loss, ax=ax_box)
sns.distplot(pix_NO_loss, ax=ax_hist)
# Remove x axis name for the boxplot
ax_hist.set(xlabel='Residuals [dB]')
ax_box.set(xlabel='')
ax_box.grid(linestyle=':', linewidth=0.5)
ax_hist.grid(linestyle=':', linewidth=0.5)
