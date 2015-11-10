#!/Users/nicolasf/anaconda3/bin/python
import sys

from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap as bm
from mpl_toolkits.basemap import maskoceans, interp

import numpy as np
import pandas as pd
import xray
from datetime import datetime, timedelta
from glob import glob
import palettable

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

dpath = '/Users/nicolasf/data/TRMM/daily/'
climpath = '/Users/nicolasf/data/TRMM/climatology/daily/'

# ### gets the cmap
cmap_anoms = palettable.colorbrewer.diverging.BrBG_11.mpl_colormap

### defines all the functions here
def read_netcdfs(files, dim, transform_func=None):
    def process_one_path(path):
        # use a context manager, to ensure the file gets closed after use
        with xray.open_dataset(path) as ds:
            # transform_func should do some sort of selection or
            # aggregation
            if transform_func is not None:
                ds = transform_func(ds)
            # load all data from the transformed dataset, to ensure we can
            # use it after closing each original file
            ds.load()
            return ds

    #paths = sorted(glob(files))
    datasets = [process_one_path(p) for p in files]
    combined = xray.concat(datasets, dim)
    return combined

def round_10(x, how='down'):
    import math
    if how == 'up':
        return int(math.ceil(x / 10.0)) * 10
    elif how == 'down':
        return int(math.floor(x / 10.0)) * 10

def plot_cities(ax, cities):
    for k in cities:
        ax.plot(cities[k][1],cities[k][0], 'ro', transform=ax.transData)
        ax.text(cities[k][1]+0.1,cities[k][0]+0.01, k, transform=ax.transData, fontsize=14, color='#0B0B61')

def get_limits(Dataarray, robust=True, robust_percentile=2, center=None):
        # ravel and removes nans for calculation of intervals etc
        calc_data = Dataarray.data
        calc_data = np.ravel(calc_data[np.isfinite(calc_data)])

        # the following is borrowed from xray
        # see: plot.py in xray/xray/plot
        vmin = np.percentile(calc_data, robust_percentile) if robust else calc_data.min()
        vmax = np.percentile(calc_data, 100 - robust_percentile) if robust else calc_data.max()

        del(calc_data)

        # Simple heuristics for whether these data should  have a divergent map
        divergent = ((vmin < 0) and (vmax > 0)) or center is not None

        # Now set center to 0 so math below makes sense
        if center is None:
            center = 0

        # A divergent map should be symmetric around the center value
        if divergent:
            vlim = max(abs(vmin - center), abs(vmax - center))
            vmin, vmax = -vlim, vlim

        # Now add in the centering value and set the limits
        vmin += center
        vmax += center

        vmin = round_10(vmin)
        if center == 0:
            vmax = round_10(vmax, how='up')
        else:
            vmax = round_10(vmax)


        step = round_10((vmax - vmin) / 12)

        return (vmin, vmax, step)

def plot_map(m, Dataarray, vmin=None, vmax=None, step=None, title="", units="", draw_cities=True, cmap=plt.get_cmap('Blues'), mask_oceans=True, extend='max'):

    lon = Dataarray.lon
    lat = Dataarray.lat

    nlats = interp_factor * len(lat)
    nlons = interp_factor * len(lon)

    lons = np.linspace(float(min(lon)),float(max(lon)),nlons)
    lats = np.linspace(float(min(lat)),float(max(lat)),nlats)

    lons, lats = np.meshgrid(lons, lats)

    offset = np.diff(lons)[0][0] / 2.

    x, y = m(lons-offset, lats-offset)

    if vmin is None or vmax is None or step is None:
        vmin, vmax, step = get_limits(Dataarray)

    if ((vmin < 0) and (vmax > 0)):
        ticks_range = list(np.arange(vmin, 0, step)) + list(np.arange(vmin, 0, step)[::-1]*-1)
        ticks_range = np.array(ticks_range)
    else:
        ticks_range = np.arange(vmin, vmax+step, step)

    interp_array = interp(Dataarray.data,lon.data.flatten(),lat.data.flatten(),lons,lats,order=1)

    if mask_oceans:
        # interpolate land/sea mask to topo grid, mask ocean values.
        interp_array_m = maskoceans(lons, lats, interp_array, resolution='f', grid=1.25)
    else:
        interp_array_m = interp_array
    # make contour plot (ocean values will be masked)

    f, ax = plt.subplots(figsize=(20,20 * np.divide(*interp_array_m.shape)))
    f.subplots_adjust(left=0.1)

    m.ax = ax
    m.drawcoastlines(color='k')
    m.drawcountries(linewidth=1)

    meridians = np.arange(140., 160., 5)
    parallels = np.arange(-10., 0., 2.5)

    m.drawparallels(parallels, labels=[1,0,0,0], fontsize=18, linewidth=0.8, color='.8')
    m.drawmeridians(meridians, labels=[0,0,0,1], fontsize=18, linewidth=0.8, color='.8')

    m.drawrivers(color='steelblue')

    im=m.contourf(x,y,interp_array_m,np.arange(vmin, vmax+step, step), cmap=cmap, extend=extend)

    m.drawmapboundary(fill_color='steelblue')

    # add the copyright blob
    text = u'\N{Copyright Sign}' + ' PNG National Weather Service \n http://www.pngmet.gov.pg'
    ax.text(0.77,0.9275, text, transform = ax.transAxes, fontdict={'size':14}, bbox=dict(facecolor='w', alpha=0.5))

    # set up the colorbar
    cb = m.colorbar(im, location='right', ticks=ticks_range,  size="3%",\
     boundaries=ticks_range, drawedges=True, extend=extend)
    [l.set_fontsize(18) for l in cb.ax.get_yticklabels()];
    cb.set_label(units, fontsize=20)

    ax.set_title(title, fontsize=24)

    plot_cities(ax, cities)

    return f

# ### set up the domain

domain = {'latmin':-12., 'lonmin':140., 'latmax':0., 'lonmax':160.}


# ### defines the cities to plot on the anomalies maps
cities = {}
cities['Port Moresby'] = (-9.513639, 147.218806)
cities['Lae'] = (-6.73333, 147)
cities['Alotau'] = (-10.316667, 150.433333)
cities['Daru'] = (-9.083333, 143.2)
cities['Madang'] = (-5.216667, 145.8)
cities['Wewak'] = (-3.55, 143.633333)


# ### original resolution of the dataset
res = 0.25

# ### factor by which the resolution will be increased
interp_factor = 10

# ### set up the map
m = bm(projection='cyl', llcrnrlon=domain['lonmin']+0.25,\
 llcrnrlat=domain['latmin']+0.25,\
  urcrnrlon=domain['lonmax']-0.25,\
   urcrnrlat=domain['latmax']-0.25, resolution='f')

# get the date of today (UTC)
today = datetime.utcnow()

lag = 1

trmm_date = today - timedelta(days=lag)

# loads the virtual stations file
Virtual_Stations = pd.read_excel('../data/PNG_stations_subset.xls')

for ndays in [30, 60, 90]:

    realtime = pd.date_range(start=trmm_date - timedelta(days=ndays-1), end=trmm_date)

    print("calculating realtime data for {:%Y-%m-%d} to {:%Y-%m-%d}".format(realtime[0], realtime[-1]))

    lfiles = []
    for d in realtime:
        fname  = dpath + "3B42RT_daily.{}.nc".format(d.strftime("%Y.%m.%d"))
        lfiles.append(fname)


    dset_realtime = read_netcdfs(lfiles, 'time')

    # ### selects the domain
    dset_realtime = dset_realtime.sel(lat=slice(domain['latmin'], domain['latmax']), lon=slice(domain['lonmin'], domain['lonmax']))

    # ### reads the climatology
    clim_files = []
    for d in realtime:
        fname  = climpath + "3B42_daily.{}.nc".format(d.strftime("%m.%d"))
        clim_files.append(fname)

    dset_clim = read_netcdfs(clim_files, 'time')


    # ### selects the domain
    dset_clim = dset_clim.sel(latitude=slice(domain['latmin'], domain['latmax']), longitude=slice(domain['lonmin'], domain['lonmax']))


    # ### calculates the averages and cumulative rainfall
    clim_ave = dset_clim.mean('time')
    clim_sum = dset_clim.sum('time')

    realtime_ave = dset_realtime.mean('time')
    realtime_sum = dset_realtime.sum('time')

    realtime_ave['clim'] = (['lat','lon'], clim_ave['hrf'].data)
    realtime_sum['clim'] = (['lat','lon'], clim_sum['hrf'].data)

    # ### calculates the anomalies
    raw = realtime_sum['trmm']

    anoms = realtime_sum['trmm'] - realtime_sum['clim']

    anomsp = ((realtime_sum['trmm'] - realtime_sum['clim']) / realtime_sum['clim']) * 100.

    pp = realtime_sum['trmm'] / realtime_sum['clim'] * 100.

    # ### plots the long term climatology

    title = 'Normal rainfall amounts (2000-2014) for the last {} days [{:%d %B} to {:%d %B}]\nsource: TRMM / TMPA-RT 3B42RT Rainfall estimates'.format(ndays, realtime[0],realtime[-1])

    vmin, vmax, step = get_limits(realtime_sum['clim'])

    f = plot_map(m, realtime_sum['clim'], title=title, units="mm", draw_cities=True,              cmap=plt.get_cmap('BuGn'), extend='max')

    f.savefig('../images/realtime_maskocean_CLIM_{}.png'.format(ndays), dpi=200)

    # ### plots the last N days observed rainfall

    title = 'Observed rainfall amounts for the last {} days [{:%d %B %Y} to {:%d %B %Y}]\nsource: TRMM / TMPA-RT 3B42RT Rainfall estimates'.format(ndays, realtime[0],realtime[-1])

    f = plot_map(m, raw, vmin=vmin, vmax=vmax, step=step, title=title, units='mm', draw_cities=True,              cmap=plt.get_cmap('BuGn'), extend='both')

    f.savefig('../images/realtime_maskocean_OBS_{}.png'.format(ndays), dpi=200)

    # ### plots the anomalies in mm

    title = 'Rainfall anomalies amounts for the last {} days [{:%d %B %Y} to {:%d %B %Y}]\nsource: TRMM / TMPA-RT 3B42RT Rainfall estimates'.format(ndays, realtime[0],realtime[-1])

    if ndays == 90:
        f = plot_map(m, anoms, title=title, units='mm', draw_cities=True, cmap=cmap_anoms, vmin=-900, vmax=900, step=100, extend='both')
    if ndays == 60:
        f = plot_map(m, anoms, title=title, units='mm', draw_cities=True, cmap=cmap_anoms, vmin=-600, vmax=600, step=60, extend='both')
    if ndays == 30:
        f = plot_map(m, anoms, title=title, units='mm', draw_cities=True, cmap=cmap_anoms, vmin=-300, vmax=300, step=30, extend='both')

    f.savefig('../images/last{}days_maskocean_anoms_mm.png'.format(ndays), dpi=200)

    title = 'Percentage of normal rainfall for the last {} days [{:%d %B %Y} to {:%d %B %Y}]\nsource: TRMM / TMPA-RT 3B42RT Rainfall estimates'.format(ndays, realtime[0],realtime[-1])

    f = plot_map(m, pp, vmin=10, vmax=190, step=10, title=title, units='% normal', draw_cities=True, cmap=cmap_anoms, extend='both')

    f.savefig('../images/last{}days_maskocean_anoms_percent_normal.png'.format(ndays), dpi=200)

    title = 'Rainfall anomalies (percent. point) for the last {} days [{:%d %B %Y} to {:%d %B %Y}]\nsource: TRMM / TMPA-RT 3B42RT Rainfall estimates'.format(ndays, realtime[0],realtime[-1])

    f = plot_map(m, anomsp, title=title, units='percent. point anomalies (%)', draw_cities=True, cmap=cmap_anoms, extend='both')

    f.savefig('../images/last{}days_maskocean_anoms_pp.png'.format(ndays), dpi=200)

    for i, row in Virtual_Stations.iterrows():
        stn_name = row['Stn Name']
        lat_V = row['Latitude']
        lon_V = row['Longitude']


        clim_ts = dset_clim.sel(latitude=lat_V, longitude=lon_V, method='nearest')['hrf']
        realtime_ts = dset_realtime.sel(lat=lat_V, lon=lon_V, method='nearest')['trmm']
        df_ts = realtime_ts.to_dataframe()[['trmm']]
        df_ts.loc[:,'clim'] = clim_ts.data
        df_ts.columns = ['observed','climatology']


        f = plt.figure(figsize=(12,5))
        ax1 = f.add_axes([0.1,0.25,0.7,0.65])

        ax1.plot(df_ts['climatology'].index, df_ts['climatology'], color='g', label='clim.')
        ax1.plot(df_ts['observed'].index, df_ts['observed'], color='b', label='obs.')

        ax1.fill_between(df_ts['climatology'].index, 0, df_ts['climatology'], color='g', alpha=0.6, label='clim.')
        ax1.fill_between(df_ts['observed'].index, 0, df_ts['observed'], color='b', alpha=0.6, label='obs.')

        [l.set_rotation(90) for l in ax1.xaxis.get_ticklabels()]
        [l.set_fontsize(12) for l in ax1.xaxis.get_ticklabels()]
        [l.set_fontsize(12) for l in ax1.yaxis.get_ticklabels()]

        ax1.legend(fontsize=12, loc=2)

        ax1.set_xticks(df_ts['climatology'].index[np.arange(1,len(df_ts-5), 5)])

        ax1.set_ylabel("mm", fontsize=12)
        ax1.grid('on')

        ax1.set_title('TRMM / TMPA_RT rainfall estimates for the last {} days to {:%d %B %Y}\nVirtual Station {}: LAT: {}, LON: {}'\
                     .format(ndays, trmm_date, stn_name, realtime_ts.lat.data, realtime_ts.lon.data), fontsize=12)

        # calculates the sum

        sums = df_ts.sum()

        # plots the cumulative rainfall as barplots

        ax2 = f.add_axes([0.8,0.25,0.14,0.65])
        ax2.bar(np.arange(0.5,2.5), sums.values, width=0.7, color=['b','g'], alpha=0.6, align='center')
        ax2.set_xticks([0.5, 1.5])
        ax2.set_xticklabels(['obs.', 'clim.'], fontsize=12)
        ax2.yaxis.tick_right()
        ax2.set_ylabel("mm", fontsize=12)
        ax2.yaxis.set_label_position("right")
        # ax2.set_yticks(None)
        ax2.set_title("{:4.1f} % of normal".format(np.divide(*sums.values) * 100), fontdict={'weight':'bold'})
        [l.set_fontsize(12) for l in ax2.yaxis.get_ticklabels()]

        stn_name_f = stn_name.replace(" ","_")

        df_ts.to_csv('../outputs/Virtual_Station_{}_{}ndays.csv'.format(stn_name_f, ndays))

        f.savefig('../images/Virtual_Station_{}_{}ndays.png'.format(stn_name_f, ndays), dpi=100)

        plt.close(f)
