# Collections of functions that help perform routine tasks related to SSRS.
#
# Add to notebooks using
#   sys.path.append(os.path.abspath('/home/rthedin/utilities/'))
#   from helper import runNtrackdirMwdir

# Regis Thedin
#

import numpy as np
import pandas as pd
import xarray as xr
import os, sys
from dataclasses import replace
from ssrs import Simulator, Config


def runNtrackdirPar(sim_object_dict, wdir, N=16, startboxsize=1):
    '''
    Run a sweep of scenarios with varying starting location and direction intent. 
    The intent of direction is always opposite to the starting point, crossing the
    domain through the central point.
    
    Parameters
    ==========
    sim_object_dict: dict
        SSRS dictionary with settings
    N: scalar
        Number of starting points around the circle. Default 16
    startboxsize: scalar
        Size of the starting box, in km. Each box is located at each
        of the N starting locations. For 25-30 km regions, 1 km box is
        appropriate.
    
    Output
    ======
    combined_map: xarray.Dataset
        Dataset containing all individual presence maps
    
    '''

    # Change the object with the requested wdir
    sim_object_dict = replace(
                       sim_object_dict, 
                       uniform_winddirn = wdir,
                       uniform_winddirn_h = wdir,
                       uniform_winddirn_href = wdir,
                     )
    
    # Get dimentions from object
    extentx = sim_object_dict.region_width_km[0]
    extenty = sim_object_dict.region_width_km[0]
    startboxsize = 1 # 1 km boxes

    # Get starting location information
    theta = np.deg2rad( np.linspace(0,360,N, endpoint=False) )  # 8 points = N, S, W, E, NW, NE, SW, SE

    # Get track direction wrt to each box. First box is N (0), so track direction should be S (180),
    # 3rd box is at E (90), so track direction should be W (270). Create this offset here
    dirns = np.rad2deg(theta)+180

    # create 1x1km starting boxes at the locations indicated by theta
    x = (extentx/2 - startboxsize) * np.sin(theta)
    y = (extenty/2 - startboxsize) * np.cos(theta)

    start_regions = [(extentx/2 + ix - startboxsize/2, # xmin
                      extentx/2 + ix + startboxsize/2, # xmax
                      extenty/2 + iy - startboxsize/2, # ymin
                      extenty/2 + iy + startboxsize/2) # ymax
                     for ix,iy in zip(x,y)]
    
    
    # Execute the N starting points and accumulate results
    sim = []
    for start_region, dirn in zip(start_regions, dirns):
        print(f'\n\nRunning starting region box {start_region}, track direction {dirn}')
        temp = replace(sim_object_dict, track_direction = dirn, track_start_region = start_region)
        isim = Simulator(temp)
        isim.simulate_tracks()
        sim.append(isim)
        
    # Get grid for xarray
    xgrid, ygrid = sim[0].get_terrain_grid(sim[0].resolution, sim[0].gridsize)

    # Plot elevation, slope, aspect, and orographic updraft maps
    sim[0].plot_updrafts(show=False, apply_threshold=False, plot='pcolormesh')
    sim[0].plot_terrain_elevation(show=False)
    sim[0].plot_terrain_aspect(show=False, plot='pcolormesh')
    sim[0].plot_terrain_slope(show=False, plot='pcolormesh')
    
    # Loop through tracks accumulating and plotting individual tracks
    pres_maps = []
    pres_maps_xr = []
    for i in range(N):
        fig, ax = sim[i].plot_simulated_tracks_altamont(show=False, in_alpha=0.1, plot_turbs=False)
        fig2, ax2, pres_map = sim[i].plot_presence_map_altamont(show=False, radius=200., minval=0.01)
        # Save figure with tracks and presence map for current direction
        sim[i].save_fig(fig, os.path.join(sim[i].fig_dir, f'simulated_tracks_wdir{sim[i].uniform_winddirn_href}_alpha01_dir{dirns[i]}.png'))
        sim[i].save_fig(fig2, os.path.join(sim[i].fig_dir, f'presence_map_wdir{sim[i].uniform_winddirn_href}_radius200_dir{dirns[i]}.png'))
        # Get grid for xarray (within the loop for generality)
        xgrid, ygrid = sim[i].get_terrain_grid(sim[i].resolution, sim[i].gridsize)
        # Accumulate
        pres_maps.append(pres_map)
        # Accumulate in xr
        #pres_maps_xr.append( xr.DataArray(pres_map).expand_dims('trackdir').assign_coords({'trackdir':[sim[i].track_direction]}) )
        pres_maps_xr.append( xr.DataArray(pres_map, dims=['x','y'], coords=[xgrid,ygrid]).
                             expand_dims('trackdir').assign_coords({'trackdir':[sim[i].track_direction]})
                           )

    # Concat the appended dataset
    pres_maps_xr = xr.concat(pres_maps_xr, dim='trackdir').to_dataset(name='pres_map')

    # Loop through again just to plot the combined track
    for i in range(N):
        if i==0: fig, ax = sim[i].plot_simulated_tracks_altamont(show=False, in_alpha=0.1, plot_turbs=False)
        else:    _, _    = sim[i].plot_simulated_tracks_altamont(show=False, fig=fig, axs=ax, in_alpha=0.1, plot_turbs=False)
    sim[i].save_fig(fig, os.path.join(sim[i].fig_dir, f'simulated_tracks_wdir{sim[i].uniform_winddirn_href}_alpha01_all{N}dir.png'))


    # Get mean presence map
    #pres_maps = np.array(pres_maps)
    summary_pres_map = np.mean(pres_maps,axis=0)
    summary_pres_map /= np.amax(summary_pres_map)

    # Save combined presence map
    fig, ax = sim[0]._plot_presence_altamont(summary_pres_map, 0.1)
    sim[0].save_fig(fig, os.path.join(sim[i].fig_dir, f'presence_map_wdir{sim[0].uniform_winddirn_href}_radius200_all{N}dir.png'))

    # convert presence map do xarray
    combined_map = pres_maps_xr.expand_dims(['wdir','wspd']).assign_coords({'wdir':[wdir],'wspd':[sim[0].uniform_windspeed_href]})

    return combined_map




def runNtrackdir(sim_object_dict, N=16, startboxsize=1):
    '''
    Run a sweep of scenarios with varying starting location and direction intent. 
    The intent of direction is always opposite to the starting point, crossing the
    domain through the central point.
    
    Parameters
    ==========
    sim_object_dict: dict
        SSRS dictionary with settings
    N: scalar
        Number of starting points around the circle. Default 16
    startboxsize: scalar
        Size of the starting box, in km. Each box is located at each
        of the N starting locations. For 25-30 km regions, 1 km box is
        appropriate.
    
    Output
    ======
    No output. Images are saved.
    
    '''
    
    # Get dimentions from object
    extentx = sim_object_dict.region_width_km[0]
    extenty = sim_object_dict.region_width_km[0]
    startboxsize = 1 # 1 km boxes

    # Get starting location information
    theta = np.deg2rad( np.linspace(0,360,N, endpoint=False) )  # 8 points = N, S, W, E, NW, NE, SW, SE

    # Get track direction wrt to each box.
    # First box is N (0), so track direction should be S (180), 3rd box is at E (90), so track direction should be W (270). Create this offset here
    dirns = np.rad2deg(theta)+180

    # create 1x1km starting boxes at the locations indicated by theta
    x = (extentx/2 - startboxsize) * np.sin(theta)
    y = (extenty/2 - startboxsize) * np.cos(theta)

    start_regions = [(extentx/2 + ix - startboxsize/2, # xmin
                      extentx/2 + ix + startboxsize/2, # xmax
                      extenty/2 + iy - startboxsize/2, # ymin
                      extenty/2 + iy + startboxsize/2) # ymax
                     for ix,iy in zip(x,y)]
    
    
    # Execute the N starting points and accumulate results
    sim = []
    for start_region, dirn in zip(start_regions, dirns):
        print(f'\n\nRunning starting region box {start_region}, track direction {dirn}')
        temp = replace(sim_object_dict, track_direction = dirn, track_start_region = start_region)
        isim = Simulator(temp)
        isim.simulate_tracks()
        sim.append(isim)
        

    # Plot elevation, slope, aspect, and orographic updraft maps
    sim[0].plot_updrafts(show=False, apply_threshold=False, plot='pcolormesh')
    sim[0].plot_terrain_elevation(show=False)
    sim[0].plot_terrain_aspect(show=False, plot='pcolormesh')
    sim[0].plot_terrain_slope(show=False, plot='pcolormesh')
    
    # Loop through tracks accumulating and plotting individual tracks
    pres_maps = []
    for i in range(N):
        fig, ax = sim[i].plot_simulated_tracks_altamont(show=False, in_alpha=0.1, plot_turbs=False)
        fig2, ax2, pres_map = sim[i].plot_presence_map_altamont(show=True, radius=200., minval=0.01)
        # Save figure with tracks and presence map for current direction
        sim[i].save_fig(fig, os.path.join(sim[i].fig_dir, f'simulated_tracks_wdir{sim[i].uniform_winddirn_href}_alpha01_dir{dirns[i]}.png'))
        sim[i].save_fig(fig2, os.path.join(sim[i].fig_dir, f'presence_map_wdir{sim[i].uniform_winddirn_href}_radius200_dir{dirns[i]}.png'))
        # Accumulate
        pres_maps.append(pres_map)


    # Loop through again just to plot the combined track
    for i in range(N):
        if i==0: fig, ax = sim[i].plot_simulated_tracks_altamont(show=False, in_alpha=0.1, plot_turbs=False)
        else:    _, _    = sim[i].plot_simulated_tracks_altamont(show=False, fig=fig, axs=ax, in_alpha=0.1, plot_turbs=False)
    sim[i].save_fig(fig, os.path.join(sim[i].fig_dir, f'simulated_tracks_wdir{sim[i].uniform_winddirn_href}_alpha01_all{N}dir.png'))


    # Get mean presence map
    #pres_maps = np.array(pres_maps)
    summary_pres_map = np.mean(pres_maps,axis=0)
    summary_pres_map /= np.amax(summary_pres_map)

    # Save combined presence map
    fig, ax = sim[0]._plot_presence_altamont(summary_pres_map, 0.1)
    sim[0].save_fig(fig, os.path.join(sim[i].fig_dir, f'presence_map_wdir{sim[0].uniform_winddirn_href}_radius200_all{N}dir.png'))




