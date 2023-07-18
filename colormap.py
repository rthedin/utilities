# Code and tools related to colormaps
# Written by R. Thedin

# To use the code below:
# sys.path.append(os.path.abspath('/home/rthedin/utilities/'))
# from colormap import createColormap

import matplotlib
import matplotlib.colors as mcolors
import numpy as np

# Register a RdGr colormap
reds = matplotlib.colormaps['Reds_r'].resampled(128)
greens = matplotlib.colormaps['Greens'].resampled(128)
redgreen = np.vstack((reds(np.linspace(0, 1, 128)), greens(np.linspace(0, 1, 128))))
RdGr   = mcolors.ListedColormap(redgreen, name='RdGr')
RdGr_r = RdGr.reversed(name='RdGr_r')
# Register colormaps
matplotlib.colormaps.register(cmap=RdGr)
matplotlib.colormaps.register(cmap=RdGr_r)


# Create a sequential colormap with every CSS4 color, named <color>s.
# Colors available at https://matplotlib.org/stable/gallery/color/named_colors.html
for c in mcolors.CSS4_COLORS:
    # Create colormap with white beginning
    cmap   = mcolors.LinearSegmentedColormap.from_list(name=f'{c}s', colors=['white',c])
    cmap_r = RdGr.reversed(name=f'{c}s_r')

    # Register colormaps
    matplotlib.colormaps.register(cmap=cmap)
    matplotlib.colormaps.register(cmap=cmap_r)


def createColormap(colors):
    '''
    Registers new colormap and its reversed form.
    Each color is of the same length on the final map.
    
    Example call:
    -------------
    createColormap(['firebrick','white','darkgreen'])
    The map "firebrickwhitedarkgreen" is now available and is
    the same as RdGr above.

    Input:
    ------
    colors: array of str
        Named colors (CSS4_COLORS) to create the colormap
    '''
    
    if isinstance(colors, str):
        raise ValueError (f'Colors should be a list of 2 or more colors')
        
    cmapstr = ''.join(colors)
    
     # Create colormap with white beginning
    cmap   = mcolors.LinearSegmentedColormap.from_list(name=cmapstr, colors=colors)
    cmap_r = RdGr.reversed(name=f'{cmapstr}_r')

    # Register colormaps
    try:
        matplotlib.colormaps.register(cmap=cmap)
        matplotlib.colormaps.register(cmap=cmap_r)
        print(f'Colormap called "{cmapstr}" and "{cmapstr}_r" registered.')
    except ValueError as e:
        if str(e) == f'A colormap named "{cmapstr}" is already registered.':
            print(f'Colormap "{cmapstr}" and "{cmapstr}_r" already registered. Skipping.')
        else:
            raise
            
