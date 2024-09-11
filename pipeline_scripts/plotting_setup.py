import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colors 
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.font_manager as fm
from matplotlib.patches import Circle
from matplotlib.lines import Line2D


save_folder = '/groups/astro/kxm508/codes/python_dispatch/graphics/'
sink_colors = ['firebrick', 'orangered', 'gold', 'chartreuse', 'darkgreen', 'dodgerblue', 'mediumblue', 'darkviolet', 'dimgrey']
markers = ['o', 's', 'd', '^', 'v', '<', '>', 'p', 'P']

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['xtick.minor.size'] = 4
plt.rcParams['ytick.minor.size'] = 4



# Set som plotting font-standards:
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 12}
mpl.rc('font', **font)

