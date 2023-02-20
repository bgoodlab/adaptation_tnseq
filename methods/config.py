from pathlib import Path
import os
path = os.getcwd().strip('/scripts')
plot_dir = str(Path(path+'/plots/'))
biorxiv_dir = plot_dir

mpl_configs = {
    'lines.linewidth': 0.75,
    'font.family': 'Helvetica',
    'font.size': 8,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'savefig.dpi': 300,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'xtick.major.pad': 2,
    'xtick.minor.pad': 1.5,
    'ytick.major.pad': 2,
    'ytick.minor.pad': 1.5,
    'figure.facecolor': (1,1,1,1), #white
    'figure.edgecolor': (1,1,1,1), #white
    'figure.figsize': (6.5, 4),
    'axes.titlesize': 10,
    'legend.fontsize': 6,
    'axes.labelpad': 0}
BACTERIA = ['BWH2', 'Bovatus', 'BtVPI', 'Bt7330']
BAC_FORMAL_NAMES = {'BWH2':'$\it{Bc}$',
                   'Bovatus':'$\it{Bo}$',
                   'BtVPI':'$\it{Bt}$-VPI',
                   'Bt7330':'$\it{Bt}$-7330'}
DIET_MICE_MAP = {'HF':[6,7,8,9,10], 'LF':[16,18,19], 'HLH':[1,2,3,5], 'LHL':[11,12,13,15]}

KELLY_COLORS = ['#FFB300', '#803E75', '#FF6800', '#A6BDD7',
                '#C10020', '#CEA262', '#817066', '#007D34',
                '#F6768E', '#00538A', '#FF7A5C', '#53377A',
                '#FF8E00', '#B32851', '#F4C800', '#7F180D',
                '#93AA00', '#593315', '#F13A13', '#232C16']

DIET_COLORS = {'HF': KELLY_COLORS[2], 'LF': KELLY_COLORS[9], 'LHL': KELLY_COLORS[1], 'HLH': KELLY_COLORS[1]}


