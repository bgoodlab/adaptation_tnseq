root_name = 'tnseq_adaptation'
import os ## make sure path to root of project directory
cwd = os.getcwd(); directory_root = cwd[:cwd.index(root_name)+len(root_name)]
import sys; sys.path.insert(0, directory_root)

plot_dir = str(directory_root+'/plots')
data_dir = str(directory_root+'/data/data')

diet_mice_map = {'HF': [6, 7, 8, 9, 10],
         'LF': [16, 18, 19],
         'HLH': [1, 2, 3, 5],
         'LHL': [11, 12, 13, 15],
         'all': [6, 7, 8, 9, 10, 16, 18, 19, 1, 2, 3, 5, 11, 12, 13, 15],
         'H4': [6, 7, 8, 9, 10, 1, 2, 3, 5],
         'H10': [6, 7, 8, 9, 10, 11, 12, 13, 15],
         'H16': [6, 7, 8, 9, 10, 1, 2, 3, 5],
         'L4': [16, 18, 19, 11, 12, 13, 15],
         'L10': [16, 18, 19, 1, 2, 3, 5],
         'L16':  [16, 18, 19, 11, 12, 13, 15],
        }
diet_colors = {'HF':'red', 'LF':'blue', 'HLH':'orange', 'LHL':'purple'}

BACTERIA = ['BWH2', 'Bovatus', 'BtVPI', 'Bt7330']
BAC_FORMAL_NAMES = {'BWH2':'$\it{Bc}$',
                   'Bovatus':'$\it{Bo}$',
                   'BtVPI':'$\it{Bt}$-VPI',
                   'Bt7330':'$\it{Bt}$-7330'}
DIET_NAMES = {'HF': 'HF/HS', 'LF': 'LF/HPP', 'HLH': 'HF/HS', 'LHL': 'LF/HPP'}
MEDIA = ['Arabinose', 'Glucose', 'TYG', 'WAX', 'Xylose']

DIET_MICE_MAP = {'HF':[6,7,8,9,10], 'LF':[16,18,19], 'HLH':[1,2,3,5], 'LHL':[11,12,13,15]}
days = [4,10,16]
mouse2diet = {}
for diet in ['HF', 'HLH', 'LF', 'LHL']:
    for mouse in diet_mice_map[diet]:
        mouse2diet[mouse] = diet
KELLY_COLORS = ['#FFB300', '#803E75', '#FF6800', '#A6BDD7',
                '#C10020', '#CEA262', '#817066', '#007D34',
                '#F6768E', '#00538A', '#FF7A5C', '#53377A',
                '#FF8E00', '#B32851', '#F4C800', '#7F180D',
                '#93AA00', '#593315', '#F13A13', '#232C16']
def get_kelly_color(i):
    return KELLY_COLORS[i%len(KELLY_COLORS)]
DIET_COLORS = {'HF': KELLY_COLORS[2], 'LF': KELLY_COLORS[9], 'LHL': KELLY_COLORS[1], 'HLH': KELLY_COLORS[1]}

## packages used by every script
import pickle
import numpy as np; rnd = np.random.default_rng()
import matplotlib as mpl
import matplotlib.pyplot as plt
if True:
    mpl.rcParams['lines.linewidth'] = 0.75
    mpl.rcParams['font.family'] = 'Helvetica'
    mpl.rcParams['font.size'] = 8
    mpl.rcParams['savefig.bbox'] = 'tight'
    mpl.rcParams['savefig.pad_inches'] = 0.05
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['xtick.labelsize'] = 6
    mpl.rcParams['ytick.labelsize'] = 6
    mpl.rcParams['figure.facecolor'] = (1,1,1,1) #white
    mpl.rcParams['figure.edgecolor'] = (1,1,1,1) #white
    mpl.rcParams['axes.titlesize'] = 10
    mpl.rcParams['legend.fontsize'] = 6
