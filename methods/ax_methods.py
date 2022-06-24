import matplotlib as mpl

def turn_off_ax(ax):
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

def make_marker_obj(**kwargs):
    return mpl.lines.Line2D((0, 0), (0, 0), **kwargs)
