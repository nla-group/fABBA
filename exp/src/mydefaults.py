import matplotlib as mpl
import matplotlib.font_manager

def mydefaults(fig, ax, r=0.51, s=1):
    """
    Parameters
    ----------
    fig, ax : figure and axes handle from matplotlib
    r : height/width ratio
    s : scaling of font size

    Example
    -------
    from mydefaults import mydefaults
    fig, ax = mpl.pyplot.subplots()
    fig, ax = mydefaults(fig, ax)
    """
    #fig, ax = mpl.pyplot.subplots()

    # Specify fig size
    fig.set_size_inches(s*(13.2/2.54), s*r*(13.2/2.54), forward=True)

    # Use tex and correct font
    #mpl.rcParams['font.family'] = 'Serif'
    mpl.rcParams['font.serif'] = ['computer modern roman']
    #mpl.rcParams['text.usetex'] = True # makes zeros bold?
    mpl.rcParams['font.size'] = 11
    mpl.rcParams['font.weight'] = 'normal'

    # MATLAB default (see MATLAB Axes Properties documentation)
    mpl.rcParams['axes.titlesize'] = 1.1*11
    mpl.rcParams['axes.titleweight'] = 'bold'

    # MATLAB default (see MATLAB Axes Properties documentation)
    mpl.rcParams['axes.labelsize'] = 1.1*11
    mpl.rcParams['axes.labelweight'] = 'normal'

    # MATLAB default (see MATLAB Axes Properties documentation)
    mpl.rcParams['legend.fontsize'] = 0.9*11

    # remove margine padding on axis
    mpl.rcParams['axes.xmargin'] = 0
    mpl.rcParams['axes.ymargin'] = 0

    # switch tick direction like MATLAB
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'

    mpl.pyplot.tight_layout(pad=1.3) # padding as fraction of font size
    if isinstance(ax, tuple):
        for axi in ax:
            axi.tick_params(axis='both', which='both', direction='in')
    else:
        ax.tick_params(axis='both', which='both', direction='in')

    # Save fig with transparent background
    mpl.rcParams['savefig.transparent'] = True

    # Make legend frame border black and face white
    mpl.rcParams['legend.edgecolor'] = 'k'
    mpl.rcParams['legend.facecolor'] = 'w'
    mpl.rcParams['legend.framealpha'] = 1

    # Change colorcycle to MATLABS
    c = mpl.cycler(color=['#0072BD', '#D95319', '#EDB120',  '#4DBEEE', '#77AC30', '#7E2F8E', '#A2142F'])

    if isinstance(ax, tuple):
        for axi in ax:
            axi.set_prop_cycle(c)
    else:
        ax.set_prop_cycle(c)
    # mpl.rcParams['axes.prop_cycle'] = c # doesnt work?

    return fig, ax
