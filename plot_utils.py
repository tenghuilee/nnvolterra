import matplotlib as mpl
import numpy as np

mymarkers = [".", "^", "+", "x", "*", ",", "|", "d", "_", "v"]

mpl.rcParams.update({
    'figure.figsize': (8, 4.5),
    'legend.fontsize' : 18,
    'legend.handlelength' : 3,
    'axes.labelsize': 16,
    'xtick.major.size': 3,
    "xtick.major.pad": 5,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'savefig.dpi': 300,
})


def savefig(fig, tag, dpi=300, subplot_adjust=False):
    """savefig: 
    write matplotlib.figure to 
    - ./result/`tag`.eps
    - ./result/`tag`.jpg
    """
    print("save tag", tag)
    # print(plt.rcParams)
    fig.tight_layout()
    if subplot_adjust:
        fig.subplots_adjust(top=0.99,
                            bottom=0.06,
                            right=0.99,
                            left=0.06,
                            hspace=0,
                            wspace=0)
    # dpi: 160 240 320 460 640
    fig.savefig(f"result/{tag}.eps",
                dpi=dpi,
                bbox_inches='tight',
                transparent=True,
                pad_inches=0.1)
    # fig.savefig(f"result/{tag}.pdf", dpi=600, bbox_inches='tight', transparent=True, pad_inches=0)
    # fig.savefig(f"result/{tag}.png", dpi=600, bbox_inches='tight', transparent=True, pad_inches=0)
    fig.savefig(f"result/{tag}.jpg",
                dpi=dpi,
                bbox_inches='tight',
                transparent=True,
                pad_inches=0.1)
