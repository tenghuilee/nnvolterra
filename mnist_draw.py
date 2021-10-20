import pandas as pd
import numpy as np
import sys


import argparse
_arg = argparse.ArgumentParser()
_arg.add_argument("--hack_layer", type=int, default=3, help="hack layer in [1, 2, 3, 4]")
args = _arg.parse_args()
HACK_LAYER = args.hack_layer

print("Hack layer", HACK_LAYER)

recFile = pd.read_csv(f"result/mnist-hack-{HACK_LAYER}-predict.csv")

outTemplate = r"""
\begin{tikzpicture}
    \node[inner sep=0pt] (I) at (0,0) {%%
    \includegraphics[width=0.08\linewidth]{mnist-hack-%s-img-%d}%%
    \includegraphics[width=0.04\linewidth]{mnist-hack-%s-patch-%d}%%
    };%%
    \node[inner xsep=0pt, inner ysep=4pt, yshift=2pt, anchor=north, scale=0.70] at (I.south) {%%
    \begin{tabular}{c|ccc}
    \hline
    \multirow{2}{*}{C} & L & %d & %d \\
    \cline{2-4}
    & V & %.4f & %.4f \\
    \hline
    \multirow{2}{*}{P} & L & %d & %d \\
    \cline{2-4}
    & V & %.4f & %.4f \\
    \hline
    \end{tabular}
    };%% 
\end{tikzpicture}"""

recout = open(f"result/mnist-hack-{HACK_LAYER}-draw.tex", "w")

for index in range(12):
    digit = recFile.loc[index*2][1:11].to_numpy()
    patch = recFile.loc[index*2+1][1:11].to_numpy()

    digsort = np.argsort(digit)
    patsort = np.argsort(patch)


    targs = (HACK_LAYER, index, HACK_LAYER, index, 
        digsort[-1], digsort[-2], digit[digsort[-1]], digit[digsort[-2]], 
        patsort[-1], patsort[-2], patch[patsort[-1]], patch[patsort[-2]])
    
    recout.write(outTemplate%targs)

recout.close()

print("done")

preview = r"""
\documentclass{article}
\usepackage{tikz}
\usepackage{multirow}
\usepackage[verbose=true,a4paper]{geometry}
\newgeometry{left=25mm, right=25mm, top=25mm, bottom=25mm}
\graphicspath{ {result/} }
\begin{document}
\begin{figure*}[htbp]
    \centering
    \include{result/mnist-hack-%s-draw.tex}
\end{figure*}
\end{document}"""

with open("result/mnist_preview.tex", "w") as f:
    f.write(preview%(HACK_LAYER))
