import pandas as pd
import numpy as np

import argparse

_arg = argparse.ArgumentParser()
_arg.add_argument("--hack_layer",
                  type=int,
                  default=3,
                  help="hack layer in [1, 2, 3, 4]")
_arg.add_argument("--mask", type=str, default="", help="tag. none or 'dd'")
args = _arg.parse_args()
HACK_LAYER = args.hack_layer

if len(args.mask) > 0:
    args.mask = "-" + args.mask

print("Hack layer", HACK_LAYER)

recFile = pd.read_csv(f"result/mnist-hack-{HACK_LAYER}-predict{args.mask}.csv")

outTemplate = r"""
\begin{tikzpicture}
    \node[inner sep=0pt] (I) at (0,0) {%%
    \includegraphics[width=0.08\linewidth]{mnist-hack-%s-img%s-%d.jpg}%%
    \includegraphics[width=0.08\linewidth]{mnist-hack-%s-img-patch%s-%d.jpg}%%
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

recout = open(f"result/mnist-hack-{HACK_LAYER}-draw{args.mask}.tex", "w")

for index in range(12):
    digit = recFile.loc[index * 2][1:11].to_numpy()
    patch = recFile.loc[index * 2 + 1][1:11].to_numpy()

    digsort = np.argsort(digit)
    patsort = np.argsort(patch)

    targs = (HACK_LAYER, args.mask, index, HACK_LAYER, args.mask, index,
             digsort[-1], digsort[-2], digit[digsort[-1]], digit[digsort[-2]],
             patsort[-1], patsort[-2], patch[patsort[-1]], patch[patsort[-2]])
    
    _fig = outTemplate % targs

    if index > 9: # 10, 11
        _fig = _fig.replace(r"\cline", r"\cdashline")\
            .replace(r"\hline", r"\hdashline")\
            .replace("begin{tabular}{c|ccc}", "begin{tabular}{c:ccc}")

    recout.write(_fig)

recout.close()

preview = r"""
\documentclass{article}
\usepackage{tikz}
\usepackage{multirow}
\usepackage{array}
\usepackage{longtable} 
\usepackage{colortab}
\usepackage{colortbl}
\usepackage{arydshln}
\usepackage[verbose=true,a4paper]{geometry}
\newgeometry{left=35mm, right=35mm, top=25mm, bottom=25mm}
\graphicspath{ {result/} }
\begin{document}
\begin{figure*}[htbp]
    \centering
    \include{mnist-hack-%s-draw%s.tex}
\end{figure*}
\end{document}"""

with open("result/mnist_preview.tex", "w") as f:
    f.write(preview % (HACK_LAYER, args.mask))

import os

os.system("xelatex result/mnist_preview.tex")

print("done")