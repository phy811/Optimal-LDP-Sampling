# A numerical result for finite data space

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from argparse import ArgumentParser
from div import kl_f, tv_f, hellinger_f

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.serif' : 'Computer Modern Roman',
    'figure.constrained_layout.use' : True,
    'axes.formatter.useoffset' : False,
    'axes.labelsize'  : 10,
    'xtick.labelsize' : 10,
    'ytick.labelsize' : 10,
    'legend.fontsize' : 10 
})

def max_pmf_unifMollifier(k, eps):
    return np.minimum(np.exp(eps/2)/k, 1-(k-1)/k*np.exp(-eps/2))


def worst_fDiv_ours(k, eps, f):
    max_pmf = np.exp(eps)/(np.exp(eps)+k-1)
    return (1-max_pmf)*f(0) + max_pmf*f(1/max_pmf)

def worst_fDiv_prev(k, eps, f):
    max_pmf = max_pmf_unifMollifier(k, eps)
    return (1-max_pmf)*f(0) + max_pmf*f(1/max_pmf)



def main(k:int, figname:str|None):
    if figname is None:
        figname = f'result_finite_k{k}.eps'
    
    eps_list = np.array([0.1, 0.5, 1, 2, 5])
    f_list = [kl_f, tv_f, hellinger_f]
    f_name_list = ["KL", "TV", "Sq. Hel"]
    
    fig = plt.figure()

    for i in range(1, 4):
        plt.subplot(1, 3, i)
        f = f_list[i-1]
        plt.title(f_name_list[i-1])

        ours = worst_fDiv_ours(k, eps_list, f)
        prev = worst_fDiv_prev(k, eps_list, f)

        plot_data = pd.DataFrame({
            "Privacy budget $\\epsilon$" : np.concatenate([eps_list, eps_list]),
            "Worst $f$-divergence" : np.concatenate([ours, prev]),
            "mech" : ["Proposed"] * 5 + ["Baseline"] * 5
        })
        plot = sns.barplot(data=plot_data, x = "Privacy budget $\\epsilon$", y = "Worst $f$-divergence", hue = "mech")
        plot.get_legend().remove()

        if i != 2:
            plot.set(xlabel=None)

        if i != 1:
            plot.set(ylabel=None)

    handles, labels = plot.get_legend_handles_labels()
    fig.legend(loc='lower center', ncol=2, handles=handles, labels=labels, bbox_to_anchor=(0.25, 0.005))
    fig.set_figheight(2.5)
    plt.tight_layout()

    plt.savefig(figname)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--figname', type=str)
    
    main(**vars(parser.parse_args()))