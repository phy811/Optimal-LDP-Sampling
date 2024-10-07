# Plotting the results of the 1D Gaussian mixture experiment
# Important: 
#   Make sure to run the experiment by exp_1DGaussMix.py for all specified eps values in eps_list 
#   before running this script

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
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


def main(figname:str):
    f_list = [kl_f, tv_f, hellinger_f]
    f_name_list = ["KL", "TV", "Sq. Hel"]

    eps_list = ['0.1', '0.5', '1.0', '2.0', '5.0']
    results = []
    for eps in eps_list:
        data = np.load(f'data_1DGaussMix_eps{eps}.npy')
        worst_fdiv = data.max(axis=0)
        results.append(pd.DataFrame({
                    'Privacy budget $\\epsilon$': [float(eps)]*6,
                    'Worst $f$-divergence': worst_fdiv,
                    'mech': ['Proposed', 'Baseline'] * 3,
                    'f': ['KL', 'KL', 'TV', 'TV', 'Sq. Hel', 'Sq. Hel']
        }))

        fig = plt.figure()
        
    result = pd.concat(results)

    for i in range(1, 4):
        plt.subplot(1, 3, i)
        f = f_list[i-1]
        plt.title(f_name_list[i-1])

        plot_data = result[result['f'] == f_name_list[i-1]]
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
    parser.add_argument('--figname', type=str, default='result_1DGaussMix.eps')
    
    main(**vars(parser.parse_args()))