# Visualize the mechanism for finite data space, Figure 2 in the paper

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.serif' : 'Computer Modern Roman',
    'figure.constrained_layout.use' : True,
    'axes.formatter.useoffset' : False,
    'axes.labelsize'  : 15,
    'xtick.labelsize' : 15,
    'ytick.labelsize' : 10,
    'legend.fontsize' : 14 
})

def perturb_discrete(input_dist, eps):
    k = np.size(input_dist)
    exp_eps = np.exp(eps)
    r1 = 1
    r2 = (exp_eps + k - 1)/exp_eps
    while True:
        r = (r1 + r2) / 2
        output_dist = np.maximum(input_dist/r, 1/(exp_eps+k-1))
        dist_sum = np.sum(output_dist)
        if np.isclose(dist_sum, 1):
            break
        
        if dist_sum > 1:
            r1 = r
        else:
            r2 = r
            
    return output_dist


if __name__ == "__main__":    
    k = 5
    eps_list = [1.0, 0.5, 0.1]
    input_dist = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    
    plot_data = pd.DataFrame({
        'Data value $x$' : np.arange(1, k+1),
        'Probability' : input_dist,
        '$\\epsilon$' : 'Original pmf $P(x)$'}
        )
    
    # create a bar chart comparing the original pmf and the private pmf over different eps
    for eps in eps_list:
        output_dist = perturb_discrete(input_dist, eps)
        plot_data = pd.concat([plot_data, pd.DataFrame({'Data value $x$': np.arange(1, k+1), 'Probability': output_dist, '$\\epsilon$': r"$\mathbf{Q}_{k,\epsilon}^* (x|P)$" + f", $\\epsilon = {eps}$"})])

    sns.barplot(data=plot_data, x='Data value $x$', y='Probability', hue='$\\epsilon$')
    plt.legend(loc='upper right', ncol=2)
    
    plt.savefig("finite_mech_visualize.eps")