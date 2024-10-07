# Optimal-LDP-Sampling

Experiments for the paper [Exactly Minimax-Optimal Locally Differentially Private Sampling]


## Requirements

The code is written in Python 3. The complete list of external libraries used in this project is as follows:

```
numpy torch scipy pandas matplotlib seaborn tqdm
```

For linux with anaconda installed, we can use the following command to setup the environment with required libraries:

```setup
conda env create -f environment.yaml
```
After that, activate the environment by the following command:
```
conda activate ldpSampling
```


By default, $\TeX$ fonts are used in the figures, which requires $\LaTeX$ to be installed. To disable this option, remove the following lines in the codes:
```
matplotlib.use("pgf")

"pgf.texsystem": "pdflatex",
'font.family': 'serif',
'text.usetex': True,
'pgf.rcfonts': False,
'font.serif' : 'Computer Modern Roman',
```


## Generating the figures about the comparison in Gaussian ring

To generate the figure about the comparison in Gaussian ring (Figure 1), run this command:

```
python plot_GaussRing.py
```


## Generating the figure for visualizing the mechanism in finite space
To generate the figure for visualizing the mechanism in finite space (Figure 2), run this command:
```
visualize_finiteSpace.py
```

## Theoretical comparison in finite space

To generate the figures to compare the theoretical worst-case utilities in finite space like Figures 3,5,6,7, run this command:

```
python plot_finite.py --k <k>
```
where `<k>` is the size of the space $\mathcal{X}$.

To generate Figures 3,5,6,7, run these commands, respectively:
```
python plot_finite.py --k 10
python plot_finite.py --k 5
python plot_finite.py --k 20
python plot_finite.py --k 100
```

## Empirical comparison in 1D Gaussian mixture

To perform experiment about the 1D Gaussian mixture, run this command:
```
python exp_1DGaussMix.py --eps <eps> --size <N> --seed <seed> 
```
where `<eps>`, `<N>` is the value of the $\epsilon, N$, respectively, and `<seed>` is the random seed. If unspecified, the default values are `<eps> = 1.0, <N> = 100, <seed> = 1`. The result of the experiment is saved as a file
`data_1DGaussMix_eps<eps>.npy`.

To generate Figure 4, first run the following commands (These can be run in any order or in parallel):
```
python exp_1DGaussMix.py --eps 0.1 --seed 1
python exp_1DGaussMix.py --eps 0.5 --seed 2
python exp_1DGaussMix.py --eps 1.0 --seed 3
python exp_1DGaussMix.py --eps 2.0 --seed 4
python exp_1DGaussMix.py --eps 5.0 --seed 5
```
Check that all of the following five files are created: (The repository already contains these files)
```
data_1DGaussMix_eps0.1.npy
data_1DGaussMix_eps0.5.npy
data_1DGaussMix_eps1.0.npy
data_1DGaussMix_eps2.0.npy
data_1DGaussMix_eps5.0.npy
```

Then, run the following command:

```
plot_1DGaussMix.py
```


