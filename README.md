# Recursive-VineCop

This is the code for my MSc dissertation "[Online Probabilistic Weather Forecasting with Vine Copulas](https://kckhchen.com/assets/docs/msc-dissertation.pdf)", submitted in partial fulfillment of the requirements for the degree of MSc in Statistics at the University of Warwick.

There are four runnable Python scripts: `generate_arma.py`, `generate_lorenz.py`, and `generate_weatherbench.py` are for generating datasets used in the dissertation. Datasets used in the dissertation can alternatively be found in the `data` folder.
`main.py` will test the models on the datasets and report the results (plots and performance metrics).

A `run.sh` script file is also included, which contains shell commands that can be used to conveniently generate results shown in the dissertation.

`trainers.py` and `utils.py` contain background helper functions and training processes that will be called by `main.py`.

`requirements.txt` lists all dependencies for the code. The dependencies can be installed with
```
pip install -r requirements.txt
```
The code was run on Python version 3.10.14.
