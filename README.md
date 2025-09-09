# Recursive-VineCop

This is the code for my MSc dissertation "Online Probabilistic Forecasting with Vine Copulas", submitted in partial fulfillment of the requirements for the degree of MSc in Statistics in the University of Warwick.

There are four runnable Python scripts: `generate_arma.py`, `generate_lorenz.py`, and `generate_weatherbench.py` are for generating datasets used in the dissertation. Datasets used in the dissertation can alternatively be found in the `data` folder.
`main.py` will test the models on the datasets and report the results (plots and performance metrics).

A `run.sh` script file is also included, which contains shell commands that can be used to conveniently generate results shown in the dissertation.

`trainers.py` and `utils.py` contain background helper functions that will be called from the four Python scripts above.

`requirements.txt` lists all dependencies for the code. The dependencies can be installed with
```
pip install -r requirements.txt
```
The code was run on Python version 3.10.4.