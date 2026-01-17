# Recursive-VineCop

This is the code for my MSc dissertation "[Probabilistic Weather Forecasting with Recursive Algorithms and Vine Copulas](https://kckhchen.com/assets/docs/msc-dissertation.pdf)", submitted in partial fulfillment of the requirements for the degree of MSc in Statistics at the University of Warwick.

## About

This dissertation explores the possibility of utilizing copulas, a statistical tool that captures non-linear dependence between variables, to produce probabilistic forecasts on weather data using the PyTorch framework. It also makes use of a Quasi-Bayesian recursive density estimation algorithm to allow for quick updates of estimates when new data arrives, enabling "online" forecasts where every new observation can be used to refine subsequent predictions without the need to re-train the model.

Specifically, this dissertation employs vine copulas, a graphical tool used to construct and estimate densities of multivariate copulas, to capture the complex relationships and dependencies between meteorological data at different time steps. The involvement of vine copulas allows the model to look further back into the history, extract deeper dependency structures, and make better forecasts than other more forgetful models.

This prediction framework has been tested on a simple autoregressive dataset (AR(1) process), two chaotic models (Lorenz63 and Lorenz96 systems), and a real-world meteorological dataset (the 500hPa geopotential, or Z500) from [WeatherBench 2](https://sites.research.google/gr/weatherbench/).

## Project Structure and Usage

There are four runnable Python scripts: `generate_arma.py`, `generate_lorenz.py`, `generate_weatherbench.py`, and `main.py`.

`generate_arma.py`, `generate_lorenz.py`, and `generate_weatherbench.py` are for generating datasets used in the dissertation. Datasets used in the dissertation can alternatively be found in the `data` folder.
`main.py` will test the models on the datasets and report the results (plots and performance metrics).

A `run.sh` script is also included, which contains shell commands that can be used to conveniently generate results shown in the dissertation.

`trainers.py` and `utils.py` contain background helper functions and training processes that will be called by `main.py`.

## Prerequisites

`requirements.txt` lists all dependencies for the code. The dependencies can be installed with

```
pip install -r requirements.txt
```

## Citation

If you would like to cite this work or use the code in your research, you can use the following citation format:

```
@mastersthesis{Chen2025OnlineProbabilistic,
  author  = {Kuan-Hung Chen},
  title   = {Probabilistic Weather Forecasting with Recursive Algorithms and Vine Copulas},
  school  = {University of Warwick},
  year    = {2024}
}
```

---

The code was run on Python version 3.10.14.
