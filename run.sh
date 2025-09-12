# To generate data used in the dissertation, use the commands below. Change any arguments to try different data.
# Alternatively, use the pre-generated datasets in the data/ folder.

# AR(3) data
python3 generate_arma.py --folder "data" --name "AR3" --ar_params "0.1, 0.3, 0.5" --ma_params "0." --n_samples 5000 --drift 0 --sigma 1 --burn_in 500 --seed 25
# Lorenz63
python3 generate_lorenz.py --folder "data" --type "L63" --name "L63" --n_samples 5000 --burn_in 1000 --discard_interval 20
# Lorenz96
python3 generate_lorenz.py --folder "data" --type "L96" --name "L96" --n_samples 5000 --burn_in 2000 --discard_interval 50
# Z500 at Uni of Warwick. Note that downloading 5,000 data from the WeatherBench2 cloud storage may take up to an hour.
python3 generate_weatherbench.py --folder "data" --name "UW_Z500" --n_samples 5000 --n_batches 50 --longitude 358.5 --latitude 52.5

# To reproduce results in the dissertation, use the commands below. Type "python3 main.py --help" to see all available arguments.
python3 main.py --data_name "AR3" --init_dist "Normal" --init_loc 0 --init_scale 1 --init_rho 0.1
python3 main.py --data_name "L63" --init_dist "Normal" --init_loc 0 --init_scale 1 --init_rho 0.3
python3 main.py --data_name "L96" --component 1 --init_dist "Normal" --init_loc 0 --init_scale 1 --init_rho 0.3
python3 main.py --data_name "UW_Z500" --init_dist "Normal" --init_loc 0 --init_scale 1 --init_rho 0.3