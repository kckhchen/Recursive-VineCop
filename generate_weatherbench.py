import xarray as xr
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser(description='Recursive-VineCop training')
parser.add_argument('--folder', type=str,  help='Data folder name', default="./data")
parser.add_argument('--name', type=str, help='Data name', default="unnamed_weatherbench")
parser.add_argument('--n_samples', type=int, help='Number of samples', default=5000)
parser.add_argument('--n_batches', type=int, help='Number of batches to fetch data, must divide n_samples', default=50)
parser.add_argument('--longitude', type=float, help='Longitude', default=358.5)
parser.add_argument('--latitude', type=float, help='Latitude', default=52.5)
args = parser.parse_args()

folder = args.folder
data_name = args.name
n_samples= args.n_samples
n_batches = args.n_batches
long = args.longitude
lat = args.latitude

if not os.path.exists(folder):
    os.makedirs(folder)

assert n_samples % n_batches == 0, "n_samples not divisible by n_batches"
batch_size = int(n_samples / n_batches)

dataset = xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr')
z500_data = dataset.geopotential.sel(level=500)
z500_data = z500_data.transpose('time', 'latitude', 'longitude')

all_data = []
for bin_idx in range(n_batches):
    bin_start = (n_batches - bin_idx) * -batch_size
    bin_end = (n_batches - bin_idx - 1) * -batch_size
    print("fetching data from WeatherBench2 cloud storage: batch", bin_idx + 1)
    part_data = z500_data.sel(latitude=lat, longitude=long).isel(time=range(bin_start, bin_end)).load()
    all_data.append(part_data)

print("Finish downloading.")
final_time_series = xr.concat(all_data, dim='time')

pd.DataFrame(final_time_series).to_csv(folder + "/" + data_name + ".csv")