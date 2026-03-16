#
# Script to use fit_dists to fit sample data to distributions
#
from fit_dists import *
import argparse
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit distributions to data from a CSV file.")
    parser.add_argument("csv_file", nargs="?", default="data/OutputTable1.csv", help="Path to the CSV file")
    parser.add_argument("column", nargs="?", default="WeibullSample", help="Name of the numerical column to fit")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file, skipinitialspace=True)
    print(df.head())
    data = df[args.column].to_numpy(dtype=float)
    print(f"\nFitting column {args.column}:")
    report = fit_dists(data, verbose=False)
    print(report)
