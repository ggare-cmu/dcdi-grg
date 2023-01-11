import numpy as np

import os

import sys

import subprocess


if __name__ == "__main__":


    trial = "trial-2"
    data_split_type = "obs_interv" #"obs" #"obs_inter" #"obs_plus_interv"

    exp_dir = f"./exp/data_mlp_gen/{trial}/{data_split_type}"
    data_dir = f"./data_mlp_gen/{trial}/{data_split_type}"
    i_dataset = f"1-data_mlp_gen_{data_split_type}"

    adj_mat = np.load(os.path.join(data_dir, f"DAG{i_dataset}.npy"))
    num_vars = adj_mat.shape[0]
    
    
    causal_discovery_args = [

                    "--random-seed", "42",
                    "--exp-path", os.path.abspath(exp_dir),
                    "--data-path", os.path.abspath(data_dir),
                    "--train",
                    "--i-dataset", i_dataset, #//"1",
                    "--num-vars", f"{num_vars}", #//"42",
                    "--intervention",
                    #// "--dcd", #//"Use DCD (DCDI with a loss not taking into account the intervention)"
                    "--intervention-type", "perfect", #//"imperfect", #//"perfect",
                    "--intervention-knowledge", "known", #//"unknown", #//"known",
                    "--coeff-interv-sparsity", "1e-8", #//"1e-8",
                    "--plot-freq", "1000", #//"100",
                    "--gpu",
                    "--float",
                    "--model", "DCDI-DSF", #//"DCDI-G", #//"DCDI-DSF",
                    "--lr", "1e-3", #//"1e-3",
            ]

    run_args = ["python", "/home/grg/Research/dcdi-grg/main.py"] + causal_discovery_args


    # subprocess.run(['python3.9', f'{pipeline}', '--config', f'{exps_path / "config.toml"}', '--train', '--change_val'], check=True)
    subprocess.run(run_args, check=True)

    print(f"Finished causal discovery.")
