import numpy as np

import os



def main():

    dataset_path = "/home/grg/Research/dcdi-grg/data/perfect/data_p10_e10_n10000_linear_perfect/data_p10_e10_n10000_linear_struct/"

    data_id = "10"

    adj_mat = np.load(os.path.join(dataset_path, f"DAG{data_id}.npy"))

    only_obs_data = np.load(os.path.join(dataset_path, f"data{data_id}.npy"))

    obs_interv_data = np.load(os.path.join(dataset_path, f"data_interv{data_id}.npy"))



    interv_mask = np.genfromtxt(os.path.join(dataset_path, f"intervention{data_id}.csv"), delimiter=",")

    regimes = np.genfromtxt(os.path.join(dataset_path, f"regime{data_id}.csv"), delimiter=",")

    print(f"adj_mat.shape = {adj_mat.shape}")

    print(f"only_obs_data.shape = {only_obs_data.shape}")

    print(f"obs_interv_data.shape = {obs_interv_data.shape}")

    print(f"interv_mask.shape = {interv_mask.shape}")

    print(f"regimes.shape = {regimes.shape}")

    obs_idx = np.where(regimes == 0)
    num_obs = obs_idx[0].shape

    print(f"num_obs = {num_obs}")
    
    assert interv_mask.shape[0] + num_obs[0] == regimes.shape[0]
    
    pass



if __name__ == "__main__":
    print(f"Started...")
    main()
    print(f"Finished!")