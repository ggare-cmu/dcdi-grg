#Code adopted from: https://github.com/ggare-cmu/TabPFN_grg/blob/main/tabpfn/priors/mlp.py 

import random
import math

import torch
from torch import nn
import numpy as np

import zero 


import sys
sys.path.append(".")

import utils

# from .utils import get_batch_to_dataloader



def get_general_config(max_features, bptt, eval_positions=None):
    """
    Returns the general PFN training hyperparameters.
    """
    config_general = {
        "lr": CSH.UniformFloatHyperparameter('lr', lower=0.0001, upper=0.00015, log=True),
        "dropout": CSH.CategoricalHyperparameter('dropout', [0.0]),
        "emsize": CSH.CategoricalHyperparameter('emsize', [2 ** i for i in range(8, 9)]), ## upper bound is -1
        "batch_size": CSH.CategoricalHyperparameter('batch_size', [2 ** i for i in range(6, 8)]),
        "nlayers": CSH.CategoricalHyperparameter('nlayers', [12]),
        "num_features": max_features,
        "nhead": CSH.CategoricalHyperparameter('nhead', [4]),
        "nhid_factor": 2,
        "bptt": bptt,
        "eval_positions": None,
        "seq_len_used": bptt,
        "sampling": 'normal',#hp.choice('sampling', ['mixed', 'normal']), # uniform
        "epochs": 80,
        "num_steps": 100,
        "verbose": False,
        "mix_activations": False,
        "pre_sample_causes": True,
        "multiclass_type": 'rank'
    }

    return config_general


def get_diff_causal():
    """"
    Returns the configuration parameters for a differentiable wrapper around MLP / Causal mixture.
    """
    diff_causal = {
        #"mix_activations": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        #"num_layers": {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': 6, 'min_mean': 1, 'round': True,
        #               'lower_bound': 2},
        "num_layers": {'distribution': 'meta_gamma', 'max_alpha': 2, 'max_scale': 3, 'round': True,
                       'lower_bound': 2},
        # Better beta?
        #"prior_mlp_hidden_dim": {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': 130, 'min_mean': 5,
        #                         'round': True, 'lower_bound': 4},
        "prior_mlp_hidden_dim": {'distribution': 'meta_gamma', 'max_alpha': 3, 'max_scale': 100, 'round': True, 'lower_bound': 4},

        "prior_mlp_dropout_prob": {'distribution': 'meta_beta', 'scale': 0.6, 'min': 0.1, 'max': 5.0},
    # This mustn't be too high since activations get too large otherwise

        "noise_std": {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': .3, 'min_mean': 0.0001, 'round': False,
                      'lower_bound': 0.0},
        "init_std": {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': 10.0, 'min_mean': 0.01, 'round': False,
                     'lower_bound': 0.0},
        #"num_causes": {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': 12, 'min_mean': 1, 'round': True,
        #               'lower_bound': 1},
        "num_causes": {'distribution': 'meta_gamma', 'max_alpha': 3, 'max_scale': 7, 'round': True,
                                 'lower_bound': 2},

        "is_causal": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        "pre_sample_weights": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        "y_is_effect": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        "sampling": {'distribution': 'meta_choice', 'choice_values': ['normal', 'mixed']},
        "prior_mlp_activations": {'distribution': 'meta_choice_mixed', 'choice_values': [
            torch.nn.Tanh
            , torch.nn.Identity
            , torch.nn.ReLU
        ]},
        "block_wise_dropout": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        "sort_features": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        "in_clique": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        #'pre_sample_causes': {'distribution': 'meta_choice', 'choice_values': [True, False]},
    }

    return diff_causal



class GaussianNoise(nn.Module):
    def __init__(self, std, device):
        super().__init__()
        self.std = std
        self.device=device

    def forward(self, x):
        return x + torch.normal(torch.zeros_like(x), self.std)


def causes_sampler_f(num_causes):
    means = np.random.normal(0, 1, (num_causes))
    std = np.abs(np.random.normal(0, 1, (num_causes)) * means)
    return means, std

'''
GRG - Notes:

seq_len             : denotes the dataset size i.e. number of samples
batch_size          : is used to get the number of fixed sub-samples in the given seq_len
new_mlp_per_example : every other batch can be sampled from a new mlp if this flag is set

'''
def get_batch(batch_size, seq_len, num_features, hyperparameters, device='cuda:1', num_outputs=1, sampling='normal'
              , epoch=None, **kwargs):
    if 'multiclass_type' in hyperparameters and hyperparameters['multiclass_type'] == 'multi_node':
        num_outputs = num_outputs * hyperparameters['num_classes']

    if not (('mix_activations' in hyperparameters) and hyperparameters['mix_activations']):
        s = hyperparameters['prior_mlp_activations']()
        hyperparameters['prior_mlp_activations'] = lambda : s

    class MLP(torch.nn.Module):
        def __init__(self, hyperparameters):
            super(MLP, self).__init__()

            self.num_layers = hyperparameters['num_layers']
            self.is_causal = hyperparameters['is_causal']
            self.verbose = hyperparameters['verbose']
            self.pre_sample_causes = hyperparameters['pre_sample_causes']
            self.random_feature_rotation = hyperparameters['random_feature_rotation']
            self.block_wise_dropout = hyperparameters['block_wise_dropout']
            self.noise_std = hyperparameters['noise_std']
            self.init_std = hyperparameters['init_std']
            self.prior_mlp_scale_weights_sqrt = hyperparameters['prior_mlp_scale_weights_sqrt']
            self.sampling = hyperparameters['sampling']
            self.in_clique = hyperparameters['in_clique']
            self.sort_features = hyperparameters['sort_features']

            with torch.no_grad():

                for key in hyperparameters:
                    setattr(self, key, hyperparameters[key])

                assert (self.num_layers >= 2)

                if 'verbose' in hyperparameters and self.verbose:
                    print({k : hyperparameters[k] for k in ['is_causal', 'num_causes', 'prior_mlp_hidden_dim'
                        , 'num_layers', 'noise_std', 'y_is_effect', 'pre_sample_weights', 'prior_mlp_dropout_prob'
                        , 'pre_sample_causes']})

                if self.is_causal:
                    self.prior_mlp_hidden_dim = max(self.prior_mlp_hidden_dim, num_outputs + 2 * num_features)
                else:
                    self.num_causes = num_features

                # This means that the mean and standard deviation of each cause is determined in advance
                if self.pre_sample_causes:
                    self.causes_mean, self.causes_std = causes_sampler_f(self.num_causes)
                    self.causes_mean = torch.tensor(self.causes_mean, device=device).unsqueeze(0).unsqueeze(0).tile(
                        (seq_len, 1, 1))
                    self.causes_std = torch.tensor(self.causes_std, device=device).unsqueeze(0).unsqueeze(0).tile(
                        (seq_len, 1, 1))

                def generate_module(layer_idx, out_dim):
                    # Determine std of each noise term in initialization, so that is shared in runs
                    # torch.abs(torch.normal(torch.zeros((out_dim)), self.noise_std)) - Change std for each dimension?
                    noise = (GaussianNoise(torch.abs(torch.normal(torch.zeros(size=(1, out_dim), device=device), float(self.noise_std))), device=device)
                         if self.pre_sample_weights else GaussianNoise(float(self.noise_std), device=device))
                    return [
                        nn.Sequential(*[self.prior_mlp_activations()
                            , nn.Linear(self.prior_mlp_hidden_dim, out_dim)
                            , noise])
                    ]

                self.layers = [nn.Linear(self.num_causes, self.prior_mlp_hidden_dim, device=device)]
                self.layers += [module for layer_idx in range(self.num_layers-1) for module in generate_module(layer_idx, self.prior_mlp_hidden_dim)]
                if not self.is_causal:
                    self.layers += generate_module(-1, num_outputs)
                self.layers = nn.Sequential(*self.layers)

                # Initialize Model parameters
                for i, (n, p) in enumerate(self.layers.named_parameters()):
                    if self.block_wise_dropout:
                        if len(p.shape) == 2: # Only apply to weight matrices and not bias
                            nn.init.zeros_(p)
                            # TODO: N blocks should be a setting
                            n_blocks = random.randint(1, math.ceil(math.sqrt(min(p.shape[0], p.shape[1]))))
                            w, h = p.shape[0] // n_blocks, p.shape[1] // n_blocks
                            keep_prob = (n_blocks*w*h) / p.numel()
                            for block in range(0, n_blocks):
                                nn.init.normal_(p[w * block: w * (block+1), h * block: h * (block+1)], std=self.init_std / keep_prob**(1/2 if self.prior_mlp_scale_weights_sqrt else 1))
                    else:
                        if len(p.shape) == 2: # Only apply to weight matrices and not bias
                            dropout_prob = self.prior_mlp_dropout_prob if i > 0 else 0.0  # Don't apply dropout in first layer
                            dropout_prob = min(dropout_prob, 0.99)
                            nn.init.normal_(p, std=self.init_std / (1. - dropout_prob**(1/2 if self.prior_mlp_scale_weights_sqrt else 1)))
                            p *= torch.bernoulli(torch.zeros_like(p) + 1. - dropout_prob)

        def forward(self):
            def sample_normal():
                if self.pre_sample_causes:
                    causes = torch.normal(self.causes_mean, self.causes_std.abs()).float()
                else:
                    causes = torch.normal(0., 1., (seq_len, 1, self.num_causes), device=device).float()
                return causes

            if self.sampling == 'normal':
                causes = sample_normal()
            elif self.sampling == 'mixed':
                zipf_p, multi_p, normal_p = random.random() * 0.66, random.random() * 0.66, random.random() * 0.66
                def sample_cause(n):
                    if random.random() > normal_p:
                        if self.pre_sample_causes:
                            return torch.normal(self.causes_mean[:, :, n], self.causes_std[:, :, n].abs()).float()
                        else:
                            return torch.normal(0., 1., (seq_len, 1), device=device).float()
                    elif random.random() > multi_p:
                        x = torch.multinomial(torch.rand((random.randint(2, 10))), seq_len, replacement=True).to(device).unsqueeze(-1).float()
                        x = (x - torch.mean(x)) / torch.std(x)
                        return x
                    else:
                        x = torch.minimum(torch.tensor(np.random.zipf(2.0 + random.random() * 2, size=(seq_len)),
                                            device=device).unsqueeze(-1).float(), torch.tensor(10.0, device=device))
                        return x - torch.mean(x)
                causes = torch.cat([sample_cause(n).unsqueeze(-1) for n in range(self.num_causes)], -1)
            elif self.sampling == 'uniform':
                causes = torch.rand((seq_len, 1, self.num_causes), device=device)
            else:
                raise ValueError(f'Sampling is set to invalid setting: {sampling}.')

            outputs = [causes]
            for layer in self.layers:
                outputs.append(layer(outputs[-1]))
            outputs = outputs[2:]

            if self.is_causal:
                ## Sample nodes from graph if model is causal
                outputs_flat = torch.cat(outputs, -1)

                if self.in_clique:
                    random_perm = random.randint(0, outputs_flat.shape[-1] - num_outputs - num_features) + torch.randperm(num_outputs + num_features, device=device)
                else:
                    random_perm = torch.randperm(outputs_flat.shape[-1]-1, device=device)

                #GRG-Bug Note: y and x can map to the same index! this would imply we lose a feature and label is included in training!!!
                random_idx_y = list(range(-num_outputs, -0)) if self.y_is_effect else random_perm[0:num_outputs]
                random_idx = random_perm[num_outputs:num_outputs + num_features]

                if self.sort_features:
                    random_idx, _ = torch.sort(random_idx)
                y = outputs_flat[:, :, random_idx_y]

                x = outputs_flat[:, :, random_idx]
            else:
                y = outputs[-1][:, :, :]
                x = causes

            if bool(torch.any(torch.isnan(x)).detach().cpu().numpy()) or bool(torch.any(torch.isnan(y)).detach().cpu().numpy()):
                print('Nan caught in MLP model x:', torch.isnan(x).sum(), ' y:', torch.isnan(y).sum())
                print({k: hyperparameters[k] for k in ['is_causal', 'num_causes', 'prior_mlp_hidden_dim'
                    , 'num_layers', 'noise_std', 'y_is_effect', 'pre_sample_weights', 'prior_mlp_dropout_prob'
                    , 'pre_sample_causes']})

                x[:] = 0.0
                y[:] = -100 # default ignore index for CE

            # random feature rotation
            if self.random_feature_rotation:
                x = x[..., (torch.arange(x.shape[-1], device=device)+random.randrange(x.shape[-1])) % x.shape[-1]] #GRG - Bug note: This does not guarantee all feature index are sampled - as it can repeat thus ignoring some features! 

            #GRG
            self.outputs = outputs
            self.outputs_flat = outputs_flat
            self.random_idx = random_idx
            self.random_idx_y = random_idx_y

            return x, y


    if hyperparameters.get('new_mlp_per_example', False):
        get_model = lambda: MLP(hyperparameters).to(device)
    else:
        model = MLP(hyperparameters).to(device)
        get_model = lambda: model

    # sample = [get_model()() for _ in range(0, batch_size)]

    models = [get_model() for _ in range(0, batch_size)]

    sample = [m() for m in models]

    x, y = zip(*sample)
    y = torch.cat(y, 1).detach().squeeze(2) #seq_len x batch_size x num_outputs
    x = torch.cat(x, 1).detach() #seq_len x batch_size x num_features

    print(f"x.shape = {x.shape}; y.shape = {y.shape}")

    return x, y, models


# DataLoader = get_batch_to_dataloader(get_batch)


import os 
import networkx as nx
import pandas as pd

def drawMLPasDAG(mlp, num_inputs, dag_path, float_precision = 2):

    #Process mlp

    features_ids = mlp.random_idx
    label_ids = mlp.random_idx_y
    
    outputs =  mlp.outputs
    outputs_flat =  mlp.outputs_flat


    adjacency_matrix = np.zeros((num_inputs, num_inputs))

    modules_list = [ m for m in mlp.named_modules()]

    mlp_vars_count = 0 
    for (name, module) in modules_list[1:]:

        if isinstance(module, nn.Linear):

            wt = module.weight
            b = module.bias

            current_vars = wt.shape[1]

            adjacency_matrix = np.pad(adjacency_matrix, (0, wt.shape[0]))
            
            for i in range(wt.shape[0]):
                
                for v in range(wt.shape[1]):
                
                    #Update adjacency matrix
                    edge_val = float(f"{wt[i,v]:.{float_precision}f}")
                    if abs(edge_val) > 0:
                        adjacency_matrix[v + mlp_vars_count, i + mlp_vars_count + current_vars] = edge_val

            
            mlp_vars_count = mlp_vars_count + current_vars

    
    num_vars = adjacency_matrix.shape[0]

    vars_ids = [v for v in range(num_vars)]
    vars_list = [f'z{v}' for v in range(num_vars)]

    start_idx = sum(modules_list[2][1].weight.shape)
    updated_features_ids = features_ids + start_idx

    assert label_ids[0] < 0, "Error! label id index start from reverse i.e. negative"
    updated_label_ids = [vars_ids[l] for l in label_ids]

    adj_labels_dict = {}
    features_list = []
    labels_list = []
    latents_list = []

    for idx in vars_ids:

        if idx in updated_label_ids:
            var_id = f'y{len(labels_list)+1}'
            labels_list.append(var_id) 
            adj_labels_dict[idx] = var_id

        elif idx in updated_features_ids:
            var_id = f'x{len(features_list)+1}'
            features_list.append(var_id) 
            adj_labels_dict[idx] = var_id

        else:
            var_id = f'z{len(latents_list)+1}'
            latents_list.append(var_id) 
            adj_labels_dict[idx] = var_id


    assert len(features_ids) == len(features_list), "Error! Missing var."
    assert len(latents_list) + len(features_list) + len(label_ids) == adjacency_matrix.shape[0], "Error! Missing var."
    assert len(adj_labels_dict.keys()) == adjacency_matrix.shape[0], "Error! Missing var."

    adj_labels = [adj_labels_dict[v] for v in vars_ids]

    ##Plot Graph

    #Draw original graph with latents
    drawGraph(adjacency_matrix, adj_labels, dag_path)

    #Filter out latents
    adj_labels_to_ids_dict = {}
    for k,v in adj_labels_dict.items():
        adj_labels_to_ids_dict[v] = k

    non_latent_vars = features_list+labels_list
    non_latent_vars_ids = [adj_labels_to_ids_dict[f] for f in non_latent_vars]
    # non_latent_adjacency_matrix = adjacency_matrix[non_latent_vars_ids, :] #This gives edges from non_latent_vars_ids; If we did adj_mat[:, ids] we get incoming edges to non_latent_vars_ids 
    non_latent_adjacency_matrix = np.zeros((len(non_latent_vars_ids), len(non_latent_vars_ids)), dtype = adjacency_matrix.dtype)
    for idx, var in enumerate(non_latent_vars_ids):
        non_latent_adjacency_matrix[idx] = adjacency_matrix[var, non_latent_vars_ids]

    non_latent_adj_labels = [adj_labels_dict[v] for v in non_latent_vars_ids]
    non_latent_dag_path = dag_path[:-4] + "_non_latent" + dag_path[-4:]

    #Draw graph without latents
    drawGraph(non_latent_adjacency_matrix, non_latent_adj_labels, non_latent_dag_path)


    return features_list, labels_list, latents_list, adjacency_matrix, non_latent_adjacency_matrix, adj_labels_dict


def drawGraph(adjacency_matrix, adj_labels, dag_path):

    adjacency_matrix_pd = pd.DataFrame(adjacency_matrix, index=adj_labels, columns=adj_labels)
    # Graph = nx.from_pandas_adjacency(adjacency_matrix_pd)
    Graph = nx.from_pandas_adjacency(adjacency_matrix_pd, create_using = nx.DiGraph)

    # plt.figure()
    labels = nx.get_edge_attributes(Graph, "weight")
    
    #Change float precision
    for k,v in labels.items():
        labels[k] = f'{v:0.2f}'
        # edge_val = float(f'{v:0.2f}')
        # if abs(edge_val) > 0:
        #     labels[k] = edge_val

    # # pos = nx.spring_layout(G)
    # # pos = nx.spring_layout(G, scale=3)
    # # pos = nx.spectral_layout(G)
    # # pos = nx.spiral_layout(Graph, scale=3)
    # # pos = nx.nx_agraph.graphviz_layout(Graph, prog="dot")
    # pos = nx.nx_agraph.graphviz_layout(Graph, prog="dot", args = "-Gnodesep=0.1 -size='20,20'")

    # nx.draw_networkx(Graph, with_labels=True, pos=pos)
    # nx.draw_networkx_edge_labels(Graph, pos=pos, edge_labels=labels)

    # plt.savefig(os.path.join(reports_path, f"learnt_model_{candidate_num}_injection.png"))

    # plt.figure()
    A = nx.nx_agraph.to_agraph(Graph)        # convert to a graphviz graph
    A.layout(prog='dot')            # neato layout
    #A.draw('test3.pdf')

    root_nodes = np.unique([e1 for (e1, e2), v in labels.items()])
    root_nodes_colors = {}

    for idx, node in enumerate(root_nodes):
        color =  "#"+''.join([hex(np.random.randint(0,16))[-1] for i in range(6)])
        root_nodes_colors[node] = color

    for (e1, e2), v in labels.items():
        edge = A.get_edge(e1,e2)
        edge.attr['weight'] = v
        edge.attr['label'] = str(v)
        # edge.attr['color'] = "red:blue"
        edge.attr['color'] = root_nodes_colors[e1]
        
    A.draw(dag_path,
            args='-Gnodesep=1.0 -Granksep=9.0 -Gfont_size=1', prog='dot' )  




def get_obs_inter_split(data, data_split_type, vars_list, class_list):
    
    cls_start_dix = len(vars_list)

    data_ft, data_lb = data[:, :cls_start_dix], data[:, cls_start_dix:]
    assert data_ft.shape == (data.shape[0], len(vars_list)), "Error! Feature-label split incorrect."
    assert data_lb.shape == (data.shape[0], len(class_list)), "Error! Feature-label split incorrect."

    cls_labels = np.unique(data_lb)

    cls_dict = {} 
    for idx, cls in enumerate(class_list):
            
        cls_index = np.where(data_lb[:, idx] == 1)
        
        cls_dict[cls] = {"ft": data_ft[cls_index], "lb": data_lb[cls_index]}


    features_list = vars_list + class_list

    # inter_data = []
    intervened_vars = []
    num_inter_samples_per_cls = cls_dict[class_list[-1]]['ft'].shape[0]
    inter_data = np.zeros((len(features_list), num_inter_samples_per_cls, len(features_list)), dtype = data.dtype)
    for idx, (cls, cls_data) in enumerate(cls_dict.items()):

        assert np.unique(cls_data["lb"].argmax(1), return_counts = True) == (idx, cls_data["ft"].shape[0]), "Error! Labels not correctly split."

        if idx == 0:
            obs_ft = cls_data["ft"]
            obs_lb = cls_data["lb"]
            obs_data = np.concatenate((obs_ft, obs_lb), axis = 1)
        else:
            inter_ft = cls_data["ft"]
            inter_lb = cls_data["lb"]
            inter_cls_data = np.concatenate((inter_ft, inter_lb), axis = 1)

            # intervened_var = np.zeros(len(features_list))
            cls_feature_idx = features_list.index(cls)
            # intervened_var[cls_feature_idx] = 1

            # assert intervened_var.argmax() == cls_feature_idx, "Error! Intervened var set incorrectly."

            # np.concatenate((intervened_var, inter_cls_data), axis = 1)
            # np.tile(intervened_var[:, np.newaxis, np.newaxis], (1, inter_cls_data.shape[0], inter_cls_data.shape[1])).shape

            # np.concatenate((intervened_var[:, np.newaxis, np.newaxis], inter_cls_data[np.newaxis, :, :]), axis = 0).shape

            inter_data[cls_feature_idx] = inter_cls_data
            intervened_vars.append(cls)

    print(f"dtype of data = {data.dtype}, obs_data = {obs_data.dtype}, inter_data = {inter_data.dtype}")

    if data_split_type == "obs_plus_interv":
        obs_data = data
    elif data_split_type == "obs_interv":
        pass
    else:
        raise Exception(f"Error! Unsupported data_split_type = {data_split_type}")

    return obs_data, inter_data, intervened_vars



def saveDataInDCIDFormat(data_obs, data_split_type, adj_matrix, variables, classes, feature_type, results_dir):

    results_dir = os.path.join(results_dir, data_split_type)
    utils.createDirIfDoesntExists(results_dir)

    trial = results_dir.split('/')[1] + "_" + results_dir.split('/')[-1]

    # data_path = os.path.join(results_dir, "train_data.npz")
    # data_path = os.path.join(results_dir, f"{trial}_{data_split_type}.npz")
    data_path = os.path.join(results_dir, f"{trial}.npz")


    if data_split_type == "obs":
        
        # Save data for ENCO
        np.savez(data_path, data_obs = data_obs, data_int = [], adj_matrix = adj_matrix, 
            vars_list = variables, class_list = classes, feature_type = feature_type, intervened_vars = [])

        # Save data for DCDI
        np.save(f"{results_dir}/data1-{trial}.npy", data_obs)

        np.save(f"{results_dir}/DAG1-{trial}.npy", adj_matrix)

        return


    train_obs, train_inter, train_intervened_vars = get_obs_inter_split(data_obs, data_split_type, variables, classes)

    ## Save data for ENCO
    # np.savez(data_path, data_obs = data_obs, data_int = data_int, adj_matrix = adj_matrix, 
    #         vars_list = variables, class_list = classes, feature_type = feature_type)
    np.savez(data_path, data_obs = train_obs, data_int = train_inter, adj_matrix = adj_matrix, 
            vars_list = variables, class_list = classes, feature_type = feature_type, intervened_vars = train_intervened_vars)

    
    ## Save data for DCDI format 

    features_list = variables + classes

    dcdi_train_obs_interv = []
    regimes = []
    interv_mask = []

    #Iterate in reverse order
    for idx, inter_var in enumerate(train_intervened_vars[::-1]):

        inter_var_idx = features_list.index(inter_var)

        inter_data = train_inter[inter_var_idx] 

        dcdi_train_obs_interv.extend(inter_data)

        regime_id = len(train_intervened_vars) - idx
        regimes += [regime_id] * inter_data.shape[0]

        #Note - intervention mask ids start with index 0
        interv_mask += [inter_var_idx] * inter_data.shape[0]

    #Add observational data - regime_id is '0' for observation data
    dcdi_train_obs_interv.extend(train_obs)
    regimes += [0] * train_obs.shape[0]

    dcdi_train_obs_interv = np.array(dcdi_train_obs_interv)
    regimes = np.array(regimes, dtype = np.int)
    interv_mask = np.array(interv_mask, dtype = np.int)

    assert regimes.shape[0] == (len(train_intervened_vars)*train_inter.shape[1]) + train_obs.shape[0], "Error! Size incorrect."
    assert interv_mask.shape[0] == (len(train_intervened_vars)*train_inter.shape[1]), "Error! Size incorrect."
    assert dcdi_train_obs_interv.shape == ((len(train_intervened_vars)*train_inter.shape[1])  + train_obs.shape[0], train_obs.shape[1]), "Error! Size incorrect."
    assert adj_matrix.shape == (dcdi_train_obs_interv.shape[1], dcdi_train_obs_interv.shape[1]), "Error! Size incorrect."

    
    np.save(f"{results_dir}/data_interv1-{trial}.npy", dcdi_train_obs_interv)

    np.save(f"{results_dir}/DAG1-{trial}.npy", adj_matrix)

    np.savetxt(f"{results_dir}/regime1-{trial}.csv", regimes, fmt = "%d")

    np.savetxt(f"{results_dir}/intervention1-{trial}.csv", interv_mask, fmt = "%d")

    file1 = open(f"{results_dir}/intervention1-{trial}.csv", "a")  # append mode
    for i in range(train_obs.shape[0]):
        file1.write("\n")
    file1.close()


def getFeatureType(X, vars_list, class_list):

    ## Set variable feature type 

    feature_type = {}
    for var, x in zip(vars_list, X.T):
        
        var_unique = np.unique(x).tolist()
        
        if var_unique == [0,1]:
            var_type = 'binary'
        elif len(var_unique) <= 20:
            var_type = 'categorical'
        elif min(var_unique) == 0 and max(var_unique) == 1:
            var_type = 'continous'
        else:
            var_type = 'continous'
        
        # print(f"{var} feature type set as {var_type}; np.unique({var}) = {var_unique}")
        print(f"{var} feature type set as {var_type};")
        
        feature_type[var] = var_type
        

    for cls in class_list:
        # feature_type[cls] = 'binary'
        feature_type[cls] = 'binary-class'

    return feature_type




def get_combined_features_labelOHE(data_features, data_labels, classes, num_features):

    assert np.array_equal(np.unique(data_labels), np.arange(len(classes))), "Error! Data labels not in correct format"

    #convert to one-hot encoding
    # data_labels_ohe = np.zeros((data_labels.shape[0], len(classes)))
    data_labels_ohe = np.zeros((data_labels.shape[0], len(classes)), dtype = data_features.dtype)
    data_labels_ohe[np.arange(data_labels.shape[0]), data_labels] = 1
    assert np.array_equal(data_labels_ohe.argmax(1), data_labels), "Error! One-hot-encoding incorrectly done."

    data_obs = np.concatenate((data_features, data_labels_ohe), axis = 1)
    assert data_obs.shape == (data_features.shape[0], num_features), "Error! Dataset not in correct format."

    return data_obs


if __name__ == "__main__":

    seed = 46 #46 #42

    #Set seed to improve reproducibility 
    zero.improve_reproducibility(seed)

    results_path = "./data_mlp_gen/trial-2"
    utils.createDirIfDoesntExists(results_path)

    hyperparams = {
        # "prior_mlp_activations": {'distribution': 'meta_choice_mixed', 'choice_values': [
        #     torch.nn.Tanh
        #     , torch.nn.Identity
        #     , torch.nn.ReLU
        # ]},
        "prior_mlp_activations": torch.nn.ReLU,
        "is_causal": True,
        "verbose": True,
        "pre_sample_causes": True,
        "random_feature_rotation": False,
        
        'num_causes': 2, #4, 
        'prior_mlp_hidden_dim': 4, #10,
        'num_layers': 3, #3, 
        'noise_std': 0.03, 
        'y_is_effect': True, 
        'pre_sample_weights': True, 
        'prior_mlp_dropout_prob': 0.66, #0.2,

        'block_wise_dropout': False,
        'init_std': 0.3,
        'prior_mlp_scale_weights_sqrt': True,
        'sampling': 'normal',
        'in_clique': False,
        'sort_features': True,
    }

    num_of_mlps = 1
    num_of_samples_per_mlp = 10000 #100
    num_features = 3 #12
    num_outputs = 2 #1

    hyperparams['seed'] = seed
    hyperparams['num_of_mlps'] = num_of_mlps
    hyperparams['num_of_samples_per_mlp'] = num_of_samples_per_mlp
    hyperparams['num_features'] = num_features
    hyperparams['num_outputs'] = num_outputs


    x, y, models = get_batch(batch_size=num_of_mlps, seq_len=num_of_samples_per_mlp, num_features=num_features, hyperparameters=hyperparams, num_outputs=num_outputs)

    #Draw the Model DAG
    mlp = models[0]

    num_causes = hyperparams['num_causes']
    dag_path = os.path.join(results_path, f"mlp_generator_dag.png")

    ## Save results

    #Draw the MLP as DAG
    features_list, labels_list, latents_list, adjacency_matrix, non_latent_adjacency_matrix, adj_labels_dict = drawMLPasDAG(mlp, num_causes, dag_path)

    #Save model
    torch.save(mlp.state_dict(), os.path.join(results_path, f'mlp.pt'))

    #Save hyperparams
    hyperparams["prior_mlp_activations"] = f"{hyperparams['prior_mlp_activations']()}"
    utils.writeJson(hyperparams, os.path.join(results_path, f"hyperparams.json"))


    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()

    #Save dataset
    np.savez(os.path.join(results_path, f"original_data.npz"), x = x, y = y, 
        features_list = features_list, labels_list = labels_list, latents_list = latents_list, 
        adjacency_matrix = adjacency_matrix, non_latent_adjacency_matrix = non_latent_adjacency_matrix, 
        adj_labels_dict = adj_labels_dict)
    
    y_label = y.squeeze(1).argmax(1)

    print(f"np.unique(y_label, return_counts = True) = {np.unique(y_label, return_counts = True)}")
    
    data_obs = get_combined_features_labelOHE(data_features = x.squeeze(1), data_labels = y_label, classes = labels_list, num_features = len(features_list) + len(labels_list))

    feature_type = getFeatureType(X = x, vars_list = features_list, class_list = labels_list)
    print(f"feature_type = {feature_type}")

    #Save data in DCDI format

    data_split_type = "obs_interv" #"obs_plus_interv" #"obs_plus_interv" #"obs_interv" #obs

    saveDataInDCIDFormat(data_obs, data_split_type, adj_matrix = non_latent_adjacency_matrix, 
            variables = features_list, classes = labels_list, 
            feature_type = feature_type, results_dir = results_path)

    pass
