import numpy as np


import torch

import os

import pandas as pd
import networkx as nx

from graph_utils import adj_matrix_to_edges, edges_or_adj_matrix, sort_graph_by_vars, get_node_relations

import utils


import random



def set_seed(seed = 42):
    """
    Sets the seed for all libraries used.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




def preprocessAdjMatrix(adj_matrix, num_vars, num_classes):

    ##Process adjacency matrix   

    REVERSE_OUTGOING_CLASS_EDGES = True

    if not REVERSE_OUTGOING_CLASS_EDGES:
        #Mask outgoing edges that start from severity to other features
        adj_matrix[-num_classes:, :] = 0 #Rows represent outgoing edges and Columns represent incoming edges #Classes
    else:
        #Reverse the edge direction of severity classes; convert outgoing edges to incoming edges 
        for s_idx in range(num_vars, num_vars+num_classes): #Classes
            outgoing_edges = adj_matrix[s_idx, :]
            outgoing_nodes = np.where(outgoing_edges == 1)
            adj_matrix[outgoing_nodes, s_idx] = 1
        
        #TODO-GRG: Check this
        #Now remove outgoing edges - otherwise this introduces cycles
        
        #Mask outgoing edges that start from severity to other features
        adj_matrix[-num_classes:, :] = 0 #Rows represent outgoing edges and Columns represent incoming edges #Classes

    return adj_matrix



def getSamplingOrder(adj_matrix, variables, logger):


    G = pd.DataFrame(adj_matrix, index = variables, columns = variables)
    G = nx.from_pandas_adjacency(G, create_using=nx.DiGraph)

    #Node degrees 
    node_degrees = {}
    for var in variables:
        node_degrees[var]= G.degree[var]

    #Node in-degrees 
    node_in_degrees = {}
    for var in variables:
        node_in_degrees[var]= G.in_degree[var]

    in_degree_sort_idx = np.argsort(list(node_in_degrees.values()))
    ascending_in_degree_nodes =  np.array(variables.copy())[in_degree_sort_idx]

    #Node out-degrees 
    node_out_degrees = {}
    for var in variables:
        node_out_degrees[var]= G.out_degree[var]

    out_degree_sort_idx = np.argsort(list(node_out_degrees.values()))
    ascending_out_degree_nodes =  np.array(variables.copy())[out_degree_sort_idx]

    #Node parents
    node_parents = {}
    for var in variables:
        node_parents[var]= list(G.predecessors(var))

    unconnected_nodes = [v for v,d in node_degrees.items() if d == 0]
    logger.log(f"Unconnected nodes = {unconnected_nodes}")

    root_nodes = [n for n,d in G.in_degree() if d == 0 and G.degree(n) != 0] 
    logger.log(f"Root nodes = {root_nodes}")

    # nx.ancestors(G, 'd0')

    sorted_variables, edges, adj_matrix, sorted_idxs = sort_graph_by_vars(variables, adj_matrix = adj_matrix)

    # sorted_variables = np.array(variables)[sorted_idxs]

    # paths_dict = {}
    # for node in G:
    #     if G.out_degree(node)==0: #it's a leaf
    #         paths_dict[node] = nx.shortest_path(G, root, node)

    for var in sorted_variables:
        logger.log(f"{var}: {node_parents[var]}")

    # return G, sorted_variables, node_parents, root_nodes, unconnected_nodes

    graph_params = {"G": G, "sorted_variables": sorted_variables, "node_parents": node_parents, "root_nodes": root_nodes, "unconnected_nodes": unconnected_nodes}

    return graph_params


#For saving & loading sklearn model - Ref: https://scikit-learn.org/stable/modules/model_persistence.html#security-maintainability-limitations
from joblib import dump, load

def train_generative_models(sorted_variables, node_parents, root_nodes, unconnected_nodes, train_data, category_classes, 
            causal_feature_mapping, feature_type_dict, hidden_layer_sizes = (128, 64, 32), max_iter = 200, 
            ADD_NOISE = False, reports_path = "."):

    trained_models = {}

    for var in sorted_variables:

        if var in unconnected_nodes:
            print(f"Skipping unconnected node {var}")

            train_labels = train_data[:, causal_feature_mapping[var]]
            train_label_classes = np.unique(train_labels)
            trained_models[var] = {'label_class': train_label_classes}

            continue
        elif var in root_nodes:
            print(f"Skipping root node {var}")

            train_labels = train_data[:, causal_feature_mapping[var]]
            train_label_classes = np.unique(train_labels)
            trained_models[var] = {'label_class': train_label_classes}

            continue

        parents = node_parents[var]
        print(f"Train for node {var} with parents {parents}")


        parents_idx = [causal_feature_mapping[v] for v in parents]

        train_sub_features = train_data[:, parents_idx]

        # train_labels = train_data[:, category_classes]
        train_labels = train_data[:, causal_feature_mapping[var]]

        train_label_classes = np.unique(train_labels)

        if ADD_NOISE:
            train_sub_features = train_sub_features + rng.normal(0, 0.01, train_sub_features.shape)

        ##Train model
        model, accuracy, ml_predictions, ml_prob_predictions = fitLargeMLP(
                                    train_label_ft =  train_sub_features, 
                                    test_label_ft = train_sub_features, 
                                    gt_train_scores = train_labels, gt_test_scores = train_labels, 
                                    label_classes = train_label_classes,
                                    feature_type = feature_type_dict[var],
                                    n_trial = 3,
                                    # hidden_layer_sizes = (128, 64, 32),
                                    hidden_layer_sizes = hidden_layer_sizes,
                                    max_iter = max_iter,
                                    verbose = False
                                )

        #Train model

        #Save trained model
        dump(model, os.path.join(reports_path, f"Gen_ML_Model_var_{var}.joblib")) 

        trained_models[var] = {'model': model, 'label_class': train_label_classes}

    return trained_models



def load_generative_models(sorted_variables, node_parents, root_nodes, unconnected_nodes, train_data, 
            causal_feature_mapping, reports_path = "."):

    trained_models = {}

    for var in sorted_variables:

        if var in unconnected_nodes:
            print(f"Skipping unconnected node {var}")

            train_labels = train_data[:, causal_feature_mapping[var]]
            train_label_classes = np.unique(train_labels)
            trained_models[var] = {'label_class': train_label_classes}

            continue
        elif var in root_nodes:
            print(f"Skipping root node {var}")

            train_labels = train_data[:, causal_feature_mapping[var]]
            train_label_classes = np.unique(train_labels)
            trained_models[var] = {'label_class': train_label_classes}

            continue

        parents = node_parents[var]
        print(f"Load model for node {var} with parents {parents}")


        #Save trained model
        model = load(os.path.join(reports_path, f"Gen_ML_Model_var_{var}.joblib")) 

        trained_models[var] = {'model': model, 'label_class': train_label_classes}

    return trained_models



#Numpy new way to sample from distributions: Ref: https://numpy.org/doc/stable/reference/random/index.html#random-quick-start
from numpy.random import default_rng
rng = default_rng()

def generateNewData(trained_models, sorted_variables, variables, node_parents, root_nodes, unconnected_nodes, 
            category_classes, feature_type_dict, num_samples = 1000):

    generated_data_dict = {} 

    for var in sorted_variables:

        feature_type = feature_type_dict[var]

        if var in unconnected_nodes:
            print(f"Generating unconnected node {var}")

            # if feature_type == "binary":
            #     sampled_var = rng.binomial(n = 1, p = 0.5, size = num_samples)

            # elif feature_type == "categorical":

            #     model_dict = trained_models[var]
            #     label_class = model_dict['label_class']

            #     sampled_var = rng.choice(label_class, size = num_samples)

            # elif feature_type == "continous":
                
            #     sampled_var = rng.uniform(low = 0, high = 1, size = num_samples)

            # else:
            #     raise Exception(F"Error! Unsupported feature type = {feature_type}")

            sampled_var = np.zeros(num_samples)
            print(f"Sampling constant zero for unconnected node {var}")

            sampled_var = sampled_var[:, np.newaxis]
            generated_data_dict[var] = sampled_var

            continue

        elif var in root_nodes:
            print(f"Generating root node {var}")

            if feature_type == "binary":
                sampled_var = rng.binomial(n = 1, p = 0.5, size = num_samples)
            
            elif feature_type == "binary-class":
                # sampled_var = rng.binomial(n = 1, p = 0.5, size = num_samples)
                sampled_var = rng.uniform(low = 0, high = 1, size = num_samples)
                print(f"Error! Class var {var} in root node.")
            
            elif feature_type == "categorical":

                model_dict = trained_models[var]
                label_class = model_dict['label_class']

                sampled_var = rng.choice(label_class, size = num_samples)

            elif feature_type == "continous":
                
                sampled_var = rng.uniform(low = 0, high = 1, size = num_samples)

            else:
                raise Exception(F"Error! Unsupported feature type = {feature_type}")

            sampled_var = sampled_var[:, np.newaxis]
            generated_data_dict[var] = sampled_var
        
            continue

        parents = node_parents[var]
        print(f"Generating data for node {var} with parents {parents}")

        
        # test_features = np.array([generated_data[p] for p in parents]).T
        test_features = np.hstack([generated_data_dict[p] for p in parents])
        print(f'test_featurs.shape = {test_features.shape}')

        ##Eval model

        model_dict = trained_models[var]

        model = model_dict['model']
        label_class = model_dict['label_class']

        model, ml_predictions, ml_prob_predictions = predLargeMLP(model, test_features, label_class, feature_type)

        
        if feature_type == "binary":

            sampled_var = ml_predictions

        elif feature_type == "binary-class":
            
            # if len(category_classes) == 2:
            #     sampled_var = ml_predictions
            # else:
            #     assert ml_prob_predictions.shape[1] == 2, "Error! Prob pred dim should be 2 (binary)."
            #     sampled_var = ml_prob_predictions[:,1]

            assert ml_prob_predictions.shape[1] == 2, "Error! Prob pred dim should be 2 (binary)."
            sampled_var = ml_prob_predictions[:,1]

        elif feature_type == "categorical":

            sampled_var = ml_predictions

        elif feature_type == "continous":

            sampled_var = ml_prob_predictions

            # sampled_var = sampled_var.argmax(1)

        else:
            raise Exception(F"Error! Unsupported feature type = {feature_type}")


        sampled_var = sampled_var[:, np.newaxis]

        generated_data_dict[var] = sampled_var

    
    generated_data = np.hstack([generated_data_dict[v] for v in variables])

    return generated_data





def main():

    exp_name = "generateContinousSamples"
    
    # task = 'heart-disease' #'heart-disease-binary' #'heart-disease' #'parity5' #'labor'
    # task = 'cifar-t1'
    task = 'higgs_small-t1'

    # causal_discovery_exp_dir = f"/home/grg/Research/ENCO/checkpoints/2022_26_Acyclic_{task}_TrainUpsampledPatient"
    # causal_discovery_exp_dir = f"/home/grg/Research/ENCO-grg/checkpoints/2022_10_4_Acyclic_{task}_TrainUpsampledPatient_T1"
    causal_discovery_exp_dir = f"/home/grg/Research/ENCO-grg/checkpoints/2022_10_22_Acyclic_{task}_TrainUpsampledPatient_T1"


    # set_seed()
    ShouldTrain = False #True
    load_model_dir = "/home/grg/Research/ENCO-grg/checkpoints/2022_10_22_Acyclic_higgs-small-t1_TrainUpsampledPatient_T1/generateContinousSamples_10000"

    num_samples = 10000 #60000 #10000 #50 #200 #3000 #500 #1000
    # Binary_Features = False #True
    # features_type = 'continous' #categorical, binary, continous

    ADD_NOISE = False #False

    DRAW_GRAPH = False #False
    
    causal_mlp_hidden_layer_sizes = (128, 64, 32) #(32, 16) #(128, 64, 32)
    max_iter = 200 #200

    results_dir = os.path.join(causal_discovery_exp_dir, f"{exp_name}_{num_samples}")
    utils.createDirIfDoesntExists(results_dir)

    acyclic_adj_matrix_path = f"binary_acyclic_matrix_001_{task}_TrainUpsampledPatient.npy"

    acyclic_adj_matrix = np.load(os.path.join(causal_discovery_exp_dir, acyclic_adj_matrix_path))

    print(f"acyclic_adj_matrix.shape = {acyclic_adj_matrix.shape}")

    adj_matrix = acyclic_adj_matrix 
        
    ### Load training data
    
    train_data_path = f"datasets/{task}_TrainUpsampledPatient.npz"

    train_dataset = np.load(train_data_path, allow_pickle = True)

    train_data = train_dataset['data_obs']


    vars_list = train_dataset['vars_list']
    class_list = train_dataset['class_list']

    feature_type_dict = train_dataset['feature_type'].item()

    features_list = vars_list.tolist() + class_list.tolist()
    


    node_names_mapping = {}
    for idx, var in enumerate(features_list):
        # node_names_mapping[f"$X_\\{{idx}\\}$"] = var
        node_names_mapping["$X_{" + str(idx+1) + "}$"] = var

    causal_feature_mapping = {}
    for idx, var in enumerate(features_list):
        causal_feature_mapping[var] = idx

    num_vars = len(vars_list)
    num_classes = len(class_list)

    binary_cross_entropy = num_classes == 2

    num_categs = 1 + train_data.max(axis=0) #Bug-Fix GRG: Max should be taken along num_samples axis not num_vars axis
    assert len(num_categs) == adj_matrix.shape[0], "Error! Num categories does not match num of variables."
    new_categs_func = lambda i : num_categs[i]


    # class_names = [ 'normal', 'covid', 'interstetial', 'copd asthma', 'chf', 'other-lung', 'others', ]
    class_names = class_list


    # category_classes = list(range(38, 38+num_classes))
    category_classes = list(range(num_vars, num_vars+num_classes))



    #Calculate Scores 
    reports_path = "."
    exp_name = "Using Synthetic Data"
    report_path = os.path.join(results_dir, f"classification_report_{task}.txt")
    logger = utils.Logger(report_path)

    logger.log(f"Classification report")

    logger.log(f"Exp name: {exp_name}")



    ##Process adjacency matrix   

    REVERSE_OUTGOING_CLASS_EDGES = True

    if not REVERSE_OUTGOING_CLASS_EDGES:
        #Mask outgoing edges that start from severity to other features
        # adj_matrices[-4:, :] = 0 #Rows represent outgoing edges and Columns represent incoming edges #Severity classes
        adj_matrix[-num_classes:, :] = 0 #Rows represent outgoing edges and Columns represent incoming edges #Disease classes
    else:
        #Reverse the edge direction of severity classes; convert outgoing edges to incoming edges 
        # for s_idx in [38, 39, 40, 41]: #Severity classes
        # for s_idx in [38, 39, 40, 41, 42, 43, 44]: #Disease classes
        for s_idx in range(num_vars, num_vars+num_classes): #Disease classes
            outgoing_edges = adj_matrix[s_idx, :]
            outgoing_nodes = np.where(outgoing_edges == 1)
            adj_matrix[outgoing_nodes, s_idx] = 1
        
        #TODO-GRG: Check this
        #Now remove outgoing edges - otherwise this introduces cycles
        
        #Mask outgoing edges that start from severity to other features
        # adj_matrices[-4:, :] = 0 #Rows represent outgoing edges and Columns represent incoming edges #Severity classes
        adj_matrix[-num_classes:, :] = 0 #Rows represent outgoing edges and Columns represent incoming edges #Disease classes

    # # TODO-GRG: Need to check if we need to transpose the adj_matrix or not!!!
    # # Transpose for mask because adj[i,j] means that i->j
    # mask_adj_matrices = adj_matrix.transpose(1, 2)

    # variables = list(range(45))
    # variables = list(vars_mapping.values())
    variables = features_list

    #Get sampling order
    G, sorted_variables, node_parents, root_nodes, unconnected_nodes = getSamplingOrder(adj_matrix, variables, logger, results_dir, DRAW_GRAPH)


    ### Train Generative Models

    #TODO-GRG: Add noise while learning generator model
    
    if ShouldTrain:
        trained_gen_models = train_generative_models(sorted_variables, node_parents, root_nodes, unconnected_nodes, train_data, 
                            category_classes, causal_feature_mapping, feature_type_dict, 
                            hidden_layer_sizes = causal_mlp_hidden_layer_sizes, 
                            max_iter = max_iter,
                            ADD_NOISE = ADD_NOISE,
                            reports_path = results_dir)
    else:
        trained_gen_models = load_generative_models(sorted_variables, node_parents, root_nodes, unconnected_nodes, train_data, 
                causal_feature_mapping, reports_path = load_model_dir) #results_dir)
    

    logger.close()


    return accuracy, only_synthetic_accuracy, synthetic_accuracy



if __name__ == "__main__":
    print("Started...")

    accuracy, only_synthetic_accuracy, synthetic_accuracy = main()
    
    print("Finished!")