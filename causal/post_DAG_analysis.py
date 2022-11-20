import numpy as np

import os 

import networkx as nx
import pandas as pd

import zero 

import sys
sys.path.append("./")

import utils
import causal.utils_graph as utils_graph
import causal.causal_graph_lib as cgraph_lib
import causal.classifier_on_causal_parents_heart_disease_exp as classifier

def graphEdgesStats(adj_matrix, variables, class_list, logger):


    G = pd.DataFrame(adj_matrix, index = variables+class_list, columns = variables+class_list)
    pred_graph = nx.from_pandas_adjacency(G, create_using=nx.DiGraph)

    logger.log(f"# of Edges = {len(pred_graph.edges)}")
    logger.log(f"Edges = {pred_graph.edges}")

    class_edges = {}
    for cls in class_list:
        class_edges[cls] = [i for i in pred_graph.edges if cls in str(i)]
        logger.log(f"class {cls} edges = {class_edges[cls]}")
    
    # class_edges_list = np.vstack(list(class_edges.values())).tolist()
    class_edges_list = [v for v in class_edges.values() if len(v) > 0]
    if len(class_edges_list) > 0:
        class_edges_list = np.vstack(class_edges_list)


    class_correlated_nodes = []
    for a, b in class_edges_list:

        if a not in class_list:
            class_correlated_nodes.append(a)
        elif b not in class_list:
            class_correlated_nodes.append(b)
        else:
            print(f"Error! Detected edge between classes {(a,b)}")
            # raise Exception(f"Error! Detected edge between severity {(a,b)}")

    class_correlated_nodes = np.unique(class_correlated_nodes).tolist()

    logger.log(f"Found {len(class_correlated_nodes)} class_correlated_nodes = {class_correlated_nodes}")



def getClassCausalParents(node_parents, class_names, features_list, ignore_latents):

    causal_feature_mapping = {}
    for idx, var in enumerate(features_list):

        causal_feature_mapping[var] = idx


    class_parents = [node_parents[c] for c in class_names]

    class_parents = np.hstack(class_parents)
    class_parents = np.unique(class_parents)
    class_parents = sorted(class_parents, key = lambda x: int(x[1:]))

    print(f"Class parents : {class_parents}")
    if ignore_latents:
        class_parents = [p for p in class_parents if 'r' not in p]
        print(f"Class parents - post removing latents : {class_parents}")

    class_parents_idx = [causal_feature_mapping[var] for var in class_parents]
    print(f"Class parents idx : {class_parents_idx}")

    if ignore_latents:
        num_latents = len([p for p in features_list if 'r' in p])
        
        class_parents_idx = [i-num_latents for i in class_parents_idx]
        print(f"Class parents idx - post removing latents : {class_parents_idx}")

        assert len([i for i in class_parents_idx if i < 0]) == 0, "Error! Negative index encountered."

    return class_parents, class_parents_idx








if __name__ == "__main__":

    seed = 42
    zero.improve_reproducibility(seed)

    interv_type = ""
    # interv_type = "_imperfect"
    interv_type = "_unknown"

    # exp_trial = "heart-disease-binary-t1-obs_only"
    exp_trial = "heart-disease-binary-t1-obs_inter"
    # exp_trial = "heart-disease-binary-t1-obs_plus_inter"
    # exp_trial = "heart-disease-binary-t1-latent_obs_inter"
    # exp_trial = "heart-disease-binary-t1-latent_obs_plus_inter"
    
    # exp_pth = "/home/grg/Research/dcdi-grg/exp/heart-disease-binary-t1-obs_only_TrainUpsampledPatient"
    # exp_pth = "/home/grg/Research/dcdi-grg/exp/heart-disease-binary-t1-obs_inter_TrainUpsampledPatient"
    # exp_pth = "/home/grg/Research/dcdi-grg/exp/heart-disease-binary-t1-obs_plus_inter_TrainUpsampledPatient"
    # exp_pth = "/home/grg/Research/dcdi-grg/exp/heart-disease-binary-t1-latent_obs_inter_TrainUpsampledPatient"
    # exp_pth = "/home/grg/Research/dcdi-grg/exp/heart-disease-binary-t1-latent_obs_plus_inter_TrainUpsampledPatient"
    exp_pth = f"/home/grg/Research/dcdi-grg/exp/{exp_trial}{interv_type}_TrainUpsampledPatient"
    
    dataset_path = "/home/grg/Research/dcdi-grg/data/grg"
    # i_dataset = "1-heart-disease-binary-t1-obs_only_TrainUpsampledPatient"
    # i_dataset = "1-heart-disease-binary-t1-obs_inter_TrainUpsampledPatient"
    # i_dataset = "1-heart-disease-binary-t1-obs_plus_inter_TrainUpsampledPatient"
    # i_dataset = "1-heart-disease-binary-t1-latent_obs_inter_TrainUpsampledPatient"
    # i_dataset = "1-heart-disease-binary-t1-latent_obs_plus_inter_TrainUpsampledPatient"
    i_dataset = f"1-{exp_trial}_TrainUpsampledPatient"
    
    adj_matrix_path = os.path.join(exp_pth, "train", f"DAG.npy")

    adj_matrix = np.load(adj_matrix_path)

    graph_path = os.path.join(exp_pth, "train", f"Discovered_DAG.png")

    # drawAdjGraph(adj_matrix, dataset_path, i_dataset, graph_path)

    data = np.load( os.path.join(dataset_path, f"{i_dataset[2:]}.npz"))

    variables = data['vars_list'].tolist()
    classes = data['class_list'].tolist()

    
    logger = utils.Logger(filename = os.path.join(exp_pth, "graph_logs.txt"))


    ## Write graph edge stats
    graphEdgesStats(adj_matrix, variables, classes, logger)


    ## Check for cycles 
    
    adj_matrix = cgraph_lib.preprocessAdjMatrix(adj_matrix, num_vars = len(variables), num_classes = len(classes))


    #Draw graph
    DRAW_GRAPH = True
    if DRAW_GRAPH:

        proc_graph_path = os.path.join(exp_pth, "train", f"processed_adjacency_matrix.png")
        utils_graph.drawAdjGraphWithVars(adj_matrix, variables, classes, proc_graph_path)


    features_list = variables + classes

    #Get sampling order
    graph_params = cgraph_lib.getSamplingOrder(adj_matrix, features_list, logger)

    
    ## Run classifier on found causal class parents

    node_parents = graph_params["node_parents"]
    class_parents, class_parents_idx = getClassCausalParents(node_parents, classes, features_list, ignore_latents = True)


    task = '-'.join(i_dataset[2:].split('_')[0].split('-')[:-1])
    
    classifier.main(
        class_parents, class_parents_idx, seed,
        causal_discovery_exp_dir = exp_pth, 
        task = task,
        dataset_path = dataset_path,
        # exp_name = "Classifer_Causal_parents",
        # trial = 'T1',
    )

    logger.close()



