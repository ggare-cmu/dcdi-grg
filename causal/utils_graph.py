import numpy as np

import os 

import networkx as nx
import pandas as pd


import utils



def drawAdjGraph(acyclic_adj_matrix, dataset_path, i_dataset, graph_path):

    data = np.load( os.path.join(dataset_path, f"{i_dataset[2:]}.npz"))

    variables = data['vars_list'].tolist()
    classes = data['class_list'].tolist()

    drawAdjGraphWithVars(acyclic_adj_matrix, variables, classes, graph_path)


def drawAdjGraphWithVars(acyclic_adj_matrix, variables, classes, graph_path):

    G = pd.DataFrame(acyclic_adj_matrix, index = variables+classes, columns = variables+classes)
    G = nx.from_pandas_adjacency(G, create_using=nx.DiGraph)

    drawGraph(G, graph_path)


def drawGraph(Graph, graph_path = ".", ):


    labels = nx.get_edge_attributes(Graph, "weight")
    
    #Change float precision
    for k,v in labels.items():
        labels[k] = f'{v:0.2f}'

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
        
    A.draw(graph_path,
            args='-Gnodesep=1.0 -Granksep=9.0 -Gfont_size=1', prog='dot' )  



if __name__ == "__main__":

    # exp_pth = "/home/grg/Research/dcdi-grg/exp/heart-disease-binary-t1-obs_only_TrainUpsampledPatient"
    # exp_pth = "/home/grg/Research/dcdi-grg/exp/heart-disease-binary-t1-obs_inter_TrainUpsampledPatient"
    # exp_pth = "/home/grg/Research/dcdi-grg/exp/heart-disease-binary-t1-obs_plus_inter_TrainUpsampledPatient"
    # exp_pth = "/home/grg/Research/dcdi-grg/exp/heart-disease-binary-t1-latent_obs_inter_TrainUpsampledPatient"
    exp_pth = "/home/grg/Research/dcdi-grg/exp/heart-disease-binary-t1-latent_obs_plus_inter_TrainUpsampledPatient"

    dataset_path = "/home/grg/Research/dcdi-grg/data/grg"
    # i_dataset = "1-heart-disease-binary-t1-obs_only_TrainUpsampledPatient"
    # i_dataset = "1-heart-disease-binary-t1-obs_inter_TrainUpsampledPatient"
    # i_dataset = "1-heart-disease-binary-t1-obs_plus_inter_TrainUpsampledPatient"
    # i_dataset = "1-heart-disease-binary-t1-latent_obs_inter_TrainUpsampledPatient"
    i_dataset = "1-heart-disease-binary-t1-latent_obs_plus_inter_TrainUpsampledPatient"
    
    adj_matrix_path = os.path.join(exp_pth, "train", f"DAG.npy")

    adj_matrix = np.load(adj_matrix_path)

    graph_path = os.path.join(exp_pth, "train", f"Discovered_DAG.png")

    # drawAdjGraph(adj_matrix, dataset_path, i_dataset, graph_path)

    data = np.load( os.path.join(dataset_path, f"{i_dataset[2:]}.npz"))

    variables = data['vars_list'].tolist()
    class_list = data['class_list'].tolist()

    G = pd.DataFrame(adj_matrix, index = variables+class_list, columns = variables+class_list)
    pred_graph = nx.from_pandas_adjacency(G, create_using=nx.DiGraph)
        
    logger = utils.Logger(filename = os.path.join(exp_pth, "graph_logs.txt"))

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


    logger.close()



