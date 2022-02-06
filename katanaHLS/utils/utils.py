import numpy as np
import os, errno, json, random
import torch

from rdkit import Chem, DataStructs
from rdkit.DataStructs import *

from katanaHLS.models import SimGNNConfig

try:
	from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors
except:
	raise ImportError("Please install pip install git+https://github.com/bp-kelley/descriptastorus and pip install pandas-flavor")


def fix_randomness(seed=0, use_cuda=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if not use_cuda:
        torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    np.random.RandomState(seed) 
    random.seed(0)

def print_graph_stats(dataset):
    data = dataset.sim_graph  # Get the first graph object.

    print()
    print(data)
    print('===============================================')
    print("Similarity Graph Stats.")

    # Print some statistics about the similarity graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Number of training nodes: {data.train_mask.sum()}')
    print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')
    

def print_exp_settings(args):
    print()
    print('Settings for the experiments:')    
    print('==============================')
    print('Dataset: {}'.format(args['dataset']))
    print('Metric: {}'.format(args['metric']))
    print('Similarity Criteria: {}'.format(args['similarity']))    
    # print('GNN Model: {}'.format(args['model']))
    # print('Model config filepath: {}'.format(args['model_config_path']))
    # print('Max Number of Epochs: {}'.format(args['num_epochs']))
    print('Using CUDA: {}'.format(args['use_cuda']))    
    

def add_val_mask(dataset, train, val, id_maps):        
    smiles_list = val['Drug'].values    

    for smiles in smiles_list:
        for id in id_maps[smiles]:
            dataset.sim_graph.val_mask[id] = True
            dataset.sim_graph.train_mask[id] = False

def split_dataset(train, val, test, id_maps):
    
    train_smiles_list = train['Drug'].values
    val_smiles_list = val['Drug'].values
    test_smiles_list = test['Drug'].values
    
    train_idx = [index for smiles in train_smiles_list for index in id_maps[smiles]]
    val_idx = [index for smiles in val_smiles_list for index in id_maps[smiles]]
    test_idx = [index for smiles in test_smiles_list for index in id_maps[smiles]]
    
    return np.unique(train_idx), np.unique(val_idx), np.unique(test_idx)
    
    
    
def make_dir(path):
    try:
        os.makedirs(path)        
    except OSError as e:
        if e.errno != errno.EEXIST or not os.path.isdir(path):
            raise


def write_results(args, results, final_score):
    out_path = args['trial_path']
    make_dir(out_path)

    with open( out_path + '/sim_metric.{}.thres.{}.results.json'.format(
        args['similarity'], args['thres']), 'w') as f:
        json.dump(results, f, indent=2)
    
    with open( out_path +  '/sim_metric.{}.thres.{}.final_score.json'.format(
        args['similarity'], args['thres']), 'w') as f:
        json.dump(final_score, f, indent=2)


def load_model_config(args):
    """ Query for the manually specified configuration"""
    config= dict()    
    model_path = args['model_config_path']
    print('Loading model configurations from {}'.format(model_path))
    sim_gnn_path = os.path.join(model_path, args['sim_gnn']+ '_sim.json')
    with open(sim_gnn_path, 'r') as f:
        sim_config = json.load(f)
    config.update(sim_config)
    return config


def load_sim_gnn_hp(args):
    config= dict()    
    # if hyperparams:
    #     config.update(hyperparams)
    # else:
    #     """ Query for the manually specified configuration"""
    #     model_path = args['model_config_path']
    #     with open('{}/{}_sim.json'.format(model_path,args['model']), 'r') as f:
    #         config = json.load(f)
    
    sim_gcn_config = SimGNNConfig(in_channels = args['in_sim_node_feats'], gnn = args['sim_gnn'])
    sim_gcn_config.hidden_channels = args['sim_gnn_hidden_channels']
    sim_gcn_config.num_layers = args['sim_gnn_num_layers']
    sim_gcn_config.batchnorm = args['sim_gnn_batchnorm']
    sim_gcn_config.dropout = args['sim_gnn_dropout']  
    sim_gcn_config.predictor_hidden_feats = args['sim_gnn_predictor_hidden_feats']         
    
    args["batch_size"] = args["sim_gnn_batch_size"]
    args["lr"] = args["sim_gnn_lr"]
    args["weight_decay"] = args["sim_gnn_weight_decay"]
    args["patience"] = args["sim_gnn_patience"]

    return sim_gcn_config




def smile_to_fps(smile):
    r""" Given a smile string, convert it into a RDKit Fingerprint
    """
    molecule = Chem.MolFromSmiles(smile)
    return Chem.RDKFingerprint(molecule)                


def construct_sim_matrices (smiles, sim_metric):
    r""" Given a list of smile strings corresponding to a tdc dataset,
    returns the pairwise similarity of all the molecules in the list
    as a numpy matrix. The similarity measure is computed with respect 
    to the similarity function.
    """
    
    print('Constructing similarity matrix. This might take a while...')
    n = len(smiles)
    sim_score_mat = np.zeros((n,n))
    
    fps = [smile_to_fps(smile) for smile in smiles]

    for i in range(n):
        for j in range(n):
            sim_score_mat[i,j] = DataStructs.FingerprintSimilarity(fps[i], fps[j], sim_metric)                

    print('Similarity matrix constructed.')
    
    return sim_score_mat

def smiles2rdkit2d(smiles):    
    try:
        generator = rdNormalizedDescriptors.RDKit2DNormalized()
        features = np.array(generator.process(smiles)[1:])
        NaNs = np.isnan(features)
        features[NaNs] = 0
    except:
        print('descriptastorus not found this smiles: ' + smiles  + ' convert to all 0 features')
        features = np.zeros((200, ))
    return np.array(features)

# Atom Features. The basic settings borrowed from DGL Life Science
def one_hot_encoding(x, allowable_set, encode_unknown=False):
    """One hot encoding of the element x with respect to the allowable set.
    If encode unknown is true, and x is not in the allowable set,
    then x is added to the allowable set.
    Args:
        :param x (elem) : [the element to encode]
        :param allowable_set (list): [ The list of elements]
        :param encode_unknown (bool, optional): [Whether to add x to the allowable list,
        if x is already not present]. Defaults to False.
        :return one hot encoding of x
    """
    if encode_unknown and (x not in allowable_set):
        allowable_set.append(x)

    return list(map(lambda s: x == s, allowable_set))


def construct_features(smiles):
    r""" Given a list of smile strings, 
    return their corresponding rdkit2d features. Useful for adding molecular features
    in node classification tasks. 
    """
    print('Generating molecular features....')
    feat = [smiles2rdkit2d(smile) for smile in smiles]    
    print('Feature constructions complete.')
    return np.array(feat)


def construct_edge_list(sim_score_mat, thres):                
    """ Constructs edge lists for a PyG graph (COO representation) 
    based on the pairwise similarity matrix of the 
    molecule and the molecular features
    """
    print('Constructing COO edge list based on similarity matrix.')
    srcs, dsts = [], []               
    n = len(sim_score_mat)
    for i in range(n):
        for j in range(n):
            if sim_score_mat[i][j] > thres:
                srcs.append(i)
                dsts.append(j)
    edge_index = torch.tensor([srcs,dsts], dtype=torch.long)
    print('Done.')
    return edge_index

import  matplotlib.pyplot as plt 

def plot_train_history(history):
    acc = history['train_score']
    val_acc = history['val_score']
    loss = history['loss']    
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'b', label='Training score')
    plt.plot(epochs, val_acc, 'r', label='Validation score')
    plt.title('Training and validation score')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.title('Training loss')
    plt.legend()
    plt.show()
    
def plot_tdc_results():
    import numpy as np
    import matplotlib.pyplot as plt
    
    Katana = [0.878, 0.921, 0.737, 0.932]
    MLP = [0.841, 0.875, 0.672, 0.889]
    
    n=4
    r = np.arange(n)
    width = 0.25
    
    
    plt.bar(r, Katana, color = 'b',
            width = width, edgecolor = 'black',
            )
    plt.bar(r + width, MLP, color = 'orange',
            width = width, edgecolor = 'black',
            )
    plt.legend(['Katana', 'TDC'], loc=2, prop={'size': 14})
    
    plt.xlabel("Datasets")
    plt.ylabel("Accuracy")
    plt.title("TDC Benchmark Datasets")
    
    # plt.grid(linestyle='--')
    plt.ylim([0.6,1])
    plt.xticks(r + width/2,['hERG','DILI','Bioavailability_Ma','BBB_Martins'])
    # plt.legend()
    
    plt.show()