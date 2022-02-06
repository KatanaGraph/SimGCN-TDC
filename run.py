import sys, os

from tdc.benchmark_group import admet_group

from katanaHLS.data.dataset import SIPGraph
from katanaHLS.models.SimGNN import SimGCN
from katanaHLS.utils.arg_parser import parse_args
import katanaHLS.utils.utils as utils
import trainer

def run_tdc_exp(args):
    
    # For reproducibility, fix the randomness
    utils.fix_randomness(seed=12345, use_cuda = args['use_cuda'])
    
    group = admet_group(path = 'tdc_data/')
    benchmark = group.get(args['dataset']) 
    name, train_val, test = benchmark['name'], benchmark['train_val'], benchmark['test']    
    
    # Copy all the arguments and hyperparameters in the model configurations
    model_config = dict()
    model_config.update(args)
    model_config.update(utils.load_model_config(args))        
    
    # 1. Prepare the dataset
    # Define the graph dataset based on similarity       
    # The similarity graph is accessed via dataset.sim_graph
    # The molecular graphs for each smiles can directly be accessed 
    # by the dataset object
    dataset = SIPGraph( root = 'tdc_data/admet_group' ,name=name, 
                            sim_metric = args['similarity'], threshold = model_config['thres'],
                            train_val = train_val, test = test, load_cache=True)

    if args['verbose']:
        utils.print_graph_stats(dataset)

    predictions_list = []    
    results_map = dict()

    for seed in [1, 2, 3, 4, 5]:
        train, val = group.get_train_valid_split(benchmark = name, split_type = 'default', seed = seed)

        # Set the validation masks in the similarity graph 
        utils.add_val_mask(dataset, train, val, dataset.map_smiles_to_idx)
            
        # Training on Similarity Graph: Define a GNN based Model
        # for supervised learning on the similarity graph.
        # The problem is set up as a node classification task. 
        
        model_config['in_sim_node_feats'] = dataset.sim_graph.num_features
        sim_gcn_config = utils.load_sim_gnn_hp(model_config)    
        model = SimGCN(sim_gcn_config) 
    
        model.to(model_config['device'])
        dataset.sim_graph = dataset.sim_graph.to(model_config['device'])
                
        model_path, train_score, valid_score  = trainer.train_model (model, model_config,
                                                            dataset, 
                                                            train_graph_rep=False
                                                            )
        test_score, y_pred_test = trainer.test_model (model, model_config,
                                                dataset, 
                                                test_graph_rep=False   
                                            )
        if model_config['verbose']:
            print('Similarity GNN test score:{}'.format(test_score))

        # 4. Store the model accuracy        
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  
        results_map.update({
            seed : {'seed': seed,
            'train_score': train_score.astype(float),
            'val_score': valid_score.astype(float),
            'test_score': test_score.astype(float),
            'model_path': model_path,
            }
        })
        if args['verbose']:
            print(results_map)

        predictions = {}        
        predictions[name] = y_pred_test
        predictions_list.append(predictions)
    
    #Use the tdc methods for evaluating the model        
    output = group.evaluate_many(predictions_list)
    output.update({'num_params' : num_params})

    print(output)
    utils.write_results(model_config, results_map, output)



def main(args):
    args = parse_args(args)    
    args['trial_path'] = os.path.join(args['out_path'], args['dataset'], 
                                      args['sim_gnn'])
    args['verbose'] = True
    utils.print_exp_settings(args)

    run_tdc_exp(args)

if __name__ == '__main__':
    main(sys.argv[1:])  

