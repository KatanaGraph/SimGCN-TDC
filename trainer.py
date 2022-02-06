import time, os, math, random, string
import numpy as np
import torch
from tqdm import tqdm
import tdc
from katanaHLS.utils.utils import *
from katanaHLS.utils.early_stop import EarlyStopping
from katanaHLS.utils.eval import Evaluator

from torch_geometric.loader import NeighborLoader
from torch_geometric.data import DataLoader
from torch_geometric.data import Batch


def evaluate (val_loader, model, metric, train_graph_rep=False):

    model.eval()
    pred_y = []   
    labels = []
    with torch.no_grad():
        for batch in val_loader:
            if train_graph_rep: # graph classification task
                logits, _  = model(batch)
                pred_y.append(logits.detach().cpu())               
                labels.append(batch.y.float().detach().cpu())
                
            else: # node classification task
                logits, _ = model(batch.x, batch.edge_index)
                logits = logits[:batch.batch_size]
                pred_y.append(logits.detach().cpu())               
                labels.append(batch.y[:batch.batch_size].detach().cpu())
                
    pred_y = torch.cat(pred_y, dim=0)
    labels = torch.cat(labels, dim=0)
    labels = labels.reshape(-1,1)

    if metric in ['roc-auc', 'pr-auc']:
        pred_y = torch.sigmoid(pred_y)
        
    eval_meter = Evaluator()
    eval_meter.update(pred_y, labels)
    score2 = np.mean(eval_meter.compute_metric(metric))
    
    evaluator = tdc.Evaluator(name = metric)
    score = evaluator(labels.detach().cpu().numpy(), pred_y.detach().cpu().numpy())

    # assert(math.isclose(score, score2))
    
    return score, pred_y


def train_epoch (train_loader, model, loss_fnc, optimizer, train_graph_rep = False):    
    model.train()
    
    total_loss = total_example = batch_size =  0    
    for batch in train_loader:    
        if train_graph_rep:  # training for graph classification
            batch_size = len(batch.y)
            logits = model(batch)
            loss = loss_fnc(logits, batch.y.float()).mean()        
            
        else: # training for node classification
            batch_size = batch.batch_size
            logits = model(batch.x, batch.edge_index)[:batch.batch_size]
            loss = loss_fnc(logits, batch.y[:batch.batch_size]).mean()        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_example += batch_size

    return total_loss / total_example


def train_model(model, config, 
                dataset, train_graph_rep=False, 
                train_idx=None, val_idx = None):
        
    # loss_fnc = torch.nn.CrossEntropyLoss()
    # Define the loss function
    if config['metric'] in ['roc_auc_score', 'pr_auc_score']:
        loss_fnc = torch.nn.BCEWithLogitsLoss(reduction='none')
    elif config['metric'] in ['mae']:
        loss_fnc = torch.nn.SmoothL1Loss(reduction='none')
    else:
        loss_fnc = torch.nn.MSELoss(reduction='none')
    
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],
                            weight_decay=config['weight_decay'])

    # Create a directory to store the model in the trial path                            
    current_time = str(round(time.time() * 100000))
    result_dir =  os.path.join(config['trial_path'], 'model', current_time)
    make_dir(result_dir)
    
    # Define the early stopping criteria to avoid overfitting
    stopper = EarlyStopping(patience=config['patience'],
                                filename=result_dir + '/model.pth',
                                metric=config['metric'], verbose=config['verbose'])
      
    # There are two types of training: one for node classification and
    # the other one is for graph classification.
    # If the train_graph flag is true, then the training loop 
    # will train molecular graphs. Otherwise it will train similarity graphs.
    
    if train_graph_rep:
        # dataset is a collection of graphs in this case                
        # Define the dataloaders for training and validation data
        train_loader = DataLoader(dataset[train_idx], 
                                  batch_size=config['batch_size'],
                                  shuffle=True)
        
        val_loader = DataLoader(dataset[val_idx], 
                                  batch_size=config['batch_size'],
                                  shuffle=False)
    
    else:        
        # Set the graph for node classification
        g = dataset.sim_graph
        # Define the data loader for mini batch sampling
        train_loader = NeighborLoader(g,
                            num_neighbors=[-1] * model.config.num_layers,                
                            batch_size= config['batch_size'],
                            input_nodes=g.train_mask,
                            shuffle=True
                        ) 
        # Define the data loader for mini batch sampling
        val_loader = NeighborLoader(g,
                            num_neighbors=[-1] * model.config.num_layers,                
                            batch_size=config['batch_size'],
                            input_nodes=g.val_mask,
                            shuffle=False
                        ) 

    dur = []
    pbar = tqdm(range(config['num_epochs']), position=1, leave=True)
    for epoch in pbar:
        t0 = time.time()
        model.train()                 
        loss = train_epoch(train_loader, model, loss_fnc, optimizer, train_graph_rep)
        dur.append(time.time() - t0)

        model.eval()        
        val_score, _ = evaluate (val_loader, model, config['metric'], train_graph_rep)
                
        early_stop, counter = stopper.step(val_score, model)
        if config['verbose']:
            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Val Score {:.4f} | "
            .format(epoch, np.mean(dur), loss, val_score))
        
        pbar.set_postfix(loss=loss, val_score = val_score, early_stop=counter)
        
        if early_stop:
                break
        
        
    stopper.load_checkpoint(model)
    
    train_score, _ = evaluate (train_loader, model, config['metric'], train_graph_rep)
    val_score, _ = evaluate (val_loader, model, config['metric'], train_graph_rep)
    # train_score, _ = evaluate(train_node_loader, model, g, config)    
    # val_score, _ = evaluate(valid_node_loader, model, g, config)
    print('---------------------------------------------------------------------------------------')
    print('Training score: {:.4f}, validation score: {:.4f}, best val score: {:.4f}'.format(train_score, val_score, stopper.best_score))    
    print('---------------------------------------------------------------------------------------')
    
    # Return the path to the trained model
    return result_dir, train_score, val_score


def test_model(model, config, 
                dataset, test_graph_rep=False, 
                test_idx=None):
    if test_graph_rep:
        test_loader = DataLoader(dataset[test_idx], 
                                  batch_size=config['batch_size'],
                                  shuffle=False)
    else:
        # Set the graph for node classification
        g = dataset.sim_graph
        test_loader = NeighborLoader(g,
                            num_neighbors=[-1] * model.config.num_layers,                
                            batch_size=config['batch_size'],
                            input_nodes=g.test_mask,
                            shuffle=False
                        ) 
    model.eval()
    test_score, pred_y = evaluate (test_loader, model, config['metric'], test_graph_rep)    
    
    return test_score, pred_y


def generate_emb (model, config, dataset, graph_level=True):
    if graph_level:
        loader = DataLoader(dataset, 
                        batch_size=config['batch_size'],
                        shuffle=False)
    else: # node level embeddings
        g = dataset.sim_graph
        loader = NeighborLoader(g,
                            num_neighbors=[-1] * model.config.num_layers,                
                            batch_size=config['batch_size'],
                            shuffle=False
                        )
    emb_list = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader):
            if graph_level:
                _, emb  = model(batch)
            else:
                _, emb  = model(batch.x, batch.edge_index)
                emb = emb[:batch.batch_size]
            emb_list.append(emb.detach().cpu())
    emb_list = torch.cat(emb_list, dim=0)
    return emb_list
        