from argparse import ArgumentParser, ArgumentTypeError
import os, torch


def str2bool(token):
    if isinstance(token, bool):
        return token
    if token.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif token.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

def default_args(dataset:str='hERG'):
    args = {}
    args['dataset'] = dataset
    args['sim_gnn'] = 'GCN'
    args['similarity'] = 'Tanimoto'
    args['thres'] = 0.65
    args['model_config_path'] = 'katanaHLS/configs'
    args['num_epochs'] = 500
    args['out_path'] = 'result'    
    args['use_cuda'] = False
    args['verbose'] = False
    args['trial_path'] = os.path.join(args['out_path'], args['dataset'], 
                                      args['mol_gnn'] + '.' + args['sim_gnn'])
    args['model_config_path'] = args['model_config_path'] + '/' +  args['dataset']
    if torch.cuda.is_available() and args['use_cuda']:
        args['use_cuda'] = True
        args['device'] = torch.device('cuda:0')
    else:
        args['use_cuda'] = False
        args['device'] = torch.device ('cpu')
        
    # Update the metric for the tdc single instance predictions datasets    
    if args['dataset'] in ['Bioavailability_Ma', 'HIA_Hou','Pgp_Broccatelli', 'BBB_Martins', 'CYP3A4_Substrate_CarbonMangels', 'hERG', 'AMES', 'DILI']:
        args['metric'] = 'roc-auc'
    elif args['dataset'] in ['CYP2C9_Veith', 'CYP2D6_Veith', 'CYP3A4_Veith', 'CYP2C9_Substrate_CarbonMangels', 'CYP2D6_Substrate_CarbonMangels']:
        args['metric'] = 'pr-auc'
    elif args['dataset'] in ['Caco2_Wang', 'Lipophilicity_AstraZeneca', 'Solubility_AqSolDB', 'PPBR_AZ', 'LD50_Zhu']:
        args['metric'] = 'mae'
    elif args['dataset'] in ['VDss_Lombardo', 'Half_Life_Obach', 'Clearance_Hepatocyte_AZ', 'Clearance_Microsome_AZ']:
        args['metric'] = 'spearman'
    else:
        print('The dataset does not belong to tdc single instance prediction task. {} is used as metric.'.format(args['metric']))

    return args


def parse_args(args):

    parser = ArgumentParser('Single Property Prediction: Node Classification')
    parser.add_argument('-d', '--dataset', choices=['VDss_Lombardo', 'Half_Life_Obach', 'Clearance_Hepatocyte_AZ', 'Clearance_Microsome_AZ',
                                'Caco2_Wang',
                                'HIA_Hou', 'Pgp_Broccatelli', 'Bioavailability_Ma',
                                'BBB_Martins',
                                'CYP2C9_Veith', 'CYP2D6_Veith', 'CYP3A4_Veith',
                                'CYP2C9_Substrate_CarbonMangels', 
                                'CYP2D6_Substrate_CarbonMangels', 
                                'CYP3A4_Substrate_CarbonMangels',
                                'AMES', 'DILI', 'hERG' ,
                                'CYP2C9_Veith', 'CYP2D6_Veith', 'CYP3A4_Veith', 'CYP2C9_Substrate_CarbonMangels', 'CYP2D6_Substrate_CarbonMangels',
                                'Caco2_Wang', 'Lipophilicity_AstraZeneca', 'Solubility_AqSolDB', 'PPBR_AZ', 'LD50_Zhu'],
                                default='hERG',                                                   
                                help='Predefine TDC Datasets classification tasks')        
    parser.add_argument('-sg', '--sim-gnn', choices=['None', 'GCN', 'GAT'],
                            default='GCN',                                                   
                            help='Models to use for the similarity graph')
    
    parser.add_argument('-sim', '--similarity', choices=['Tanimoto', 'Dice', 'Cosine',
                                    "Sokal",
                                    "Russel",
                                    "RogotGoldberg",
                                    "AllBit",
                                    "Kulczynski",
                                    "McConnaughey",
                                    "Asymmetric",
                                    "BraunBlanquet"],
                            default='Tanimoto',                                                   
                            help='The molecular similarity metric')                            
    parser.add_argument('-th', '--thres', type=float, default=0.6,
                            help='The threshold for constructing the molecular similarity graph.')
    parser.add_argument('-mp', '--model-config-path', type=str, default='katanaHLS/configs',
                            help='Path to load pre-defined model hyperparameters (default: katanaHLS/configs)')    
    parser.add_argument('-me', '--metric', choices=['roc_auc_score', 'pr_auc_score', 'mae'],
                            default='roc_auc_score',
                            help='Metric for evaluation (default: roc_auc_score)')                        
    parser.add_argument('-n', '--num-epochs', type=int, default=2000,
                            help='Maximum number of epochs for training. '
                                'We set a large number by default as early stopping '
                                'will be performed. (default: 1000)')
    parser.add_argument('-op', '--out-path', type=str, default='result',
                            help='Path to save training results (default: result)')
    parser.add_argument('-uc', '--use-cuda', type=str2bool, default=True,
                            help='Option to do hyperparams search.')                                
    parser.add_argument('-v', '--verbose', type=str2bool, default=True,
                            help='Option to print details.')                                
    
    args = parser.parse_args(args).__dict__

    args['model_config_path'] = args['model_config_path'] + '/' +  args['dataset']
    if torch.cuda.is_available() and args['use_cuda']:
        args['use_cuda'] = True
        args['device'] = torch.device('cuda:0')
    else:
        args['use_cuda'] = False
        args['device'] = torch.device ('cpu')
    
    # Update the metric for the tdc single instance predictions datasets    
    if args['dataset'] in ['Bioavailability_Ma', 'HIA_Hou','Pgp_Broccatelli', 'BBB_Martins', 'CYP3A4_Substrate_CarbonMangels', 'hERG', 'AMES', 'DILI']:
        args['metric'] = 'roc-auc'
    elif args['dataset'] in ['CYP2C9_Veith', 'CYP2D6_Veith', 'CYP3A4_Veith', 'CYP2C9_Substrate_CarbonMangels', 'CYP2D6_Substrate_CarbonMangels']:
        args['metric'] = 'pr-auc'
    elif args['dataset'] in ['Caco2_Wang', 'Lipophilicity_AstraZeneca', 'Solubility_AqSolDB', 'PPBR_AZ', 'LD50_Zhu']:
        args['metric'] = 'mae'
    elif args['dataset'] in ['VDss_Lombardo', 'Half_Life_Obach', 'Clearance_Hepatocyte_AZ', 'Clearance_Microsome_AZ']:
        args['metric'] = 'spearman'
    else:
        raise ValueError('The dataset does not belong to tdc single instance prediction task. {} is used as metric.'.format(args['metric']))

    return args
