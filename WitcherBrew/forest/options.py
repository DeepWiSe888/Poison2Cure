"""
Parameters setting
"""

import argparse
import numpy as np

def options_p2p():
    """Construct the central argument parser, filled with useful defaults.

    The first block is essential to test poisoning in different scenarios.
    The options following afterwards change the algorithm in various ways and are set to reasonable defaults.
    """
    parser = argparse.ArgumentParser(description='Construct poisoned training data for the given network and dataset')
    ## Please set these parameters !!!!!!===============================================================================
    parser.add_argument('--dataset_dir', default='.../data_for_poison',
                        type=str)  # Dataset path, npz. Please set you path!!!!
    ## =================================================================================================================
    # Central:
    parser.add_argument('--net', default='raw_gru_FT_0', type=lambda s: [str(item) for item in s.split(',')], help='Network architecture(s) to use')
    parser.add_argument('--dataset', default='data_for_poison', type=str, choices=['Test'])  # Should be same as 'get model'
    parser.add_argument('--recipe', default='gradient-matching', type=str, choices=['gradient-matching','metapoison']) # Brew poison method
    parser.add_argument('--threatmodel', default='p2p-v1', type=str, choices=['single-class', 'third-party', 'random-subset'])

    # Parameters about the dataset:
    parser.add_argument('--num_classes', default=10, type=int, help='Number of classes in the dataset.')
    parser.add_argument('--batch_size_UE', default=64, type=int, help='Batch size for UE validation.')

    # Reproducibility management:
    parser.add_argument('--poisonkey', default=None, type=str, help='Initialize poison setup with this key.')  # Also takes a triplet 0-3-1
    parser.add_argument('--modelkey', default=10, type=int, help='Initialize the model with this key.')        # Random seed
    parser.add_argument('--deterministic', default=True, help='Disable CUDNN non-determinism.')

    # Poison properties / controlling the strength of the attack:
    parser.add_argument('--budget', default=0.1, type=float, help='Fraction of training data that is poisoned')

    # Files and folders ==================================== data =================================
    parser.add_argument('--name', default='v1', type=str, help='Name tag for the result table and possibly for export folders.')
    parser.add_argument('--table_path', default='tables/', type=str)
    parser.add_argument('--poison_path', default='poisons/', type=str)
    parser.add_argument('--dataset_name', default='data_for_poison', type=str)     # Name of the dataset root, shuold be same as datasets.construct_datasets function.
    parser.add_argument('--auth_act_list', default=[0, 1, 2, 3, 6, 7, 8, 9], type=list)      # Authorized activities
    parser.add_argument('--priv_act_list', default=[4,5], type=list)                         # Privacy activities

    ###########################################################################
    # Poison brewing:
    parser.add_argument('--attackoptim', default='signAdam', type=str, choices=['signAdam'])
    parser.add_argument('--attackiter', default=150, type=int)    # poison epoch
    parser.add_argument('--init', default='all-zero', type=str)
    parser.add_argument('--tau', default=0.01, type=float)
    parser.add_argument('--scheduling', action='store_false', help='Disable step size decay.')
    parser.add_argument('--target_criterion', default='cross-entropy-p2p', type=str, help='Loss criterion for target loss')
    parser.add_argument('--restarts', default=1, type=int, help='How often to restart the attack.')

    parser.add_argument('--pbatch', default=64, type=int, help='Poison batch size during optimization')
    parser.add_argument('--pshuffle', action='store_true', help='Shuffle poison batch during optimization')
    parser.add_argument('--paugment', action='store_true', help='Augment poison batch during optimization')
    parser.add_argument('--auth_weight', default=1., type = float, help='Weight of the auth loss in the total loss function') # authorized action loss weight (to private action).
    parser.add_argument('--data_aug', type=str, default='none', choices=['csi_noise_aug', 'none'], help='Mode of diff. data augmentation.')

    # Poisoning algorithm changes
    parser.add_argument('--ensemble', default=1, type=int, help='Ensemble of networks to brew the poison on')
    parser.add_argument('--stagger', action='store_true', help='Stagger the network ensemble if it exists')
    parser.add_argument('--step', action='store_true', help='Optimize the model for one epoch.') # For one epoch
    parser.add_argument('--max_epoch', default=None, type=int, help='Train only up to this epoch before poisoning.')
    
    # Gradient Matching - Specific Options
    parser.add_argument('--loss', default='similarity', choices=['scalar_product', 'similarity', 'similarity-narrow', 'cosine1', 'SE', 'MSE'], type=str)  # similarity is stronger in  difficult situations; 使用的地方位于witch_matching.py:: _passenger_loss()里面。

    # These are additional regularization terms for gradient matching. We do not use them, but it is possible
    # that scenarios exist in which additional regularization of the poisoned data is useful.
    parser.add_argument('--centreg', default=0, type=float)
    parser.add_argument('--normreg', default=0, type=float)
    parser.add_argument('--repel', default=0, type=float)

    # Validation behavior
    parser.add_argument('--vruns', default=1, type=int, help='How often to re-initialize and check target after retraining')
    parser.add_argument('--flag_clean_vruns', default=True, type=bool, help='Whether to try some clean validation runs.')
    parser.add_argument('--vnet', default=None, type=lambda s: [str(item) for item in s.split(',')], help='Evaluate poison on this victim model. Defaults to --net')
    parser.add_argument('--retrain_from_init', action='store_true', help='Additionally evaluate by retraining on the same model initialization.')

    # Optimization setup
    # pre-trained model path
    parser.add_argument('--pretrained', default=True, type= bool, help='Use pretrained models.')

    parser.add_argument('--optimization', default='basic_p2p', type=str, help='Optimization Strategy') # basic_p2p / basic_p2p_joint_FT
    # Strategy overrides:
    parser.add_argument('--epochs', default=None, type=int)
    parser.add_argument('--noaugment', action='store_true', help='Do not use data augmentation during training.')
    parser.add_argument('--gradient_clip', default=None, type=float, help='Add custom gradient clip during training.')

    # Debugging:
    parser.add_argument('--dryrun', default=False, type=bool)
    parser.add_argument('--save', default='numpy', help='Export poisons into a given format. Options are full/limited/automl/numpy.')   # default=None

    # Distributed Computations
    parser.add_argument("--local_rank", default=None, type=int, help='Distributed rank. This is an INTERNAL ARGUMENT! '
                                                                     'Only the launch utility should set this argument!')

    return parser


def options_gru_constraint_update(step_list: list):

    parser = options_p2p()
    args = parser.parse_args()
    args.ensemble = 1
    args.multiStep_flag = True
    args.optimization = 'basic_p2p_joint_FT'
    args.FT_step = step_list
    args.attackoptim = 'signAdam'
    args.attackiter = 100
    args.craft_with_UE_eval = False
    args.valid_with_UE_eval = True
    args.var_snr_constraint = 25 # unit is dB, ratio between FT_data power and delta_poison power
    args.dfs_ub_constraint = 5  # detele the dfs features with index larger than this
    args.dfs_ub_const_dim_time = 16
    args.dfs_up_cons_poison_freq_dim = 256
    return args


def options_gru_ensble_M(num_ensemble: int, trial_idx: int):

    parser = options_p2p()
    parser.add_argument('--true_feat_len', default=64, type=int, help='The size of the feature vector of the GRU model')
    parser.add_argument('--true_layer_gru', default=2, type=int, help='The number of layers in the GRU model')
    parser.add_argument('--num_trial', default=6, type=int, help='The number of trials for the experiment')
    parser.add_argument('--ensemble_case', default='M1', type=str,
                        help='Case M(with true model)/M1(no true model) indicate using M surrogate model')
    parser.add_argument('--num_cases', default=6, type=int, help='Number of prepared cases')
    parser.add_argument('--use_FT_model_flag', default=True, type=bool, help='Use the FT or PT model')
    args = parser.parse_args()
    args.attackoptim = 'signAdam'
    args.attackiter = 500
    args.ensemble = num_ensemble
    args.ensemble_flag = True
    args.optimization = 'basic_p2p_joint_en'
    args.trial_index = trial_idx
    args.dataset = 'data_for_ensemble'
    args.tau = 0.05
    args.flag_average_or_max = 'average'  # 'max'

    np.random.seed(0)
    full_index_list = np.array(
        [np.random.choice(args.num_cases, num_ensemble, replace=False) for _ in range(args.num_trial)])  # num_ensemble: model number, num_trial: experiment number

    index_list = full_index_list[trial_idx, :]

    args.feat_len_ensemble, args.layer_gru_ensemble = check_list_ensemble_surrogate_model(index_list)


    return args

def check_list_ensemble_surrogate_model(index_list):

    list_feat_size = [38,32,85,37,64,48]  # 6 random case
    list_layer_gru = [3,3,3,2,2,3]

    return [list_feat_size[i] for i in index_list], [list_layer_gru[i] for i in index_list]
