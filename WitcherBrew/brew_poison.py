"""
Brew Poison
"""

import torch
import scipy.io
import datetime
import time
import forest
import os
import multiprocessing as mp
from os.path import exists


os.environ['CUDA_VISIBLE_DEVICES'] = "0"     # cuda index
torch.backends.cudnn.benchmark = forest.consts.BENCHMARK
# torch.backends.cudnn.benchmark = False
# torch.multiprocessing.set_sharing_strategy(forest.consts.SHARING_STRATEGY) # file_system or file_descriptor
from visualize.draw_poison_delta_distribution import save_draw_poison_delta_distribution


# Parse input arguments
current_directory = os.getcwd()            # Get current directory
args = forest.options_p2p().parse_args()   # Parameters setting
args.craft_with_UE_eval = False            # Brew poison with UE evaluation dataset or not.
args.valid_with_UE_eval = True             # Verify poison with UE evaluation dataset or not.
args.log_file_dir = current_directory + '/log_file' # log save path
parent_directory = os.path.abspath(os.path.join(os.getcwd(), ".."))
args.pretrained_dir = parent_directory + "/Res_GRU/Results_tr_ft_D0"  # Pretrained model directory
if args.deterministic:
    forest.utils.set_deterministic()

if __name__ == "__main__":
    setup = forest.utils.system_startup(args)   # Parameters about torch.to(): cuda number and torch data format
    model = forest.Victim(args, setup=setup)    # Victim mddel, i.e., the trained neural network model.
    data = forest.Kettle(args, model.defs.batch_size, model.defs.augmentations, setup=setup) # model.defs.augmentations: None.
    witch = forest.Witch(args, setup=setup)     # Witch_matching
    if exists(args.log_file_dir) != True:
        os.makedirs(args.log_file_dir)
        print(f'Created log file directory {args.log_file_dir}')
    else:
        # delete all files in the log file directory
        for file in os.listdir(args.log_file_dir):
            os.remove(os.path.join(args.log_file_dir, file))
        print(f'Cleaned log file directory {args.log_file_dir}')   
        
    if args.flag_clean_vruns:
        print('Running clean validation runs...')
        stats_results = model.validate(data, None)

    if args.pretrained:
        print('Loading pretrained model...')
        train_time = time.time()
        model.load_pretrained(args.pretrained_dir) 

    start_time = time.time()
    poison_delta,poison_loss_value = witch.brew(model, data)
    brew_time = time.time()
    save_draw_poison_delta_distribution(poison_delta,poison_loss_value, args.log_file_dir)   # passenger loss during brewing poison

    if args.vruns > 0:
        stats_results = model.validate(data, poison_delta)   # FT model with poisoned data
    else:
        stats_results = None
    test_time = time.time()
    print(str(datetime.timedelta(seconds=brew_time - train_time)).replace(',', ''))
    timestamps = dict(train_time=str(datetime.timedelta(seconds=start_time -train_time)).replace(',', ''),
                      brew_time=str(datetime.timedelta(seconds=brew_time - train_time)).replace(',', ''),
                      test_time=str(datetime.timedelta(seconds=test_time - brew_time)).replace(',', ''))
    time_path = current_directory + '/log_file/timestamps.mat'
    scipy.io.savemat(time_path, timestamps)
    exit(0)
    # Save run to table
    results = stats_results
    forest.utils.record_results(data, witch.stat_optimal_loss, results,
                                args, model.defs, model.model_init_seed, extra_stats=timestamps)

    # Export
    if args.save is not None:
        data.export_poison(poison_delta, path=args.poison_path, mode=args.save)

    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print('---------------------------------------------------')
    print(f'Finished computations with train time: {str(datetime.timedelta(seconds=train_time - start_time))}')
    print(f'--------------------------- brew time: {str(datetime.timedelta(seconds=brew_time - train_time))}')
    print(f'--------------------------- test time: {str(datetime.timedelta(seconds=test_time - brew_time))}')
    print('-------------Job finished.-------------------------')
