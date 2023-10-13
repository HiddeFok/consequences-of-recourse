import os

from argparse import ArgumentParser

from models.models_experiments import models
from models.models_experiments import recourse_models

from utils.experiment import do_single_real_experiment, save_single_experiment_data
from utils.utils import set_seed

EXPERIMENT_SETTINGS = {
    'adult': {
        'N_total': 48832,  
        'N_cond_train': 30000,  
        'N_cond_calib': 10000,  
        'N_train': 5000,  
        'N_test': 1000,  
    },
    'credit': {
        'N_total': 115527,  
        'N_cond_train': 40000,  
        'N_cond_calib': 10000,  
        'N_train': 5000,  
        'N_test': 1000,  
    },
    'heloc': {
        'N_total': 9871,  
        'N_cond_train': 5000,  
        'N_cond_calib': 2000,  
        'N_train': 2000,  
        'N_test': 1000,  
    }
}

if __name__ == "__main__":
    SEEDS = [124, 142, 253, 103, 24, 105, 85, 91, 95, 56]
    REPEATS = 10

    parser = ArgumentParser()
    parser.add_argument('--data_set', type=str,
                        choices=['adult', 'credit', 'heloc'])
    parser.add_argument('--classifier', type=str, 
                        choices=list(models.keys()))
    parser.add_argument('--recourse', type=str,
                        choices=list(recourse_models.keys()))
    parser.add_argument('--experiment_nr', type=int,
                        choices=list(range(10)))

    args = parser.parse_args()

    N_total = EXPERIMENT_SETTINGS[args.data_set]["N_total"] 
    N_cond_train = EXPERIMENT_SETTINGS[args.data_set]["N_cond_train"]
    N_cond_calib = EXPERIMENT_SETTINGS[args.data_set]["N_cond_calib"]
    N_train = EXPERIMENT_SETTINGS[args.data_set]["N_train"]
    N_test  = EXPERIMENT_SETTINGS[args.data_set]["N_test"]

    fracs = fracs = {
        "cond_train": N_cond_train / N_total, 
        "cond_calib": N_cond_calib / N_total, 
        "train": N_train / N_total, 
        "test": N_test / N_total, 
    }

    experiment_nr = args.experiment_nr

    checkpoint_dir = f"checkpoints/{args.data_set}_cluster"
    os.makedirs(checkpoint_dir, exist_ok=True)

    classifier = models[args.classifier]
    recourse = recourse_models[args.recourse]

    print(f"Starting Experiment: {args.data_set}, {args.classifier}, {args.recourse}")

    set_seed(SEEDS[experiment_nr])
    print(f"Working on {experiment_nr} of {REPEATS} repeated experiments")
    
    experiment_dir = os.path.join(checkpoint_dir, f"experiment_{experiment_nr+1}")

    os.makedirs(os.path.join(experiment_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "data"), exist_ok=True)
    
    result_dict = do_single_real_experiment(
        args.data_set,
        fracs,
        classifier, 
        recourse
    )

    save_single_experiment_data(
        experiment_dir, 
        result_dict, 
        args.classifier,
        args.recourse,
        None,
        None,
        two_dim=False
    )
