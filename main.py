from config import parse_args
from Data.read_data import *
import datetime
from Utils.running import *
from Utils.utils import *
import warnings
import torch
from opacus.accountants.utils import get_noise_multiplier
from opacus import PrivacyEngine
from opacus.accountants import create_accountant

warnings.filterwarnings("ignore")


def run(args, current_time, device):
    # read data
    print('Using noise scale: {}, clip: {}'.format(args.ns, args.clip))

    if args.dataset == 'adult':
        train_df, test_df, male_df, female_df, feature_cols, label, z = read_adult(args)
        args.feature = feature_cols
        args.target = label
        args.z = z
        args.input_dim = len(feature_cols)
        args.output_dim = 1
        print(feature_cols)
    elif args.dataset == 'bank':
        train_df, test_df, male_df, female_df, feature_cols, label, z = read_bank(args)
        args.feature = feature_cols
        args.target = label
        args.z = z
        args.input_dim = len(feature_cols)
        args.output_dim = 1
        print(feature_cols)
    elif args.dataset == 'stroke':
        train_df, test_df, male_df, female_df, feature_cols, label, z = read_stroke(args)
        args.feature = feature_cols
        args.target = label
        args.z = z
        args.input_dim = len(feature_cols)
        args.output_dim = 1
        print(feature_cols)

    # running process
    if args.mode == 'clean':
        if args.debug:
            run_clean(fold=0, train_df=train_df, test_df=test_df, args=args, device=device, current_time=current_time)
        else:
            for fold in range(args.folds):
                run_clean(fold=fold, train_df=train_df, test_df=test_df, args=args, device=device,
                          current_time=current_time)
    elif args.mode == 'dpsgd':
        if args.debug:
            run_dpsgd(fold=0, train_df=train_df, test_df=test_df, args=args, device=device, current_time=current_time)
        else:
            for fold in range(args.folds):
                run_dpsgd(fold=fold, train_df=train_df, test_df=test_df, args=args, device=device,
                          current_time=current_time)
    elif args.mode == 'fair':
        if args.debug:
            run_fair(fold=0, train_df=train_df, male_df=male_df, female_df=female_df, test_df=test_df, args=args, device=device,
                     current_time=current_time)
        else:
            for fold in range(args.folds):
                run_fair(fold=fold, train_df=train_df, male_df=male_df, female_df=female_df, test_df=test_df, args=args, device=device,
                         current_time=current_time)
    elif args.mode == 'fairdp':
        # fold, train_df, test_df, male_df, female_df, args, device, current_time)
        if args.debug:
            run_fair_dpsgd(fold=0, train_df=train_df, test_df=test_df, male_df=male_df, female_df=female_df, args=args,
                           device=device,
                           current_time=current_time)
        else:
            for fold in range(args.folds):
                run_fair_dpsgd(fold=fold, male_df=male_df, female_df=female_df, test_df=test_df, train_df=train_df,
                               args=args, device=device,
                               current_time=current_time)
    elif args.mode == 'fairdp_epoch':
        if args.debug:
            run_fair_dp(fold=0, train_df=train_df, test_df=test_df, male_df=male_df, female_df=female_df, args=args,
                           device=device,
                           current_time=current_time)
        else:
            for fold in range(args.folds):
                run_fair_dp(fold=fold, train_df=train_df, test_df=test_df, male_df=male_df, female_df=female_df, args=args,
                           device=device,
                           current_time=current_time)
    elif args.mode == 'test':
        sigma = get_noise_multiplier(target_epsilon=args.tar_eps, target_delta=args.tar_delt,
                                     sample_rate=args.batch_size / len(train_df), epochs=args.epochs,
                                     accountant='rdp')
        print("Noise scale to use: {}".format(sigma))

    elif args.mode == 'geteps':
        print(len(train_df))
        accountant = create_accountant(mechanism='rdp')
        sampler_rate = args.batch_size / len(train_df)
        steps = int((args.epochs + 1) / sampler_rate)
        accountant.history = [(args.ns, sampler_rate, steps)]
        eps = accountant.get_epsilon(delta=args.tar_delt)
        print("Epsilon used: {}".format(eps))
    elif args.mode == 'fairdp_track':
        # fold, train_df, test_df, male_df, female_df, args, device, current_time)
        if args.debug:
            run_fair_dpsgd_track_grad(fold=0, train_df=train_df, test_df=test_df, male_df=male_df, female_df=female_df, args=args,
                           device=device,
                           current_time=current_time)
        else:
            for fold in range(args.folds):
                run_fair_dpsgd_track_grad(fold=fold, male_df=male_df, female_df=female_df, test_df=test_df, train_df=train_df,
                               args=args, device=device,
                               current_time=current_time)
    elif args.mode == 'func':
        # fold, train_df, test_df, male_df, female_df, args, device, current_time)
        if args.debug:
            run_functional_mechanism_logistic_regression(fold=0, train_df=train_df, test_df=test_df, male_df=male_df, female_df=female_df, args=args,
                           device=device,
                           current_time=current_time)
        else:
            for fold in range(args.folds):
                run_functional_mechanism_logistic_regression(fold=fold, male_df=male_df, female_df=female_df, test_df=test_df, train_df=train_df,
                               args=args, device=device,
                               current_time=current_time)
    elif args.mode == 'proposed':
        if args.debug:
            run_fair_dpsgd_alg2(fold=0, male_df=male_df, female_df=female_df, test_df=test_df, args=args, device=device,
                                current_time=current_time)
        else:
            for fold in range(args.folds):
                run_fair_dpsgd_alg2(fold=fold, male_df=male_df, female_df=female_df, test_df=test_df, args=args,
                                    device=device,
                                    current_time=current_time)


if __name__ == "__main__":
    current_time = datetime.datetime.now()
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run(args, current_time, device)
