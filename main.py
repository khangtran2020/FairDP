from config import parse_args
from Data.read_data import *
from Data.datasets import *
import datetime
from Utils.running import *
from Utils.utils import seed_everything
import warnings

warnings.filterwarnings("ignore")


def run(args, current_time, device):
    # read data
    if args.dataset == 'adult':
        train_df, test_df, male_df, female_df, feature_cols, label = read_adult(args)
        args.feature = feature_cols
        args.target = label
        args.input_dim = len(feature_cols)
        args.output_dim = 1
        print(feature_cols)
    elif args.dataset == 'bank':
        train_df, test_df, male_df, female_df, feature_cols, label = read_bank(args)
        args.feature = feature_cols
        args.target = label
        args.input_dim = len(feature_cols)
        args.output_dim = 1
        print(feature_cols)
    elif args.dataset == 'stroke':
        train_df, test_df, male_df, female_df, feature_cols, label = read_stroke(args)
        args.feature = feature_cols
        args.target = label
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
            run_fair_v2(fold=0, male_df=male_df, female_df=female_df, test_df=test_df, args=args, device=device,
                        current_time=current_time)
        else:
            for fold in range(args.folds):
                run_fair_v2(fold=fold, male_df=male_df, female_df=female_df, test_df=test_df, args=args, device=device,
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
    elif args.mode == 'alg1':
        if args.debug:
            run_fair_dpsgd_alg1(fold=0, male_df=male_df, female_df=female_df, test_df=test_df, args=args, device=device,
                                current_time=current_time)
        else:
            for fold in range(args.folds):
                run_fair_dpsgd_alg1(fold=fold, male_df=male_df, female_df=female_df, test_df=test_df, args=args,
                                    device=device,
                                    current_time=current_time)
    elif args.mode == 'opacus':
        if args.debug:
            run_opacus(fold=0, train_df=train_df, test_df=test_df, args=args, device=device, current_time=current_time)
        else:
            for fold in range(args.folds):
                run_fair_dpsgd_alg1(fold=fold, train_df=train_df, test_df=test_df, args=args, device=device,
                                    current_time=current_time)
    elif args.mode == 'dpmanual':
        if args.debug:
            run_dpsgd_without_optimizer(fold=0, train_df=train_df, test_df=test_df, args=args, device=device,
                                        current_time=current_time)
        else:
            for fold in range(args.folds):
                run_dpsgd_without_optimizer(fold=fold, train_df=train_df, test_df=test_df, args=args, device=device,
                                            current_time=current_time)
    elif args.mode == 'onebatch':
        if args.debug:
            run_fair_dpsgd_alg2_one_batch(fold=0, male_df=male_df, female_df=female_df, test_df=test_df, args=args, device=device,
                                        current_time=current_time)
        else:
            for fold in range(args.folds):
                run_fair_dpsgd_alg2_one_batch(fold=fold, male_df=male_df, female_df=female_df, test_df=test_df, args=args, device=device,
                                            current_time=current_time)


if __name__ == "__main__":
    current_time = datetime.datetime.now()
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run(args, current_time, device)
