from config import parse_args
from Data.read_data import *
from Data.datasets import *
import datetime
from Utils.running import *


def run(args, current_time, device):
    # read data
    if args.dataset == 'adult':
        train_df, test_df, male_df, female_df, feature_cols, label = read_adult(args)
    # running process
    if args.mode == 'clean':
        if args.debug:
            run_clean(fold=0, df=train_df, args=args, device=device, current_time=current_time)
        else:
            for fold in range(args.folds):
                run_clean(fold=fold, df=train_df, args=args, device=device, current_time=current_time)
    elif args.mode == 'dpsgd':
        if args.debug:
            run_dpsgd(fold=0, df=train_df, args=args, device=device, current_time=current_time)
        else:
            for fold in range(args.folds):
                run_dpsgd(fold=fold, df=train_df, args=args, device=device, current_time=current_time)
    elif args.mode == 'fair':
        if args.debug:
            run_fair_v2(fold=0, male_df=male_df, female_df=female_df, args=args, device=device,
                        current_time=current_time)
        else:
            for fold in range(args.folds):
                run_fair_v2(fold=fold, male_df=male_df, female_df=female_df, args=args, device=device,
                            current_time=current_time)
    elif args.mode == 'proposed':
        if args.debug:
            run_fair_dpsgd_alg2(fold=0, male_df=male_df, female_df=female_df, args=args, device=device,
                        current_time=current_time)
        else:
            for fold in range(args.folds):
                run_fair_dpsgd_alg2(fold=fold, male_df=male_df, female_df=female_df, args=args, device=device,
                            current_time=current_time)


if __name__ == "__main__":
    current_time = datetime.datetime.now()
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run(args, current_time, device)
