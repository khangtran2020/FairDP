from config import parse_args
from Data.read_data import *
from Data.datasets import *
import datetime
from Utils.running import *
from Utils.utils import seed_everything
from Model.models import NeuralNetwork
import warnings
import torch

warnings.filterwarnings("ignore")


def run(args, current_time, device):
    # read data
    print('Using noise scale: {}, clip: {}'.format(args.ns, args.clip))
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
    elif args.mode == 'test':
        train_dataset = Adult(
            train_df[args.feature].values,
            train_df[args.target].values
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            pin_memory=True,
            drop_last=True,
            num_workers=0
        )

        # Defining Model for specific fold
        model = NeuralNetwork(args.input_dim, args.n_hid, args.output_dim)
        model.to(device)

        # DEfining criterion
        criterion = torch.nn.BCELoss()
        criterion.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=args.epochs,
            target_epsilon=args.tar_eps,
            target_delta=args.tar_delt,
            max_grad_norm=args.clip,
        )
        print(f"Using sigma={optimizer.noise_multiplier} and C={args.clip}")


if __name__ == "__main__":
    current_time = datetime.datetime.now()
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run(args, current_time, device)
