import matplotlib.pyplot as plt
import numpy as np
from Utils.utils import *


def print_history(fold,history,num_epochs, args, current_time):
    # plt.figure(figsize=(15,5))
    save_name = args.plot_path + '{}_{}_fold_{}_{}{}{}_{}{}{}.jpg'.format(args.dataset, args.mode, fold, current_time.day,
                                                                     current_time.month,
                                                                     current_time.year, current_time.hour,
                                                                     current_time.minute,
                                                                     current_time.second)
    fig, axs = plt.subplots(1, 2, figsize=(15,5))

    axs[0].plot(
        np.arange(num_epochs),
        history['train_history_acc'],
        '-o',
        label='Train ACC',
        color='#ff7f0e'
    )

    axs[0].plot(
        np.arange(num_epochs),
        history['val_history_acc'],
        '--o',
        label='Val ACC',
        color='#1f77b4'
    )

    axs[0].plot(
        np.arange(num_epochs),
        history['test_history_acc'],
        ':*',
        label='Test ACC',
        color='deeppink'
    )

    x = np.argmax(history['val_history_acc'])
    y = np.max(history['val_history_acc'])

    xdist = axs[0].get_xlim()[1] - axs[0].get_xlim()[0]
    ydist = axs[0].get_ylim()[1] - axs[0].get_ylim()[0]

    axs[0].scatter(x, y, s=200, color='#1f77b4')

    axs[0].text(
        x-0.03*xdist,
        y-0.13*ydist,
        'max acc\n%.2f'%y,
        size=14
    )

    axs[0].set_ylabel('ACC', size=14)
    axs[0].set_xlabel('Epoch', size=14)
    axs[0].set_title(f'FOLD {fold + 1}',size=18)
    axs[0].legend(loc=2)

    # plt2 = plt.gca().twinx()

    axs[1].plot(
        np.arange(num_epochs),
        history['train_history_loss'],
        '-o',
        label='Train Loss',
        color='#2ca02c'
    )

    axs[1].plot(
        np.arange(num_epochs),
        history['val_history_loss'],
        '--o',
        label='Val Loss',
        color='#d62728'
    )
    axs[1].plot(
        np.arange(num_epochs),
        history['test_history_loss'],
        ':*',
        label='Test Loss',
        color='deeppink'
    )

    x = np.argmin(history['val_history_loss'])
    y = np.min(history['val_history_loss'])

    xdist = axs[1].get_xlim()[1] - axs[1].get_xlim()[0]
    ydist = axs[1].get_ylim()[1] - axs[1].get_ylim()[0]

    axs[1].scatter(x, y, s=200, color='#d62728')

    axs[1].text(
        x-0.03*xdist,
        y+0.05*ydist,
        'min loss',
        size=14
    )

    axs[1].set_ylabel('Loss', size=14)
    axs[1].set_xlabel('Epochs', size=14)
    axs[1].set_title(f'FOLD {fold + 1}',size=18)

    axs[1].legend(loc=3)
    plt.savefig(save_name)

def print_history_fair(fold,history,num_epochs, args, current_time):
    # plt.figure(figsize=(15,5))
    save_name = args.plot_path + '{}_{}_fold_{}_{}{}{}_{}{}{}.jpg'.format(args.dataset, args.mode, fold, current_time.day,
                                                                     current_time.month,
                                                                     current_time.year, current_time.hour,
                                                                     current_time.minute,
                                                                     current_time.second)
    fig, axs = plt.subplots(1, 4, figsize=(22,5))

    axs[0].plot(
        np.arange(num_epochs),
        history['train_history_acc'],
        '-o',
        label='Train ACC',
        color='#ff7f0e'
    )

    axs[0].plot(
        np.arange(num_epochs),
        history['val_history_acc'],
        '--o',
        label='Val ACC',
        color='#1f77b4'
    )

    axs[0].plot(
        np.arange(num_epochs),
        history['test_history_acc'],
        ':*',
        label='Test ACC',
        color='deeppink'
    )

    x = np.argmax(history['val_history_acc'])
    y = np.max(history['val_history_acc'])

    xdist = axs[0].get_xlim()[1] - axs[0].get_xlim()[0]
    ydist = axs[0].get_ylim()[1] - axs[0].get_ylim()[0]

    axs[0].scatter(x, y, s=200, color='#1f77b4')

    axs[0].text(
        x-0.03*xdist,
        y-0.13*ydist,
        'max acc\n%.2f'%y,
        size=14
    )

    axs[0].set_ylabel('ACC', size=14)
    axs[0].set_xlabel('Epoch', size=14)
    axs[0].set_title(f'FOLD {fold + 1}',size=18)
    axs[0].legend(loc=2)

    # plt2 = plt.gca().twinx()

    axs[1].plot(
        np.arange(num_epochs),
        history['train_history_loss'],
        '-o',
        label='Train Loss',
        color='#2ca02c'
    )

    axs[1].plot(
        np.arange(num_epochs),
        history['val_history_loss'],
        '--o',
        label='Test Loss',
        color='deeppink'
    )

    axs[1].plot(
        np.arange(num_epochs),
        history['test_history_loss'],
        ':*',
        label='Val Loss',
        color='#d62728'
    )

    x = np.argmin(history['val_history_loss'])
    y = np.min(history['val_history_loss'])

    xdist = axs[1].get_xlim()[1] - axs[1].get_xlim()[0]
    ydist = axs[1].get_ylim()[1] - axs[1].get_ylim()[0]

    axs[1].scatter(x, y, s=200, color='#d62728')

    axs[1].text(
        x-0.03*xdist,
        y+0.05*ydist,
        'min loss',
        size=14
    )

    axs[1].set_ylabel('Loss', size=14)
    axs[1].set_xlabel('Epochs', size=14)
    axs[1].set_title(f'FOLD {fold + 1}',size=18)

    axs[1].legend(loc=3)

    axs[2].plot(
        np.arange(num_epochs),
        history['prob_male'],
        '-o',
        label='P(Y = 1| Male)',
        color='#2ca02c'
    )

    axs[2].plot(
        np.arange(num_epochs),
        history['prob_female'],
        '-o',
        label='P(Y = 1| Female)',
        color='#d62728'
    )

    axs[2].plot(
        np.arange(num_epochs),
        history['demo_parity'],
        '-o',
        label='Demographic Parity',
        color='blue'
    )

    axs[2].set_ylabel('Prob/Demographic Parity', size=14)
    axs[2].set_xlabel('Epochs', size=14)
    axs[2].set_title(f'FOLD {fold + 1}',size=18)

    axs[2].legend(loc=3)

    axs[3].plot(
        np.arange(num_epochs),
        history['male_tpr'],
        '-o',
        label='Male TPR',
        color='#2ca02c'
    )

    axs[3].plot(
        np.arange(num_epochs),
        history['female_tpr'],
        '-o',
        label='Female TPR',
        color='#d62728'
    )

    axs[3].plot(
        np.arange(num_epochs),
        history['equal_odd'],
        '-o',
        label='Equality of Odds',
        color='blue'
    )

    axs[3].set_ylabel('TPR/Equality of Odds', size=14)
    axs[3].set_xlabel('Epochs', size=14)
    axs[3].set_title(f'FOLD {fold + 1}',size=18)

    axs[3].legend(loc=3)
    plt.savefig(save_name)

def print_history_fair_(fold,history,num_epochs, args, current_time):
    # plt.figure(figsize=(15,5))
    save_name = args.plot_path + '{}_{}_fold_{}_{}{}{}_{}{}{}.jpg'.format(args.dataset, args.mode, fold, current_time.day,
                                                                     current_time.month,
                                                                     current_time.year, current_time.hour,
                                                                     current_time.minute,
                                                                     current_time.second)
    fig, axs = plt.subplots(1, 3, figsize=(17,5))

    axs[0].plot(
        np.arange(num_epochs),
        history['train_male_history_acc'],
        '-o',
        label='Train male ACC',
        color='#ff7f0e'
    )

    axs[0].plot(
        np.arange(num_epochs),
        history['val_male_history_acc'],
        '--o',
        label='Val male ACC',
        color='#ff7f0e'
    )

    axs[0].plot(
        np.arange(num_epochs),
        history['train_female_history_acc'],
        '-*',
        label='Train female ACC',
        color='#1f77b4'
    )

    axs[0].plot(
        np.arange(num_epochs),
        history['val_female_history_acc'],
        '--*',
        label='Val female ACC',
        color='#1f77b4'
    )

    axs[0].plot(
        np.arange(num_epochs),
        history['test_history_acc'],
        ':s',
        label='Test ACC',
        color='deeppink'
    )

    # x = np.argmax(history['val_history_acc'])
    # y = np.max(history['val_history_acc'])

    # xdist = axs[0].get_xlim()[1] - axs[0].get_xlim()[0]
    # ydist = axs[0].get_ylim()[1] - axs[0].get_ylim()[0]

    # axs[0].scatter(x, y, s=200, color='#1f77b4')

    # axs[0].text(
    #     x-0.03*xdist,
    #     y-0.13*ydist,
    #     'max acc\n%.2f'%y,
    #     size=14
    # )

    axs[0].set_ylabel('ACC', size=14)
    axs[0].set_xlabel('Epoch', size=14)
    axs[0].set_title(f'FOLD {fold + 1}',size=18)
    axs[0].legend()

    # plt2 = plt.gca().twinx()

    axs[1].plot(
        np.arange(num_epochs),
        history['train_male_history_loss'],
        '-o',
        label='Train male Loss',
        color='#2ca02c'
    )

    axs[1].plot(
        np.arange(num_epochs),
        history['val_male_history_loss'],
        '--o',
        label='Val male Loss',
        color='#2ca02c'
    )

    axs[1].plot(
        np.arange(num_epochs),
        history['train_female_history_loss'],
        '-*',
        label='Train female Loss',
        color='#d62728'
    )

    axs[1].plot(
        np.arange(num_epochs),
        history['val_female_history_loss'],
        '--*',
        label='Val female Loss',
        color='#d62728'
    )

    axs[1].plot(
        np.arange(num_epochs),
        history['test_history_loss'],
        ':s',
        label='Test Loss',
        color='deeppink'
    )

    # x = np.argmin(history['val_history_loss'])
    # y = np.min(history['val_history_loss'])

    # xdist = axs[1].get_xlim()[1] - axs[1].get_xlim()[0]
    # ydist = axs[1].get_ylim()[1] - axs[1].get_ylim()[0]

    # axs[1].scatter(x, y, s=200, color='#d62728')

    # axs[1].text(
    #     x-0.03*xdist,
    #     y+0.05*ydist,
    #     'min loss',
    #     size=14
    # )

    axs[1].set_ylabel('Loss', size=14)
    axs[1].set_xlabel('Epochs', size=14)
    axs[1].set_title(f'FOLD {fold + 1}',size=18)

    axs[1].legend()

    # axs[2].plot(
    #     np.arange(num_epochs),
    #     history['prob_male'],
    #     '-o',
    #     label='P(Y = 1| Male)',
    #     color='#2ca02c'
    # )

    # axs[2].plot(
    #     np.arange(num_epochs),
    #     history['prob_female'],
    #     '-o',
    #     label='P(Y = 1| Female)',
    #     color='#d62728'
    # )

    # axs[2].plot(
    #     np.arange(num_epochs),
    #     history['demo_parity'],
    #     '-o',
    #     label='Demographic Parity',
    #     color='blue'
    # )

    # axs[2].set_ylabel('Prob/Demographic Parity', size=14)
    # axs[2].set_xlabel('Epochs', size=14)
    # axs[2].set_title(f'FOLD {fold + 1}',size=18)

    # axs[2].legend()

    # axs[3].plot(
    #     np.arange(num_epochs),
    #     history['male_tpr'],
    #     '-o',
    #     label='Male TPR',
    #     color='#2ca02c'
    # )

    # axs[3].plot(
    #     np.arange(num_epochs),
    #     history['female_tpr'],
    #     '-o',
    #     label='Female TPR',
    #     color='#d62728'
    # )

    # axs[3].plot(
    #     np.arange(num_epochs),
    #     history['equal_odd'],
    #     '-o',
    #     label='Equality of Odds',
    #     color='blue'
    # )

    # axs[3].set_ylabel('TPR/Equality of Odds', size=14)
    # axs[3].set_xlabel('Epochs', size=14)
    # axs[3].set_title(f'FOLD {fold + 1}',size=18)

    # axs[3].legend()

    axs[2].plot(
        np.arange(num_epochs),
        history['male_norm'],
        '-o',
        label='Male norm',
        color='#2ca02c'
    )

    axs[2].plot(
        np.arange(num_epochs),
        history['female_norm'],
        '-o',
        label='Female norm',
        color='#d62728'
    )

    value_bound = bound(args)

    axs[2].plot(
        np.arange(num_epochs),
        np.ones(num_epochs)*value_bound,
        '-o',
        label='Bound',
        color='blue'
    )

    axs[2].set_ylabel('L1 norm', size=14)
    axs[2].set_xlabel('Epochs', size=14)
    axs[2].set_title(f'FOLD {fold + 1}',size=18)

    axs[2].legend()
    plt.savefig(save_name)

def print_history_fair_v2(fold,history,num_epochs, args, current_time):
    save_name = args.plot_path + '{}_{}_fold_{}_{}{}{}_{}{}{}.jpg'.format(args.dataset, args.mode, fold,
                                                                          current_time.day,
                                                                          current_time.month,
                                                                          current_time.year, current_time.hour,
                                                                          current_time.minute,
                                                                          current_time.second)
    fig, axs = plt.subplots(1, 3, figsize=(17, 5))

    # axs[0].plot(
    #     np.arange(num_epochs),
    #     history['train_male_history_acc'],
    #     '-o',
    #     label='Train male ACC',
    #     color='#ff7f0e'
    # )

    # axs[0].plot(
    #     np.arange(num_epochs),
    #     history['val_male_history_acc'],
    #     '-o',
    #     label='Val male ACC',
    #     color='#1f77b4'
    # )

    # axs[0].plot(
    #     np.arange(num_epochs),
    #     history['train_female_history_acc'],
    #     '--s',
    #     label='Train female ACC',
    #     color='#ff7f0e'
    # )

    # axs[0].plot(
    #     np.arange(num_epochs),
    #     history['val_female_history_acc'],
    #     '--s',
    #     label='Val female ACC',
    #     color='#1f77b4'
    # )

    axs[0].plot(
        np.arange(num_epochs),
        history['train_global_history_acc'],
        '-o',
        label='Train Global ACC',
        color='#ff7f0e'
    )

    axs[0].plot(
        np.arange(num_epochs),
        history['val_global_history_acc'],
        '--^',
        label='Val Global ACC',
        color='#1f77b4'
    )

    axs[0].plot(
        np.arange(num_epochs),
        history['test_history_acc'],
        ':*',
        label='Test ACC',
        color='deeppink'
    )

    # x = np.argmax(history['val_history_acc'])
    # y = np.max(history['val_history_acc'])

    # xdist = axs[0].get_xlim()[1] - axs[0].get_xlim()[0]
    # ydist = axs[0].get_ylim()[1] - axs[0].get_ylim()[0]

    # axs[0].scatter(x, y, s=200, color='#1f77b4')

    # axs[0].text(
    #     x-0.03*xdist,
    #     y-0.13*ydist,
    #     'max acc\n%.2f'%y,
    #     size=14
    # )

    axs[0].set_ylabel('ACC', size=14)
    axs[0].set_xlabel('Epoch', size=14)
    axs[0].set_title(f'FOLD {fold + 1}', size=18)
    axs[0].legend()

    # plt2 = plt.gca().twinx()

    # axs[1].plot(
    #     np.arange(num_epochs),
    #     history['train_male_history_loss'],
    #     '-o',
    #     label='Train male Loss',
    #     color='#2ca02c'
    # )

    # axs[1].plot(
    #     np.arange(num_epochs),
    #     history['val_male_history_loss'],
    #     '-o',
    #     label='Val male Loss',
    #     color='#d62728'
    # )

    # axs[1].plot(
    #     np.arange(num_epochs),
    #     history['train_female_history_loss'],
    #     '--s',
    #     label='Train female Loss',
    #     color='#2ca02c'
    # )

    # axs[1].plot(
    #     np.arange(num_epochs),
    #     history['val_female_history_loss'],
    #     '--s',
    #     label='Val female Loss',
    #     color='#d62728'
    # )

    axs[1].plot(
        np.arange(num_epochs),
        history['train_global_history_loss'],
        '-o',
        label='Train Global Loss',
        color='#2ca02c'
    )

    axs[1].plot(
        np.arange(num_epochs),
        history['val_global_history_loss'],
        '--o',
        label='Val Global Loss',
        color='#d62728'
    )

    axs[1].plot(
        np.arange(num_epochs),
        history['test_history_loss'],
        ':*',
        label='Test Loss',
        color='deeppink'
    )

    # x = np.argmin(history['val_history_loss'])
    # y = np.min(history['val_history_loss'])

    # xdist = axs[1].get_xlim()[1] - axs[1].get_xlim()[0]
    # ydist = axs[1].get_ylim()[1] - axs[1].get_ylim()[0]

    # axs[1].scatter(x, y, s=200, color='#d62728')

    # axs[1].text(
    #     x-0.03*xdist,
    #     y+0.05*ydist,
    #     'min loss',
    #     size=14
    # )

    axs[1].set_ylabel('Loss', size=14)
    axs[1].set_xlabel('Epochs', size=14)
    axs[1].set_title(f'FOLD {fold + 1}', size=18)

    axs[1].legend()

    axs[2].plot(
        np.arange(num_epochs),
        history['male_norm'],
        '-o',
        label='Male norm',
        color='#2ca02c'
    )

    axs[2].plot(
        np.arange(num_epochs),
        history['female_norm'],
        '-o',
        label='Female norm',
        color='#d62728'
    )

    value_bound = bound(args)

    axs[2].plot(
        np.arange(num_epochs),
        np.ones(num_epochs) * value_bound,
        '-o',
        label='Bound',
        color='blue'
    )

    axs[2].set_ylabel('L1 norm', size=14)
    axs[2].set_xlabel('Epochs', size=14)
    axs[2].set_title(f'FOLD {fold + 1}', size=18)

    axs[2].legend()
    plt.savefig(save_name)

def print_history_fair_v3(fold,history,num_epochs, args, current_time):
    save_name = args.plot_path + '{}_{}_fold_{}_{}{}{}_{}{}{}.jpg'.format(args.dataset, args.mode, fold,
                                                                          current_time.day,
                                                                          current_time.month,
                                                                          current_time.year, current_time.hour,
                                                                          current_time.minute,
                                                                          current_time.second)
    fig, axs = plt.subplots(1, 3, figsize=(17,5))

    axs[0].plot(
        np.arange(num_epochs),
        history['train_male_history_acc'],
        '-o',
        label='Train male ACC',
        color='#ff7f0e'
    )

    axs[0].plot(
        np.arange(num_epochs),
        history['val_male_history_acc'],
        '--o',
        label='Val male ACC',
        color='#ff7f0e'
    )

    axs[0].plot(
        np.arange(num_epochs),
        history['train_female_history_acc'],
        '-*',
        label='Train female ACC',
        color='#1f77b4'
    )

    axs[0].plot(
        np.arange(num_epochs),
        history['val_female_history_acc'],
        '--*',
        label='Val female ACC',
        color='#1f77b4'
    )

    axs[0].plot(
        np.arange(num_epochs),
        history['val_global_history_acc'],
        ':^',
        label='Val global ACC',
        color='green'
    )

    axs[0].plot(
        np.arange(num_epochs),
        history['test_history_acc'],
        '-.s',
        label='Test ACC',
        color='deeppink'
    )

    # x = np.argmax(history['val_history_acc'])
    # y = np.max(history['val_history_acc'])

    # xdist = axs[0].get_xlim()[1] - axs[0].get_xlim()[0]
    # ydist = axs[0].get_ylim()[1] - axs[0].get_ylim()[0]

    # axs[0].scatter(x, y, s=200, color='#1f77b4')

    # axs[0].text(
    #     x-0.03*xdist,
    #     y-0.13*ydist,
    #     'max acc\n%.2f'%y,
    #     size=14
    # )

    axs[0].set_ylabel('ACC', size=14)
    axs[0].set_xlabel('Epoch', size=14)
    axs[0].set_title(f'FOLD {fold + 1}',size=18)
    axs[0].legend()

    # plt2 = plt.gca().twinx()

    axs[1].plot(
        np.arange(num_epochs),
        history['train_male_history_loss'],
        '-o',
        label='Train male Loss',
        color='#2ca02c'
    )

    axs[1].plot(
        np.arange(num_epochs),
        history['val_male_history_loss'],
        '--o',
        label='Val male Loss',
        color='#2ca02c'
    )

    axs[1].plot(
        np.arange(num_epochs),
        history['train_female_history_loss'],
        '-*',
        label='Train female Loss',
        color='#d62728'
    )

    axs[1].plot(
        np.arange(num_epochs),
        history['val_female_history_loss'],
        '--*',
        label='Val female Loss',
        color='#d62728'
    )

    axs[1].plot(
        np.arange(num_epochs),
        history['val_global_history_loss'],
        ':^',
        label='Val global Loss',
        color='blue'
    )

    axs[1].plot(
        np.arange(num_epochs),
        history['test_history_loss'],
        '-.s',
        label='Test Loss',
        color='deeppink'
    )


    # x = np.argmin(history['val_history_loss'])
    # y = np.min(history['val_history_loss'])

    # xdist = axs[1].get_xlim()[1] - axs[1].get_xlim()[0]
    # ydist = axs[1].get_ylim()[1] - axs[1].get_ylim()[0]

    # axs[1].scatter(x, y, s=200, color='#d62728')

    # axs[1].text(
    #     x-0.03*xdist,
    #     y+0.05*ydist,
    #     'min loss',
    #     size=14
    # )

    axs[1].set_ylabel('Loss', size=14)
    axs[1].set_xlabel('Epochs', size=14)
    axs[1].set_title(f'FOLD {fold + 1}',size=18)

    axs[1].legend()

    # axs[2].plot(
    #     np.arange(num_epochs),
    #     history['prob_male'],
    #     '-o',
    #     label='P(Y = 1| Male)',
    #     color='#2ca02c'
    # )

    # axs[2].plot(
    #     np.arange(num_epochs),
    #     history['prob_female'],
    #     '-o',
    #     label='P(Y = 1| Female)',
    #     color='#d62728'
    # )

    # axs[2].plot(
    #     np.arange(num_epochs),
    #     history['demo_parity'],
    #     '-o',
    #     label='Demographic Parity',
    #     color='blue'
    # )

    # axs[2].set_ylabel('Prob/Demographic Parity', size=14)
    # axs[2].set_xlabel('Epochs', size=14)
    # axs[2].set_title(f'FOLD {fold + 1}',size=18)

    # axs[2].legend()

    # axs[3].plot(
    #     np.arange(num_epochs),
    #     history['male_tpr'],
    #     '-o',
    #     label='Male TPR',
    #     color='#2ca02c'
    # )

    # axs[3].plot(
    #     np.arange(num_epochs),
    #     history['female_tpr'],
    #     '-o',
    #     label='Female TPR',
    #     color='#d62728'
    # )

    # axs[3].plot(
    #     np.arange(num_epochs),
    #     history['equal_odd'],
    #     '-o',
    #     label='Equality of Odds',
    #     color='blue'
    # )

    # axs[3].set_ylabel('TPR/Equality of Odds', size=14)
    # axs[3].set_xlabel('Epochs', size=14)
    # axs[3].set_title(f'FOLD {fold + 1}',size=18)

    # axs[3].legend()

    axs[2].plot(
        np.arange(num_epochs),
        history['male_norm'],
        '-o',
        label='Male norm',
        color='#2ca02c'
    )

    axs[2].plot(
        np.arange(num_epochs),
        history['female_norm'],
        '-o',
        label='Female norm',
        color='#d62728'
    )

    value_bound = bound(args)

    axs[2].plot(
        np.arange(num_epochs),
        np.ones(num_epochs)*value_bound,
        '-o',
        label='Bound',
        color='blue'
    )

    axs[2].set_ylabel('L1 norm', size=14)
    axs[2].set_xlabel('Epochs', size=14)
    axs[2].set_title(f'FOLD {fold + 1}',size=18)

    axs[2].legend()
    plt.savefig(save_name)

def print_history_fair_alg1(fold,history,num_epochs, args, current_time):
    # plt.figure(figsize=(15,5))
    save_name = args.plot_path + '{}_{}_fold_{}_{}{}{}_{}{}{}.jpg'.format(args.dataset, args.mode, fold, current_time.day,
                                                                     current_time.month,
                                                                     current_time.year, current_time.hour,
                                                                     current_time.minute,
                                                                     current_time.second)
    fig, axs = plt.subplots(1, 3, figsize=(17,5))

    axs[0].plot(
        np.arange(num_epochs),
        history['train_male_history_acc'],
        '-o',
        label='Train male ACC',
        color='#ff7f0e'
    )

    axs[0].plot(
        np.arange(num_epochs),
        history['val_male_history_acc'],
        '--o',
        label='Val male ACC',
        color='#ff7f0e'
    )

    axs[0].plot(
        np.arange(num_epochs),
        history['train_female_history_acc'],
        '-*',
        label='Train female ACC',
        color='#1f77b4'
    )

    axs[0].plot(
        np.arange(num_epochs),
        history['val_female_history_acc'],
        '--*',
        label='Val female ACC',
        color='#1f77b4'
    )

    axs[0].plot(
        np.arange(num_epochs),
        history['test_history_acc'],
        '-.s',
        label='Test ACC',
        color='deeppink'
    )

    # x = np.argmax(history['val_history_acc'])
    # y = np.max(history['val_history_acc'])

    # xdist = axs[0].get_xlim()[1] - axs[0].get_xlim()[0]
    # ydist = axs[0].get_ylim()[1] - axs[0].get_ylim()[0]

    # axs[0].scatter(x, y, s=200, color='#1f77b4')

    # axs[0].text(
    #     x-0.03*xdist,
    #     y-0.13*ydist,
    #     'max acc\n%.2f'%y,
    #     size=14
    # )

    axs[0].set_ylabel('ACC', size=14)
    axs[0].set_xlabel('Epoch', size=14)
    axs[0].set_title(f'FOLD {fold + 1}',size=18)
    axs[0].legend()

    # plt2 = plt.gca().twinx()

    axs[1].plot(
        np.arange(num_epochs),
        history['train_male_history_loss'],
        '-o',
        label='Train male Loss',
        color='#2ca02c'
    )

    axs[1].plot(
        np.arange(num_epochs),
        history['val_male_history_loss'],
        '--o',
        label='Val male Loss',
        color='#2ca02c'
    )

    axs[1].plot(
        np.arange(num_epochs),
        history['train_female_history_loss'],
        '-*',
        label='Train female Loss',
        color='#d62728'
    )

    axs[1].plot(
        np.arange(num_epochs),
        history['val_female_history_loss'],
        '--*',
        label='Val female Loss',
        color='#d62728'
    )

    axs[1].plot(
        np.arange(num_epochs),
        history['test_history_loss'],
        '-.s',
        label='Test Loss',
        color='deeppink'
    )

    # x = np.argmin(history['val_history_loss'])
    # y = np.min(history['val_history_loss'])

    # xdist = axs[1].get_xlim()[1] - axs[1].get_xlim()[0]
    # ydist = axs[1].get_ylim()[1] - axs[1].get_ylim()[0]

    # axs[1].scatter(x, y, s=200, color='#d62728')

    # axs[1].text(
    #     x-0.03*xdist,
    #     y+0.05*ydist,
    #     'min loss',
    #     size=14
    # )

    axs[1].set_ylabel('Loss', size=14)
    axs[1].set_xlabel('Epochs', size=14)
    axs[1].set_title(f'FOLD {fold + 1}',size=18)

    axs[1].legend()

    # axs[2].plot(
    #     np.arange(num_epochs),
    #     history['prob_male'],
    #     '-o',
    #     label='P(Y = 1| Male)',
    #     color='#2ca02c'
    # )

    # axs[2].plot(
    #     np.arange(num_epochs),
    #     history['prob_female'],
    #     '-o',
    #     label='P(Y = 1| Female)',
    #     color='#d62728'
    # )

    # axs[2].plot(
    #     np.arange(num_epochs),
    #     history['demo_parity'],
    #     '-o',
    #     label='Demographic Parity',
    #     color='blue'
    # )

    # axs[2].set_ylabel('Prob/Demographic Parity', size=14)
    # axs[2].set_xlabel('Epochs', size=14)
    # axs[2].set_title(f'FOLD {fold + 1}',size=18)

    # axs[2].legend()

    # axs[3].plot(
    #     np.arange(num_epochs),
    #     history['male_tpr'],
    #     '-o',
    #     label='Male TPR',
    #     color='#2ca02c'
    # )

    # axs[3].plot(
    #     np.arange(num_epochs),
    #     history['female_tpr'],
    #     '-o',
    #     label='Female TPR',
    #     color='#d62728'
    # )

    # axs[3].plot(
    #     np.arange(num_epochs),
    #     history['equal_odd'],
    #     '-o',
    #     label='Equality of Odds',
    #     color='blue'
    # )

    # axs[3].set_ylabel('TPR/Equality of Odds', size=14)
    # axs[3].set_xlabel('Epochs', size=14)
    # axs[3].set_title(f'FOLD {fold + 1}',size=18)

    # axs[3].legend()

    axs[2].plot(
        np.arange(num_epochs),
        history['male_norm'],
        '-o',
        label='Male norm',
        color='#2ca02c'
    )

    axs[2].plot(
        np.arange(num_epochs),
        history['female_norm'],
        '-o',
        label='Female norm',
        color='#d62728'
    )

    value_bound = bound(args)
    # value_bound_alg1 = bound_alg1(args)

    axs[2].plot(
        np.arange(num_epochs),
        np.ones(num_epochs)*value_bound,
        '-o',
        label='Proposed Bound',
        color='blue'
    )

    # axs[2].plot(
    #     np.arange(num_epochs),
    #     np.arange(num_epochs) * value_bound,
    #     '-o',
    #     label='Bound with T',
    #     color='blue'
    # )

    axs[2].set_ylabel('L1 norm', size=14)
    axs[2].set_xlabel('Epochs', size=14)
    axs[2].set_title(f'FOLD {fold + 1}',size=18)

    axs[2].legend()
    plt.savefig(save_name)