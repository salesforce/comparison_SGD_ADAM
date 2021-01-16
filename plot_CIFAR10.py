import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
import torch
import math
import pdb
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
def average_tensor(x):
    average_ = []
    for i in range(len(x)):
        average_.append(torch.mean(x[i]).data.numpy())

    return np.array(average_)


def tensor_numpy_(x):
    average_ = []
    for i in range(len(x)):
        average_.append(x[i].data.numpy())

    return np.array(average_)

if __name__ == '__main__':

    font = {'family': 'monospace',
            'weight': 'bold',
            'size': 'larger'}

    params = {'legend.fontsize': 'x-large',
              'figure.figsize': (7, 3),
              'axes.labelsize': 'x-large',
              'axes.titlesize': 'x-large',
              'xtick.labelsize': 'large',
              'ytick.labelsize': 'large',
              "legend.markerscale": 2.5
              }
    lengend_size='medium'
    marker0 = ":"
    marker32 = "--"
    size_point = 0
    linewide = 1
    linewide2 = 8


    color1 = (87/255.0, 150/255.0, 204/255.0)
    color2 = (217/255.0, 83/255.0, 22/255.0)
    color3 = (233/255.0, 180/255.0, 22/255.0)
    color4 = (116/255.0, 170/255.0, 48/255.0)
    color7 = (1, 0, 0)

    dataset = 'cifar10'
    ddd1, ddd2 = 150, 1000

    iddx = 1

    ## SGD
    optimizer = 'sgd'
    idx_threshold = 1000
    threshold, scale_ratio = 2.1, 0.14
    ## load SGD data
    sgd_evaluation_history_TEST = torch.load('./exp_data/SGD_evaluation_history_TEST.hist', map_location='cpu')
    sgd_evaluation_history_TRAIN = torch.load('./exp_data/SGD_evaluation_history_TRAIN.hist', map_location='cpu')

    sgd_evaluation_history_TEST = np.array(sgd_evaluation_history_TEST)
    sgd_evaluation_history_TRAIN = np.array(sgd_evaluation_history_TRAIN)

    x_idx_sgd = [math.log10(sgd_evaluation_history_TRAIN[i][0]+1) for i in range(0, len(sgd_evaluation_history_TRAIN))]
    x_idx2_sgd = [math.log10(sgd_evaluation_history_TEST[i][0]+1) for i in range(0, len(sgd_evaluation_history_TEST))]

    ## scale data such that the phenomenon can be well observed.
    x_idx_sgd2 = np.array(x_idx_sgd)
    idx = np.where(x_idx_sgd2 < threshold)[0]
    IDD = x_idx_sgd2[idx[-1]]
    x_idx_sgd2 = np.append(x_idx_sgd2[idx] * scale_ratio, x_idx_sgd2[idx[-1]+1:] - (1 - scale_ratio) * IDD)

    x_idx2_sgd2 = np.array(x_idx2_sgd)
    idx = np.where(x_idx2_sgd2 < threshold)[0]
    IDD_sgd = x_idx2_sgd2[idx[-1]]
    x_idx2_sgd2 = np.append(x_idx2_sgd2[idx] * scale_ratio, x_idx2_sgd2[idx[-1] + 1:] - (1 - scale_ratio) *IDD_sgd)


    x_left_sgd, x_right_sgd = 1.3, x_idx2_sgd2[-1] + 0.1

    ## load ADAM data
    adam_evaluation_history_TEST = torch.load('./exp_data/ADAM_evaluation_history_TEST.hist', map_location='cpu')
    adam_evaluation_history_TRAIN = torch.load('./exp_data/ADAM_evaluation_history_TRAIN.hist', map_location='cpu')

    adam_evaluation_history_TEST = np.array(adam_evaluation_history_TEST)
    adam_evaluation_history_TRAIN = np.array(adam_evaluation_history_TRAIN)

    x_idx_adam = [math.log10(adam_evaluation_history_TRAIN[i][0]+1) for i in range(0, len(adam_evaluation_history_TRAIN))]
    x_idx2_adam = [math.log10(adam_evaluation_history_TEST[i][0]+1) for i in range(0, len(adam_evaluation_history_TEST))]

    ## scale data such that the phenomenon can be well observed.
    x_idx_adam2 = np.array(x_idx_adam)
    idx = np.where(x_idx_adam2 < threshold)[0]
    IDD = x_idx_adam2[idx[-1]]
    x_idx_adam2 = np.append(x_idx_adam2[idx] * scale_ratio, x_idx_adam2[idx[-1]+1:] - (1 - scale_ratio) * IDD)

    x_idx2_adam2 = np.array(x_idx2_adam)
    idx = np.where(x_idx2_adam2 < threshold)[0]
    IDD_adam = x_idx2_adam2[idx[-1]]
    x_idx2_adam2 = np.append(x_idx2_adam2[idx] * scale_ratio, x_idx2_adam2[idx[-1] + 1:] - (1 - scale_ratio) * IDD_adam)

    x_left_adam, x_right_adam = 1.3, x_idx2_adam2[-1] + 0.1

    print('\n accuracy sgd %.3f adam %.3f'%(sgd_evaluation_history_TRAIN[:,2].max(),adam_evaluation_history_TRAIN[:,2].max()))
    print('\n accuracy sgd %.3f adam %.3f'%(sgd_evaluation_history_TRAIN[-1,2],adam_evaluation_history_TRAIN[-1,2]))

    print('\n accuracy sgd %.3f adam %.3f'%(sgd_evaluation_history_TEST[:,2].max(),adam_evaluation_history_TEST[:,2].max()))
    print('\n accuracy sgd %.3f adam %.3f'%(sgd_evaluation_history_TEST[-1,2],adam_evaluation_history_TEST[-1,2]))

    x_left, x_right = 0, max(x_right_sgd,x_right_adam)
    x_left2, x_right2 = 2, max(x_idx2_sgd[-1] + 0.1,x_idx2_adam[-1] + 0.1)

    if dataset == 'mnist':
        loss_left,loss_right = 0.0, 2.4
        alpha_left, alpha_right = 0.5, 1.3
        accuracy_left, accuracy_right = 8.5, 99.99999
        noise_left, noise_right = 0.0, 7.0
    elif dataset == 'cifar10':
        loss_left,loss_right = 0.0, 2.4
        alpha_left, alpha_right = 0.8, 1.5
        accuracy_left, accuracy_right = 5.0, 59.0
        noise_left, noise_right = 0.0, 3.0
    else:
        print('\n error dataset\n')


    plt.close()
    plt.rcParams.update(params)
    fig = plt.figure()


    ax=plt.subplot(1,2,1)
    plt.plot(x_idx_sgd2, sgd_evaluation_history_TRAIN[0:,3], 'o-', lw=linewide, markersize=size_point, color=color1)
    plt.plot(x_idx2_sgd2, sgd_evaluation_history_TEST[0:,3], "p-.",lw=linewide, markersize=size_point, color=color2)
    plt.plot(x_idx_adam2, adam_evaluation_history_TRAIN[0:, 3], 'o-', lw=linewide, markersize=size_point,
                  color=color3)
    plt.plot(x_idx2_adam2, adam_evaluation_history_TEST[0:, 3], "p-.", lw=linewide, markersize=size_point,
                  color=color4)


    if np.log10(ddd1)<threshold:
        aa = np.log10(ddd1) * scale_ratio
    else:
        aa= np.log10(ddd1)- (1 - scale_ratio) *IDD_sgd

    if np.log10(ddd2)<threshold:
        bb = np.log10(ddd2) * scale_ratio
    else:
        bb= np.log10(ddd2)- (1 - scale_ratio) *IDD_sgd


    plt.plot([aa, aa], [alpha_left, 1.35* alpha_left],'--', lw=linewide, markersize=size_point, color=color7)
    plt.plot([aa, aa], [1.46 * alpha_left, alpha_right], '--', lw=linewide, markersize=size_point, color=color7)
    plt.plot([bb, bb], [alpha_left, 1.35* alpha_left],'--', lw=linewide, markersize=size_point, color=color7)
    plt.plot([bb, bb], [1.47 * alpha_left, alpha_right], '--', lw=linewide, markersize=size_point, color=color7)

    plt.text(aa-0.23, 0.81, r'$t_1$', fontsize=12)
    plt.text(bb+0.05, 0.81, r'$t_2$', fontsize=12)

    plt.xlabel('Iterations',fontweight='bold')
    plt.ylabel(r"Estimated $\alpha$", fontweight='bold')
    plt.ylim(alpha_left, alpha_right)
    plt.xlim(x_left,x_right)
    plt.xticks([2.2 - (1 - scale_ratio) *IDD_sgd, 3 - (1 - scale_ratio) *IDD_sgd, 4 -  (1 - scale_ratio) *IDD_sgd, 5 - (1 - scale_ratio) *IDD_sgd],
               (r"$10^{2.2}}$", r"$10^3$", r"$10^4$", r"$10^5$"))

    ## plot 2∂∂
    ax = plt.subplot(1, 2, 2)
    s1 = plt.plot(x_idx_sgd2, sgd_evaluation_history_TRAIN[0:, 2], 'o-', lw=linewide, markersize=size_point,
                  color=color1)
    s2 = plt.plot(x_idx2_sgd2, sgd_evaluation_history_TEST[0:, 2], "p--", lw=linewide, markersize=size_point,
                  color=color2)
    dddd = 46
    d2 = -0.19
    font_size11=11
    plt.text(0.65-d2, 72-dddd, r'SGD: training acc. 99.83%', fontsize=font_size11)
    plt.text(1.27-d2, 68-dddd, r'test acc. 55.98%', fontsize=font_size11)
    plt.text(0.65-d2 -0.2, 62-dddd, r'ADAM: training acc. 99.89%', fontsize=font_size11)
    plt.text(1.43-d2-0.2, 58-dddd, r'test acc. 54.29%', fontsize=font_size11)

    s3 = plt.plot(x_idx_adam2, adam_evaluation_history_TRAIN[0:, 2], 'o-', lw=linewide, markersize=size_point,
                  color=color3)
    s4 = plt.plot(x_idx2_adam2, adam_evaluation_history_TEST[0:, 2], "p--", lw=linewide, markersize=size_point,
                  color=color4)


    plt.plot([aa, aa], [accuracy_left, 2.7* accuracy_left],'--', lw=linewide, markersize=size_point, color=color7)
    plt.plot([aa, aa], [3.5 * accuracy_left, accuracy_right], '--', lw=linewide, markersize=size_point, color=color7)
    plt.plot([bb, bb], [accuracy_left, accuracy_right],'--', lw=linewide, markersize=size_point, color=color7)
    plt.text(aa-0.23, 6, r'$t_1$', fontsize=12)
    plt.text(bb+0.05, 6, r'$t_2$', fontsize=12)

    plt.xlabel('Iterations', fontweight='bold')
    plt.ylabel(r"Accuracy (%)", fontweight='bold')
    plt.ylim(accuracy_left, accuracy_right)
    plt.xlim(x_left, x_right)
    plt.xticks([ 2.2 - (1 - scale_ratio) *IDD_sgd, 3 - (1 - scale_ratio) *IDD_sgd, 4 -  (1 - scale_ratio) *IDD_sgd, 5 - (1 - scale_ratio) *IDD_sgd],
               ( r"$10^{2.2}}$", r"$10^3$", r"$10^4$", r"$10^5$"))

    line_labels = ["SGD train", "SGD test", "ADAM train", "ADAM test"]
    pro = matplotlib.font_manager.FontProperties(weight='bold', size='large')
    leg = fig.legend(labels=line_labels,
                     frameon=False,
                     ncol=4,
                     bbox_to_anchor=(0.99, 1.04),
                     prop=pro
                     )
    legend_line_width_scale = 2
    for line in leg.get_lines():
        line.set_linewidth(legend_line_width_scale)

    fig.tight_layout()
    # plt.show()
    plt.savefig("./exp_data/comparison.png")



