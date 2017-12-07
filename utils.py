import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import sys
import os


def plt_learn(hist_list, fname=None):
    colors = ['b', 'c', 'y']
    fig, ax = plt.subplots()
    for i, hist in enumerate(hist_list):
        ax.plot(hist['epoch'], hist['train_loss'], color=colors[i],
                                                   label=hist['optimizer'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True)
    if fname is not None:
        fig.savefig(os.path.join('./img', fname))
        plt.close(fig)
    plt.show()


def plt_representations(auenc, inputs, labels, fname=None):
    auenc.net.compute_outputs(inputs)
    mid_idx = int(len(auenc.net.layers) / 2) - 1
    reprs = auenc.net.get_activations(mid_idx).T

    opt = auenc.hist['optimizer']

    c = list(cm.rainbow(np.linspace(0, 5, 10)))
    mnist = {'1': ([], c[0]), '2': ([], c[1]), '3': ([], c[2]),
             '4': ([], c[3]), '5': ([], c[4]), '6': ([], c[5]),
             '7': ([], c[6]), '8': ([], c[7]), '9': ([], c[8]),
             '0': ([], c[9])}

    for i in range(len(labels)):
        label = int(labels[i, 0])
        mnist[str(label)][0].append(reprs[i, :])

    fig, ax = plt.subplots()
    for key in mnist.keys():
        x = [p[0] for p in mnist[key][0]]
        y = [p[1] for p in mnist[key][0]]
        ax.scatter(x, y, color=mnist[key][1], label=key, alpha=0.3)
    # ax.title('Inputs representation, oprtimizer: {}'.format(opt))
    ax.legend()
    ax.grid(True)
    if fname is not None:
        fig.savefig(os.path.join('./img', fname))
        plt.close(fig)
    else:
        plt.show()
