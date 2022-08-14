import collections
import matplotlib.pyplot as plt
import json

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import torch

matplotlib.use("Agg")


def load_json(file):
    try:
        with open(file, "r") as f:
            return json.load(f)
    except:
        raise IOError("json file {} not found".format(file))


def save_json(data, file):
    try:
        with open(file, "w") as f:
            json.dump(data, f, indent=4)
    except:
        raise IOError("json file {} write error".format(file))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


class ResultGatherer(object):
    def __init__(self):
        self.reset()
        self.output = []
        self.target = []

    def reset(self):
        self.output = []
        self.target = []

    def update(self, output, target):
        if isinstance(output, torch.Tensor) and isinstance(target, torch.Tensor):
            self.output.append(output.detach().cpu().numpy().flatten())
            self.target.append(target.detach().cpu().numpy().flatten())
        else:
            self.output.append(output)
            self.target.append(target)

    def finalise(self):
        self.output = np.concatenate(self.output, axis=0)
        self.target = np.concatenate(self.target, axis=0)


class ResultCounter(object):
    def __init__(self):
        self.reset()
        self.counter = collections.defaultdict(int)

    def reset(self):
        self.counter = collections.defaultdict(int)

    def count(self, data):
        if isinstance(data, np.ndarray):
            for i in range(data.shape[0]):
                self.counter[data[i]] += 1
        else:
            self.counter[data] += 1

    def update(self, output):
        if isinstance(output, torch.Tensor):
            output = output.detach().cpu().numpy().flatten()

        self.count(output)

    def sortcounter(self):
        self.counter = collections.OrderedDict(sorted(self.counter.items()))

    def normalisecounter(self):
        total_sum = max(self.counter.values())
        for key in self.counter.keys():
            self.counter[key] = (self.counter[key]) / total_sum


class GtVsPredPlotter(ResultGatherer):
    def plot_gt_vs_pred(self):
        self.finalise()
        plt.scatter(self.target, self.output, color="g")

        min_val, max_val = min([np.min(self.target), np.min(self.output)]), max(
            [np.max(self.target), np.max(self.output)]
        )

        values = np.linspace(min_val, max_val, 25)
        plt.plot(values, values, "b--", linewidth=5)
        plt.xlabel("GT")
        plt.ylabel("Pred")
        plt.grid(True)

        self.reset()

        return plt.gcf()


class ConfMatrixPlotter(ResultGatherer):
    def __init__(self, conf_matrix_func, figsize=(10, 8), annot=False, save_path=None):
        super().__init__()
        self.conf_matrix_func = conf_matrix_func
        self.figsize = figsize
        self.annot = annot
        self.save_path = save_path

    def plot_confusion_matrix(self):
        self.finalise()
        figure = make_confusion_matrix_plot(
            self.output,
            self.target,
            self.conf_matrix_func,
            figsize=self.figsize,
            annot=self.annot,
            save_path=self.save_path,
        )
        self.reset()
        return figure


def make_confusion_matrix_plot(
    output, target, conf_matrix_func, figsize=(20, 17), annot=True, save_path=None
):
    output = output.astype(np.int32)
    target = target.astype(np.int32)

    output = torch.from_numpy(output)
    target = torch.from_numpy(target)

    min_val, max_val = min(torch.min(output), torch.min(target)), max(
        torch.max(output), torch.max(target)
    )
    # for conf matrix func, target has to be non-negative
    output = output - min_val
    target = target - min_val
    conf_matrix = np.round(conf_matrix_func(output, target).numpy() * 100)

    vals = [x for x in range(min_val, max_val + 1, 1)]
    # print(vals)

    # following based on: https://stackoverflow.com/a/42265865/798093
    df_cm = pd.DataFrame(conf_matrix, vals, vals)
    plt.figure(figsize=figsize)
    plt.ticklabel_format(useOffset=False)

    sns.set(font_scale=2.5)  # for label size
    sns.heatmap(
        df_cm, annot=annot, annot_kws={"size": 24}, fmt="g", vmin=0, vmax=100
    )  # font size
    fontsize = 24
    plt.xticks(fontsize=fontsize, rotation=45)
    plt.yticks(fontsize=fontsize, rotation=45)
    plt.xlabel("Pred", fontsize=fontsize + 10, rotation=45)
    plt.ylabel("GT", fontsize=fontsize + 10, rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        return plt.gcf()
    else:
        return plt.gcf()
