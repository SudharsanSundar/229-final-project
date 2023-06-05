from collections import OrderedDict
import re
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm.notebook import tqdm
import numpy as np

from eval import get_run_metrics, read_run_dir, get_model_from_run
from plot_utils import basic_plot, collect_results, relevant_model_names

from samplers import get_data_sampler
from tasks import get_task_sampler


run_dir = "../models"

df = read_run_dir(run_dir)
print("Models available:\n", df)

task = "linear_regression"

run_id = "pretrained"  # if you train more models, replace with the run_id from the table above

run_path = os.path.join(run_dir, task, run_id)
recompute_metrics = False

if recompute_metrics:
    get_run_metrics(run_path)  # these are normally precomputed at the end of training

model, conf = get_model_from_run(run_path)

n_dims = conf.model.n_dims
batch_size = conf.training.batch_size

print("Model: ", run_path, "\nTask: linear regression\nTask dim: ", n_dims, "\nBatch size: ", batch_size, "\n")


# Automates and simplifies creating scale experiment inputs
def create_scale_exps(scales):
    scale_exp_list = []
    for item in scales:
        if type(item) == int or type(item) == float:
            scale_exp_list.append({'scale': item})
        else:
            start = int(item.split(':')[0])
            end = int(item.split(':')[1])
            for i in range(start, end, 1):
                scale_exp_list.append({'scale': i})
    print(scale_exp_list)
    return scale_exp_list


# Tests loaded model in test settings with different input scaling. Plots and visualizes results
def run_scale_exp(scale_exp_list,
                  scale_inputs=True,
                  scale_functions=False,
                  scale_only_query=False,
                  plot_ind_exp=False,
                  plot_scale_vs_loss=True,
                  plot_scale_vs_sided_err=True,
                  plot_scale_vs_norms=True,
                  save_figs=False,
                  suffix=''):

    if scale_only_query and not scale_inputs:
        print('ERROR: Indicated wanted to scale query but not to scale inputs. Exiting...')
        exit

    if scale_only_query:
        title_stub = 'Input scaling (query only) '
    elif scale_functions:
        title_stub = 'Function vector scaling '
    else:
        title_stub = 'Input scaling '

    scale_wise_test_loss = []
    abs_sided_err = []
    output_norms = []

    for scale_exp in scale_exp_list:
        scale = scale_exp['scale']
        print("\nTest setting: scale = ", scale)

        # Determines how input vectors are sampled
        if scale_inputs:
            data_sampler = get_data_sampler(conf.training.data, n_dims, **scale_exp)
        else:
            data_sampler = get_data_sampler(conf.training.data, n_dims)

        # Determines how function vectors are sampled
        if scale_functions:
            task_sampler = get_task_sampler(
                conf.training.task,
                n_dims,
                batch_size,
                **scale_exp
            )
        else:
            task_sampler = get_task_sampler(
                conf.training.task,
                n_dims,
                batch_size,
            )

        task = task_sampler()
        # b_size is the number of test examples. n_points is max num of IC examples
        if scale_only_query:
            xs = data_sampler.sample_query_scale_xs(b_size=batch_size, n_points=conf.training.curriculum.points.end)
        else:
            xs = data_sampler.sample_xs(b_size=batch_size, n_points=conf.training.curriculum.points.end)
        # Shape: 64 (examples) by 41 (each example's input's output)
        ys = task.evaluate(xs)

        with torch.no_grad():
            pred = model(xs, ys)

        metric = task.get_metric()
        loss = metric(pred, ys).numpy()

        pred_np = pred.numpy()
        ys_np = ys.numpy()
        sum_abs_error = 0

        # Determines if error is an absolute 'undershot' (pos) or 'overshot' (neg)
        for i in range(len(ys_np[:, -1:])):
            if np.abs(ys_np[:, -1:][i]) >= np.abs(pred_np[:, -1:][i]):
                sum_abs_error += np.abs(ys_np[:, -1:][i] - pred_np[:, -1:][i])
            else:
                sum_abs_error -= np.abs(ys_np[:, -1:][i] - pred_np[:, -1:][i])

        abs_sided_err.append(sum_abs_error)
        output_norms.append(np.linalg.norm(pred_np[:, -1:]))
        scale_wise_test_loss.append(loss.mean(axis=0)[-1:])

        sparsity = conf.training.task_kwargs.sparsity if "sparsity" in conf.training.task_kwargs else None
        baseline = {
            "linear_regression": n_dims,
            "sparse_linear_regression": sparsity,
            "relu_2nn_regression": n_dims,
            "decision_tree": 1,
        }[conf.training.task]

        if plot_ind_exp:
            plt.plot(loss.mean(axis=0), lw=2, label="Transformer")
            plt.axhline(baseline*scale, ls="--", color="gray", label="zero estimator")
            plt.xlabel("# in-context examples " + str(scale_exp['scale']))
            plt.ylabel("squared error")
            plt.legend()
            plt.show()

    index_list = [elem['scale'] for elem in scale_exp_list]
    if plot_scale_vs_loss:
        plt.plot(index_list, scale_wise_test_loss)
        plt.title(title_stub + 'vs. Test loss')
        plt.xlabel('Scaling factor')
        plt.ylabel('Test loss (MSE)')
        if save_figs:
            plt.savefig('scale_vs_loss' + suffix + '.png')
        plt.show()


    # simple_index = [i for i in range(len(exp_off_by))]
    if plot_scale_vs_sided_err:
        plt.plot(index_list, abs_sided_err)
        plt.title(title_stub + 'vs. absolute, sided error')
        plt.xlabel('Scaling factor')
        plt.ylabel('Absolute total sided error')
        if save_figs:
            plt.savefig('scale_vs_sided_err' + suffix + '.png')
        plt.show()

    if plot_scale_vs_norms:
        plt.plot(index_list, output_norms)
        plt.title(title_stub + 'vs. prediction norms')
        plt.xlabel('Scaling factor')
        plt.ylabel('Mean prediction norm')
        if save_figs:
            plt.savefig('scale_vs_norm' + suffix + '.png')
        plt.show()

    print("Output norms: ", output_norms)


scale_exps = create_scale_exps([0.0001, 0.001, 0.01, 0.1, '1:20'])
test_input = [{'scale': 10, 'bias': 15}]
run_scale_exp(test_input)
# run_scale_exp(scale_exps, scale_only_query=True)
# run_scale_exp(scale_exps, scale_inputs=False, scale_functions=True)



