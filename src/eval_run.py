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

run_id = "finetuned"
# run_id = "pretrained"

run_path = os.path.join(run_dir, task, run_id)
recompute_metrics = False

if recompute_metrics:
    get_run_metrics(run_path)  # these are normally precomputed at the end of training

model, conf = get_model_from_run(run_path, step=10000)
# model, conf = get_model_from_run(run_path)

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
                for j in range(4):
                    scale_exp_list.append({'scale': i + 0.25 * j})
    # print(scale_exp_list)
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

    if save_figs and suffix == '':
        print('ERROR: No save file suffix specified, so cannot save. Exiting...')
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
    correct_norms = []

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
        # Shape: 64 (example docs) by 41 (each example doc's x position input's output, since 40 IC examples per doc)
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

        pred_norms = np.abs(pred_np[:, -1:].mean())
        output_norms.append(pred_norms)
        correct_norms.append(np.abs(ys_np[:, -1:].mean()))

        loss_term = loss.mean(axis=0)[-1:]
        # print(loss.mean(axis=0).shape)
        print(loss_term)
        loss_term = loss_term/((scale**2)*20)
        print(loss_term)
        scale_wise_test_loss.append(loss_term)

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

    if plot_scale_vs_sided_err:
        plt.plot(index_list, abs_sided_err)
        plt.title(title_stub + 'vs. absolute, sided error')
        plt.xlabel('Scaling factor')
        plt.ylabel('Absolute total sided error')
        if save_figs:
            plt.savefig('scale_vs_sided_err' + suffix + '.png')
        plt.show()

    if plot_scale_vs_norms:
        # plt.plot(index_list, output_norms)
        # plt.plot(index_list, correct_norms)
        # plt.title(title_stub + 'vs. prediction norms')
        # plt.xlabel('Scaling factor')
        # plt.ylabel('Mean prediction norm')

        fig, ax = plt.subplots()
        ax.plot(index_list, output_norms, label='Predicted output')
        ax.plot(index_list, correct_norms, label='Ground truth output')
        ax.set_title(title_stub + 'vs. prediction norms')
        ax.set_xlabel('Scaling factor (query is scaled ones vector)')
        ax.set_ylabel('Mean prediction norm')
        ax.legend()

        if save_figs:
            plt.savefig('scale_vs_norm' + suffix + '.png')
        plt.show()

    print("Output norms: ", output_norms)


def run_function_sweep(max_scale, scale_examples=False, save_fig=False, suffix=None):
    data_sampler = get_data_sampler(conf.training.data, n_dims)

    if save_fig and suffix is None:
        print('ERROR: want to save fig, but no suffix specified. exiting...')
        exit()

    task_sampler = get_task_sampler(
        conf.training.task,
        n_dims,
        1)

    task = task_sampler()

    xs = data_sampler.sample_xs(b_size=1, n_points=41)
    print('XS', xs.numpy().shape)
    pred = []
    actual = []

    for i in range(max_scale):
        test_xs = xs
        if scale_examples:
            test_xs *= ((i+1)/4)
        test_xs[:, -1:, :] = torch.ones(20)*((i+1)/4)

        # test_xs *= 0

        all_ys = task.evaluate(test_xs)
        # print(xs[-2:])
        with torch.no_grad():
            all_pred = model(test_xs, all_ys)

        # print('test XS', test_xs.numpy().shape, test_xs)
        # print('PRED ', all_pred.numpy().shape, all_pred)
        # print('YS ', all_ys.numpy().shape, all_ys)

        pred.append(all_pred[:, -1:])
        actual.append(all_ys[:, -1:])

    simple_index = [((i+1)/4) for i in range(max_scale)]
    fig, ax = plt.subplots()
    ax.plot(simple_index, actual, label='Ground truth output')
    ax.plot(simple_index, pred, label='Predicted output')
    if scale_examples:
        ax.set_title('Function Sweep: fixed IC examples and function vector; scaled examples and query)')
        ax.set_xlabel('Scaling factor (query is scaled ones vector)')
        suffix += '_scaled_examples'
    else:
        ax.set_title('Function sweep: fixed ID IC examples and function vector; scaled query)')
        ax.set_xlabel('Query scaling factor (scaled ones vector)')

    ax.set_ylabel('Output value')
    ax.legend()

    if save_fig:
        plt.savefig('function_sweep_IC_scaled'+suffix)

    plt.show()


scale_exps = create_scale_exps([0.1, 0.5, '1:20'])
# scale_exps = create_scale_exps(['1:20'])
# test_input = [{'scale': 10, 'bias': 15}]

run_function_sweep(80, save_fig=True, suffix='_ft_high_res')
run_function_sweep(80, scale_examples=True, save_fig=True, suffix='_ft_high_res')

# Run scaling experiments
run_scale_exp(scale_exps, save_figs=True, suffix='_ft_high_res')
run_scale_exp(scale_exps, scale_only_query=True, save_figs=True, suffix='_ft_high_res')
# run_scale_exp(scale_exps, save_figs=True, suffix='_normed')
# run_scale_exp(scale_exps, scale_only_query=True, save_figs=True)
# run_scale_exp(scale_exps, scale_inputs=False, scale_functions=True, save_figs=True)


# Plot losses from training
# ft_5k_10m_3v_losses = np.load('loss_series_each_step_5k_copy.npy')
# ft_10k_losses = np.load('ft_losses_10k_10_1_copy.npy')
# index_5k = [i+1 for i in range(10067)]
# plt.plot(index_5k, ft_10k_losses)
# plt.show()





# LOSS TESTING
#
# dummy_pred = np.ones(20) * 0.5
# for i in range(20):
#     # print((np.zeros(64) - np.ones(64)*(i+1)).mean(axis=0))
#     dummy_loss = (np.zeros(64) - np.ones(64)*(i+1)).mean(axis=0) / ((i+1))
#     print(dummy_loss)





