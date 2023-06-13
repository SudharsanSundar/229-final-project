from collections import OrderedDict
import re
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm.notebook import tqdm
import numpy as np

from eval import get_run_metrics, read_run_dir, get_model_from_run, eval_model
from plot_utils import basic_plot, collect_results, relevant_model_names

from samplers import get_data_sampler
from tasks import get_task_sampler


# Automates and simplifies creating scale experiment inputs
def create_scale_exps_hires(scales):
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


def create_scale_exps(scales):
    scale_exp_list = []
    for item in scales:
        if type(item) == int or type(item) == float:
            scale_exp_list.append({'scale': item})
        else:
            start = int(item.split(':')[0])
            end = int(item.split(':')[1])
            for i in range(start, end, 10):
                scale_exp_list.append({'scale': i})
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
                  suffix=None):

    if scale_only_query and not scale_inputs:
        print('ERROR: Indicated wanted to scale query but not to scale inputs. Exiting...')
        exit

    if save_figs and suffix is None:
        print('ERROR: No save file suffix specified, so cannot save. Exiting...')
        exit

    if scale_only_query:
        title_stub = 'Input scaling (query only) '
    elif scale_functions:
        title_stub = 'Function vector scaling '
    else:
        title_stub = 'input scaling '

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
        loss_term = loss_term/((scale**2)*n_dims)
        print(loss_term)

        naive_loss = (np.zeros([64, 1]) - ys_np[:, -1:])**2
        naive_loss = naive_loss.mean()
        naive_loss /= ((scale**2)*n_dims)
        print('0 guess loss, should be 1: ', naive_loss)

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
            plt.axhline(1, ls="--", color="gray", label="zero estimator")
            plt.xlabel("# in-context examples " + str(scale_exp['scale']))
            plt.ylabel("squared error")
            plt.legend()
            plt.show()

    index_list = [elem['scale'] for elem in scale_exp_list]

    if suffix is not None:
        if scale_only_query:
            suffix += '_scaled_query'
        elif scale_functions:
            raise NotImplementedError
        else:
            suffix += '_scaled_all'

    if plot_scale_vs_loss:
        fig, ax = plt.subplots()
        ax.plot(index_list, scale_wise_test_loss)
        plt.axhline(1, ls="--", color="gray", label="zero estimator")
        # ax.set_title(title_stub + 'vs. test loss')
        ax.set_xlabel('scaling factor')
        ax.set_ylabel('loss (MSE/scale^2)')
        # ax.legend()
        ax.set_ylim(-0.1, 1.25)

        if save_figs:
            plt.savefig('scale_vs_loss_' + suffix + '.png')
        plt.show()

    if plot_scale_vs_sided_err:
        plt.plot(index_list, abs_sided_err)
        # plt.title(title_stub + 'vs. absolute, sided error')
        plt.xlabel('scaling factor')
        plt.ylabel('absolute total sided error')
        if save_figs:
            plt.savefig('scale_vs_sided_err_' + suffix + '.png')
        plt.show()

    if plot_scale_vs_norms:
        fig, ax = plt.subplots()
        ax.plot(index_list, output_norms, label='Predicted output')
        ax.plot(index_list, correct_norms, label='Ground truth output')
        ax.set_title(title_stub + 'vs. prediction norms')
        ax.set_xlabel('scaling factor')
        # ax.set_ylabel('mean prediction norm')
        # ax.legend()

        if save_figs:
            plt.savefig('scale_vs_norm_' + suffix + '.png')
        plt.show()

    print("Output norms: ", output_norms)


def run_function_sweep(start, max_scale, scale_examples=True, save_fig=False, suffix=None, title_in=None):
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
    pred_scaled_ex = []
    actual = []

    for i in range(start, max_scale*4, 1):
        test_xs = xs.detach()
        if i > 0 and scale_examples:
            test_xs *= (4 / i)
        # print('example initial coord, normal: ', test_xs[0, 0, 0], ((i + 1) / 4))
        test_xs[:, -1:, :] = torch.ones(20) * ((i + 1) / 4)

        all_ys = task.evaluate(test_xs)
        with torch.no_grad():
            all_pred = model(test_xs, all_ys)

        pred.append(all_pred[:, -1:])
        actual.append(all_ys[:, -1:])

        if scale_examples:
            test_xs *= ((i+1)/4)
            # print('example initial coord, scale: ', test_xs[0, 0, 0], ((i+1)/4))

            test_xs[:, -1:, :] = torch.ones(20)*((i+1)/4)

            all_ys_ex = task.evaluate(test_xs)
            with torch.no_grad():
                all_pred_ex = model(test_xs, all_ys_ex)

            # print('test XS', test_xs.numpy().shape, test_xs)
            # print('PRED ', all_pred.numpy().shape, all_pred)
            # print('YS ', all_ys.numpy().shape, all_ys)

            pred_scaled_ex.append(all_pred_ex[:, -1:])

    simple_index = [((i+1)/4) for i in range(start, max_scale*4, 1)]
    fig, ax = plt.subplots()
    ax.plot(simple_index, actual, label='Ground truth output')
    # ax.plot(simple_index, pred, label='Predicted output, query scaling')
    if scale_examples:
        ax.plot(simple_index, pred_scaled_ex, label='Predicted output, query and example scaling')

    # ax.set_title(title_in)
    ax.set_xlabel('scaling factor')

    if suffix is not None:
        if scale_examples:
            suffix += '_scaled_all'
        else:
            suffix += '_scaled_query'

    ax.set_ylabel('output value')
    ax.legend(loc='best')

    if save_fig:
        plt.savefig('function_sweep_'+suffix)

    plt.show()


def run_function_sweep_ext(start, max_scale, scale_examples=True, save_fig=False, suffix=None):
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
    pred_scaled_ex = []
    actual = []

    for i in range(start, max_scale, 1):
        test_xs = xs.detach()
        if i > 0 and scale_examples:
            test_xs *= (1 / i)
        # print('example initial coord, normal: ', test_xs[0, 0, 0], ((i + 1) / 4))
        test_xs[:, -1:, :] = torch.ones(n_dims) * ((i + 1))

        all_ys = task.evaluate(test_xs)
        with torch.no_grad():
            all_pred = model(test_xs, all_ys)

        pred.append(all_pred[:, -1:])
        actual.append(all_ys[:, -1:])

        if scale_examples:
            test_xs *= ((i+1))
            # print('example initial coord, scale: ', test_xs[0, 0, 0], ((i+1)))

            test_xs[:, -1:, :] = torch.ones(n_dims)*((i+1))

            all_ys_ex = task.evaluate(test_xs)
            with torch.no_grad():
                all_pred_ex = model(test_xs, all_ys_ex)

            # print('test XS', test_xs.numpy().shape, test_xs)
            # print('PRED ', all_pred.numpy().shape, all_pred)
            # print('YS ', all_ys.numpy().shape, all_ys)

            pred_scaled_ex.append(all_pred_ex[:, -1:])

    simple_index = [((i+1)) for i in range(start, max_scale, 1)]
    fig, ax = plt.subplots()
    ax.plot(simple_index, actual, label='Ground truth output')
    # ax.plot(simple_index, pred, label='Predicted output, query scaling')
    if scale_examples:
        ax.plot(simple_index, pred_scaled_ex, label='Predicted output, query and example scaling')

    # ax.set_title('Function sweep: fixed examples/function, scaled inputs')
    ax.set_xlabel('Scaling factor')

    if suffix is not None:
        if scale_examples:
            suffix += '_scaled_all'
        else:
            suffix += '_scaled_query'

    ax.set_ylabel('Output value')
    ax.legend(loc='best')

    if save_fig:
        plt.savefig('function_sweep_'+suffix)

    plt.show()


def run_cheat_mode(start, max_scale, scale_examples=False, save_fig=False, suffix=None, cheat_pos=0):
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
    pred_scaled_ex = []
    actual = []

    for i in range(start, max_scale * 4, 1):
        test_xs = xs.detach()
        if i > 0 and scale_examples:
            test_xs *= (4 / i)
        # print('example initial coord, normal: ', test_xs[0, 0, 0], ((i + 1) / 4))
        test_xs[:, -1:, :] = test_xs[:, cheat_pos:cheat_pos+1, :] * ((i + 1) / 4)
        # print(test_xs[:, -4:, :], test_xs[:, -1:, :])

        all_ys = task.evaluate(test_xs)
        with torch.no_grad():
            all_pred = model(test_xs, all_ys)

        pred.append(all_pred[:, -1:])
        actual.append(all_ys[:, -1:])

        if scale_examples:
            test_xs *= ((i + 1) / 4)
            # print('example initial coord, scale: ', test_xs[0, 0, 0], ((i+1)/4))

            test_xs[:, -1:, :] = test_xs[:, cheat_pos:cheat_pos+1, :]
            # print(test_xs[:, :1, :], test_xs[:, -1:, :])

            all_ys_ex = task.evaluate(test_xs)
            with torch.no_grad():
                all_pred_ex = model(test_xs, all_ys_ex)

            # print('test XS', test_xs.numpy().shape, test_xs)
            # print('PRED ', all_pred.numpy().shape, all_pred)
            # print('YS ', all_ys.numpy().shape, all_ys)

            pred_scaled_ex.append(all_pred_ex[:, -1:])

    simple_index = [((i + 1) / 4) for i in range(start, max_scale * 4, 1)]
    fig, ax = plt.subplots()
    ax.plot(simple_index, actual, label='Ground truth output')
    ax.plot(simple_index, pred, label='Predicted output, query scaling')
    if scale_examples:
        ax.plot(simple_index, pred_scaled_ex, label='Predicted output, query and example scaling')

    ax.set_title('Function sweep: fixed examples/function, scaled inputs / CHEAT pos ' + str(cheat_pos))
    ax.set_xlabel('Scaling factor (query is scaled ones vector)')

    if suffix is not None:
        if scale_examples:
            suffix += '_scaled_all'
        else:
            suffix += '_scaled_query'

    ax.set_ylabel('Output value')
    ax.legend(loc='best')

    if save_fig:
        plt.savefig('function_sweep_' + suffix)

    plt.show()


# Model Setup
run_dir = "../models"

df = read_run_dir(run_dir)
print("Models available:\n", df)

task = "linear_regression"

# CHOOSE MODEL:
run_id = "finetuned"
# run_id = "pretrained"

run_path = os.path.join(run_dir, task, run_id)

model_name = 'model_100000.pt'
if model_name is not None:
    model, conf = get_model_from_run(run_path, name=model_name)
else:
    model, conf = get_model_from_run(run_path)

n_dims = conf.model.n_dims
batch_size = conf.training.batch_size

print("Model: ", run_path, "\nTask: linear regression\nTask dim: ", n_dims, "\nBatch size: ", batch_size, "\n")

# RESULTS
scale_exps = create_scale_exps_hires(['1:20'])
scale_exps_ex = create_scale_exps(['1:1000'])

# run_function_sweep(3, 20, scale_examples=True, save_fig=True, suffix='A_final', title_in='Model A function approximation')
# run_function_sweep(3, 20, scale_examples=True, save_fig=True, suffix='B_final')
# run_function_sweep(3, 20, scale_examples=True, save_fig=True, suffix='C_final')
# run_function_sweep(3, 20, scale_examples=True, save_fig=True, suffix='D_final')
run_function_sweep_ext(1, 1000, scale_examples=True, save_fig=True, suffix='E_final')

# run_scale_exp(scale_exps, save_figs=True, suffix='A_final')
# run_scale_exp(scale_exps, save_figs=True, suffix='B_final')
# run_scale_exp(scale_exps, save_figs=True, suffix='C_final')
# run_scale_exp(scale_exps, save_figs=True, suffix='D_final')
run_scale_exp(scale_exps_ex, save_figs=True, suffix='E_final')



# metrics = eval_model(model, 'linear_regression', 'gaussian', 1, 41, 'standard', num_eval_examples=256, data_sampler_kwargs={'bias': 0, 'scale': 100})
# for item in metrics:
#     print(item, metrics[item])

# Run function sweep
# run_function_sweep(0, 20, save_fig=True, suffix='ft_high_res')
# run_function_sweep(0, 20, scale_examples=True, save_fig=True, suffix='20k_high_res')
# for i in range(5):
#     run_function_sweep(0, 20)
#     run_function_sweep(0, 20, scale_examples=True)
#     run_function_sweep_ext(0, 1000, scale_examples=True)
#     run_cheat_mode(0, 20, scale_examples=True, cheat_pos=39)
# run_function_sweep(0, 20)
# run_function_sweep(0, 20, scale_examples=True)



# Run scaling experiments
# scale_exps = create_scale_exps_hires([0.5, '1:20'])
# scale_exps = create_scale_exps(['1:1000'])

# run_scale_exp(scale_exps, save_figs=True, suffix='_200k_t1000_high_res')
# run_scale_exp(scale_exps, scale_only_query=True, save_figs=True, suffix='_ft_high_res')
# run_scale_exp(scale_exps)
# run_scale_exp(scale_exps, scale_only_query=True)


# Plot losses from training
# ft_5k_10m_3v_losses = np.load('loss_series_each_step_5k_copy.npy')
# ft_10k_losses = np.load('ft_losses_10k_10_1_copy.npy') #10067
# ft_10k_5_1_losses = np.load('loss_series_each_step_10k_5_1_copy.npy') #10049
# ft_10k_5_1_cur_losses = np.load('loss_series_each_step_cur.npy') #10092
# ft_10k_2p5_1_losses = np.load('loss_series_each_step_10k_2p5_1.npy') #10024
# ft_10k_0_10_losses = np.load('loss_series_each_step_10k_0_10.npy') #10025
# ft_20k_0_10_losses = np.load('loss_series_each_step_20k_0_10.npy') #10034
# pt_100k_0_100_losses = np.load('loss_series_each_step_100k_0_100.npy')
#
# index_10k = [i+1 for i in range(105914)]
# plt.plot(index_10k, pt_100k_0_100_losses)
# plt.show()





# LOSS TESTING
#
# dummy_pred = np.ones(20) * 0.5
# for i in range(20):
#     # print((np.zeros(64) - np.ones(64)*(i+1)).mean(axis=0))
#     dummy_loss = (np.zeros(64) - np.ones(64)*(i+1)).mean(axis=0) / ((i+1))
#     print(dummy_loss)





