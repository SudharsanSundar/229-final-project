import uuid
from collections import OrderedDict
import re
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import yaml
from tqdm.notebook import tqdm
import numpy as np

from eval import get_run_metrics, read_run_dir, get_model_from_run
from plot_utils import basic_plot, collect_results, relevant_model_names

from samplers import get_data_sampler
from tasks import get_task_sampler

from train import train

# Load in pretrained model and finetuning config
run_dir = "../models"

df = read_run_dir(run_dir)
task = "linear_regression"
pt_id = "pretrained"
pt_path = os.path.join(run_dir, task, pt_id)
ft_id = "finetuned"
ft_path = os.path.join(run_dir, task, ft_id)

model, _ = get_model_from_run(pt_path)
_, conf_ft = get_model_from_run(ft_path, only_conf=True)

print("Model: ", pt_path)
print("FT config: ")
for elem in conf_ft:
    print(elem)


# Prepare vars and paths for training
n_dims = conf_ft.model.n_dims
batch_size = conf_ft.training.batch_size

run_id = conf_ft.training.resume_id
if run_id is None:
    run_id = str(uuid.uuid4())

out_dir = os.path.join(conf_ft.out_dir, run_id)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
conf_ft.out_dir = out_dir

with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
    yaml.dump(conf_ft.__dict__, yaml_file, default_flow_style=False)


# Conduct training
model.cuda()
model.train()
train(model, conf_ft, shift=conf_ft.training.input_kwargs)


# Evaluate model on basic metrics
if not conf_ft.test_run:
    _ = get_run_metrics(conf_ft.out_dir)  # precompute metrics for eval

