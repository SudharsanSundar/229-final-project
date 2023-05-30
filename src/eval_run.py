from collections import OrderedDict
import re
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm.notebook import tqdm

from eval import get_run_metrics, read_run_dir, get_model_from_run
from plot_utils import basic_plot, collect_results, relevant_model_names

# CHANGED ::
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

# sns.set_theme('notebook', 'darkgrid')
# palette = sns.color_palette('colorblind')
# ::

run_dir = "../models"

df = read_run_dir(run_dir)
print(df)


