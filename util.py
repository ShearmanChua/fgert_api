import os

import torch
from hydra.utils import to_absolute_path

def config_to_abs_paths(config, *parameter_names):
    for param_name in parameter_names:
        param = getattr(config, param_name)
        if param is not None and param.startswith('./'):
            setattr(config, param_name, to_absolute_path(param))
