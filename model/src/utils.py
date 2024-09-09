import logging
import os
from importlib import reload
from typing import Any, Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def create_dirs(path: str):
    """Create a directory, recursively if needed.

    Parameters
    ----------
    path: str
        A path that needs to be created
    """
    try:
        os.makedirs(path)
    except:
        pass


def init_logging(logging_level: str):
    """Initialize logging with specific settings.

    Parameters
    ----------
    logging_level: str
        The desired logging level
    """
    reload(logging)
    logging.basicConfig(
        level=getattr(logging, logging_level.upper()),
        format=
        "%(asctime)s [%(levelname)s] %(module)s.%(funcName)s](%(name)s)__[L%(lineno)d] %(message)s",
        handlers=[logging.StreamHandler()])

    logging.info(f"Successfully initialized logging.")


def combine_metrics(path: str, num_epochs: int) -> pd.DataFrame:
    """Combine train/val loss metrics from lightning logs. 
    """
    df: pd.DataFrame = pd.read_csv(path)
    new_index: List[int] = [i + 1 for i in range(num_epochs)]
    epochs: Union[List[Any], np.ndarray] = df["epoch"].drop_duplicates().values
    train_loss: Union[List[Any], np.ndarray] = df["train_loss"].dropna().values
    val_loss: Union[List[Any], np.ndarray] = df["val_loss"].dropna().values
    result: pd.DataFrame = pd.DataFrame(
        {
            "epoch": epochs,
            "train_loss": train_loss,
            "val_loss": val_loss
        }, index=new_index)
    return result


def plot_losses(df: pd.DataFrame, job_type: str, output_file: str):
    """Plot loss comparision for train/val loss during model pretraining.
    """
    # plot train_loss and val_loss
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Train and Validation Losses Over Epochs ({job_type})")
    plt.legend()
    plt.savefig(output_file)
    plt.close()


def parse_configs(filename: str, config_list: List[Any]):
    """Parse given list of configs to yaml file.
    """
    # convert the class attributes to a dictionary
    config_dict: Dict[str, Any] = {}
    for config in config_list:
        for key, value in config.__annotations__.items():
            try:
                config_dict[key] = getattr(config, key)
            except KeyError:
                continue

    # write the dictionary to a YAML file
    with open(filename, 'w') as file:
        yaml.dump(config_dict, file)
