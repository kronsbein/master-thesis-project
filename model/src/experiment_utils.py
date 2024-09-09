import copy
import logging
import os
import random
from collections import OrderedDict
from itertools import product
from typing import Any, Dict, List, Set

import dill
import numpy as np
import pandas as pd
from src.preparation import load_data
from src.splitting import RepeatedRandomSubsampleInterpolationSplits
from src.utils import create_dirs


def filtered_cond_scout(target_df, other_df):
    if target_df["machine_type"].values[0] != other_df["machine_type"].values[0]:
        if target_df["data_characteristics"].values[0] != other_df["data_characteristics"].values[
                0]:

            target_size = target_df["data_size_MB"].values[0]
            other_size = other_df["data_size_MB"].values[0]

            if target_size != other_size:
                # bayes, join, lr, regression, pagerank have no / single job argument
                if any([
                        el in target_df["job_type"].values[0]
                        for el in ["bayes", "join", "lr", "regression", "pagerank"]
                ]):
                    return True
                else:
                    return (target_df["job_args"].values[0] != other_df["job_args"].values[0])

    return False


def filtered_cond_c3o(target_df, other_df):
    if target_df["machine_type"].values[0] != other_df["machine_type"].values[0]:
        if target_df["data_characteristics"].values[0] != other_df["data_characteristics"].values[
                0]:

            target_size = target_df["data_size_MB"].values[0]
            other_size = other_df["data_size_MB"].values[0]

            if not (target_size * 0.8 <= other_size and other_size <= target_size * 1.2):
                # sort and grep have no / single job argument
                if any([el in target_df["job_type"].values[0] for el in ["sort", "grep"]]):
                    return True
                else:
                    return (target_df["job_args"].values[0] != other_df["job_args"].values[0])

    return False


def alter_data_dict(path_to_dir: str, used_setup: Dict[str, Any], orig_data_dict: dict,
                    models: list, data_type: str):
    new_mod_data_dict = {}
    blacklist_keys = set()
    n_iter = used_setup["n_iter"]
    n_train_sizes = used_setup["n_train_sizes"]

    for (k1, m) in product(list(orig_data_dict.keys()), list(models)):

        the_data = orig_data_dict[k1]

        target_data = copy.deepcopy(the_data[0])
        target_data = target_data.sort_values('instance_count')
        target_data = target_data.reset_index(drop=True)

        support_data = [copy.deepcopy(d) for d in the_data[1:]]

        # get splits for target data
        split_dict: Dict[str, Any] = get_split_dict(target_data, n_iter, n_train_sizes)

        for k2 in list(split_dict.keys()):
            new_key = f"{k1}___{k2}___{m}"
            new_support_data = []
            if m in ["TorchModel-s"]:
                new_support_data = []  # no support data
            elif m == "TorchModel-p":  # use all similar executions
                new_support_data = support_data
            elif m == "TorchModel-f":  # use all similar executions, except too similar ones
                appended: bool = False
                for sd in support_data:
                    if data_type == "c3o":
                        if filtered_cond_c3o(target_data, sd):
                            new_support_data.append(sd)
                            appended = True
                    elif data_type == "scout":
                        if filtered_cond_scout(target_data, sd):
                            new_support_data.append(sd)
                            appended = True
                    else:
                        # no default
                        pass
                if not appended:
                    blacklist_keys.add(k1)

            if (m in ["TorchModel-p", "TorchModel-f"] and
                    len(new_support_data)) or (m in ["TorchModel-s"]):
                new_mod_data_dict[new_key] = {
                    "model": m,
                    "n_train": k2,
                    "splits": split_dict[k2],
                    "data": [target_data] + new_support_data,
                    "pretrain_type": f"{k1}___{m}" if "TorchModel" in m else None
                }

            # save split_dict
            path_to_splits: str = os.path.join(path_to_dir, "splits/")
            create_dirs(path_to_splits)
            path_to_split_dict: str = os.path.join(path_to_splits, f"{new_key}_split_dict.pkl")
            if not os.path.exists(path_to_split_dict):
                with open(path_to_split_dict, "wb") as dill_file:
                    dill.dump(split_dict, dill_file)

    blacklist_keys = list(blacklist_keys)

    new_orig_data_dict = {k: v for k, v in orig_data_dict.items() if k not in blacklist_keys}
    new_mod_data_dict = {
        k: v for k, v in new_mod_data_dict.items() if not any([bk in k for bk in blacklist_keys])
    }

    return new_orig_data_dict, new_mod_data_dict


def get_data_dict(data_path: str,
                  data_type: str,
                  algorithms: list,
                  group_type="machine+job+data+char"):

    logging.info(f"Used group-type: {group_type}")

    data_dict = load_data(data_path, data_type, group_type=group_type)

    orig_data_dict = copy.deepcopy({
        k: ([v]) for k, v in data_dict.items() if any([f"_{el}" in k for el in algorithms])
    })

    all_keys = list(orig_data_dict.keys())

    for k in all_keys:
        # now: add this df also to other lists
        datatype_and_job = k.split(".tsv")[0]
        for kk in all_keys:
            if datatype_and_job in kk and kk != k:
                orig_data_dict[kk].append(orig_data_dict[k][0])

    return orig_data_dict


def get_split_dict(
    target_data: List[pd.DataFrame],
    n_iter: int,
    n_train_sizes: List[int],
) -> Dict[str, Any]:

    split_dict = None
    scale_outs: List[int] = list(target_data["instance_count"].values)
    dummy_runtime = np.arange(0, len(scale_outs))
    dummy_df = pd.DataFrame.from_dict({
        "instance_count": scale_outs,
        "gross_runtime": dummy_runtime
    })
    # scout: define n_train_sizes based on unique scale_outs
    if not n_train_sizes:
        n_train_sizes = [i for i in range(0, len(list(set(scale_outs))), 1)]
    split_dict = {k: (get_splits(dummy_df, k, n_iter)) for k in n_train_sizes}

    return split_dict


def subsample_data_dict(path_to_dir: str, orig_data_dict: dict, mod_data_dict: dict,
                        algorithms: list, num_samples: int):

    random.seed(42)
    new_keys: list = []

    for algo in algorithms:
        temp_keys: list = []
        rest_keys: list = []

        poss_keys = sorted([k for k in sorted(list(orig_data_dict.keys())) if f"_{algo}" in k
                           ])  # all keys for this algorithm
        unique_machines = list(
            set([df.machine_type.values[0] for df in orig_data_dict[poss_keys[0]]
                ]))  # unique machines for this algorithm

        for um in sorted(unique_machines):
            m_keys = [el for el in poss_keys if f"_{um}_" in el]
            random_key = random.choice(m_keys)
            temp_keys.append(random_key)
            rest_keys += [el for el in m_keys if el != random_key]

        new_keys += temp_keys
        random.shuffle(rest_keys)
        new_keys += random.sample(rest_keys, int(num_samples - len(temp_keys)))

    new_keys = sorted(list(set(new_keys)))

    # delete splits from splits directory
    splits_dir: str = os.path.join(path_to_dir, "splits/")
    clean_splits_dir(splits_dir, files_to_keep=new_keys)

    new_orig_data_dict = OrderedDict(
        sorted({
            k: v for k, v in orig_data_dict.items() if k in new_keys
        }.items()))
    new_mod_data_dict = OrderedDict(
        sorted({
            k: v for k, v in mod_data_dict.items() if any([new_k in k for new_k in new_keys])
        }.items()))

    return new_orig_data_dict, new_mod_data_dict


def get_splits(df: pd.DataFrame, n_train: int, n_iter: int):
    split_set = set()
    new_splits = []
    splits = RepeatedRandomSubsampleInterpolationSplits(df, n_train, n_iter * 10)
    while len(new_splits) != n_iter:
        # n_train < 1
        if n_train == 0 and len(new_splits) == len(df):
            break
        try:
            split = next(splits)
        except StopIteration:
            # no next
            break
        set_key: str = str([np.sort(indices) for indices in split])
        if set_key not in split_set:
            split_set.add(set_key)
            new_splits.append(split)

    return new_splits


def clean_splits_dir(directory: str, files_to_keep: List[str]) -> None:
    """Function to delete unused splits after subsampling data dict
    """
    files_in_directory: List[str] = os.listdir(directory)
    substrings_to_keep: Set[str] = set(files_to_keep)

    for filename in files_in_directory:
        splitted_filename: str = filename.split("___")[0]
        if splitted_filename not in substrings_to_keep:
            file_path: str = os.path.join(directory, filename)
            os.remove(file_path)
            logging.info(f"Deleted {file_path}")
