import copy
import math
import os
from collections import OrderedDict
from typing import Any, Dict, List

import pandas as pd
from src.config import GeneralConfig, ModelConfig

col_transforms: Dict[str, Any] = {
    "Runtime": lambda x: x / 1000,  # ms to s
}

separators: Dict[str, str] = {".tsv": "\t", ".csv": ","}

group_types: Dict[str, List[str]] = {
    "machine+job+data+char": ["machine_type", "job_args", "data_size_MB", "data_characteristics"],
    "machine+job+data": ["machine_type", "job_args", "data_size_MB"],
    "machine+job": ["machine_type", "job_args"],
    "machine": ["machine_type"]
}

header_transforms: Dict[str, str] = {"Nodes": "instance_count", "Runtime": "gross_runtime"}


def transform_c3o_df(df: pd.DataFrame, workload: str, group_type: str) -> pd.DataFrame:
    """Prepares a C3O-DataFrame.

    Parameters
    ----------
    df : DataFrame
        The raw DataFrame
    workload: str
        Name of the algorithm

    Returns
    ------
    DataFrame
        the adapted DataFrame.
    """

    target_cols: List[str] = group_types.get(group_type)

    df.loc[:, 'job_type'] = df['machine_type'].map(lambda mt: f"{workload}-spark")

    df.loc[:, 'environment'] = df['machine_type'].map(lambda mt: "public cloud aws")

    if workload == "grep":
        df.loc[:, 'job_args'] = df['machine_type'].map(
            lambda mt: "computer")  # parameters: "computer" (filter-word)
        df.loc[:, 'data_characteristics'] = df[['line_length', 'lines', 'p_occurrence']].apply(
            lambda row: ' '.join([str(el) for el in row.to_list()]), axis=1)
    elif workload == "kmeans":
        df.loc[:, 'job_args'] = df['k'].map(
            lambda k: f"{k} 0.001")  # parameters: values for k + fixed convergence criterion
        df.loc[:, 'data_characteristics'] = df[['features', 'observations']].apply(
            lambda row: ' '.join([str(el) for el in row.to_list()]), axis=1)
    elif workload == "pagerank":
        df.loc[:, 'job_args'] = df['convergence_criterion'].map(
            lambda cc: cc)  # parameters: values for convergence criterion
        df.loc[:, 'data_characteristics'] = df[['pages', 'links']].apply(
            lambda row: ' '.join([str(el) for el in row.to_list()]), axis=1)
    elif workload == "sgd":
        df.loc[:,
               'job_args'] = df['iterations'].map(lambda i: i)  # parameters: values for iteration
        df.loc[:, 'data_characteristics'] = df[['features', 'observations']].apply(
            lambda row: ' '.join([str(el) for el in row.to_list()]), axis=1)
    elif workload == "sort":
        df.loc[:, 'job_args'] = df['machine_type'].map(lambda mt: "")  # parameters: None
        df.loc[:, 'data_characteristics'] = df[['line_length', 'lines']].apply(
            lambda row: ' '.join([str(el) for el in row.to_list()]), axis=1)

    df.loc[:, 'group_key'] = df[target_cols].apply(
        lambda row: ' '.join([str(el) for el in row.to_list()]), axis=1)

    return df


def transform_scout_df(df: pd.DataFrame, workload: str, group_type: str) -> pd.DataFrame:
    """Prepares a SCOUT-DataFrame.

    Parameters
    ----------
    df : DataFrame
        The raw DataFrame
    workload: str
        Name of the algorithm

    Returns
    ------
    DataFrame
        the adapted DataFrame.
    """
    target_cols: List[str] = group_types.get(group_type)
    spark_version = "2.1"
    df.loc[:, 'environment'] = df['machine_type'].map(lambda mt: "public cloud aws")

    if workload == "bayes":
        spark_version = "1.5"
        df.loc[:, 'job_args'] = df['machine_type'].map(lambda mt: "")  # parameters: None
        df.loc[:, 'data_characteristics'] = df[[
            'use_dense', 'examples', 'features', 'pages', 'classes'
        ]].apply(lambda row: ' '.join([str(el) for el in row.to_list()]), axis=1)
    elif workload == "join":
        df.loc[:, 'job_args'] = df['machine_type'].map(lambda mt: "")  # parameters: None
        df.loc[:, 'data_characteristics'] = df[['pages', 'uservisits']].apply(
            lambda row: ' '.join([str(el) for el in row.to_list()]), axis=1)
    elif workload == "kmeans":
        spark_version = "1.5"
        df.loc[:, 'job_args'] = df[['max_iteration', 'k']].apply(
            lambda row: f"{row['max_iteration']} {row['k']} 0.0",
            axis=1)  # parameters: values for max_iteration, k + fixed convergence criterion
        df.loc[:, 'data_characteristics'] = df[[
            'num_of_clusters', 'dimensions', 'num_of_samples', 'samples_per_inputfile'
        ]].apply(lambda row: ' '.join([str(el) for el in row.to_list()]), axis=1)
    elif workload == "lr":
        df.loc[:, 'job_args'] = df['machine_type'].map(lambda mt: "")  # parameters: None
        df.loc[:, 'data_characteristics'] = df[['features', 'examples']].apply(
            lambda row: ' '.join([str(el) for el in row.to_list()]), axis=1)
    elif workload == "pagerank":
        df.loc[:, 'job_args'] = df['convergence_criterion'].map(
            lambda cc: cc)  # parameters: values for convergence criterion
        df.loc[:, 'data_characteristics'] = df[['pages']].apply(
            lambda row: ' '.join([str(el) for el in row.to_list()]), axis=1)
    elif workload == "regression":
        spark_version = "1.5"
        df.loc[:, 'job_args'] = df['machine_type'].map(lambda mt: "")  # parameters: None
        df.loc[:, 'data_characteristics'] = df[['examples', 'features']].apply(
            lambda row: ' '.join([str(el) for el in row.to_list()]), axis=1)

    df.loc[:, 'job_type'] = df['machine_type'].map(lambda mt: f"{workload}-spark-{spark_version}")

    df.loc[:, 'group_key'] = df[target_cols].apply(
        lambda row: ' '.join([str(el) for el in row.to_list()]), axis=1)

    return df


df_transforms: Dict[str, Any] = {
    "c3o": transform_c3o_df,
    "scout": transform_scout_df,
}


def load_data(root_dir: str, sub_dir: str, group_type: str = "machine+job+data+char", **kwargs):
    """Loads data from disk.

    Parameters
    ----------
    root_dir : str
        The root directory of the data
    training_target: str
        What data to incorporate

    Returns
    ------
    dict
        A dict of DataFrame's.
    """
    path_to_dir: str = os.path.join(root_dir, sub_dir)
    if GeneralConfig.debug:
        path_to_dir: str = os.path.join(GeneralConfig.debug_data_folder, sub_dir)
    res_dict: OrderedDict = OrderedDict()
    for el in sorted(os.listdir(path_to_dir)):
        if ".csv" in el or ".tsv" in el:
            workload: str = ".".join(el.split(".")[:-1])
            ext: str = os.path.splitext(el)[1]
            sep: str = separators.get(ext)

            # load data
            df: pd.DataFrame = pd.read_csv(os.path.join(path_to_dir, el), sep=sep, **kwargs)
            df = df.drop(columns=ModelConfig.excluded_benchmark_scores, axis=1)

            # apply simple column transformations
            for col in df.columns:
                df[col] = df[col].apply(col_transforms.get(col, lambda x: x))

            # rename columns if needed
            df.columns = [header_transforms.get(col, col) for col in list(df.columns)]

            # further datatrame transformations if needed
            df = df_transforms.get(sub_dir, lambda x, y: x)(df, workload, group_type)

            for idx, name in enumerate(df.group_key.unique()):
                infra: str = sub_dir
                job: str = el.replace(f".{ext}", "")

                sub_df: pd.DataFrame = copy.deepcopy(df.loc[df.group_key == name, :])

                # reset index
                sub_df = sub_df.reset_index(drop=True)

                tol: float = 0.000001
                sub_df.loc[:, 'instance_count_log'] = sub_df['instance_count'].map(
                    lambda ic: math.log(float(ic)))
                sub_df.loc[:, 'instance_count_div'] = sub_df['instance_count'].map(
                    lambda ic: 1. / (float(ic) + tol))

                sub_df.loc[:, 'type'] = sub_df['machine_type'].map(lambda mt: 0)

                res_dict[f"{infra}_{job}_{sub_df.machine_type.values[0]}_No.{idx + 1:02d}"] = sub_df

    return res_dict
