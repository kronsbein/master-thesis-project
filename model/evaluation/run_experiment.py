import argparse
import copy
import glob
import logging
import os
import random
import shutil
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

parent_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONPATH"] = parent_dir + ":" + os.environ.get("PYTHONPATH", "")
sys.path.append(parent_dir)
sys.path.append(os.path.join(os.path.abspath(''), ".."))

from src.config import *
from src.experiment_utils import (alter_data_dict, get_data_dict,
                                  subsample_data_dict)
from src.pipeline import TorchPipeline
from src.pretraining import *
from src.transforms import PairData
from src.utils import create_dirs, init_logging, parse_configs


def learning_curve_func(job_identifier: str = None,
                        model: str = None,
                        splits: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
                        data: List[PairData] = None,
                        pretrain_type: str = None,
                        n_train: str = None,
                        path_to_pretrain_folder: str = None,
                        path_to_dir: str = None) -> str:
    """Save results and call cv score function on given data.

    Parameters
    ----------
    job_identifier: str
        Identifier of the job (pretrain_type)
    model: str
        Name of the model
    splits: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
        List of tuples with computed splits
    data: List[PairData]
        List of PairData objects
    pretrain_type: str
    n_train: str
    path_to_pretrain_folder: str
        Relative path to the pretrain folder
    path_to_dir: str
        Relative path to experiment analysis dir
        
    Returns
    ------
    job_identifier: str
        Identifier of the job (pretrain_type)
    
    """
    logging.info(f"Job: {job_identifier}")
    opt_args: Dict[str, Any] = {
        "model": model,
        "job_identifier": pretrain_type,
        "path_to_pretrain_folder": path_to_pretrain_folder,
        "path_to_finetune_folder": os.path.join(path_to_dir, GeneralConfig.path_to_temp_dir),
        "only_pretraining": False
    }
    path_to_pipeline: str = os.path.join(path_to_dir, f"{job_identifier}_analysis")
    pipeline: TorchPipeline = TorchPipeline.getInstance(path_to_pipeline, model, **opt_args)

    res_dict: Dict[str, Any] = cv_score_p(pipeline, data, splits, opt_args)
    res_dict.update({"model": [model] * len(splits), "n_train": [int(n_train)] * len(splits)})

    df: pd.DataFrame = pd.DataFrame.from_dict(res_dict)
    with open(os.path.join(path_to_pipeline, "results.csv"), 'w') as f:
        df.to_csv(f, header=True, index=False)

    if len(data):
        with open(os.path.join(path_to_pipeline, "my_data.csv"), 'w') as f:
            my_df = data[0]
            my_df.to_csv(f, header=True, index=False)

    if len(data) > 1:
        with open(os.path.join(path_to_pipeline, "rest_data.csv"), 'w') as f:
            rest_df = pd.concat(data[1:], axis=0, ignore_index=True)
            rest_df.to_csv(f, header=True, index=False)

    return job_identifier


def cv_score_p(pipeline: TorchPipeline, df_list: List[pd.DataFrame],
               splits: List[Tuple[np.ndarray, np.ndarray,
                                  np.ndarray]], opt_args: Dict[str, Any]) -> Dict[str, Any]:
    """Function for cv experiment, evaluates finetuning and prediction results.

    Parameters
    ----------
    pipeline: TorchPipeline
        A TorchPipeline
    df_list: List[pd.DataFrame]
        A list of DataFrames
    splits: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
        A list containing splits
    opt_args: Dict[str, Any]
        A dictionary with additional information like model name, job identifier, path to 
        pretraining folder...
        
    Returns
    ------
    response_dict: Dict[str, Any]
        A dictionary containing results from inter-/extrapolation experiments
    """
    static_keys: List[str] = [
        "n_iter", "train_time", "support_samples", "support_configs", "int_true", "int_pred",
        "int_pred_time", "ext_true", "ext_pred", "ext_pred_time"
    ]

    response_dict: Dict[str, Any] = {k: (np.ones(len(splits)) * -1) for k in static_keys}
    response_dict["ft_losses"] = [None] * len(splits)

    emb_columns: List[str] = PipelineConfig.graph_list_dict["emb_columns"]
    for el in emb_columns:
        for prefix in ["int", "ext"]:
            response_dict[f"{prefix}_emb_{el}_raw"] = [None] * len(splits)
            response_dict[f"{prefix}_emb_{el}_code"] = [None] * len(splits)

    opt_columns: List[str] = PipelineConfig.graph_list_dict["opt_columns"]
    for el in opt_columns:
        for prefix in ["int", "ext"]:
            response_dict[f"{prefix}_opt_{el}_raw"] = [None] * len(splits)
            response_dict[f"{prefix}_opt_{el}_code"] = [None] * len(splits)

    for i, (train_indices, test_indices, val_indices) in enumerate(splits):
        response_dict["n_iter"][i] = int(i)
        train_df: List[Any] = []
        test_df: List[Any] = []
        val_df: List[Any] = []
        my_df: pd.DataFrame = df_list[0]
        rest_df_list: List[pd.DataFrame] = df_list[1:]
        new_train_list: List[Any] = []

        if len(train_indices):
            train_df: pd.DataFrame = copy.deepcopy(my_df.iloc[train_indices, :])
            train_df.loc[:, 'type'] = train_df['machine_type'].map(lambda mt: 0)
            new_train_list.append(train_df)

        if len(test_indices):
            test_df: pd.DataFrame = copy.deepcopy(my_df.iloc[test_indices, :])
            test_df.loc[:, 'type'] = test_df['machine_type'].map(lambda mt: 0)

        if len(val_indices):
            val_df: pd.DataFrame = copy.deepcopy(my_df.iloc[val_indices, :])
            val_df.loc[:, 'type'] = val_df['machine_type'].map(lambda mt: 0)

        if len(rest_df_list):
            response_dict["support_configs"][i] = len(rest_df_list)
            rest_df: pd.DataFrame = pd.concat(rest_df_list, axis=0, ignore_index=True)
            response_dict["support_samples"][i] = len(rest_df)

            rest_df.loc[:, 'type'] = rest_df['machine_type'].map(lambda mt: 1)
            new_train_list.append(rest_df)

        if len(new_train_list):
            train_df: pd.DataFrame = pd.concat(new_train_list, axis=0, ignore_index=True)

        if len(train_df) < 1 and opt_args["model"] in [
                "TorchModel-s", "TorchModel-f", "TorchModel-p"
        ]:
            continue
        else:
            if len(train_df):
                sample_list: List[PairData] = pipeline.fit(
                    copy.deepcopy(train_df)).transform(train_df)
                target_scaler: Any = None
                if ModelConfig.target_scaling_enabled:
                    target_scaler = pipeline.get_transformer_by_step(step_name="target_scaler")
                    opt_args["early_stopping"] = {
                        "stopping_threshold": -(target_scaler.transform([[5.0]])[0][0]),
                    }
                # update ohe features
                if ModelConfig.ohe_enabled_machine_type:
                    one_hot_encoder_machine_type = pipeline.get_transformer_by_step(
                        "one_hot_encoder_machine_type")
                    ModelConfig.model_args["ohe_features_in_machine_type"] = len(
                        one_hot_encoder_machine_type.categories_[0])
                if ModelConfig.ohe_enabled_data_size_mb:
                    one_hot_encoder_data_size_MB = pipeline.get_transformer_by_step(
                        "one_hot_encoder_data_size_MB")
                    ModelConfig.model_args["ohe_features_in_data_size_MB"] = len(
                        one_hot_encoder_data_size_MB.categories_[0])
                model_result: Tuple[LightningModel, float] = fit_and_finetune(sample_list, opt_args)
                model: LightningModel = model_result[0]
                fit_time: float = model_result[1]
                response_dict["train_time"][i] = fit_time
                if ModelConfig.target_scaling_enabled:
                    response_dict["ft_losses"][i] = target_scaler.inverse_transform([model.losses])
                else:
                    response_dict["ft_losses"][i] = model.losses

            if len(test_df):
                response_dict.update(
                    cv_predict_helper(test_df, pipeline, model, opt_args, response_dict, i, "int"))

            if len(val_df):
                response_dict.update(
                    cv_predict_helper(val_df, pipeline, model, opt_args, response_dict, i, "ext"))

    return response_dict


def fit_and_finetune(sample_list: List[PairData],
                     opt_args: Dict[str, Any]) -> Tuple[LightningModel, float]:
    """Fitting and fine-tuning for given data.

    Parameters
    ----------
    sample_list: List[PairData]
        List of PairData objects containing train and help datasets
    opt_args: Dict[str, Any]
        A dictionary with additional information like model name, job identifier, path to 
        pretraining folder...
        
    Returns
    ------
    model, fit_time: Tuple[LightningModel, float]
        Fitted model and measured time to fit it
    
    """
    train_dataset: List[PairData] = [el for el in sample_list if el.type.item() == 0]
    help_dataset: List[PairData] = [el for el in sample_list if el.type.item() == 1]

    logging.info(f"train_dataset: {len(train_dataset)}")
    logging.info(f"help_dataset: {len(help_dataset)}")

    job_identifier: str = opt_args["job_identifier"]
    pretrain_folder: str = opt_args["path_to_pretrain_folder"]

    pretrainer: Pretrainer = Pretrainer(job_identifier, pretrain_folder)
    model: LightningModel = pretrainer.get_pretrained_model(help_dataset)

    if opt_args["only_pretraining"]:
        return

    fit_time: float = 0.0
    if len(train_dataset):
        batch_size: int = ModelConfig.batch_size

        ### create data loaders ###
        train_loader: DataLoader = DataLoader(train_dataset,
                                              shuffle=True,
                                              batch_size=batch_size,
                                              follow_batch=ModelConfig.follow_batch)

        # checkpoint callback
        finetune_folder: str = opt_args["path_to_finetune_folder"]
        filename: str = f"{job_identifier}_{datetime.now().strftime('%Y%m%d-%H%M%S')}_noice{random.randint(0, 10000):04d}"
        if os.path.exists(finetune_folder):
            shutil.rmtree(finetune_folder)
        create_dirs(finetune_folder)
        checkpoint = ModelCheckpoint(monitor="ft_loss", dirpath=finetune_folder, filename=filename)

        # early stopping with patience
        stopping_threshold: float = 5.0
        try:
            stopping_threshold = opt_args["early_stopping"]["stopping_threshold"]
        except KeyError:
            pass

        early_stopping = EarlyStopping(monitor="ft_loss",
                                       min_delta=0.001,
                                       stopping_threshold=stopping_threshold,
                                       patience=ModelConfig.early_stopping["patience"])

        ### train model ###
        model.is_finetuning = True
        model.losses = []
        trainer: Trainer = pl.Trainer(accelerator=GeneralConfig.device["device"],
                                      devices=GeneralConfig.device["devices"],
                                      logger=False,
                                      callbacks=[checkpoint, early_stopping],
                                      max_epochs=ModelConfig.epochs[1])
        t_start: float = time.time()
        trainer.fit(model, train_loader)
        t_end: float = time.time()
        fit_time = t_end - t_start

        # load model from checkpoint
        model_path: str = os.path.join(finetune_folder, f"{filename}.ckpt")
        model = LightningModel.load_from_checkpoint(model_path).double()

    return model, fit_time


def cv_predict_helper(df: pd.DataFrame, pipeline: TorchPipeline, model: LightningModel,
                      opt_args: Dict[str, Any], response_dict: Dict[str, Any], idx: int,
                      prefix: str) -> Dict[str, Any]:
    """Helper function for prediction of int/ext experiment in cv_score_p.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame holding data for inference
    pipeline: TorchPipeline
        TorchPipeline object to fit and transform data
    model: LightningModel
        LightningModel object 
    opt_args: Dict[str, Any]
        A dictionary with additional information like model name, job identifier, path to 
        pretraining folder...
    response_dict: Dict[str, Any]
        A dictionary to store results.
    idx: int
        Index from data loop
    prefix: str
        String to determine int/ext experiment
    
    Returns
    ------
    response_dict: Dict[str, Any]
        A dictionary containing results from inter-/extrapolation experiments
    """
    emb_columns: List[str] = PipelineConfig.graph_list_dict["emb_columns"]
    opt_columns: List[str] = PipelineConfig.graph_list_dict["opt_columns"]

    y_true = df["gross_runtime"].values.flatten()
    trainer: Trainer = pl.Trainer(accelerator=GeneralConfig.device["device"],
                                  devices=GeneralConfig.device["devices"],
                                  enable_checkpointing=False,
                                  logger=False)
    pred_list: List[PairData] = pipeline.transform(copy.deepcopy(df))
    pred_loader: DataLoader = DataLoader(pred_list,
                                         batch_size=ModelConfig.batch_size,
                                         follow_batch=ModelConfig.follow_batch)
    t_start: float = time.time()
    y_pred: List[Dict[str, Any]] = trainer.predict(model, pred_loader)
    t_end: float = time.time()
    predict_time: float = t_end - t_start

    if any([opt_args["model"] in m for m in ["TorchModel-s", "TorchModel-f", "TorchModel-p"]]):
        y_pred: torch.Tensor
        emb_codes: torch.Tensor
        opt_codes: torch.Tensor
        y_pred, emb_codes, opt_codes = y_pred[0]["y_pred"], y_pred[0]["emb_codes"], y_pred[0][
            "opt_codes"]

        for col, code in zip(emb_columns, emb_codes.tolist()):
            response_dict[f"{prefix}_emb_{col}_raw"][idx] = df[col].values[0]
            response_dict[f"{prefix}_emb_{col}_code"][idx] = code

        for col, code in zip(opt_columns, opt_codes.tolist()):
            response_dict[f"{prefix}_opt_{col}_raw"][idx] = df[col].values[0]
            response_dict[f"{prefix}_opt_{col}_code"][idx] = code

    response_dict[f"{prefix}_true"][idx] = y_true
    if ModelConfig.target_scaling_enabled:
        target_scaler = pipeline.get_transformer_by_step(step_name="target_scaler")
        response_dict[f"{prefix}_pred"][idx] = target_scaler.inverse_transform(y_pred)
    else:
        response_dict[f"{prefix}_pred"][idx] = y_pred
    response_dict[f"{prefix}_pred_time"][idx] = predict_time

    return response_dict


def run_full_experiment(experiment_name: str,
                        data_type: str,
                        num_configs: int = None,
                        algorithms: List[str] = None,
                        num_samples: int = None,
                        n_iter: int = None):
    """Run full inter-/extrapolation experiment

    Parameters
    ----------
    experiment_name: str
        Name of local experiment
    data_type: str
        Type of the data
    num_configs: int
        Number of configs to investigate
    algorithms: List[str]
        List of scalable analytics algorithms
    num_samples: int
        Number of samples to draw
    n_iter: int
        Number of iterations 
    """
    ##### SETUP ######
    num_samples = num_samples or 9
    n_min: int = 0
    n_max: int = num_samples
    # set fixed train size for c3o
    n_train_sizes: List[int] = None
    if data_type == "c3o":
        num_samples = 7
        n_train_sizes = [i for i in range(n_min, n_max, 1)]
    n_iter = n_iter or 200

    data_path: str = os.path.join(os.path.abspath(''), "..", GeneralConfig.path_to_data_dir)

    ##################
    logging.info(f"Number of Samples: {num_samples}")
    logging.info(f"Use algorithms: {algorithms}")

    ##### 1. SETUP EXPERIMENT DIR #####
    path_to_dir: str = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                    f"{data_type}/experiment-{experiment_name}")
    create_dirs(path_to_dir)
    sub_dirs: List[str] = filter(os.path.isdir, os.listdir(path_to_dir))
    for sub_dir in sub_dirs:
        folder_path: str = os.path.join(path_to_dir, sub_dir)
        if len(os.listdir(folder_path)) == 0:  # Check is empty..
            shutil.rmtree(folder_path)  # Delete..
    ###################################

    # save config
    general_config: GeneralConfig = GeneralConfig()
    model_config: ModelConfig = ModelConfig()
    pipeline_config: PipelineConfig = PipelineConfig()
    config_list: List[Any] = [general_config, model_config, pipeline_config]
    used_config_filename: str = os.path.join(path_to_dir, "used_config.yaml")
    parse_configs(used_config_filename, config_list)

    # save experiment setup
    used_setup: Dict[str, Any] = {
        "n_min": n_min,
        "n_max": n_max,
        "n_iter": n_iter,
        "n_train_sizes_str": str(n_train_sizes),
        "n_train_sizes": n_train_sizes,
    }
    with open(os.path.join(path_to_dir, "used_setup.yaml"), "w") as output:
        yaml.safe_dump(used_setup, output)

    ##### 2. GET DATA #####
    data_dict: Dict[str, Any] = get_data_dict(data_path,
                                              data_type,
                                              algorithms,
                                              group_type=GeneralConfig.group_type)
    data_dict: Dict[str, Any] = {
        k: v for i, (k, v) in enumerate(data_dict.items()) if i < (num_configs or len(data_dict))
    }
    logging.info(f"Prepared data dict (Length): {len(data_dict)}")
    #######################

    ##### 3. ALTER DATA #####
    alter_data_result: Tuple[Dict[str, Any],
                             Dict[str, Any]] = alter_data_dict(path_to_dir, used_setup, data_dict,
                                                               GeneralConfig.models, data_type)
    data_dict: Dict[str, Any] = alter_data_result[0]
    exec_dict: Dict[str, Any] = alter_data_result[1]
    logging.info(
        f"Data dict (Length): {len(data_dict)}, Unfolded data dict (Length): {len(exec_dict)}")
    #########################

    ##### 4. SUBSAMPLE DATA #####
    subsample_data_result: Tuple[Dict[str, Any],
                                 Dict[str,
                                      Any]] = subsample_data_dict(path_to_dir, data_dict, exec_dict,
                                                                  algorithms, num_samples)
    data_dict: Dict[str, Any] = subsample_data_result[0]
    exec_dict: Dict[str, Any] = subsample_data_result[1]
    logging.info(f"Subsampled Unfolded data dict (Length): {len(exec_dict)}")
    #######################

    if len(data_dict) != (len(algorithms) * num_samples):
        raise ValueError("There was a problem during subsampling.")

    for k, v in data_dict.items():
        logging.info(f"{k}, {v[0].shape}, {len(v)}")

    ##### 5. FILTER DATA #####
    for filename in glob.iglob(path_to_dir + '/**/results.csv', recursive=True):
        target_dir: str = "/".join(filename.split("/")[-2:-1])
        matches: List[str] = [el for el in list(exec_dict.keys()) if el in target_dir]
        if len(matches) == 1:
            del exec_dict[matches[0]]
    logging.info(f"Filtered data dict (Length): {len(exec_dict)}")
    ##########################

    ##### 6. PRETRAINING #####
    path_to_pretrain_folder: str = os.path.join(path_to_dir, GeneralConfig.path_to_pretrain_dir)
    create_dirs(path_to_pretrain_folder)
    files_in_pretrain_folder: str = list(os.listdir(path_to_pretrain_folder))

    for dd_key, dd_value in exec_dict.items():
        if dd_value["pretrain_type"]:  # needs potential pretraining
            if not any([dd_value["pretrain_type"] in f for f in files_in_pretrain_folder
                       ]):  # no pretraining yet
                if len(dd_value["data"][1:]):  # actually has something to pretrain on
                    opt_args: Dict[str, Any] = {
                        "job_identifier": dd_value["pretrain_type"],
                        "path_to_pretrain_folder": path_to_pretrain_folder,
                        "only_pretraining": True
                    }
                    path_to_pipeline: str = os.path.join(path_to_dir, f"{dd_key}_analysis")
                    pipeline: TorchPipeline = TorchPipeline.getInstance(
                        path_to_pipeline, dd_value['model'], **opt_args)

                    rest_df: pd.DataFrame = pd.concat(dd_value["data"][1:],
                                                      axis=0,
                                                      ignore_index=True)
                    rest_df.loc[:, 'type'] = rest_df['machine_type'].map(lambda mt: 1)

                    sample_list: List[PairData] = pipeline.fit(
                        copy.deepcopy(rest_df)).transform(rest_df)
                    pipeline.save()

                    if ModelConfig.ohe_enabled_machine_type:
                        one_hot_encoder_machine_type = pipeline.get_transformer_by_step(
                            "one_hot_encoder_machine_type")
                        ModelConfig.model_args["ohe_features_in_machine_type"] = len(
                            one_hot_encoder_machine_type.categories_[0])
                    if ModelConfig.ohe_enabled_data_size_mb:
                        one_hot_encoder_data_size_MB = pipeline.get_transformer_by_step(
                            "one_hot_encoder_data_size_MB")
                        ModelConfig.model_args["ohe_features_in_data_size_MB"] = len(
                            one_hot_encoder_data_size_MB.categories_[0])

                    fit_and_finetune(sample_list, opt_args)
    ##########################

    ##### 7. START INTERPOLATION / EXTRAPOLATION EXPERIMENT #####
    logging.info("Start Executors...")

    submission_list: List[Dict[str, Any]] = [{
        "job_identifier": k,
        "path_to_pretrain_folder": path_to_pretrain_folder,
        "path_to_dir": path_to_dir,
        **v
    } for k, v in list(exec_dict.items())]

    for submission in submission_list:
        learning_curve_func(**submission)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", type=str, help="Name of experiment.", required=True)
    parser.add_argument("--dataset", type=str, help="Name of the dataset.", required=True)
    parser.add_argument("--num-configs", type=int, help="Number of configurations to investigate.")
    parser.add_argument("--algorithms", nargs='*', help="Algorithms to run.", required=True)
    parser.add_argument("--num-samples", type=int, help="Number of samples to draw.")
    parser.add_argument("--num-iters", type=int, help="Number of iterations. Default 200.")
    args = parser.parse_args()

    init_logging("INFO")

    run_full_experiment(args.experiment_name, args.dataset, args.num_configs, args.algorithms,
                        args.num_samples, args.num_iters)
