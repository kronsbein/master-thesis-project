import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

parent_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONPATH"] = parent_dir + ":" + os.environ.get("PYTHONPATH", "")
sys.path.append(parent_dir)
sys.path.append(os.path.join(os.path.abspath(''), ".."))

from evaluation_utils import *
from src.utils import init_logging


def compute_eval_data_from_results_with_metric(
    workload: List[str],
    models: List[str],
    directory: str,
    target: str,
    error_metric: str,
) -> pd.DataFrame:
    """Function to compute evaluation data from result files with given error metric
    """
    data_points_list: List[int] = []
    error_values_list: List[float] = []
    model_type_list: List[str] = []
    for model in models:
        for subdir in os.listdir(directory):
            if workload in subdir and model in subdir:
                path_to_subdir = os.path.join(directory, subdir)
                # load the CSV file into a DataFrame
                filepath: str = os.path.join(path_to_subdir, "results.csv")
                df = pd.read_csv(filepath)

                # calculate error according to given metric
                if error_metric == "sMAPE":
                    # abs(y_true - y_pred) / (abs(y_true) + abs(y_pred))
                    df[error_metric] = abs(df[f"{target}_true"] - df[f"{target}_pred"]) / (
                        abs(df[f"{target}_true"]) + abs(df[f"{target}_pred"]))
                else:
                    # default MRE
                    df[error_metric] = abs(df[f"{target}_true"] - df[f"{target}_pred"]) / abs(
                        df[f"{target}_true"])

                # calculate the mean of the error values
                mean_error_per_result_file = df[error_metric].mean()

                # (data points/error_metric/model type)
                data_points_list.append(subdir.split("___")[1])
                error_values_list.append(mean_error_per_result_file)
                model_type_list.append(model)

    data = {
        "Data Points": data_points_list,
        error_metric: error_values_list,
        "Model Type": model_type_list
    }
    eval_data_df: pd.DataFrame = pd.DataFrame(data=data)
    return eval_data_df.sort_values(by=["Model Type", "Data Points"])


def compute_eval_data_from_results_with_noised_gt_comparison(
    datatype: str,
    experiment: str,
    directory: str,
) -> pd.DataFrame:
    """Function to compute evaluation data from result files with noised groundtruth comparison
    """
    eval_data_df: pd.DataFrame = pd.DataFrame()
    eval_data_list: List[Any] = []
    all_sdirs: List[str] = os.listdir(directory)
    exclude_dirs: Set[str] = {"pretrained_models", "splits", "temp_finetune_models"}
    subdirs = [sdir for sdir in all_sdirs if sdir not in exclude_dirs]
    for subdir in subdirs:
        # filter files
        if not os.path.isdir(os.path.join(directory, subdir)):
            continue
        parts = subdir.split("___")
        n_train_size = int(parts[1])
        first_part_split = parts[0].split("_")
        workload = first_part_split[1].split(".")[0]
        machine_type = first_part_split[2]
        model = parts[2].split("_")[0]

        # filter for less than 2 training points
        if n_train_size < 2:
            continue
        logging.info(f"Comparison for subdir {subdir}.")

        sub_experiment_name: str = subdir.split("TorchModel")[0]

        path_to_subdir: str = os.path.join(directory, subdir)
        results_filepath: str = os.path.join(path_to_subdir, "results.csv")
        my_data_filepath: str = os.path.join(path_to_subdir, "my_data.csv")
        split_dict_path: str = os.path.join(directory,
                                            f"splits/{sub_experiment_name}{model}_split_dict.pkl")

        # get result file, training data and split dict for sub experiment
        results_df: pd.DataFrame = pd.read_csv(results_filepath)
        my_data_df: pd.DataFrame = pd.read_csv(my_data_filepath)
        split_dict: Dict[int, Any] = pd.read_pickle(split_dict_path)[n_train_size]

        # iterate results file
        for index, result_row in results_df.iterrows():
            int_true: int = int(result_row["int_true"])
            int_pred: int = int(round(result_row["int_pred"]))

            indices: List[int] = split_dict[index][0]
            scaleout_values: pd.Series = my_data_df.loc[indices, "instance_count"]
            min_scaleout: int = scaleout_values.min()
            max_scaleout: int = scaleout_values.max()

            # indices for training data points
            rows_to_exclude: List[int] = my_data_df.index.isin(indices)

            # get range, exclude boundaries and training data points
            my_data_df_filtered = my_data_df[(my_data_df["instance_count"] > min_scaleout) &
                                             (my_data_df["instance_count"] < max_scaleout) &
                                             ~rows_to_exclude].copy()
            gt_filtered_copy: pd.DataFrame = my_data_df_filtered.copy()

            # draw noise from np.normal distribution and add to filtered df
            percentage_difference: float = abs(int_true - int_pred) / max(int_true, int_pred)
            noise: np.ndarray = np.random.normal(0, 0.1, size=my_data_df_filtered.shape[0])
            my_data_df_filtered.loc[:, "noise_value"] = noise
            my_data_df_filtered.loc[:, "noised_gross_runtime"] = np.ceil(
                my_data_df_filtered["gross_runtime"] *
                (1 + ((percentage_difference + noise) * np.sign(noise)))).astype(int)

            # get min-/max_runtime from gt
            min_runtime: int = my_data_df["gross_runtime"].min()
            max_runtime: int = my_data_df["gross_runtime"].max()
            min_max_range: np.ndarray = np.arange(start=min_runtime,
                                                  stop=max_runtime,
                                                  step=100,
                                                  dtype=int)

            # get mean runtime from gt data per scaleout and add to filtered df
            mean_values: pd.DataFrame = my_data_df.groupby(
                'instance_count')['gross_runtime'].mean().reset_index()
            mean_values['gross_runtime'] = mean_values['gross_runtime'].round().astype(int)
            mean_values.columns = ['instance_count', 'mean_gt_runtime']
            my_data_df_filtered = pd.merge(my_data_df_filtered,
                                           mean_values,
                                           on='instance_count',
                                           how='left')

            for synthetic_target in min_max_range:
                runtime_target_met: List[int] = []
                runtime_target_met_count: int = 0
                runtime_target_not_met_count: int = 0
                noised_runtimes: List[int] = []
                mean_gt_runtimes: List[int] = []
                for _, filtered_row in my_data_df_filtered.iterrows():
                    noised_runtime: int = filtered_row["noised_gross_runtime"]
                    mean_gt_runtime: int = filtered_row["mean_gt_runtime"]
                    # check if runtime target met
                    if noised_runtime <= synthetic_target and mean_gt_runtime <= synthetic_target:
                        runtime_target_met.append(1)
                        runtime_target_met_count += 1
                    else:
                        runtime_target_met.append(0)
                        runtime_target_not_met_count += 1
                    noised_runtimes.append(noised_runtime)
                    mean_gt_runtimes.append(mean_gt_runtime)

                # find best scale out for target in gt and noised data
                best_scale_out_gt: Optional[pd.Series] = find_best_scale_out(gt_filtered_copy,
                                                                             synthetic_target,
                                                                             key="gross_runtime")
                best_scale_out_noised: Optional[pd.Series] = find_best_scale_out(
                    my_data_df_filtered, synthetic_target, key="noised_gross_runtime")
                if best_scale_out_gt is None or best_scale_out_noised is None:
                    # no target config found
                    continue
                else:
                    # compute resource ratio
                    gt_resources: int = best_scale_out_gt["instance_count"] * best_scale_out_gt[
                        "gross_runtime"]
                    noised_resources: int = best_scale_out_noised[
                        "instance_count"] * best_scale_out_noised["noised_gross_runtime"]
                    resource_ratio: float = abs(gt_resources - noised_resources) / max(
                        gt_resources, noised_resources) * 100

                    row_data: List[Any] = [
                        model, workload, machine_type, n_train_size, int_true, int_pred,
                        percentage_difference, min_runtime, max_runtime, synthetic_target,
                        best_scale_out_gt["instance_count"], best_scale_out_gt["gross_runtime"],
                        gt_resources, best_scale_out_noised["instance_count"],
                        best_scale_out_noised["noised_gross_runtime"], noised_resources,
                        mean_gt_runtimes, noised_runtimes, runtime_target_met,
                        runtime_target_met_count, runtime_target_not_met_count, resource_ratio
                    ]
                eval_data_list.append(row_data)

    columns: List[str] = [
        "model", "workload", "machine_type", "n_train_size", "int_true", "int_pred",
        "percentage_difference_true_pred", "min_runtime", "max_runtime", "synthetic_target",
        "groundtruth_instance_count", "groundtruth_runtime", "groundtruth_instance_count_runtime",
        "noised_instance_count", "noised_runtime", "noised_instance_count_runtime",
        "mean_gt_runtimes", "noised_runtimes", "runtime_target_met", "target_met_count",
        "target_not_met_count", "noised_gt_runtime_resource_ratio"
    ]

    eval_data_df = pd.DataFrame(data=eval_data_list, columns=columns)
    eval_data_df = eval_data_df.sort_values(by=["model", "workload"])
    eval_data_df.to_csv(f"{datatype}/experiment-{experiment}/eval_data_noised_gt_comparison.csv")
    return eval_data_df


def find_best_scale_out(
    df: pd.DataFrame,
    runtime_target: int,
    key: str,
) -> Optional[pd.Series]:
    """Helper function to find best scale-out for a given runtime_target
    """
    # filter to get configurations that meet the runtime target
    filtered_df = df[df[key] <= runtime_target]

    if filtered_df.empty:
        return None

    # get optimal config
    optimal_config = filtered_df.sort_values(by=['instance_count', key]).iloc[0]

    return optimal_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Type of data e.g. c3o, scout...", required=True)
    parser.add_argument("--workloads", nargs='*', help="List of workloads.", required=False)
    parser.add_argument("--models",
                        nargs='*',
                        help="List of different model types: TorchModel-s/-f/-p.",
                        required=False)
    parser.add_argument(
        "--directory",
        type=str,
        help="Name of directory with sub experiments. E.g.: './experiment-baseline/'.",
        required=True)
    parser.add_argument("--experiment",
                        type=str,
                        help="Name of experiment. E.g.: 'baseline'.",
                        required=True)
    parser.add_argument("--target",
                        type=str,
                        help="Targeted result column int/ext for inter-/extrapolation.")
    parser.add_argument(
        "--metric",
        type=str,
        help="The type of error metric to use for error computation e.g. MRE, sMAPE...")
    args = parser.parse_args()

    init_logging("INFO")

    if args.metric:
        if not args.models or not args.workloads:
            parser.print_help()
            sys.exit(0)
        else:
            for workload in args.workloads:
                error_metric_df: pd.DataFrame = compute_eval_data_from_results_with_metric(
                    workload=workload,
                    models=args.models,
                    directory=args.directory,
                    target=args.target,
                    error_metric=args.metric,
                )

                plot_eval_data_with_metric(df=error_metric_df,
                                           datatype=args.data,
                                           experiment=args.experiment,
                                           target=args.target,
                                           workload=workload,
                                           error_metric=args.metric)
    else:
        # check if eval df for comparison already exists
        compare_df: pd.DataFrame = pd.DataFrame()
        path_to_df: str = f"{args.data}/experiment-{args.experiment}/eval_data_noised_gt_comparison.csv"
        if os.path.exists(path=path_to_df):
            compare_df = pd.read_csv(path_to_df)
        else:
            compare_df: pd.DataFrame = compute_eval_data_from_results_with_noised_gt_comparison(
                datatype=args.data,
                experiment=args.experiment,
                directory=args.directory,
            )

        plot_resource_ratio_by_model(df=compare_df,
                                     datatype=args.data,
                                     experiment=args.experiment,
                                     models=args.models)

        plot_runtime_target_met_by_model(df=compare_df,
                                         datatype=args.data,
                                         experiment=args.experiment,
                                         models=args.models)


if __name__ == "__main__":
    main()
