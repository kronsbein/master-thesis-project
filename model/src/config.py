from typing import Any, Dict, List


class GeneralConfig(object):
    debug: bool = False
    group_type: str = "machine+job+data+char"
    path_to_pretrain_dir: str = "pretrained_models"
    path_to_temp_dir: str = "temp_finetune_models"
    path_to_data_dir: str = "data"
    grouping_keys: List[str] = [
        "machine_type", 
        "job_args", 
        "data_size_MB", 
        "data_characteristics",
        "environment", 
        "instance_count"
    ]
    models: List[str] = [
        "TorchModel-f",
        "TorchModel-p",
        "TorchModel-s",
    ]
    device: Dict[str, Any] = {
        "device": "cpu",
        "torch_device": "cpu",
        "devices": "auto"
    }
    # device: Dict[str, Any] = {
    #     "device": "gpu", 
    #     "torch_device": "cuda:1", 
    #     "devices":[1]
    # } # change based on available gpus


class ModelConfig(GeneralConfig):
    follow_batch: List[str] = ["x_emb", "x_opt"]
    optimizer_args: Dict[str, float] = {"lr": 0.001, "weight_decay": 0.001}
    model_args: Dict[str, Any] = {
        "dropout": 0.05,
        "hidden_dim": 8,
        "encoding_dim": 40,
        "ohe_features_in_machine_type": 0,
        "ohe_features_in_data_size_MB": 0,
        "benchmark_scores": 0,
    }
    reuse_for_fine_tuning: Dict[str, Any] = {
        "model_args": True,
        "optimizer_args": False
    }
    early_stopping: Dict[str, int] = {"patience": 1000}
    epochs: List[int] = [2500, 2500]  # 0: pre-training, 1: fine-tuning
    batch_size: int = 64
    ohe_enabled_machine_type: bool = False
    ohe_enabled_data_size_mb: bool = False
    target_scaling_enabled: bool = False
    with_benchmark_scores: bool = False
    encoding_in_c_layer: bool = False
    encoding_in_scale_out_layer: bool = False
    # comment out to include scores, when with_benchmark_scores is True.
    # modify model.py ll. 102-104 accordingly. 
    excluded_benchmark_scores: List[str] = [
        "cpu_sysbench", 
        "disk_seq_fio", 
        "memory_sysbench"
    ]
    pooling: str = "mean" # add, mean, max


class PipelineConfig(GeneralConfig):
    feature_scaling_dict: Dict[str, List[str]] = {
        "in_column": [
            "instance_count", "instance_count_div", "instance_count_log"
        ],
        "out_column": [
            "instance_count", "instance_count_div", "instance_count_log"
        ],
        "transformer": "MinMaxScaler",
    }
    target_scaling_dict: Dict[str, List[str]] = {
        "in_column": ["gross_runtime"],
        "out_column": ["gross_runtime"],
        "transformer": "MinMaxScaler",
    }
    binarizer_dict: Dict[str, List[str]] = {
        "in_column": [
            "data_size_MB", 
            "memory", 
            "slots"
        ],
        "out_column": [
            "data_size_MB", 
            "memory", 
            "slots"],
        "transformer": "BinaryTransformer",
        "transformer_args": {
            "n": 39
        },
    }
    hashing_vectorizer_dict: Dict[str, Any] = {
        "in_column": [
            "machine_type", 
            "job_args",
            "job_type", 
            "data_characteristics",
            "environment"
        ],
        "out_column": [
            "machine_type", 
            "job_args",
            "job_type", 
            "data_characteristics",
            "environment"
        ],
        "transformer": "HashingVectorizer",
        "transformer_args": {
            "n_features": 39,
            "lowercase": True,
            "analyzer": "char_wb",
            "ngram_range": [1, 3],
            "norm": "l2",
            "alternate_sign": True,
        },
    }
    one_hot_encoder_dict_machine_type: Dict[str, Any] = {
        "in_column": [
            "machine_type",
        ],
        "out_column": [
            "machine_type",
        ],
        "transformer": "OneHotEncoder",
        "transformer_args": {
            "sparse_output": False,
            "handle_unknown": "ignore",
        },
    }
    one_hot_encoder_dict_datasize_mb: Dict[str, Any] = {
        "in_column": [
            "data_size_MB",
        ],
        "out_column": [
            "data_size_MB",
        ],
        "transformer": "OneHotEncoder",
        "transformer_args": {
            "sparse_output": False,
            "handle_unknown": "ignore",
        },
    }
    graph_list_dict: Dict[str, Any] = {
        "emb_columns": [
            "machine_type", 
            "data_size_MB",
            "job_args", 
            "data_characteristics",
            "environment"
        ],
        "opt_columns": ["job_type", "memory", "slots"],
        "extra_columns": [
            "instance_count", "instance_count_div", "instance_count_log",
            "type", "machine_type", "job_args", "data_size_MB",
            "data_characteristics", "environment", "job_type", "memory", "slots",
            #"cpu_sysbench", "disk_seq_fio", "memory_sysbench" # comment in when with_benchmark_scores is true
        ],
        "target_column": ["gross_runtime"]
    }
