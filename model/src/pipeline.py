import os
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import dill
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from src.config import ModelConfig, PipelineConfig
from src.transforms import *
from src.utils import create_dirs

transformer_dict: Dict[str, Any] = {
    "BinaryTransformer": BinaryTransformer,
    "HashingVectorizer": HashingVectorizer,
    "OneHotEncoder": OneHotEncoder,
    "MinMaxScaler": MinMaxScaler,
}


class TorchPipeline():

    def __init__(self, path_to_dir: str, model_type: str, **kwargs):
        """
        Parameters
        ----------
        path_to_dir : str
            The to the pipeline-object
        """

        self.path_to_dir: str = path_to_dir
        self.model_type: str = model_type
        self.is_fitted: bool = False
        create_dirs(self.path_to_dir)

        self.pipeline = self._get_pipeline()

    @classmethod
    def getInstance(cls, path_to_dir: str, model_type: str, **kwargs):
        """Returns a new instance of the TorchPipeline class.
        
        Parameters
        ----------
        path_to_dir : str
            The to the pipeline-object

        Returns
        ------
        Pipeline
            The machine learning pipeline
        """
        path_to_file: str = os.path.join(path_to_dir, f"{model_type}.pkl")
        if os.path.exists(path_to_file):
            with open(path_to_file, "rb") as f:
                return dill.load(f)
        else:
            return TorchPipeline(path_to_dir, model_type, **kwargs)

    def save(self):
        """Saves the pipeline to disk."""
        with open(os.path.join(self.path_to_dir, f"{self.model_type}.pkl"), "wb") as dill_file:
            self.is_fitted = True
            dill.dump(self, dill_file)

    def fit(self, *args, **kwargs):
        return self.pipeline.fit(*args, **kwargs)

    def transform(self, *args, **kwargs):
        return self.pipeline.transform(*args, **kwargs)

    def get_transformer_by_step(self, step_name: str) -> TransformerMixin:
        """
        Get the transformer from a pipeline based on the step name.
        
        Parameters:
        pipeline (Pipeline): 
            The pipeline object.
        step_name (str): 
            The name of the step whose transformer is to be returned.
        
        Returns:
            transformer: The transformer corresponding to the given step name.
        """
        if step_name not in self.pipeline.named_steps:
            raise ValueError(f"Step '{step_name}' not found in pipeline.")
        transformer = self.pipeline.named_steps[step_name].transformer
        return transformer

    def _get_pipeline(self) -> Pipeline:
        """Retrieves a machine learning pipeline.
        
        Returns
        ------
        Pipeline
            The machine learning pipeline
        """
        transformer_list: List[Tuple[str, Any]] = []
        step_list: List[Tuple[str, Any]] = []
        step_list.append(("scaler", PipelineConfig.feature_scaling_dict))

        if ModelConfig.target_scaling_enabled:
            step_list.append(("target_scaler", PipelineConfig.target_scaling_dict))

        if ModelConfig.ohe_enabled_machine_type:
            step_list.append(
                ("one_hot_encoder_machine_type", PipelineConfig.one_hot_encoder_dict_machine_type))

        if ModelConfig.ohe_enabled_data_size_mb:
            step_list.append(
                ("one_hot_encoder_data_size_MB", PipelineConfig.one_hot_encoder_dict_datasize_mb))

        step_list.append(("binarizer", PipelineConfig.binarizer_dict))
        step_list.append(("hasher", PipelineConfig.hashing_vectorizer_dict))

        steps: OrderedDict = OrderedDict(step_list)
        for step, config in steps.items():
            # transformer steps
            transformer_list.append(
                (step,
                 WrapperTransformer(in_column=config["in_column"],
                                    out_column=config["out_column"],
                                    transformer=transformer_dict[config["transformer"]](
                                        **config.get("transformer_args", {})))))
        # append last transform step
        transformer_list.append(
            ("to_graph_list", ToGraphListTransformer(**PipelineConfig.graph_list_dict)))
        return Pipeline(transformer_list)
