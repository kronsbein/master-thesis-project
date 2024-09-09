import logging
import os

import dill
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from src.config import GeneralConfig, ModelConfig
from src.model import LightningModel
from src.utils import create_dirs
from torch_geometric.loader import DataLoader


class Pretrainer(object):

    def __init__(self, job_identifier: str, path_to_pretrain_folder: str):
        """
        Parameters
        ----------
        training_target : str
            decide what data to use for pretraining
        """
        self.job_identifier = job_identifier
        self.path_to_pretrain_folder = path_to_pretrain_folder

    @classmethod
    def getInstance(cls, job_identifier: str, path_to_pretrain_folder: str, **kwargs):
        path_to_file = os.path.join(path_to_pretrain_folder, f"{job_identifier}.ckpt")
        create_dirs(path_to_pretrain_folder)

        if os.path.exists(path_to_file):
            with open(path_to_file, 'rb') as dill_file:
                return dill.load(dill_file)
        else:
            return Pretrainer(job_identifier, path_to_pretrain_folder, **kwargs)

    def save(self):
        with open(os.path.join(self.path_to_pretrain_folder, f"{self.job_identifier}.pkl"),
                  "wb") as dill_file:
            dill.dump(self, dill_file)

    @staticmethod
    def compute_groups(dataset: list):
        """Compute groups for data splitting.
        
        Parameters
        ----------
        dataset : list
            List of Data-objects
        
        Returns
        ----------
        list
            A list of group indices.
        """

        keys = GeneralConfig.grouping_keys
        labels = ["_".join([str(d[k]) for k in keys]) for d in dataset]

        unique_labels = list(set([str(label) for label in labels]))
        group_dict = {label: (idx + 1) for idx, label in enumerate(unique_labels)}

        return [group_dict[label] for label in labels]

    @staticmethod
    def split_data(dataset: list):
        """Split data into training and validation data.
        
        Parameters
        ----------
        dataset : list
            List of Data-objects
        
        Returns
        ----------
        list, list
            Two lists of Data-objects
        """

        groups = np.array(Pretrainer.compute_groups(dataset))
        logging.info(f"#groups for stratified-split: {len(np.unique(groups))}")

        train_indices, val_indices = train_test_split(
            np.arange(len(dataset)),  # needed, but not used
            test_size=0.2,
            shuffle=True,
            stratify=groups)

        train_list: list = [dataset[i] for i in train_indices]
        val_list: list = [dataset[i] for i in val_indices]

        return train_list, val_list

    def __pretrain__(self, help_dataset: list = []):
        """Gets called whenever a pretraining is required.
        """
        if not len(help_dataset):
            raise ValueError

        train_list, val_list = Pretrainer.split_data(help_dataset)

        batch_size = ModelConfig.batch_size

        # init model callback
        checkpoint_callback = ModelCheckpoint(monitor="val_loss",
                                              dirpath=self.path_to_pretrain_folder,
                                              filename=f"{self.job_identifier}")

        csv_logger = CSVLogger(f"{self.path_to_pretrain_folder}/lightning_logs",
                               name=f"{self.job_identifier}")

        ### create data loaders ###
        train_loader = DataLoader(train_list,
                                  shuffle=True,
                                  batch_size=batch_size,
                                  follow_batch=ModelConfig.follow_batch)

        val_loader = DataLoader(val_list,
                                shuffle=False,
                                batch_size=batch_size,
                                follow_batch=ModelConfig.follow_batch)

        ### train model ###
        model = LightningModel(self.path_to_pretrain_folder, self.job_identifier,
                               **ModelConfig.model_args).double()
        trainer = pl.Trainer(accelerator=GeneralConfig.device["device"],
                             devices=GeneralConfig.device["devices"],
                             logger=csv_logger,
                             callbacks=[checkpoint_callback],
                             max_epochs=ModelConfig.epochs[0])
        trainer.fit(model, train_loader, val_loader)

        logging.info(f"Pretraining for {self.job_identifier} finished.")

    def get_pretrained_model(self, help_dataset: list = []):
        """Get a pretrained model. Freeze layers, disable dropout, unfreeze prediction layer
        
        Parameters
        ----------
        help_dataset: list
            List of training data points. 
        """

        if len(help_dataset):
            # load model from checkpoint path
            checkpoint_suffix = f"{self.job_identifier}.ckpt"
            checkpoint_path = os.path.join(self.path_to_pretrain_folder, checkpoint_suffix)
            if not (os.path.exists(checkpoint_path)):
                logging.error(
                    f"No pretrained model. Start pretraining for {self.job_identifier} now.")
                self.__pretrain__(help_dataset=help_dataset)

            model = LightningModel.load_from_checkpoint(checkpoint_path).double()
        else:
            model = LightningModel(self.path_to_pretrain_folder, self.job_identifier,
                                   **ModelConfig.model_args).double()

        # freeze all layers
        model.freeze_all_layers()

        # disable dropout
        model.disable_dropout()

        if model.state_dict() is None:
            # unfreeze scale out layer
            model.unfreeze_scale_out_layer()

        # unfreeze final prediction layer
        model.unfreeze_c_layer()

        return model
