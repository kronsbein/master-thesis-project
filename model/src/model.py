import logging

import pytorch_lightning as pl
import torch
import torch.nn as nn
from src.config import GeneralConfig, ModelConfig, PipelineConfig
from src.losses import ObservationLoss, TrainingLoss
from torch_geometric.nn import (global_add_pool, global_max_pool,
                                global_mean_pool)
from torch_geometric.utils import to_dense_batch


def init_weights(m):
    """He Initialization"""
    if isinstance(m, torch.nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="selu")
        # biases zero
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class LightningModel(pl.LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__()

        # save hyperparameters
        self.save_hyperparameters(kwargs)

        # model configuration
        self.dropout = ModelConfig.model_args["dropout"]
        self.hidden_dim = ModelConfig.model_args["hidden_dim"]
        self.encoding_dim = ModelConfig.model_args["encoding_dim"]
        self.emb_columns = PipelineConfig.graph_list_dict["emb_columns"]

        self.lr = ModelConfig.optimizer_args["lr"]
        self.wd = ModelConfig.optimizer_args["weight_decay"]
        self.is_finetuning = False
        self.ohe_features_in_machine_type = ModelConfig.model_args["ohe_features_in_machine_type"]
        self.ohe_features_in_data_size_MB = ModelConfig.model_args["ohe_features_in_data_size_MB"]
        self.benchmark_scores = ModelConfig.model_args["benchmark_scores"]

        logging.info(f"OHE features machine_type: {self.ohe_features_in_machine_type}")
        logging.info(f"OHE features data_size_MB: {self.ohe_features_in_data_size_MB}")
        logging.info(f"Pooling: global_{ModelConfig.pooling}_pool")
        logging.info(f"Num benchmark scores: {self.benchmark_scores}")

        self.losses = []

        self.downscale_hidden_dim: int = int(self.hidden_dim / 2)
        self.upscale_hidden_dim: int = int(self.hidden_dim * 2)

        ### instance count embeddings ###
        self.ohe_features = self.ohe_features_in_machine_type + self.ohe_features_in_data_size_MB
        in_features_scale_out_layer: int = 3
        if ModelConfig.encoding_in_scale_out_layer:
            in_features_scale_out_layer += self.ohe_features + self.benchmark_scores
        self.scale_out_layer = nn.Sequential(
            nn.Linear(in_features_scale_out_layer, self.upscale_hidden_dim), nn.SELU(),
            nn.Linear(self.upscale_hidden_dim, self.hidden_dim), nn.SELU())
        self.scale_out_layer.apply(init_weights)

        ### encoder ###
        self.encoder = nn.Sequential(
            nn.Linear(self.encoding_dim, self.hidden_dim, bias=False), nn.SELU(),
            nn.AlphaDropout(p=self.dropout),
            nn.Linear(self.hidden_dim, self.downscale_hidden_dim, bias=False), nn.SELU())
        self.encoder.apply(init_weights)

        ### decoder ###
        self.decoder = nn.Sequential(
            nn.Linear(self.downscale_hidden_dim, self.hidden_dim, bias=False), nn.SELU(),
            nn.AlphaDropout(p=self.dropout),
            nn.Linear(self.hidden_dim, self.encoding_dim, bias=False), nn.Tanh())
        self.decoder.apply(init_weights)

        ### combine predictions ###
        # in_dim = output dim of scale out layer + number of encoded columns + one more for additional
        self.c_layer_in_dim = self.hidden_dim + int(
            (len(self.emb_columns) + 1) * self.downscale_hidden_dim)

        in_features_c_layer: int = self.c_layer_in_dim
        if ModelConfig.encoding_in_c_layer:
            in_features_c_layer += self.ohe_features + self.benchmark_scores
        self.c_layer = nn.Sequential(nn.Linear(in_features_c_layer, self.hidden_dim), nn.SELU(),
                                     nn.Linear(self.hidden_dim, 1), nn.SELU())
        self.c_layer.apply(init_weights)

        self.training_loss_fn = TrainingLoss()
        self.fine_tuning_loss_fn = ObservationLoss()

    def forward(self, batch):
        # additional data encodings
        data_encodings = []
        if ModelConfig.ohe_enabled_machine_type:
            data_encodings.append(batch.machine_type)
        if ModelConfig.ohe_enabled_data_size_mb:
            data_encodings.append(batch.data_size_MB)
        if ModelConfig.with_benchmark_scores:
            data_encodings.extend([batch.cpu_sysbench, batch.disk_seq_fio, batch.memory_sysbench])

        ### instance count embeddings ###
        tensor_list = [batch.instance_count_div, batch.instance_count_log, batch.instance_count]
        if ModelConfig.encoding_in_scale_out_layer:
            tensor_list.extend(data_encodings)
        instance_count = torch.cat(tensor_list, dim=-1)
        instance_count = self.scale_out_layer(instance_count)

        ### compute required embeddings ###
        x_emb_enc = self.encoder(batch.x_emb)

        emb_codes = x_emb_enc.detach().clone()

        x_emb_dec = self.decoder(x_emb_enc)

        emb_pred_dense, _ = to_dense_batch(x_emb_enc, batch.x_emb_batch)
        emb_pred = emb_pred_dense.reshape(emb_pred_dense.size(0), -1)

        ### compute optional embeddings ###
        x_opt_dec = None
        opt_codes = None
        if len(batch.x_opt):
            x_opt_enc = self.encoder(batch.x_opt)

            opt_codes = x_opt_enc.detach().clone()

            x_opt_dec = self.decoder(x_opt_enc)

            if ModelConfig.pooling == "add":
                opt_pred = global_add_pool(x_opt_enc, batch.x_opt_batch)
            elif ModelConfig.pooling == "max":
                opt_pred = global_max_pool(x_opt_enc, batch.x_opt_batch)
            else:
                # default mean
                opt_pred = global_mean_pool(x_opt_enc, batch.x_opt_batch)
        else:
            opt_pred = torch.zeros(emb_pred_dense.size(0),
                                   emb_pred_dense.size(2),
                                   dtype=x_emb_dec.dtype)

        ### combine predictions ###
        if ModelConfig.encoding_in_c_layer:
            additional_data_enc = torch.cat(data_encodings, dim=-1)
            y_pred = self.c_layer(
                torch.cat([instance_count, additional_data_enc, emb_pred, opt_pred], dim=-1))
        else:
            y_pred = self.c_layer(torch.cat([instance_count, emb_pred, opt_pred], dim=-1))

        return {
            "y_pred": y_pred,
            "x_emb_dec": x_emb_dec,
            "x_opt_dec": x_opt_dec,
            "emb_codes": emb_codes,
            "opt_codes": opt_codes
        }

    def training_step(self, batch, batch_idx):
        res_dict = self(batch)
        batch.y = torch.abs(batch.y + torch.normal(0, 0.25, size=batch.y.shape).to(
            GeneralConfig.device["torch_device"]).double())  # add some noise to target values
        if self.is_finetuning:
            loss = self.fine_tuning_loss_fn(res_dict, batch)
            loss_name = "ft_loss"
            self.losses.append(loss.item())
        else:
            loss = self.training_loss_fn(res_dict, batch)
            loss_name = "train_loss"
            self.losses.append(loss.item())
        self.log(loss_name, loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        res_dict = self(batch)
        if self.is_finetuning:
            loss = self.fine_tuning_loss_fn(res_dict, batch)
            loss_name = "ft_val_loss"
            self.losses.append(loss.item())
        else:
            loss = self.training_loss_fn(res_dict, batch)
            loss_name = "val_loss"
            self.losses.append(loss.item())
        self.log(loss_name, loss, on_step=False, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        result_list = []
        target_keys = ["y_pred", "emb_codes", "opt_codes"]
        res_dict = self(batch)
        result_list.append({k: v for k, v in res_dict.items() if k in target_keys})
        return result_list[0]

    def on_save_checkpoint(self, checkpoint):
        if self.is_finetuning:
            checkpoint["is_finetuning"] = True
        checkpoint['losses'] = self.losses

    def on_load_checkpoint(self, checkpoint):
        # if not finetuning, pass
        try:
            self.is_finetuning = checkpoint["is_finetuning"]
        except:
            pass
        self.losses = checkpoint['losses']

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)

    def freeze_all_layers(self):
        logging.info("Freeze all layers")

        for param in self.parameters():
            param.requires_grad = False

    def disable_dropout(self):
        logging.info("Disable dropout layers...")

        for module in self.modules():
            if isinstance(module, (nn.Dropout, nn.AlphaDropout)):
                module.p = 0.0

    def unfreeze_scale_out_layer(self):
        logging.info("Unfreeze scale-out-layer...")

        for param in self.scale_out_layer.parameters():
            param.requires_grad = True

    def unfreeze_c_layer(self):
        logging.info("Unfreeze c-layer...")

        for param in self.c_layer.parameters():
            param.requires_grad = True

    @property
    def all_params(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def all_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
