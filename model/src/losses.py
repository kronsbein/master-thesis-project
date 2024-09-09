import torch.nn as nn


class TrainingLoss(object):

    def __init__(self, *args, **kwargs):

        self.__dict__.update(kwargs)

        self.prediction_loss = nn.SmoothL1Loss()
        self.reconstruction_loss = nn.MSELoss()

    def __call__(self, pred_dict: dict, batch):
        """Computes the loss given a batch-object and a result-dict from the model.
        
        Parameters
        ----------
        pred_dict : dict
            A result dict from a call to the model
        batch: Batch
            A batch object from the Dataloader

        Returns
        ------
        loss
            A PyTorch loss.
        """

        pred_dict = {k: v for k, v in pred_dict.items() if v is not None and "_codes" not in k}

        loss_list: list = []
        for k, v in pred_dict.items():
            ks = "_".join(k.split('_')[:-1])

            t = batch[ks]

            loss_value = None
            if ks == "y":  # loss for runtime prediction
                loss_value = self.prediction_loss(v, t)
            else:  # loss for auto-encoder reconstruction
                loss_value = self.reconstruction_loss(v, t)

            loss_list.append(loss_value)

        return sum(loss_list)


class FineTuningLoss(object):

    def __init__(self, *args, **kwargs):

        self.loss = nn.SmoothL1Loss()

    def __call__(self, pred_dict, batch):
        """Computes the loss given a batch-object and a result-dict from the model.
        
        Parameters
        ----------
        pred_dict : dict
            A result dict from a call to the model
        batch: Batch
            A batch object from the Dataloader

        Returns
        ------
        loss
            A PyTorch loss.
        """

        y_pred = pred_dict["y_pred"]
        y_true = batch.y
        return self.loss(y_pred, y_true)


class ObservationLoss(object):

    def __init__(self, *args, **kwargs):

        self.loss = nn.L1Loss()

    def __call__(self, pred_dict, batch):

        y_pred = pred_dict["y_pred"]
        y_true = batch.y
        return self.loss(y_pred, y_true)
