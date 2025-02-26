import torch  # type: ignore
from lightning.pytorch import LightningModule # type: ignore
from pathlib import Path
from torch.utils.data import DataLoader  # type: ignore
from lightning.pytorch.cli import LightningCLI  # type: ignore
from ligmet.dataset import OnTheFlyDataSet, PreprocessedDataSet
# from ligmet.utils.sampler import get_weighted_sampler
from typing import Type
import torch.nn as nn # type: ignore


# ====== LightningModule: MyLightningModel ======
class LigMetModel(LightningModule):
    def __init__(self, model: Type[nn.Module], model_config: dict, loss_fns: dict, save_pdb_dir: Path| None = None, save_pdb_step: int = 50):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = model(**model_config)
        self.loss_fns = {
            "BCE": torch.nn.BCEWithLogitsLoss(),
            "Bin": torch.nn.CrossEntropyLoss(),
            "CE": torch.nn.CrossEntropyLoss(),
            "CEfocus": torch.nn.CrossEntropyLoss(),
        }
        # self.metal_weight = torch.tensor(metal_weights_list, dtype=torch.float32)

        self.loss_fns = {
            "BCE": torch.nn.BCEWithLogitsLoss(),
            "Bin": torch.nn.CrossEntropyLoss(),
            "CE": torch.nn.CrossEntropyLoss(),
            "CEfocus": torch.nn.CrossEntropyLoss(),
        }

    def forward(self, G):
        output, type_output, vector_pred = self.model(G)
        return output, type_output, vector_pred

    def compute_loss(self, pred, label, type_pred=None, bin_pred=None):
        loss = self.loss_fns["BCE"](pred, label[...,0].unsqueeze(-1))
        if type_pred is not None:
            loss += self.loss_fns["CE"](type_pred, label[..., 1].long())
        
        mask = label[..., 0] > 0.5
        if mask.any():
            loss += self.loss_fns["CEfocus"](type_pred[mask][..., :-1], label[..., 1][mask])
            
        if bin_pred is not None:
            bin_label = (label[..., 0] > 0.75).long() + (label[..., 0] > 0.5).long() #bin label = 1.0-0.75:2, 0.75-0.5: 1, 0.5-0.0: 0
            loss += self.loss_fns["Bin"](bin_pred, bin_label)
        return loss
    

    def training_step(self, batch, batch_idx):
        G, label, info = batch
        pred, type_pred, bin_pred = self(G.to(self.device))

        grididx = torch.where(G.ndata["grid_mask"] > 0)[0]

        # label_prob, label_type, label_vector = (
        #     label[..., 0],
        #     label[..., 1].long(),
        #     label[..., 2:5],
        # )
        total_loss = self.compute_loss(pred[grididx], label, type_pred[grididx], bin_pred[grididx])
        self.log("train_loss", total_loss, on_epoch=True, prog_bar=True, sync_dist=False)
        return total_loss

    def validation_step(self, batch, batch_idx):
        G, label, info = batch
        pred, type_pred, bin_pred = self(G.to(self.device))

        grididx = torch.where(G.ndata["grid_mask"] > 0)[0]

        # label_prob, label_type, label_vector = (
        #     label[..., 0],
        #     label[..., 1].long(),
        #     label[..., 2:5],
        # )
        total_loss = self.compute_loss(pred[grididx], label, type_pred[grididx], bin_pred[grididx])
        self.log("train_loss", total_loss, on_epoch=True, prog_bar=True, sync_dist=False)
        return total_loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer



class LigMetDataModule(LightningModule):
    def __init__(self, dataset_config: dict):
        super().__init__()
        self.save_hyperparameters()
        self.train_loader_params = dataset_config["train_loader_params"]
        self.val_loader_params = dataset_config["val_loader_params"]
    
    def setup(self, stage=None):
        if stage in ("fit", None):
            self.train_dataset = PreprocessedDataSet(self.hparams["dataset"]["preprocessed"])
            self.valid_dataset = OnTheFlyDataSet(self.hparams["dataset"]["onthefly"])
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.train_loader_params)
    
    def val_dataloader(self):
        return DataLoader(self.valid_dataset, **self.val_loader_params)


def main():
    LightningCLI(LigMetModel, LigMetDataModule)


if __name__ == "__main__":
    main()
