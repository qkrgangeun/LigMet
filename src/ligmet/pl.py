import torch  # type: ignore
from lightning import LightningModule, LightningDataModule  # type: ignore
from pathlib import Path
from torch.utils.data import DataLoader  # type: ignore
from ligmet.dataset import OnTheFlyDataSet, PreprocessedDataSet
# from ligmet.utils.sampler import get_weighted_sampler
from typing import Type, Optional
import torch.nn as nn # type: ignore
from ligmet.utils.constants import metal_counts_focus, metal_counts
import wandb

# ====== LightningModule: MyLightningModel ======
class LigMetModel(LightningModule):
    def __init__(self, model: Type[nn.Module], model_config: dict):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = model(**model_config)
        
        self.register_buffer(
            "metal_weight", 
            torch.tensor([10000 / metal_counts.get(metal, 10000) for metal in metal_counts], dtype=torch.float32)
        )
        self.register_buffer(
            "metal_weight_focus", 
            torch.tensor([10000 / metal_counts.get(metal, 10000) for metal in metal_counts_focus], dtype=torch.float32)
        )
        self.register_buffer("pos_weight", torch.tensor([300], dtype=torch.float32))
        self.register_buffer("bin_weights", torch.tensor([1, 300, 1000], dtype=torch.float32))

        self.loss_fns = nn.ModuleDict({
            "BCE": torch.nn.BCEWithLogitsLoss(),#pos_weight=self.pos_weight
            "Bin": torch.nn.CrossEntropyLoss(),#weight=self.bin_weights
            "CE": torch.nn.CrossEntropyLoss(weight=self.metal_weight),
            "CEfocus": torch.nn.CrossEntropyLoss(weight=self.metal_weight_focus),
        })

    def forward(self, G):
        output, type_output, vector_pred = self.model(G)
        return output, type_output, vector_pred

    # def compute_loss(self, pred, label, type_pred=None, bin_pred=None):
    #     loss = self.loss_fns["BCE"](pred, label[...,0].unsqueeze(-1))
    #     if type_pred is not None:
    #         loss += self.loss_fns["CE"](type_pred, label[..., 1].long())
        
    #     mask = label[..., 0] > 0.5
    #     if mask.any():
    #         loss += self.loss_fns["CEfocus"](type_pred[mask][..., :-1], label[..., 1].long()[mask])
            
    #     if bin_pred is not None:
    #         bin_label = (label[..., 0] > 0.75).long() + (label[..., 0] > 0.5).long() #bin label = 1.0-0.75:2, 0.75-0.5: 1, 0.5-0.0: 0
    #         loss += self.loss_fns["Bin"](bin_pred, bin_label)
    #     return loss
    
    def compute_loss(self, pred, label, type_pred=None, bin_pred=None):
        logs = {}

        bce_loss = self.loss_fns["BCE"](pred, label[..., 0].unsqueeze(-1))
        logs["BCE Loss"] = bce_loss.item()

        loss = bce_loss

        if type_pred is not None:
            ce_loss = self.loss_fns["CE"](type_pred, label[..., 1].long())
            logs["CE Loss"] = ce_loss.item()
            loss += ce_loss

            mask = label[..., 0] > 0.5
            if mask.any():
                ce_focus_loss = self.loss_fns["CEfocus"](type_pred[mask][..., :-1], label[..., 1].long()[mask])
                logs["CE Focus Loss"] = ce_focus_loss.item()
                loss += ce_focus_loss

        if bin_pred is not None:
            bin_label = (label[..., 0] > 0.75).long() + (label[..., 0] > 0.5).long()

            if torch.isnan(bin_pred).any() or torch.isinf(bin_pred).any():
                print("⚠️ bin_pred에 NaN/Inf 값 발견됨!")
                bin_pred = torch.nan_to_num(bin_pred, nan=0.0, posinf=1.0, neginf=-1.0)

            bin_loss = self.loss_fns["Bin"](bin_pred, bin_label)
            logs["Bin Loss"] = bin_loss.item()
            loss += bin_loss

        # WandB 로그 저장
        wandb.log(logs)

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



class LigMetDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_type: str,
        train_data_file: str,
        val_data_file: str,
        preprocessed: dict,
        onthefly: dict,
        train_loader_params: dict,
        val_loader_params: dict,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_type = dataset_type
        self.train_loader_params = train_loader_params
        self.val_loader_params = val_loader_params

        if dataset_type == "preprocessed":
            self.train_dataset = PreprocessedDataSet(
                data_file=train_data_file, **preprocessed
            )
            self.valid_dataset = PreprocessedDataSet(
                data_file=val_data_file, **preprocessed
            )
        elif dataset_type == "onthefly":
            self.train_dataset = OnTheFlyDataSet(
                data_file=train_data_file, **onthefly
            )
            self.valid_dataset = OnTheFlyDataSet(
                data_file=val_data_file, **onthefly
            )
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, collate_fn=self.train_dataset.collate, **self.train_loader_params)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, collate_fn=self.valid_dataset.collate, **self.val_loader_params)

