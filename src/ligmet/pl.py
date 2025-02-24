import torch
import numpy as np
import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from metalpred.models.all_atom_model import MyModel
from metalpred.data.dataset import DataSet, DataSetGraphCache
from metalpred.utils.args import Argument
from metalpred.utils.sampler import get_weighted_sampler
from pytorch_lightning.loggers import WandbLogger  # type: ignore
import os
import torch.nn.functional as F  # type: ignore
import time
from metalpred.utils.constants.residue_constants import group_matrix
from metalpred.utils.constants.residue_constants import metals, metal_weights_list
import math
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


# ====== LightningModule: MyLightningModel ======
class MyLightningModel(pl.LightningModule):
    def __init__(self, args, parser):
        super().__init__()
        self.metal_weight = torch.tensor(metal_weights_list, dtype=torch.float32)
        self.model = MyModel(args, parser)
        self.BCEloss = torch.nn.BCEWithLogitsLoss()
        self.Binloss = torch.nn.CrossEntropyLoss()
        self.CEloss = (
            torch.nn.CrossEntropyLoss(weight=self.metal_weight)
            if parser.use_type_prediction
            else None
        )
        # self.CEloss = torch.nn.CrossEntropyLoss(weight=self.metal_weight)
        self.CEfocusloss = torch.nn.CrossEntropyLoss()
        self.Regressionloss = torch.nn.MSELoss()
        self.Grouploss = torch.nn.CrossEntropyLoss() if parser.use_group_loss else None

        self.use_group_loss = parser.use_group_loss
        self.use_type_prediction = parser.use_type_prediction
        self.use_bin_loss = parser.use_bin_loss
        self.save_hyperparameters()  # Save arguments for logging
        self.group_matrix = torch.tensor(group_matrix, dtype=torch.float32)
        self.alpha = args.label_decay
        self.test = {
            "accuracy": {"total": 0, "correct": 0},
            "Top1": {"total": 0, "correct": 0},
        }
        self.dist = args.test["dist"]
        self.use_regression_loss = parser.use_regression_loss
        self.use_focus_type_loss = parser.use_focus_type_loss

    def forward(self, G):
        output, type_output, vector_pred = self.model(G)
        return output, type_output, vector_pred

    def _group_loss(self, type_pred, type_label):
        group_matrix_tensor = self.group_matrix.to(type_pred.device)
        group_pred = torch.matmul(type_pred, group_matrix_tensor.T)
        _, num_classes = type_pred.shape
        type_label_onehot = F.one_hot(type_label, num_classes=num_classes).to(
            dtype=type_pred.dtype, device=type_pred.device
        )
        group_label = torch.matmul(type_label_onehot, group_matrix_tensor.T)
        loss_fn = self.Grouploss
        loss = None
        if self.use_group_loss:
            loss = loss_fn(group_pred, group_label)
        return loss

    def _bin_label(self, label_prob):
        label_bin = torch.zeros_like(label_prob, dtype=torch.long)
        label_bin[label_prob > 0.5] = 1
        label_bin[label_prob > 0.75] = 2
        return label_bin

    def training_step(self, batch, batch_idx):
        G, label, info = batch
        pred, type_pred, bin_pred = self(G.to(self.device))

        grididx = torch.where(G.ndata["grid_mask"] > 0)[0]
        pred = pred[grididx]
        type_pred = type_pred[grididx]
        if bin_pred is not None:
            bin_pred = bin_pred[grididx]

        label_prob, label_type, label_vector = (
            label[..., 0],
            label[..., 1].long(),
            label[..., 2:5],
        )
        prob_loss = self.BCEloss(
            pred.to(torch.float64), label_prob.to(torch.float64).unsqueeze(-1)
        )
        total_loss = prob_loss
        self.log(
            "train_prob_loss", prob_loss, on_epoch=True, prog_bar=True, sync_dist=False
        )

        if self.use_regression_loss:
            regression_loss = self.Regressionloss(pred, label_prob.unsqueeze(-1))
            total_loss += regression_loss
            self.log(
                "train_regression_loss",
                regression_loss,
                on_epoch=True,
                prog_bar=True,
                sync_dist=False,
            )

        if self.use_type_prediction:
            type_loss = self.CEloss(type_pred, label_type)
            total_loss += type_loss
            self.log(
                "train_type_loss",
                type_loss,
                on_epoch=True,
                prog_bar=True,
                sync_dist=False,
            )

            if self.use_focus_type_loss:
                mask = label_prob > 0.5
                if mask.sum() > 0:
                    type_pred_metal = type_pred[mask][..., :-1]
                    label_type_metal = label_type[mask]
                    focus_type_loss = self.CEfocusloss(
                        type_pred_metal, label_type_metal
                    )
                    total_loss += focus_type_loss
                    self.log(
                        "train_focus_type_loss",
                        focus_type_loss,
                        on_epoch=True,
                        prog_bar=True,
                        sync_dist=False,
                    )

            if self.use_group_loss:
                group_loss = self._group_loss(type_pred, label_type)
                total_loss += group_loss
                self.log(
                    "train_group_loss",
                    group_loss,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=False,
                )

        if self.use_bin_loss:
            bin_label = self._bin_label(label_prob)
            bin_loss = self.Binloss(bin_pred, bin_label)
            total_loss += bin_loss
            self.log("train_bin_loss", bin_loss, on_epoch=True, sync_dist=False)

        self.log(
            "train_loss", total_loss, on_epoch=True, prog_bar=True, sync_dist=False
        )

        return total_loss

    def validation_step(self, batch, batch_idx):
        G, label, info = batch
        pred, type_pred, bin_pred = self(G.to(self.device))

        grididx = torch.where(G.ndata["grid_mask"] > 0)[0]
        grididx = torch.eye(len(G.ndata["grid_mask"]))[grididx].to(self.device)
        pred = torch.matmul(grididx, pred).squeeze()
        type_pred = torch.matmul(grididx, type_pred)
        if bin_pred is not None:
            bin_pred = torch.matmul(grididx, bin_pred)

        label_prob, label_type, label_vector = (
            label[..., 0],
            label[..., 1].long(),
            label[..., 2:5],
        )

        prob_loss = self.BCEloss(pred.to(torch.float64), label_prob.to(torch.float64))
        total_loss = prob_loss
        self.log(
            "val_prob_loss", prob_loss, on_epoch=True, prog_bar=True, sync_dist=True
        )

        if self.use_regression_loss:
            regression_loss = self.Regressionloss(pred, label_prob)
            total_loss += regression_loss
            self.log(
                "val_regression_loss",
                regression_loss,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        if self.use_type_prediction:
            type_loss = self.CEloss(type_pred, label_type)
            total_loss += type_loss
            self.log(
                "val_type_loss",
                type_loss,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

            if self.use_focus_type_loss:
                mask = label_prob > 0.5
                if mask.sum() > 0:
                    type_pred_metal = type_pred[mask][..., :-1]
                    label_type_metal = label_type[mask]
                    focus_type_loss = self.CEfocusloss(
                        type_pred_metal, label_type_metal
                    )
                    total_loss += focus_type_loss
                    self.log(
                        "val_focus_type_loss",
                        focus_type_loss,
                        on_epoch=True,
                        prog_bar=True,
                        sync_dist=True,
                    )

            if self.use_group_loss:
                group_loss = self._group_loss(type_pred, label_type)
                total_loss += group_loss
                self.log(
                    "val_group_loss",
                    group_loss,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )

        if self.use_bin_loss:
            bin_label = self._bin_label(label_prob)
            bin_loss = self.Binloss(bin_pred, bin_label)
            total_loss += bin_loss
            self.log("val_bin_loss", bin_loss, on_epoch=True, sync_dist=True)

        dist_correct, dist_total = self._compute_dist_accuracy(
            torch.sigmoid(pred), label_prob, dist=self.dist
        )
        top1_correct, top1_total = self._compute_top1_accuracy(
            torch.sigmoid(pred), label_prob
        )
        type_2A_correct, type_2A_total = self._compute_dist_type_accuracy(
            F.softmax(type_pred, dim=-1), label_type, label_prob
        )

        self.log("val_loss", total_loss, on_epoch=True, prog_bar=True, sync_dist=True)

        return {
            "val_loss": total_loss.detach(),
            "dist_correct": dist_correct,
            "dist_total": dist_total,
            "top1_correct": top1_correct,
            "top1_total": top1_total,
            "type_2A_correct": type_2A_correct,
            "type_2A_total": type_2A_total,
        }

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()

        dist_correct_sum = sum([x["dist_correct"] for x in outputs])
        dist_total_sum = sum([x["dist_total"] for x in outputs])
        top1_correct_sum = sum([x["top1_correct"] for x in outputs])
        top1_total_sum = sum([x["top1_total"] for x in outputs])
        type_correct_sum = sum([x["type_2A_correct"] for x in outputs])
        type_total_sum = sum([x["type_2A_total"] for x in outputs])

        dist_accuracy = dist_correct_sum / dist_total_sum if dist_total_sum > 0 else 0.0
        top1_accuracy = top1_correct_sum / top1_total_sum if top1_total_sum > 0 else 0.0
        type_accuracy = type_correct_sum / type_total_sum if type_total_sum > 0 else 0.0

        self.log("val_loss", val_loss_mean, sync_dist=True)
        self.log(f"Val {self.dist}A accuracy", dist_accuracy, sync_dist=True)
        self.log("Val top1 accuracy", top1_accuracy, sync_dist=True)
        self.log("Type 2A accuracy", type_accuracy, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def _compute_dist_accuracy(self, prob_pred, prob_label, dist):
        prob_threshold = 0.5
        correct = torch.sum(
            (prob_pred > prob_threshold) & (prob_label > prob_threshold)
        ).item()
        total = torch.sum(prob_label > prob_threshold).item()
        return correct, total

    def _compute_top1_accuracy(self, prob_pred, prob_label):
        top1_idx = torch.argmax(prob_label, dim=-1)
        top1_idx_pred = torch.argmax(prob_pred, dim=-1)
        correct = (top1_idx == top1_idx_pred).sum()
        total = 1
        return correct, total

    def _compute_dist_type_accuracy(self, type_pred, label_type, label_prob):
        mask = label_prob > 0.5
        masked_type_pred = type_pred[mask]
        masked_label_type = label_type[mask]

        if masked_type_pred.shape[0] == 0:
            return 0, 0

        pred_label = masked_type_pred.argmax(dim=-1)
        correct = (pred_label == masked_label_type).sum().item()
        total = masked_label_type.shape[0]
        return correct, total


# =========== LightningDataModule 클래스 =============
class MyDataModule(pl.LightningDataModule):
    def __init__(self, args, parser_args):
        super().__init__()
        self.args = args
        self.parser_args = parser_args

        # train/valid 샘플링 타입
        self.sampling_type_train = parser_args.sampling_type_train
        self.sampling_type_valid = parser_args.sampling_type_valid

        # DataSet을 나중(setup)에서 만들기 위해 None으로 초기화
        self.train_dataset = None
        self.valid_dataset = None

        # DataLoader 옵션
        self.loader_params = {
            "num_workers": 1,
            "pin_memory": True,
            "batch_size": self.args.nbatch,
            "worker_init_fn": np.random.seed,
            "persistent_workers": True,
            "prefetch_factor": 4,
            # 그 외 필요하면 추가
        }

    def setup(self, stage=None):
        """Lightning이 DDP init 후에 호출하는 훅(Hook). Dataset 생성 로직을 여기서."""
        if stage == "fit" or stage is None:
            self.train_dataset = DataSetGraphCache(
                self.args.dataf_train, self.args, self.parser_args
            )
            self.valid_dataset = DataSetGraphCache(
                self.args.dataf_valid, self.args, self.parser_args
            )

    def train_dataloader(self):
        """학습용 DataLoader 생성."""
        is_distributed = dist.is_initialized()

        if self.sampling_type_train in ["cluster", "metal", "combined"]:
            print(f"Train sampler: {self.sampling_type_train}")
            sampler_t = get_weighted_sampler(
                self.args.dataf_train, self.sampling_type_train
            )
            train_loader = DataLoader(
                self.train_dataset,
                sampler=sampler_t,
                collate_fn=self.train_dataset.collate,
                **self.loader_params,
            )
            return train_loader
        else:
            if is_distributed:
                print("train sampler: DistributedSampler")
                train_sampler = DistributedSampler(
                    self.train_dataset,
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                    shuffle=True,
                )
                train_loader = DataLoader(
                    self.train_dataset,
                    sampler=train_sampler,
                    collate_fn=self.train_dataset.collate,
                    **self.loader_params,
                )
            else:
                print("train sampler: None (no distributed)")
                train_loader = DataLoader(
                    self.train_dataset,
                    collate_fn=self.train_dataset.collate,
                    **self.loader_params,
                )
            return train_loader

    def val_dataloader(self):
        """검증용 DataLoader 생성."""
        is_distributed = dist.is_initialized()

        if self.sampling_type_valid in ["cluster", "metal", "combined"]:
            print(f"Valid sampler: {self.sampling_type_valid}")
            sampler_v = get_weighted_sampler(
                self.args.dataf_valid, self.sampling_type_valid
            )
            valid_loader = DataLoader(
                self.valid_dataset,
                sampler=sampler_v,
                collate_fn=self.valid_dataset.collate,
                **self.loader_params,
                shuffle=False,
            )
            return valid_loader
        else:
            if is_distributed:
                print("valid sampler: DistributedSampler")
                valid_sampler = DistributedSampler(
                    self.valid_dataset,
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                    shuffle=False,
                )
                valid_loader = DataLoader(
                    self.valid_dataset,
                    sampler=valid_sampler,
                    collate_fn=self.valid_dataset.collate,
                    **self.loader_params,
                )
            else:
                print("valid sampler: None (no distributed)")
                valid_loader = DataLoader(
                    self.valid_dataset,
                    collate_fn=self.valid_dataset.collate,
                    **self.loader_params,
                    shuffle=False,
                )
            return valid_loader


# =========== Argument Parser 설정 =============
def get_parser(args):
    parser = ArgumentParser()
    parser.add_argument(
        "--projectname", type=str, default="model", help="Name of wandb project "
    )
    parser.add_argument(
        "--modelname", type=str, default="model", help="Name of the model"
    )
    parser.add_argument(
        "--paramdir",
        type=str,
        default=args.paramdir,
        help="Path to save/load model parameters",
    )
    parser.add_argument(
        "--trainer_log", action="store_true", help="Enable wandb logging"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint if available",
    )
    parser.add_argument(
        "--use_type_prediction", action="store_true", help="Enable type prediction"
    )
    parser.add_argument(
        "--use_focus_type_loss",
        action="store_true",
        help="Enable exist_type prediction",
    )
    parser.add_argument(
        "--use_regression_loss",
        action="store_true",
        help="Enable exist_type prediction",
    )
    parser.add_argument("--use_group_loss", action="store_true")
    parser.add_argument("--use_bin_loss", action="store_true")
    parser.add_argument(
        "--sampling_type_train",
        type=str,
        default="None",
        choices=["None", "cluster", "metal", "combined"],
        help="Type of weighted sampling: 'cluster', 'metal', or 'combined'",
    )
    parser.add_argument(
        "--sampling_type_valid",
        type=str,
        default="None",
        choices=["None", "cluster", "metal", "combined"],
        help="Type of weighted sampling for validation: 'cluster', 'metal', or 'combined'",
    )
    # parser.add_argument(
    #     "--dl_feat_dir",
    #     default="/home/qkrgangeun/MetalPred/data/biolip_group/latest/features",
    # )
    return parser


# =========== 메인 함수 =============
def main():
    args = Argument()  # 기존에 사용하던 커스텀 Argument
    parser = get_parser(args)
    parser_args = parser.parse_args()

    print("Train set sampling type:", parser_args.sampling_type_train)
    print("Valid set sampling type:", parser_args.sampling_type_valid)

    # DataModule 생성
    data_module = MyDataModule(args, parser_args)

    # LightningModule (모델) 생성
    model = MyLightningModel(args, parser_args)

    # 체크포인트 경로 설정
    checkpoint_dir = Path(parser_args.paramdir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path = checkpoint_dir / f"{parser_args.modelname}_last.ckpt"
    best_checkpoint_path = checkpoint_dir / f"{parser_args.modelname}_best.ckpt"

    # WandB 로거 설정
    if parser_args.trainer_log:
        wandb_logger = WandbLogger(
            project=parser_args.projectname, name=parser_args.modelname
        )
    else:
        wandb_logger = None

    # 체크포인트 콜백
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{parser_args.modelname}_best",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    # Resume 체크
    if parser_args.resume and last_checkpoint_path.exists():
        resume_checkpoint = str(last_checkpoint_path)
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
    else:
        resume_checkpoint = None

    # Trainer 생성
    trainer = pl.Trainer(
        max_epochs=args.maxepoch,
        accelerator="auto",
        enable_progress_bar=True,
        detect_anomaly=True,
        resume_from_checkpoint=resume_checkpoint,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        # strategy="ddp",  # 멀티 GPU DDP
    )

    # Trainer로 학습 시작 (DataModule을 인자로 넘김)
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
