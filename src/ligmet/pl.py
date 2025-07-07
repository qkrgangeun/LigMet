import torch  # type: ignore
from lightning import LightningModule, LightningDataModule  # type: ignore
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import DataLoader  # type: ignore
from ligmet.dataset import OnTheFlyDataSet, PreprocessedDataSet, TestDataSet, DataSetGraphCashe, NPZCachedDataset, GraphCachedDataset, GraphFeatureCachedDataset, CachedEdgeDataset 
# from ligmet.utils.sampler import get_weighted_sampler
from typing import Type, Optional 
import torch.nn as nn # type: ignore
from ligmet.utils.constants import metal_counts_focus, metal_counts
import wandb
from torch.utils.data import DistributedSampler  # 추가
import numpy as np
from collections import defaultdict
from ligmet.utils.loss import FocalLoss # type: ignore
from ligmet.utils.sampler import WeightedSampler
from ligmet.utils.constants import metals
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
        self.register_buffer("pos_weight", torch.tensor([2], dtype=torch.float32))
        self.register_buffer("bin_weights", torch.tensor([1, 300, 1000], dtype=torch.float32))

        self.loss_fns = nn.ModuleDict({
            "BCE": torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight),#FocalLoss(alpha=0.5, reduction='mean'),#torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight),#pos_weight=self.pos_weight, reduction='none'
            "Bin": torch.nn.CrossEntropyLoss(),#weight=self.bin_weights
            "CE": torch.nn.CrossEntropyLoss(),#weight=self.metal_weight
            "CEfocus": torch.nn.CrossEntropyLoss(),#weight=self.metal_weight_focus
        })
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
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

        bce_loss = self.loss_fns["BCE"](pred.squeeze(-1), label[..., 0])
        # label_zero_mask = label[...,0] == 0
        # bce_loss_scaled = torch.where(label_zero_mask, bce_loss, 200*bce_loss)
        logs["BCE Loss"] = bce_loss
        loss = bce_loss

        if type_pred is not None:
            ce_loss = self.loss_fns["CE"](type_pred, label[..., 1].long())
            logs["CE Loss"] = ce_loss.item()
            loss += ce_loss

            mask = label[..., 0] > 0.5
            if mask.any():
                local_type_pred, local_label = type_pred[mask][..., :-1], label[..., 1].long()[mask]
                ce_focus_loss = self.loss_fns["CEfocus"](local_type_pred, local_label)
                # label_zero_mask = torch.where(torch.nn.functional.one_hot(local_label,num_classes=local_type_pred.shape[-1])==1,ce_focus_loss, 0.01*ce_focus_loss)
                logs["CE Focus Loss"] = ce_focus_loss.item()
                loss += ce_focus_loss

        # if bin_pred is not None:
        #     bin_label = (label[..., 0] > 0.75).long() + (label[..., 0] > 0.5).long()
        #     bin_loss = self.loss_fns["Bin"](bin_pred,bin_label)
        #     logs["Bin Loss"] = bin_loss.item()
        #     if torch.isnan(bin_pred).any() or torch.isinf(bin_pred).any():
        #         # print("⚠️ bin_pred에 NaN/Inf 값 발견됨!")
        #         bin_pred = torch.nan_to_num(bin_pred, nan=0.0, posinf=1.0, neginf=-1.0)

        return loss, logs

    
    def training_step(self, batch, batch_idx):
        G, label, info = batch
        pred, type_pred, bin_pred = self(G.to(self.device))

        grididx = torch.where(G.ndata["grid_mask"] > 0)[0]

        # label_prob, label_type, label_vector = (
        #     label[..., 0],
        #     label[..., 1].long(),
        #     label[..., 2:5],
        # )
        total_loss, logs = self.compute_loss(pred[grididx], label, type_pred[grididx], bin_pred[grididx])
        self.log("train_loss", total_loss, on_epoch=True, prog_bar=True, sync_dist=False)
        for key in logs:
            self.log(f"train_{key}", logs[key], on_epoch=True, prog_bar=True, sync_dist=False)
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
        preds , type_preds, bin_preds = pred[grididx], type_pred[grididx], bin_pred[grididx]
        total_loss, logs = self.compute_loss(preds, label, type_preds, bin_preds)
        for key in logs:
            self.log(f"val_{key}", logs[key], on_epoch=True, prog_bar=True, sync_dist=False)
        self.log("val_loss", total_loss, on_epoch=True, prog_bar=True, sync_dist=False)
        
        #Metrics
        label_05 = label[...,0] > 0.5
        prob = torch.sigmoid(preds.squeeze())
        pred_05 = prob > 0.5
    
        TP = torch.logical_and(label_05, pred_05).sum().item()
        total_label_05 = label_05.sum().item()
        total_pred_05 = pred_05.sum().item()
        precision = TP/(total_pred_05+1e-06)    
        recall = TP/(total_label_05+1e-06)
        
        label_type_05 = label[..., 1].long()[label_05]
        _, top1_pred = torch.max(type_preds[label_05][...,:-1],dim=1)
        TP_type = (top1_pred == label_type_05).sum().item()
        total_label_type_05 = len(label_type_05)
        type_accuracy = TP_type/(total_label_type_05+1e-06)
        self.validation_step_outputs.append({
            "loss": total_loss,
            "precision": precision,
            "recall": recall,
            "type_accuracy": type_accuracy,
            "type_preds": top1_pred.detach().cpu(),
            "type_labels": label_type_05.detach().cpu()
        })
        print("pdb_id",info.pdb_id, info.metal_types)
        print("num of grid", len(label_05))
        print('label')
        # print(label[...,0])
        print('prob')
        # print(prob)   
        # print("label_05",label_05)
        # print("top1_pred_05", top1_pred)
        # print("label_type_05",label_type_05)
        print('TP',TP)
        print("total_label_05",total_label_05)
        print("total_pred_05", total_pred_05)
        print("precision", precision)
        print("recall", recall)
        print("type_accuracy",type_accuracy)
        return total_loss

    def on_validation_epoch_end(self):
        self.validation_step_outputs
        mean_precision = torch.mean(torch.tensor([o["precision"] for o in self.validation_step_outputs]))
        mean_recall = torch.mean(torch.tensor([o["recall"] for o in self.validation_step_outputs]))
        mean_type_accuracy = torch.mean(torch.tensor([o["type_accuracy"] for o in self.validation_step_outputs]))
        mean_loss = torch.mean(torch.tensor([o["loss"] for o in self.validation_step_outputs]))

        # NEW: collect all predictions and labels
        all_preds = []
        all_labels = []
        for o in self.validation_step_outputs:
            if "type_preds" in o and "type_labels" in o:
                all_preds.append(o["type_preds"])
                all_labels.append(o["type_labels"])

        if all_preds and all_labels:
            preds = torch.cat(all_preds)
            labels = torch.cat(all_labels)
            num_classes = labels.max().item() + 1
            per_class_accs = []
            per_class_prec = []
            per_class_counts = []

            for cls in range(num_classes):
                mask = labels == cls
                mask2 = preds == cls
                count = mask.sum().item()
                per_class_counts.append(count)
                if count > 0:
                    acc = (preds[mask] == labels[mask]).float().mean()
                    pre = (preds[mask2] == labels[mask2]).float().mean()
                    per_class_accs.append(acc)
                    per_class_prec.append(pre)
                else:
                    per_class_accs.append(torch.tensor(0.0))  # 또는 생략 가능
                    per_class_prec.append(torch.tensor(0.0))
            if per_class_accs:
                for metal, acc, pre, count in zip(metals, per_class_accs, per_class_prec, per_class_counts):
                    print(f"{metal:<4} | Recall: {acc.item():.3f} | Precision: {pre.item():.3f} | Count: {count}")
                # print('metal class', metals)
                # print('per_class_accs',per_class_accs)
                macro_type_accuracy = torch.stack(per_class_accs).mean().item()
                self.log("macro_type_accuracy", macro_type_accuracy, on_epoch=True, prog_bar=False, sync_dist=False)

        self.log("val_loss", mean_loss, on_epoch=True, prog_bar=False, sync_dist=False)
        self.log("precision", mean_precision, on_epoch=True, prog_bar=False, sync_dist=False)
        self.log("recall", mean_recall, on_epoch=True, prog_bar=False, sync_dist=False)
        self.log("type_accuracy", mean_type_accuracy, on_epoch=True, prog_bar=False, sync_dist=False)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        G, label, info = batch
        pred, type_pred, bin_pred = self(G.to(self.device))

        grididx = torch.where(G.ndata["grid_mask"] > 0)[0]
        preds, type_preds, bin_preds = pred[grididx], type_pred[grididx], bin_pred[grididx]
        total_loss, logs = self.compute_loss(preds, label, type_preds, bin_preds)
        preds = torch.sigmoid(preds.squeeze())
        label_05 = label[..., 0] > 0.5
        print('target:',info.pdb_id, info.metal_types)
        print('label',label_05)
        print('pred',preds)
        threshold_metrics = []
        type_accuracy_by_threshold = []
        
        dm = self.trainer.datamodule
        base_dir = Path(dm.dl_test_result_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        # 2) PDB ID 별 하위 디렉터리 또는 파일 패스 결정
        pdb_id = info.pdb_id[0]  # e.g. '1abc'
        out_path = base_dir / f"test_{pdb_id}.npz"

        # 3) 결과 저장
        np.savez(
            out_path,
            pred=preds.cpu().numpy(),
            label=label.cpu().numpy(),
            type_pred=type_preds.cpu().numpy(),
            type_label=label[..., 1].long().cpu().numpy(),
            metal_positions=info.metal_positions.cpu().numpy(),
            metal_types=info.metal_types.cpu().numpy(),
            grid_positions=info.grids_positions.cpu().numpy()
        )
        
        for i in torch.arange(0.1,1.0,0.1):
            pred_05 = preds.squeeze() > i
            TP = torch.logical_and(label_05, pred_05).sum().item()
            total_label_05 = label_05.sum().item()
            total_pred_05 = pred_05.sum().item()

            precision = TP/(total_pred_05 + 1e-06)
            recall = TP/(total_label_05 + 1e-06)
            print(f'threshold:{i:.1f}  precision:{precision:.4f}  recall:{recall:.4f}, TP:{TP}, label:{total_label_05}, pred:{total_pred_05}')
            threshold_metrics.append({
                "threshold": round(i.item(), 1),
                "precision": precision,
                "recall": recall
            })        
        label_type_05 = label[..., 1].long()[label_05]
        logits = type_preds[..., :-1]
        type_prob = F.softmax(logits, dim=-1)
        _, top1_pred = torch.max(type_prob, dim=1)
        print('type label',label_type_05)
        print('type prob', type_prob)

        for i in torch.arange(0.5, 0.9, 0.1):
            conf_mask = label[..., 0] > i
            if conf_mask.sum() == 0:
                continue
            label_type = label[..., 1].long()[conf_mask]
            pred_type = type_prob[conf_mask].argmax(dim=-1)
            acc = (pred_type == label_type).float().mean().item()
            type_accuracy_by_threshold.append({
                "threshold": round(i.item(), 1),
                "type_accuracy": acc
            })
            print(f'threshold:{round(i.item(), 1)}  type_accuracy:{acc:.4f}')
        self.test_step_outputs.append({
            "pdb_id": info.pdb_id,
            "threshold_metrics": threshold_metrics,
            "type_accuracy_by_threshold": type_accuracy_by_threshold,
            "type_preds": type_prob.argmax(dim=-1).detach().cpu(),
            "type_labels": label[..., 1].long()[label_05].detach().cpu(),
        })
        return 
    
    def on_test_epoch_end(self):
        all_metrics = defaultdict(list)
        per_class_preds = []
        per_class_labels = []

        # per-PDB threshold logs
        for o in self.test_step_outputs:
            pdb = o["pdb_id"]
            print(f"\n=== PDB: {pdb} ===")

            for m in o["threshold_metrics"]:
                t = m["threshold"]
                all_metrics[f"precision@{t}"].append(m["precision"])
                all_metrics[f"recall@{t}"].append(m["recall"])
                print(f"[{pdb}] threshold {t:.1f} | precision: {m['precision']:.4f} | recall: {m['recall']:.4f}")

            for m in o["type_accuracy_by_threshold"]:
                t = m["threshold"]
                all_metrics[f"type_accuracy@{t}"].append(m["type_accuracy"])
                print(f"[{pdb}] threshold {t:.1f} | type_accuracy: {m['type_accuracy']:.4f}")

            # For per-class accuracy
            per_class_preds.append(o["type_preds"])
            per_class_labels.append(o["type_labels"])

        # 전체 평균 로그 (콘솔 출력용)
        print("\n=== Overall Average Metrics ===")
        for k, v in all_metrics.items():
            mean_val = torch.tensor(v).mean().item()
            print(f"{k}: {mean_val:.4f}")

        # metal type (class)별 정확도
        print("\n=== Per-Class Type Accuracy ===")
        all_preds = torch.cat(per_class_preds)
        all_labels = torch.cat(per_class_labels)
        num_classes = all_preds.max().item() + 1

        for cls in range(num_classes):
            cls_mask = all_labels == cls
            if cls_mask.sum() > 0:
                acc = (all_preds[cls_mask] == all_labels[cls_mask]).float().mean().item()
                print(f"class {cls}: {acc:.4f}")
            else:
                print(f"class {cls}: No samples")

        self.test_step_outputs.clear()

            
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer


class LigMetDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_type: str,
        train_data_file: str,
        val_data_file: str,
        test_data_file: str,
        dl_test_result_dir: str,
        preprocessed: dict,
        onthefly: dict,
        train_loader_params: dict,
        val_loader_params: dict,
        test_loader_params: dict
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_type = dataset_type
        self.train_loader_params = train_loader_params
        self.val_loader_params = val_loader_params
        self.test_loader_params = test_loader_params
        self.train_data_file = train_data_file
        self.val_data_file = val_data_file
        self.test_data_file = test_data_file
        self.dl_test_result_dir = dl_test_result_dir
        self.preprocessed = preprocessed
        self.onthefly = onthefly

        self.setup()
        
    def setup(self, stage=None):
        # stage가 None이면 train/val/test 모두 세팅
        # stage == "fit"인 경우 train/val
        # stage == "test"인 경우 test 세팅
        if stage == "fit" or stage is None:
            if self.dataset_type == "preprocessed":
                self.train_dataset = PreprocessedDataSet(
                    data_file=self.train_data_file, **self.preprocessed
                )
                self.valid_dataset = PreprocessedDataSet(
                    data_file=self.val_data_file, **self.preprocessed
                )
            elif self.dataset_type == "onthefly":
                self.train_dataset = OnTheFlyDataSet(
                    data_file=self.train_data_file, **self.onthefly
                )
                self.valid_dataset = OnTheFlyDataSet(
                    data_file=self.val_data_file, **self.onthefly
                )
            else:
                raise ValueError(f"Unknown dataset type: {self.dataset_type}")

        if stage == "test" or stage is None:
            print(self.preprocessed)
            if self.dataset_type == "preprocessed":
                self.test_dataset = PreprocessedDataSet(
                    data_file=self.test_data_file, **self.preprocessed
                )
            elif self.dataset_type == "onthefly":
                self.test_dataset = OnTheFlyDataSet(
                    data_file=self.test_data_file, **self.onthefly
                )
            else:
                raise ValueError(f"Unknown dataset type: {self.dataset_type}")
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Validation dataset size: {len(self.valid_dataset)}")
        print(f"Train file: {self.train_data_file}")

    def train_dataloader(self):
        sampler = WeightedSampler(self.train_dataset, shuffle=True, total_samples=20000)  # DistributedSampler 추가 WeightedSampler
        if isinstance(sampler, torch.utils.data.DistributedSampler):
            print('Sampler: DistributedSampler')
        else:
            print('Sampler: WeightedSampler')
        return DataLoader(self.train_dataset, collate_fn=self.train_dataset.collate, sampler=sampler, **self.train_loader_params)
    # def train_dataloader(self):
        # if self.dataset_type == "preprocessed" or self.dataset_type == "onthefly":
        #     builder = MetalSamplerBuilder(
        #         metal_to_pdbs_path="/home/qkrgangeun/LigMet/data/biolip/metal_to_pdbs.pkl",
        #         pdb_id_to_metals_path="/home/qkrgangeun/LigMet/data/biolip/pdb_id_to_metals.pkl",
        #         total_samples=self.train_loader_params.get("num_samples", 15000)
        #     )
        #     sampler = builder.build_sampler()

        #     # ⚠️ Dataset이 builder.get_pdb_ids()에 기반한 데이터셋이라야 함
        #     return DataLoader(
        #         self.train_dataset,
        #         collate_fn=self.train_dataset.collate,
        #         sampler=sampler,
        #         shuffle=False,  # sampler를 쓰면 shuffle은 False여야 함
        #         **self.train_loader_params
        #     )
        # else:
        #     raise ValueError("Weighted sampling only supported for known)
    def val_dataloader(self):
        sampler = DistributedSampler(self.valid_dataset, shuffle=False)  # Validation 데이터셋에 대해서는 shuffle=False
        return DataLoader(self.valid_dataset, collate_fn=self.valid_dataset.collate, sampler=sampler, **self.val_loader_params)
    
    def test_dataloader(self):
        sampler = DistributedSampler(self.test_dataset, shuffle=False)
        return DataLoader(
            self.test_dataset,
            collate_fn=self.test_dataset.collate,
            sampler=sampler,
            **self.test_loader_params
        )