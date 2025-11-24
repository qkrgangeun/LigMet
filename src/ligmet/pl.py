# ===== Test-only LigMet =====
from pathlib import Path
from typing import Type, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningDataModule, LightningModule
from torch.utils.data import DataLoader, DistributedSampler

from ligmet.dataset import OnTheFlyDataSet
from ligmet.utils.constants import metals,metal_counts_focus, metal_counts
from sklearn.cluster import DBSCAN

class LigMetTestModule(LightningModule):
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
            "CE": torch.nn.CrossEntropyLoss(weight=self.metal_weight),#weight=self.metal_weight
            "CEfocus": torch.nn.CrossEntropyLoss(weight=self.metal_weight_focus),#weight=self.metal_weight_focus
        })
        self.dl_threshold = 0.5  
        
    def forward(self, G):
        # 모델이 (prob_logits, type_logits, bin_logits) 형태를 반환한다는 기존 가정 유지
        output, type_output, vector_pred = self.model(G)
        return output, type_output, vector_pred
    
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

        return loss, logs
    
    def dbscan_clustering_weighted(self, coords: np.ndarray, pred: np.ndarray, type_pred: np.ndarray,
                                eps: float=2.0, min_samples: int=2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return DBSCAN cluster centers and pred-weighted type scores."""
        if len(coords) == 0:
            return np.empty((0,3)), np.empty((0,1)), np.empty((0, type_pred.shape[1]))
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        labels = db.labels_
        centers, scores, t_scores = [], [], []
        for lab in set(labels):
            if lab == -1: 
                continue
            idx = np.where(labels == lab)[0]
            w = pred[idx]
            centers.append(np.average(coords[idx], axis=0, weights=w))
            scores.append(np.max(pred[idx]))
            t_scores.append(np.average(type_pred[idx], axis=0, weights=w))
        if len(centers) == 0:
            return np.empty((0,3)), np.empty((0, type_pred.shape[1]))
        return np.vstack(centers), np.vstack(scores), np.vstack(t_scores)
    
    def grid2pdb(self, grid_positions: np.ndarray, probs: np.ndarray, types: np.ndarray, output_pdb_path: Path):
        """Save predicted grids to a PDB file."""
        with open(output_pdb_path, 'w') as file:
            for i, (coord, prob, type_score) in enumerate(zip(grid_positions, probs, types), start=1):
                type_idx = np.argmax(type_score[:-1])
                metal_element = metals[type_idx]
                line = (
                    f"HETATM{i:5d} {metal_element:<4}  {metal_element:>2} A{i:4d}    "
                    f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                    f"  {prob.item():.2f}  {prob.item():.2f}          {metal_element:>2}\n"
                )

                file.write(line)
        print(f"Predicted grids saved to: {output_pdb_path}")
        return
    @torch.no_grad()
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

        dm = self.trainer.datamodule
        base_dir = Path(dm.dl_test_result_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        # 2) PDB ID 별 하위 디렉터리 또는 파일 패스 결정
        pdb_id = info.pdb_id[0]  # e.g. '1abc'
        out_path = base_dir / f"test_{pdb_id}.npz"
        print('SAVE ',out_path)
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
        mask = preds > self.dl_threshold
        position_selected = info.grids_positions[mask].cpu().numpy()
        pred_selected = preds[mask].cpu().numpy()
        type_selected = type_preds[mask].cpu().numpy()
        centers, occup_scores, type_scores = self.dbscan_clustering_weighted(position_selected, pred_selected, type_selected)
        self.grid2pdb(centers, occup_scores, type_scores, base_dir / f"test_{pdb_id}.pdb")
        
        return


class LigMetTestDataModule(LightningDataModule):
    def __init__(
        self,
        test_data_file: str,
        dl_test_result_dir: str,
        onthefly: dict,
        test_loader_params: dict,
    ):
        super().__init__()
        self.test_data_file = test_data_file
        self.dl_test_result_dir = dl_test_result_dir
        self.onthefly = onthefly
        self.test_loader_params = test_loader_params

        # 테스트만 필요하므로 즉시 세팅
        self.setup(stage="test")

    def setup(self, stage: Optional[str] = None):
        if stage == "test" or stage is None:
            self.test_dataset = OnTheFlyDataSet(
                data_file=self.test_data_file, **self.onthefly
            )
            print(f"[Test-only] dataset size: {len(self.test_dataset)}")

    def test_dataloader(self):
        # DDP 환경이면 DistributedSampler 사용
        try:
            # world_size가 1이면 그냥 DataLoader만
            sampler = DistributedSampler(self.test_dataset, shuffle=False)
            return DataLoader(
                self.test_dataset,
                collate_fn=self.test_dataset.collate,
                sampler=sampler,
                **self.test_loader_params,
            )
        except Exception:
            # 분산 미사용 환경 대비
            return DataLoader(
                self.test_dataset,
                collate_fn=self.test_dataset.collate,
                **self.test_loader_params,
            )
