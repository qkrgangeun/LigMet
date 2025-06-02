import torch  # type: ignore
import torch.nn.functional as F
from torch import nn  # type: ignore

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.9, gamma=2.0, reduction='mean', pos_weight=None):
        """
        Args:
            alpha (float): 양성 샘플의 가중치 계수.
            gamma (float): 포커싱 파라미터. 클수록 쉬운 샘플에 대한 가중치를 낮춤.
            reduction (str): 'mean', 'sum' 또는 'none'
            pos_weight (Tensor, optional): 불균형을 보정하기 위한 pos_weight 값 (BCEWithLogitsLoss와 동일 형식)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        # BCE 손실을 element-wise로 계산
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none', pos_weight=self.pos_weight
        )
        # p_t = p if y=1, 1-p if y=0
        p_t = torch.exp(-bce_loss)
        
        # 양성과 음성 각각에 대해 손실 분리
        focal_loss_pos = self.alpha * (1 - p_t) ** self.gamma * bce_loss * targets
        focal_loss_neg = (1 - self.alpha) * (1 - p_t) ** self.gamma * bce_loss * (1 - targets)
        
        focal_loss = focal_loss_pos + focal_loss_neg

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
