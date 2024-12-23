import torch
import torch.nn as nn

class NegativeQueue:
    """
    Implements a queue for storing negative samples for contrastive learning.
    """
    def __init__(self, feature_dim: int, queue_size: int):
        self.queue = torch.randn(queue_size, feature_dim).cuda()
        self.queue = self.queue / self.queue.norm(dim=1, keepdim=True)  # 歸一化
        self.ptr = queue_size  # 初始化時將 ptr 設置為 queue_size，視為已填滿
        self.queue_size = queue_size
        self.labels = torch.ones(queue_size).cuda()  # 初始化為負樣本標籤

    def dequeue_and_enqueue(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        negative_features = features[labels == 1]  # 選取負樣本
        if negative_features.size(0) == 0:  # 如果沒有負樣本，直接返回
            return

        negative_features = negative_features / negative_features.norm(dim=1, keepdim=True)  # 歸一化
        batch_size = negative_features.size(0)

        if batch_size > self.queue_size:
            self.queue = negative_features[-self.queue_size:].detach()
            self.ptr = 0
        else:
            end_ptr = (self.ptr + batch_size) % self.queue_size
            if end_ptr < self.ptr:
                self.queue[self.ptr:] = negative_features[:self.queue_size - self.ptr].detach()
                self.queue[:end_ptr] = negative_features[self.queue_size - self.ptr:].detach()
            else:
                self.queue[self.ptr:end_ptr] = negative_features.detach()
            self.ptr = end_ptr

    def get_negatives(self) -> torch.Tensor:
        """
        Returns negative samples from the queue.

        Returns:
            torch.Tensor: Negative samples.
        """
        return self.queue


class MomentumUpdater:
    """
    Updates the parameters of a momentum-based encoder.
    """
    def __init__(self, model: nn.Module, momentum: float = 0.999):
        self.model = model
        self.momentum = momentum
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters()}

    def update(self, new_model: nn.Module) -> None:
        """
        Updates the parameters of the shadow model using momentum.

        Args:
            new_model (nn.Module): The model with updated parameters.
        """
        with torch.no_grad():
            for name, param in new_model.named_parameters():
                if name in self.shadow:
                    self.shadow[name] = self.momentum * self.shadow[name] + (1 - self.momentum) * param.data
                    param.data = self.shadow[name]
