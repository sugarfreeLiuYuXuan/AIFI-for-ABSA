from __future__ import print_function

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn


__all__ = ['InfoNCE']


class InfoNCE(nn.Module):
    def __init__(self, input_dim, scl_label, max_temperature, min_temperature, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.max_temperature = max_temperature
        self.min_temperature = min_temperature
        self.hidden_size = input_dim[-1]
        self.batch_size = input_dim[0]
        self.mlp_layers = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2).to('cuda'),
            nn.ReLU().to('cuda'),
            nn.Linear(2 * self.hidden_size, 3).to('cuda')
        ).to('cuda')
        self.scl_label = scl_label
        self.reduction = reduction
        # initiate adaptive temperature
        # self.at = nn.Parameter(torch.randn(self.batch_size,), requires_grad=True)* 0.95 + 0.05
        self.negative_mode = negative_mode
        self.scl_label = scl_label

    def forward(self, query, positive_key, v_ac, v_ae, negative_keys=None,
                reduction='mean', negative_mode='unpaired', use_static_temperature=False, temp_value=None):
        # Check input dimensionality.
        # print(f"query: {query.size()}{query.requires_grad}")
        # print(f"positive_key: {positive_key.size()}{positive_key.requires_grad}")
        query = query.view(self.batch_size, -1)
        positive_key = positive_key.view(self.batch_size, -1)
         
        if query.dim() != 2:
            raise ValueError('<query> must have 2 dimensions.')
        if positive_key.dim() != 2:
            raise ValueError('<positive_key> must have 2 dimensions.')
        if negative_keys is not None:
            if negative_mode == 'unpaired' and negative_keys.dim() != 2:
                raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
            if negative_mode == 'paired' and negative_keys.dim() != 3:
                raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

        # Check matching number of samples.
        if len(query) != len(positive_key):
            raise ValueError('<query> and <positive_key> must must have the same number of samples.')
        if negative_keys is not None:
            if negative_mode == 'paired' and len(query) != len(negative_keys):
                raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

        # Embedding vectors should have same number of components.
        if query.shape[-1] != positive_key.shape[-1]:
            raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
        if negative_keys is not None:
            if query.shape[-1] != negative_keys.shape[-1]:
                raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

        # Normalize to unit vectors
        query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
        if negative_keys is not None:
            # Explicit negative keys

            # Cosine between positive pairs
            positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

            if negative_mode == 'unpaired':
                # Cosine between all query-negative combinations
                negative_logits = query @ transpose(negative_keys)

            elif negative_mode == 'paired':
                query = query.unsqueeze(1)
                negative_logits = query @ transpose(negative_keys)
                negative_logits = negative_logits.squeeze(1)

            # First index in last dimension are the positive samples
            logits = torch.cat([positive_logit, negative_logits], dim=1)
            
            
            labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
        else:
            # Negative keys are implicitly off-diagonal positive keys.

            # Cosine between all combinations
           
            logits = query @ transpose(positive_key)
            
            # Positive keys are the entries on the diagonal
            labels = torch.arange(len(query), device=query.device)
            
        if(use_static_temperature is True):
            temperature = temp_value
            return F.cross_entropy(logits / temperature, labels, reduction=reduction)
        else:
            # Train Adaptive Temperature
            # Avg. pooling
            v_ae = torch.mean(v_ae, dim=-2)
            v_ac = torch.mean(v_ac, dim=-2)
            self.scl_label = self.scl_label.to("cuda")
            adaptive_temperature = v_ac * v_ae
            adaptive_temperature = adaptive_temperature.to('cuda')
            adaptive_temperature = self.mlp_layers(adaptive_temperature).to('cuda')
            adaptive_temperature = F.softmax(adaptive_temperature, dim=1)
            
            at_loss = F.cross_entropy(adaptive_temperature, self.scl_label.long())
            
            adaptive_temperature, _ = torch.max(adaptive_temperature, dim=1)
            
            adaptive_temperature = adaptive_temperature.unsqueeze(1).expand(-1, logits.shape[0])
            
            temperature = (self.max_temperature - self.min_temperature) * (1 - adaptive_temperature)/2 + self.min_temperature
            # self.scl_label = torch.where(self.scl_label == 2, torch.tensor(1), self.scl_label)
            return F.cross_entropy(logits / temperature, labels, reduction=reduction), at_loss


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, loss_scaling_factor=1.0):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.scaling_factor = loss_scaling_factor

    def forward(self, h_ac, h_ae):
        # 前向传播计算h_{ac}和h_{ae}的表示
        assert h_ac.size() == h_ae.size()
        
        # 计算正样本相似度
        sim_pos = torch.bmm(h_ac, h_ae.transpose(1, 2))
        sim_pos = F.log_softmax(sim_pos, dim=-1)
        
        # 生成批内负样本
        bsz, views, hid_dim = h_ac.shape
        h_ac_rep = h_ac.repeat_interleave(views - 1, dim=0)
        h_ae_rep = h_ae.repeat(bsz, 1, 1)
        mask = ~torch.eye(views, dtype=torch.bool).repeat(bsz, 1, 1)
        h_ae_neg = h_ae_rep[mask].reshape(bsz, views - 1, hid_dim)
        assert h_ac_rep.shape == h_ae_neg.shape
        
        # 计算负样本相似度
        sim_neg = torch.bmm(h_ac_rep, h_ae_neg.transpose(1, 2))
        sim_neg = F.log_softmax(sim_neg, dim=-1)
        
        # 计算InfoNCE损失函数
        loss = -(sim_pos.mean() + sim_neg.mean())
        return loss
    
class InfoNCELoss(nn.Module):
    def __init__(self, tau, neg_samples):
        super(InfoNCELoss, self).__init__()
        self.tau = tau
        self.neg_samples = neg_samples

    def forward(self, Hae, Hac):
        batch_size = Hae.size(0)

        # 计算正样本得分
        pos_score = torch.bmm(Hae.unsqueeze(1), Hac.unsqueeze(2)).squeeze() / self.tau

        # 生成负样本向量
        indices = torch.randint(high=batch_size, size=(batch_size * self.neg_samples,))
        Hae_neg = Hae[indices]
        Hac_neg = Hac[indices]

        # 计算负样本得分
        neg_score = torch.sum(Hae_neg * Hac_neg, dim=1) / self.tau
        neg_score = neg_score.view(batch_size, self.neg_samples)

        # 计算对比损失
        logits = torch.cat([pos_score.unsqueeze(1), neg_score], dim=1)
        targets = torch.zeros(batch_size, dtype=torch.long).to(Hae.device)
        loss = F.cross_entropy(logits, targets)

        return loss
