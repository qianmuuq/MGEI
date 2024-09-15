"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, mask_aa=32):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.batch_size = mask_aa

    def forward(self, features, labels=None, mask=None, min_len=None):
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
        
        # print("input", features)
        # print(features.size())
        device = (torch.device('cuda:1')
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
        # print("mask", mask.size())
        # for i in range(36):
        #     print(sum(mask[i]))
        contrast_count = features.shape[1]
        # print("feature",features)
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # print("contrast",contrast_feature)
        # print(contrast_feature.size())
        # print()
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature[:min_len]
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # print("anchor_dot",anchor_dot_contrast.size())
        # print(anchor_dot_contrast.size())
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # print("logits", logits.size())
        # tile mask
        # print("mask_repeat",anchor_count,contrast_count)
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        # print("mask", mask.size())
        # print("logits_mask",logits_mask.size())
        mask = mask * logits_mask
        mask = mask[:min_len,:]
        logits_mask = logits_mask[:min_len,:]
        # print("mask", mask.size())
        # print("logits_mask",logits_mask.size())
        # print("mask")
        # for i in range(36):
        #     print(sum(mask[i]))

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        # print("exp",exp_logits)
        # print("exp_sum",exp_logits.sum(1, keepdim=True))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # print("log_prob",log_prob)

        # compute mean of log-likelihood over positive
        # print("mean_log",(mask * log_prob),(mask * log_prob).size())
        # print(mask.sum(1))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # print(mean_log_prob_pos.size())
        # print(mean_log_prob_pos)
        # print(mean_log_prob_pos[:self.batch_size])
        # mmm = torch.zeros_like(mean_log_prob_pos)
        # for i in range(self.batch_size):
        #     mmm[i] = 1
        # mean_log_prob_pos = mmm*mean_log_prob_pos
        # print(mean_log_prob_pos)
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # loss = loss.view(anchor_count, batch_size).mean()
        loss = loss.mean()
        

        return loss
