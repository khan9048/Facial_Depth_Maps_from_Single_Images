import torch
import torch.nn as nn


class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.95):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target):
        valid_mask = (target > 0).detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))

        return loss

def scale_invariant_loss(valid_out, valid_gt):
    logdiff = torch.log(valid_out) - torch.log(valid_gt)
    scale_inv_loss = torch.sqrt((logdiff ** 2).mean() - 0.85*(logdiff.mean() ** 2))*10.0
    return scale_inv_loss

def make_mask(depths, crop_mask, dataset):
    # masking valied area
    if dataset == 'kitti':
        valid_mask = depths > 0.001
    else:
        valid_mask = depths > 0.001
    if dataset == "kitti":
        if (crop_mask.size(0) != valid_mask.size(0)):
            crop_mask = crop_mask[0:valid_mask.size(0), :, :, :]
        final_mask = crop_mask | valid_mask
    else:
        final_mask = valid_mask

    return valid_mask, final_mask

# class silog_loss(nn.Module):
#     def __init__(self, variance_focus):
#         super(silog_loss, self).__init__()
#         self.variance_focus = variance_focus
#
#     def forward(self, depth_est, depth_gt, mask):
#         d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
#         return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0

