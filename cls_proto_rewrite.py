import torch
from torch import nn
import torch.nn.functional as F
import math



def get_one_hot_label(gt,num_classes,ignored_index):

    # batch_size,h,w
    not_ignored_index = (gt != ignored_index)
    # batch_size,h,w,k
    one_hot_gt = F.one_hot((gt * not_ignored_index), num_classes)
    # batch_size,h,w,k
    one_hot_gt = one_hot_gt * (not_ignored_index.unsqueeze(-1))

    return one_hot_gt


def get_inter_p2c_loss(one_hot_label,distance_l2):

    """
    Args:
        one_hot_label (b,h,w,k) , distance_l2 (b,k,h,w)
        
    """

    B, H, W, K = one_hot_label.size()
    # b,k,h,w
    distance_l2 = F.interpolate(distance_l2,size=(H, W), mode="bilinear", align_corners=False)
    # b,h,w,k
    distance_l2 = distance_l2.permute(0,2,3,1).contiguous()

    # b,h,w,k
    selected_distance = distance_l2 * one_hot_label

    non_zero_map = torch.count_nonzero(one_hot_label.view(B,-1,K),dim = 1)

    sum_distance = torch.sum(selected_distance.view(B,-1,K),dim = 1)

    return torch.sum(sum_distance,dim = -1) / (torch.sum(non_zero_map,dim = -1) * B)


def get_outer_p2c_loss(one_hot_label,distance_l2,gt,ignored_index):

    """
    Args:
        one_hot_label (b,h,w,k) , distance_l2 (b,k,h,w)
        
    """
    B, H, W, K = one_hot_label.size()
    # b,k,h,w
    distance_l2 = F.interpolate(distance_l2,size=(H, W), mode="bilinear", align_corners=False)
    # b,h,w,k
    distance_l2 = distance_l2.permute(0,2,3,1).contiguous()

    not_gt_label = (1 - one_hot_label) * ((gt != ignored_index).unsqueeze(-1))

    other_distance = distance_l2 * not_gt_label

    loss = torch.min(other_distance)

    return loss


#gt_mask
gt = torch.tensor([[[1, 2, 3, 4],
                        [5, 6, 4, 2],
                        [6, 0, 1, 1],
                        [0, 3, 6, 2]],

                       [[1, 1, 1, 1],
                        [1, 1, 1, 2],
                        [2, 2, 2, 2],
                        [0, 3, 6, 2]]])

x = torch.randn(2, 6, 4, 4)

one_hot_label = get_one_hot_label(gt,6,6)

loss = get_inter_p2c_loss(one_hot_label,x)

loss2 = get_outer_p2c_loss(one_hot_label,x,gt,6)

print("sss")








