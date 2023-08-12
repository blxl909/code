import torch
from torch import nn
import torch.nn.functional as F
import math
from geoseg.losses.dice import *



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
        one_hot_label (b,h,w,k) , distance_l2 (b,k,h,w)  实际上是余弦距离
        
    """

    B, H, W, K = one_hot_label.size()
    # b,k,h,w
    distance_l2 = F.interpolate(distance_l2,size=(H, W), mode="bilinear", align_corners=False)
    # b,h,w,k
    distance_l2 = distance_l2.permute(0,2,3,1).contiguous()
    # 余弦相似度，需要转换为正的，并且越靠近1（会有scale影响）（相似度最大）越小
    distance_l2 = 1.0 / (torch.exp(distance_l2) + 1.0)

    # b,h,w,k
    selected_distance = distance_l2 * one_hot_label

    non_zero_map = torch.count_nonzero(one_hot_label.view(B,-1,K),dim = 1)

    sum_distance = torch.sum(selected_distance.view(B,-1,K),dim = 1)

    loss = torch.sum(sum_distance,dim = -1) / torch.sum(non_zero_map,dim = -1)

    return torch.sum(loss,dim = -1) / B


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

    loss = torch.max(other_distance)

    return loss


class local_center_loss(nn.Module):
    def __init__(self,
                 alpha = 0.5,
                 beta = 1.0,
                 ignore_index = 6,
                 num_classes = 6,
                 num_prototype_per_class = 8):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.dice = DiceLoss(ignore_index = ignore_index)
        self.ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.num_prototype_per_class = num_prototype_per_class
        self.aux = None

    def forward(self,predition, gt_mask):
        if self.training:
              pred,coarse_pred,self.aux, prototypes = predition
        else:
              pred,prototypes = predition

        batch_size, h, w = gt_mask.size()
        
        if self.training:
            coarse_pred = F.interpolate(coarse_pred,size=(h, w), mode="bilinear", align_corners=False)
            dice_loss = self.dice(coarse_pred, gt_mask)

        


        

        





#=================================================
        #optimize orthogonality of prototype_vector
        c = prototypes.size()[-1]
        prototypes = prototypes.view(-1,c)
        #[num_classes*num_prototypes_per_class,c]
        cur_basis_matrix = prototypes 
        #[num_classes,num_prototypes_per_class,c]
        subspace_basis_matrix = cur_basis_matrix.reshape(self.num_classes,self.num_prototype_per_class,-1)
        #[num_classes,c,num_prototypes_per_class]
        subspace_basis_matrix_T = torch.transpose(subspace_basis_matrix,1,2)
        #[num_classes,num_prototypes_per_class,num_prototypes_per_class]
        orth_operator = torch.matmul(subspace_basis_matrix,subspace_basis_matrix_T)
        #[num_prototypes_per_class,num_prototypes_per_class]
        I_operator = torch.eye(subspace_basis_matrix.size(1),subspace_basis_matrix.size(1)).cuda()
        #[num_classes,num_prototypes_per_class,num_prototypes_per_class] broadcast
        difference_value = orth_operator - I_operator
        # a number  （关于这个，是不是要除以维度和类别数来进行归一化？）
        orth_cost = torch.sum(torch.relu(torch.norm(difference_value,p=1,dim=[1,2]) - 0))

        orth_cost = orth_cost / (c * self.num_classes)

        #===========================

        #subspace sep
        #[num_classes,c,c]
        projection_operator = torch.matmul(subspace_basis_matrix_T,subspace_basis_matrix)
        #[num_classes,1,c,c]
        projection_operator_1 = torch.unsqueeze(projection_operator,dim=1)
        #[1,num_classes,c,c]
        projection_operator_2 = torch.unsqueeze(projection_operator, dim=0)
        #[num_classes,num_classes]
        pairwise_distance =  torch.norm(projection_operator_1-projection_operator_2+1e-10,p='fro',dim=[2,3])
        # a number
        subspace_sep = 0.5 * torch.norm(pairwise_distance,p=1,dim=[0,1],dtype=torch.double) / torch.sqrt(torch.tensor(2,dtype=torch.double)).cuda()

        subspace_sep = subspace_sep / (c * self.num_classes)





#==============================================



        one_hot_label = get_one_hot_label(gt_mask,self.num_classes,self.ignore_index)

        inter_p2c_loss = get_inter_p2c_loss(one_hot_label,pred)

        outer_p2c_loss = get_outer_p2c_loss(one_hot_label,pred,gt_mask,self.ignore_index)

        ce_loss = self.ce(pred,gt_mask)
        if self.training:
              aux_loss = self.ce(self.aux,gt_mask)
   
              return   inter_p2c_loss + ce_loss  + self.beta * dice_loss+ 0.4 * aux_loss - 0.01 * subspace_sep + 0.1*orth_cost + 0.1*outer_p2c_loss
        else:
              #return p2c_loss + self.beta * dice_loss + ce_loss
              #return self.alpha * c2c_loss + self.beta * dice_loss + ce_loss
              #return p2c_loss + self.alpha * c2c_loss + self.beta * dice_loss + ce_loss
              return   inter_p2c_loss + ce_loss - 0.01 * subspace_sep + 0.1*orth_cost +0.1* outer_p2c_loss










