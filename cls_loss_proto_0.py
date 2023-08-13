import torch
from torch import nn
import torch.nn.functional as F
import math
from geoseg.losses.dice import *







def get_one_hot_label(gt,num_classes,num_prototype_per_class,ignored_index):

    b,h,w = gt.size()

    # batch_size,h,w
    not_ignored_index = (gt != ignored_index)
    # batch_size,h,w,k
    one_hot_gt = F.one_hot((gt * not_ignored_index), num_classes)
    # batch_size,h,w,k
    one_hot_gt = one_hot_gt * (not_ignored_index.unsqueeze(-1))
    # batch_size,k,h,w
    one_hot_gt = one_hot_gt.permute(0,3,1,2).contiguous()

    return one_hot_gt


def get_inter_p2c_loss(one_hot_label,distance_l2):

    """
    Args:
        one_hot_label (b,k,h,w) , distance_l2 (b,m*k,h,w) 
        
    """

    B, k, h, w = one_hot_label.size()
    # b,m*k,h,w
    distance_l2 = F.interpolate(distance_l2,size=(h, w), mode="bilinear", align_corners=False)
    # b,m,k,h,w
    distance_l2 = distance_l2.view(B,-1,k,h,w)
    # b,k,h,w
    distance_l2 = torch.min(distance_l2,dim = 1)[0]
    # b,k,h,w
    selected_distance = distance_l2 * one_hot_label
    # b,k
    non_zero_map = torch.count_nonzero(one_hot_label.view(B,-1,h*w),dim = -1)
    # b,k
    sum_distance = torch.sum(selected_distance.view(B,-1,h*w),dim = -1)

    loss = torch.sum(sum_distance,dim = -1) / torch.sum(non_zero_map,dim = -1)

    return torch.sum(loss,dim = -1) / B


def get_outer_p2c_loss(one_hot_label,distance_l2,gt,ignored_index):

    """
    Args:
        one_hot_label (b,k,h,w) , distance_l2 (b,m*k,h,w) 
        
    """
    B, k, h, w = one_hot_label.size()
    # b,k*m,h,w
    distance_l2 = F.interpolate(distance_l2,size=(h, w), mode="bilinear", align_corners=False)
    # b,m,k,h,w
    distance_l2 = distance_l2.view(B,-1,k,h,w)
    # b,k,h,w
    distance_l2 = torch.min(distance_l2,dim = 1)[0]

    not_gt_label = (1 - one_hot_label) * ((gt != ignored_index).unsqueeze(1))

    other_distance = distance_l2 * not_gt_label

    non_zero_values = other_distance[other_distance != 0]

    loss = torch.min(non_zero_values)

    return 1.0 / (loss + 1e-3)



class local_center_loss(nn.Module):
    def __init__(self,
                 alpha = 0.5,
                 beta = 1.0,
                 ignore_index = 6,
                 num_classes = 6,
                 patch_size = (4,4),
                 num_prototype_per_class = 8):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.dice = DiceLoss(ignore_index = ignore_index)
        self.ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.num_patch = patch_size[0] * patch_size[1]
        self.num_prototype_per_class = num_prototype_per_class

        self.aux = None

    def forward(self,predition, gt_mask):
        if self.training:
              pred,coarse_pred,adaptive_class_center,self.aux, p2c_sim_map,distance_l2,prototypes = predition
        else:
              pred,coarse_pred,adaptive_class_center, p2c_sim_map ,distance_l2,prototypes = predition
        batch_size, h, w = gt_mask.size()
        _,_,h_head,w_head = coarse_pred.size()

        coarse_pred = F.interpolate(coarse_pred,size=(h, w), mode="bilinear", align_corners=False)



#=================================================
        #optimize orthogonality of prototype_vector
        c = prototypes.size()[-1]
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


        
        dice_loss = self.dice(coarse_pred, gt_mask)
        #c2c_loss = compute_c2c_loss(adaptive_class_center)
        #p2c_loss = compute_p2c_loss(gt_mask, p2c_sim_map,self.ignore_index,self.num_classes,self.num_prototype_per_class)
        #distance_p2c_loss = compute_distance_p2c_loss(gt_mask,distance_l2,self.ignore_index,self.num_classes,self.num_prototype_per_class)
        ce_loss = self.ce(pred,gt_mask)
        one_hot_label = get_one_hot_label(gt_mask,self.num_classes,self.num_prototype_per_class,self.ignore_index)
        
        distance_l2 = distance_l2.view(batch_size,-1,h_head,w_head)
        distance_l2 = F.interpolate(distance_l2,size=(h, w), mode="bilinear", align_corners=False)
        inter_p2c_loss = get_inter_p2c_loss(one_hot_label,distance_l2)
        outer_p2c_loss = get_outer_p2c_loss(one_hot_label,distance_l2,gt_mask,self.ignore_index)

        if self.training:
              aux_loss = self.ce(self.aux,gt_mask)

              
              #return p2c_loss + self.beta * dice_loss + ce_loss + 0.4 * aux_loss
              #return self.alpha * c2c_loss + self.beta * dice_loss + ce_loss + 0.4 * aux_loss  no_p2c_loss
              
              #return p2c_loss + self.alpha * c2c_loss + self.beta * dice_loss + ce_loss + 0.4 * aux_loss   

            #   print("inter_p2c_loss:",inter_p2c_loss*0.1)
            #   print("\n")
            #   print("dice_loss:",self.beta * dice_loss)
            #   print("\n")
            #   print("aux_loss:",0.4* aux_loss)
            #   print("\n")
            #   print("subspace_sep:",- 0.08 * subspace_sep)
            #   print("\n")
            #   print("orth_cost:",orth_cost)
            #   print("\n")
            #   print("outer_p2c_loss:",outer_p2c_loss)
            #   print("\n")
              return  0.1*inter_p2c_loss + ce_loss  + self.beta * dice_loss+ 0.4 * aux_loss - 0.08 * subspace_sep + orth_cost + outer_p2c_loss
        else:
              #return p2c_loss + self.beta * dice_loss + ce_loss
              #return self.alpha * c2c_loss + self.beta * dice_loss + ce_loss
              #return p2c_loss + self.alpha * c2c_loss + self.beta * dice_loss + ce_loss
              return  0.1*inter_p2c_loss + ce_loss  + self.beta * dice_loss- 0.08 * subspace_sep + orth_cost + outer_p2c_loss


