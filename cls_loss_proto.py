import torch
from torch import nn
import torch.nn.functional as F
import math
from geoseg.losses.dice import *


def square_mean_loss(loss, inverse_order=True):
    if inverse_order:
        return torch.square(torch.mean(loss))
    else:
        return torch.mean(torch.square(loss))

def get_class_diag(num_class, dtype = torch.float32):
    ones = torch.ones(num_class, dtype=torch.long)
    diag = torch.diag(ones)
    diag = diag.type(dtype)

    return diag

def get_inter_class_relations(query, key = None, apply_scale = True):

    if key is None:
        key = torch.clone(query)
    
    key = key.permute(0,2,1) # [N, C, class]
    key = key.detach()

    attention = torch.matmul(query, key) # [N, class, class]

    num_class = key.shape[-1]
    diag = get_class_diag(num_class, query.dtype)

    if apply_scale:
        attention_scale = torch.sqrt(torch.tensor(query.shape[-1], dtype=query.dtype))
        attention /= (attention_scale + 1e-4)

    attention = F.softmax(attention, dim=-1)

    return attention, diag

def compute_c2c_loss(class_features_query, class_features_key = None, inter_c2c_loss_threshold = 0.5):
    # class_features [N, class, C]

    class_relation, diag = get_inter_class_relations(
        class_features_query,class_features_key
    )  # [N, class, class]
    
    class_relation = class_relation.to("cuda")
    diag = diag.to("cuda")

    num_class = class_relation.shape[-1]

    other_relation = class_relation * (1 - diag)  # [N, class, class]

    threshold = inter_c2c_loss_threshold / (num_class - 1 )

    other_relation = torch.where(other_relation > threshold, other_relation - threshold, torch.zeros_like(other_relation))

    loss = other_relation.sum(dim=-1)  # [N, class]

    loss = torch.clamp(loss, min=torch.finfo(loss.dtype).eps, max=1 - torch.finfo(loss.dtype).eps)

    loss = square_mean_loss(loss)

    return loss




def compute_p2c_loss(gt_mask, p2c_sim_map, ignored_index = 6, num_classes = 6,num_prototype_per_class = 8):
    # gt_mask batch_size*h*w（1024*1024）
    # p2c_sim_map batch_size*km*h*w
    batch_size, h, w = gt_mask.size()
    # batch_size*hw
    gt_mask = gt_mask.view(batch_size, -1)

    p2c_sim_map = F.interpolate(p2c_sim_map,size=(h, w), mode="bilinear", align_corners=False)
    #batch_size*km*h*w
    p2c_sim_map = torch.softmax(p2c_sim_map,dim=1)
    # batch_size*km*h*w
    RSimi = p2c_sim_map

    RSimi = torch.clamp(RSimi,1e-6,1)
    # batch_size*km*hw
    distance = - torch.log(RSimi)
    distance = distance.view(batch_size,num_classes*num_prototype_per_class,-1)
    distance = F.normalize(distance, p=2, dim=1, eps=1e-6, out=None)

    #===============================

    # batch_size*hw
    # 验证了一下，具体见test.py
    mask = gt_mask != ignored_index
    # mask.unsqueeze(1) batch_size*1*hw
    # feats batch_size*c*hw
    #feats = feats * mask.unsqueeze(1)
    # batch_size*hw -> batch_size*hw*k
    gt_mask = F.one_hot((gt_mask * mask).to(torch.long), num_classes) 
    # gt_mask [batch_size*k*hw] * mask.unsqueeze(1) [batch_size*1*hw] => [batch_size*k*hw]
    gt_mask = gt_mask.permute(0, 2, 1) * mask.unsqueeze(1)  # batch_size*k*HW
    # batch_size*k*1*HW
    gt_mask = gt_mask.unsqueeze(2)
    # batch_size*k*m*HW
    gt_mask = gt_mask.expand(batch_size,num_classes,num_prototype_per_class,h*w).contiguous()
    # batch_size*km*HW
    gt_mask = gt_mask.view(batch_size,num_classes * num_prototype_per_class, h*w)
    #================================
    # batch_size*km*hw
    # 过滤出gt对应的distance
    distance_masked = gt_mask * distance
      
    sum_not_ignore_pix = torch.count_nonzero(mask, dim=-1) * num_prototype_per_class


    zero_mask = (sum_not_ignore_pix == 0)
    sum_not_ignore_pix[zero_mask] = 1

    p2c_loss = torch.sum(distance_masked,dim = (1,2)) / sum_not_ignore_pix

    p2c_loss = torch.sum(p2c_loss) / batch_size

    return p2c_loss


def compute_distance_p2c_loss(gt_mask, distance_l2, ignored_index = 6, num_classes = 6,num_prototype_per_class = 8):
    # gt_mask batch_size*h*w（1024*1024）
    # p2c_sim_map batch_size*km*h*w
    batch_size, h, w = gt_mask.size()
    _,_,h_head_plus_w_head = distance_l2.size()
    h_head = int(math.sqrt(h_head_plus_w_head))
    w_head = h_head
    # batch_size*hw
    gt_mask = gt_mask.view(batch_size, -1)

    distance_l2 = distance_l2.view(batch_size,-1,h_head,w_head)

    distance_l2 = F.interpolate(distance_l2,size=(h, w), mode="bilinear", align_corners=False)
    distance_l2 = F.normalize(distance_l2,dim = -1)
    #batch_size*km*h*w
    distance_l2 = torch.softmax(distance_l2,dim=1)
    # batch_size*km*h*w
    RSimi = distance_l2

    RSimi = torch.clamp(RSimi,1e-6,1)
    # batch_size*km*hw
    distance = - torch.log(RSimi)
    distance = distance.view(batch_size,num_classes*num_prototype_per_class,-1)
    distance = F.normalize(distance, p=2, dim=1, eps=1e-6, out=None)

    #===============================

    # batch_size*hw
    # 验证了一下，具体见test.py
    mask = gt_mask != ignored_index
    # mask.unsqueeze(1) batch_size*1*hw
    # feats batch_size*c*hw
    #feats = feats * mask.unsqueeze(1)
    # batch_size*hw -> batch_size*hw*k
    gt_mask = F.one_hot((gt_mask * mask).to(torch.long), num_classes) 

    gt_mask = 1 - gt_mask
    # gt_mask [batch_size*k*hw] * mask.unsqueeze(1) [batch_size*1*hw] => [batch_size*k*hw]
    gt_mask = gt_mask.permute(0, 2, 1) * mask.unsqueeze(1)  # batch_size*k*HW
    # batch_size*k*1*HW
    gt_mask = gt_mask.unsqueeze(2)
    # batch_size*k*m*HW
    gt_mask = gt_mask.expand(batch_size,num_classes,num_prototype_per_class,h*w).contiguous()
    # batch_size*km*HW
    gt_mask = gt_mask.view(batch_size,num_classes * num_prototype_per_class, h*w)
    #================================
    # batch_size*km*hw
    # 过滤出gt对应的distance
    distance_masked = gt_mask * distance
      
    sum_not_ignore_pix = torch.count_nonzero(gt_mask, dim=(1,2))


    zero_mask = (sum_not_ignore_pix == 0)
    sum_not_ignore_pix[zero_mask] = 1

    p2c_loss = torch.sum(distance_masked,dim = (1,2)) / sum_not_ignore_pix

    p2c_loss = torch.sum(p2c_loss) / batch_size

    return p2c_loss






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
        p2c_loss = compute_p2c_loss(gt_mask, p2c_sim_map,self.ignore_index,self.num_classes,self.num_prototype_per_class)
        distance_p2c_loss = compute_distance_p2c_loss(gt_mask,distance_l2,self.ignore_index,self.num_classes,self.num_prototype_per_class)
        ce_loss = self.ce(pred,gt_mask)
        if self.training:
              aux_loss = self.ce(self.aux,gt_mask)
              #return p2c_loss + self.beta * dice_loss + ce_loss + 0.4 * aux_loss
              #return self.alpha * c2c_loss + self.beta * dice_loss + ce_loss + 0.4 * aux_loss  no_p2c_loss
              
              #return p2c_loss + self.alpha * c2c_loss + self.beta * dice_loss + ce_loss + 0.4 * aux_loss   
              return  p2c_loss + ce_loss  + self.beta * dice_loss+ 0.4 * aux_loss - 0.08 * subspace_sep + orth_cost + 0.5*distance_p2c_loss
        else:
              #return p2c_loss + self.beta * dice_loss + ce_loss
              #return self.alpha * c2c_loss + self.beta * dice_loss + ce_loss
              #return p2c_loss + self.alpha * c2c_loss + self.beta * dice_loss + ce_loss
              return  p2c_loss + ce_loss+self.beta * dice_loss  - 0.08 * subspace_sep + orth_cost +0.5* distance_p2c_loss


