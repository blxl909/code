import torch
from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

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

def get_inter_class_relative_loss(class_features_query, class_features_key = None, inter_c2c_loss_threshold = 0.5):
    # class_features [N, class, C]

    class_relation, diag = get_inter_class_relations(
        class_features_query,class_features_key
    )  # [N, class, class]
    
    class_relation = class_relation.to('cuda')
    diag = diag.to('cuda')

    num_class = class_relation.shape[-1]

    other_relation = class_relation * (1 - diag)  # [N, class, class]

    threshold = inter_c2c_loss_threshold / (num_class - 1 )

    other_relation = torch.where(other_relation > threshold, other_relation - threshold, torch.zeros_like(other_relation))

    loss = other_relation.sum(dim=-1)  # [N, class]

    loss = torch.clamp(loss, min=torch.finfo(loss.dtype).eps, max=1 - torch.finfo(loss.dtype).eps)

    loss = square_mean_loss(loss)

    return loss

def get_intra_class_absolute_loss(x, avg_value, remove_max_value=False, not_ignore_spatial_mask=None):
    avg_value = avg_value.detach() # stop gradient
    value_diff = torch.abs(avg_value - x) # [N, HW, C]

    if not_ignore_spatial_mask is not None:
        value_diff *= not_ignore_spatial_mask.float()

    value_diff = value_diff.transpose(1, 2) # [N, C, HW]
    assert value_diff.ndim == 3, "ndim must be 3"

    if remove_max_value:
        value_diff, _ = torch.sort(value_diff, dim=-1) # [N, C, HW]
        threshold = 1 # torch.tensor(value_diff.shape[-1], dtype=torch.float32) * 0.8).to(torch.long)
        value_diff = value_diff[..., :-threshold] # [N, C, HW - 1]

    loss = value_diff
    loss = square_mean_loss(loss)
    loss = torch.clamp_min(loss, torch.finfo(loss.dtype).eps)

    return loss

def get_pixel_inter_class_relative_loss(x, class_avg_feature, one_hot_label, inter_c2p_loss_threshold=0.25):

    # x : [N, HW, C]
    # class_avg_feature : # [N, class, C]
    # one_hot_label : [N, HW, class]

    class_avg_feature = class_avg_feature.detach()
    class_avg_feature = class_avg_feature.permute(0, 2, 1)  # [N, C, class]

    energy = torch.matmul(x, class_avg_feature)  # [N, HW, class]

    self_energy = class_avg_feature * class_avg_feature  # [N, C, class]
    self_energy = torch.sum(self_energy, dim=1, keepdim=True)  # [N, 1, class]

    other_label_mask = 1 - one_hot_label.type(torch.long).to(energy.device)

    energy *= other_label_mask  # [N, HW, class]
    energy += self_energy * one_hot_label

    energy_scale = torch.sqrt(torch.tensor(x.shape[-1], dtype=x.dtype, device=x.device))
    energy /= (energy_scale + 1e-4)
    inter_c2p_relation = F.softmax(energy, dim=-1)  # [N, HW, class]

    num_class = inter_c2p_relation.shape[-1]
    num_class = torch.tensor(num_class).float()

    threshold = inter_c2p_loss_threshold / (num_class - 1)

    other_c2p_relation = inter_c2p_relation * other_label_mask  # [N, HW, class]
    other_c2p_relation = torch.where(other_c2p_relation > threshold, other_c2p_relation - threshold, torch.zeros_like(other_c2p_relation))
    other_c2p_relation = torch.sum(other_c2p_relation, dim=-1)  # [N, HW]

    other_c2p_relation = torch.clamp(other_c2p_relation, torch.finfo(other_c2p_relation.dtype).eps, 1 - torch.finfo(other_c2p_relation.dtype).eps)

    loss = other_c2p_relation
    loss = square_mean_loss(loss)

    return loss



def get_flatten_one_hot_label(label, num_class = 6, ignore_index = 6):

    label = torch.tensor(label, dtype=torch.long)  # [N, HW, 1]
    label = label.squeeze(dim=-1) # [N, HW]
    # [N,HW]
    mask = label != ignore_index
    # [N,HW,K]
    label = F.one_hot((label * mask).to(torch.long), num_class) 
    # [N,HW,K]
    one_hot_label = label * mask.unsqueeze(-1)


    return one_hot_label

def get_class_sum_features_and_counts(features, one_hot_label):
    # features [N, HW, C]
    # label [N, HW, class] long

    class_mask = torch.transpose(one_hot_label, 1, 2)  # [N, class, HW]

    non_zero_map = torch.count_nonzero(class_mask, dim=-1)
    non_zero_map = non_zero_map.unsqueeze(-1)
    non_zero_map = non_zero_map.to(features.dtype)  # [N, class, 1]

    class_mask = class_mask.to(features.dtype)

    class_sum_feature = torch.matmul(class_mask, features)  # [N, class, C]

    return class_sum_feature, non_zero_map



class carnet_loss(nn.Module):
    def __init__(self,
                 num_classes = 6,
                 use_inter_class_loss = True,
                 inter_c2c_loss_threshold = 0.5,
                 inter_c2p_loss_threshold = 0.25,
                 inter_class_loss_rate = 1,
                 use_intra_class_loss = True,
                 intra_class_loss_rate = 1,
                 intra_class_loss_remove_max = False
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.use_inter_class_loss = use_inter_class_loss
        self.inter_c2c_loss_threshold = inter_c2c_loss_threshold
        self.inter_c2p_loss_threshold = inter_c2p_loss_threshold
        self.inter_class_loss_rate = inter_class_loss_rate
        self.use_intra_class_loss = use_intra_class_loss
        self.intra_class_loss_rate = intra_class_loss_rate
        self.intra_class_loss_remove_max = intra_class_loss_remove_max


    def forward(self, inputs, label):
        feats = inputs.permute(0,2,3,1) # [N, C, H, W] to  [N, H, W, C]
        batch_size, height, width, channels = feats.size()

        # lables : [N,1, H, W ]
        label = label.unsqueeze(1).float()
        label = F.interpolate(label, size=(height, width), mode='nearest')
        # lables : [N, H, W, 1]
        label = label.permute(0,2,3,1).contiguous()

        not_ignore_spatial_mask = label.to(torch.int32) != self.num_classes  # [N, H, W, 1]
        not_ignore_spatial_mask = not_ignore_spatial_mask.view(batch_size,height*width,1)


        if not torch.isfinite(feats).all():
            raise ValueError("features contains nan or inf")
        
        flatten_features = feats.view(batch_size, height * width, -1).contiguous() #[N, HW, C]
        label = label.view(batch_size, height * width, -1).contiguous() #[N, HW, 1]

        one_hot_label = get_flatten_one_hot_label(label = label, num_class = self.num_classes,ignore_index=self.num_classes) # [N, HW, class]

        class_sum_features, class_sum_non_zero_map = get_class_sum_features_and_counts(
            flatten_features, one_hot_label
        )  # [N, class, C]

        class_sum_features_in_cross_batch = torch.sum(class_sum_features,dim=0,keepdim=True)
        class_sum_non_zero_map_in_cross_batch = torch.sum(class_sum_non_zero_map,dim=0,keepdim=True).to(torch.long)

        class_sum_non_zero_map_in_cross_batch = torch.where(class_sum_non_zero_map_in_cross_batch == 0, 
                                                     torch.tensor([1], device=class_sum_non_zero_map_in_cross_batch.device, dtype=torch.long),
                                                     class_sum_non_zero_map_in_cross_batch)
        class_avg_features_in_cross_batch = class_sum_features_in_cross_batch / class_sum_non_zero_map_in_cross_batch

        class_avg_features = class_avg_features_in_cross_batch
        


        if not torch.isfinite(class_avg_features).all():
            raise ValueError("features contains nan or inf")

        total_loss = 0

        if self.use_inter_class_loss:
            inter_class_relative_loss = 0.0
            inter_class_relative_loss += get_inter_class_relative_loss(
                class_avg_features, inter_c2c_loss_threshold=self.inter_c2c_loss_threshold,
            )

            inter_class_relative_loss += get_pixel_inter_class_relative_loss(
                flatten_features, class_avg_features, one_hot_label, inter_c2p_loss_threshold=self.inter_c2p_loss_threshold,
            )

            total_loss += inter_class_relative_loss * self.inter_class_loss_rate

        if self.use_intra_class_loss:
            one_hot_label = one_hot_label.float()
            same_avg_value = torch.matmul(one_hot_label, class_avg_features)
            if torch.isnan(same_avg_value).any() or torch.isinf(same_avg_value).any():
                raise ValueError("same_avg_value contains NaN or Inf")
            
            self_absolute_loss = get_intra_class_absolute_loss(
                flatten_features,
                same_avg_value,
                remove_max_value=self.intra_class_loss_remove_max,
                not_ignore_spatial_mask = not_ignore_spatial_mask,
            )

            total_loss += self_absolute_loss * self.intra_class_loss_rate
        #print('total_loss = ' , total_loss)
        return total_loss




class cross_entropy_with_carnet_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.carnet_loss = carnet_loss()
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index = 6)
        self.aux = None

    def forward(self, prediction, labels):
        if self.training:    
            pred, self.aux, car_loss_inputs = prediction
        else:
            pred, car_loss_inputs = prediction
        car_loss_val = self.carnet_loss(car_loss_inputs, labels)
        cross_entropy_loss_val = self.cross_entropy_loss(pred, labels)

        if self.training:
            aux_loss = self.cross_entropy_loss(self.aux,labels)
            return car_loss_val +  cross_entropy_loss_val + 0.4 * aux_loss
        else:
            return car_loss_val +  cross_entropy_loss_val







