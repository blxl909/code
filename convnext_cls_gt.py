from geoseg.models.ConvNext import *
import torch
from torch import nn 
from torch.nn import functional as F

class LN_dim3(nn.Module):
    def __init__(self, channels):
        super(LN_dim3, self).__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self,x):
        #x = x.permute(0,2,1)
        x = self.norm(x)
        #x = x.permute(0,2,1)
        return x

class LN_dim4(nn.Module):
    def __init__(self, channels):
        super(LN_dim4, self).__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self,x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x

        

class SpatialGatherModule(nn.Module):
    def __init__(self, scale=1):
        super(SpatialGatherModule, self).__init__()
        self.scale = scale
    
    def forward(self, features, probs):
        batch_size, num_classes, h, w = probs.size()
        probs = probs.view(batch_size, num_classes, -1) # batch * k * hw
        probs = F.softmax(self.scale * probs, dim=2)

        features = features.view(batch_size, features.size(1), -1)
        features = features.permute(0, 2, 1) # batch * hw * c
        
        ocr_context = torch.matmul(probs, features) #(B, k, c)
        ocr_context = ocr_context.permute(0, 2, 1).contiguous() #(B, C, K)
        return ocr_context





class CLSNet(nn.Module):
    def __init__(self,
                 in_channels = 768,
                 decoder_out_channels = 128,
                 num_classes = 6,
                 patch_size = (4,4),
                 momentum = 0.999,
                 num_prototype_per_class = 8):
        super(CLSNet,self).__init__()
        self.in_channels = in_channels
        self.decoder_out_channels = decoder_out_channels
        self.backbone = ConvNeXt()
        self.num_classes = num_classes
        self.num_patch = patch_size[0] * patch_size[1]
        self.patch_size = patch_size
        self.momentum = momentum
        self.num_prototype_per_class = num_prototype_per_class
        self.first_training = True

        self.norm_feats = nn.Sequential(
            LN_dim4(self.decoder_out_channels),
            nn.GELU(),
        )

        self.dsn = nn.Sequential(
            nn.Conv2d(384, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, 1, stride=1, padding=0, bias=True)
        )

        self.bottle_neck = FPNHEAD( (96, 192, 384, 768), 128)

        self.coarse_pred = nn.Sequential(
            nn.Conv2d(self.decoder_out_channels,num_classes,kernel_size=1),
        )

        self.get_local_center = SpatialGatherModule()

        self.local_center_norm = nn.Sequential(
            LN_dim3(self.decoder_out_channels),
            nn.GELU(),
        )

        self.layer_norm_last = nn.Sequential(
            LN_dim4(self.num_classes*self.num_prototype_per_class),
            nn.GELU(),
        )

        self.prototype = nn.Parameter(torch.randn(self.num_prototype_per_class*self.num_classes,self.decoder_out_channels),requires_grad=True)
        
        
        #decoder_channal to km
        #self.mask_conv = nn.Conv2d(self.decoder_out_channels, self.num_classes * self.num_prototype_per_class, kernel_size = 1, stride = 1,padding=0, bias = True)

    def patch_split(self, input, patch_size):
        """
        input: (B, C, H, W)
        output: (B*num_h*num_w, C, patch_h, patch_w)
        """
        b, c, h, w = input.size()
        num_h, num_w = patch_size
        patch_h, patch_w = h // num_h , w // num_w
        out = input.view(b, c, num_h, patch_h, num_w, patch_w)
        # (b*num_h*num_w,c,patch_h,patch_w)
        out = out.permute(0,2,4,1,3,5).contiguous().view(-1,c,patch_h,patch_w)

        return out

    def momentum_update(self, old_value, new_value):
        # 动量更新公式
        return self.momentum * old_value + (1 - self.momentum) * new_value


    def forward(self, inputs,gt):
        batch_size, channels, h_org, w_org = inputs.size()

        backbone_outputs = self.backbone(inputs)
        feats = backbone_outputs[-1]
        feats = self.bottle_neck(backbone_outputs)

        dsn_feats = backbone_outputs[-2]
        pred_dsn = self.dsn(dsn_feats)
        
        _, c, h_head, w_head = feats.size()


        # lables : [b,1, H, W ]
        label = gt.unsqueeze(1).float()
        label = F.interpolate(label, size=(h_head, w_head), mode='nearest')
        label = label.squeeze(1)# [b, H, W]
        not_ignore_spatial_mask = label.to(torch.int32) != self.num_classes  # [b, H, W]
        one_hot_label = F.one_hot((label * not_ignore_spatial_mask).to(torch.long), self.num_classes) # [b, H, W,k]
        one_hot_label = one_hot_label.view(batch_size,-1,h_head,w_head)
        one_hot_label = one_hot_label.float()

        # b,k,h',w'
        pred = one_hot_label
        
        # b*num_h*num_w,c,h,w
        patch_feats = self.patch_split(feats, self.patch_size)
        # b*num_h*num_w,k,h,w
        pacth_pred = self.patch_split(pred, self.patch_size)
        # b*num_h*num_w,c,k
        class_local_center = self.get_local_center(patch_feats, pacth_pred)

        # b,num_h*num_w*k,c
        class_local_center = class_local_center.permute(0,2,1).contiguous().view(batch_size,-1,c)
        # b,num_h*num_w,k,c
        class_local_center = class_local_center.view(batch_size, -1, self.num_classes, c)
        # b,num_h*num_w,k,c
        normalize_class_local_center = self.local_center_norm(class_local_center)
        # b*num_h*num_w,k,c
        normalize_class_local_center = normalize_class_local_center.view(-1, self.num_classes, c)
        # b,num_h*num_w*k,c
        normalize_class_local_center_copy = normalize_class_local_center.view(batch_size, -1, c)
        # b,k*num_proto,c
        prototype_expand = self.prototype.unsqueeze(0).expand(batch_size,self.num_classes*self.num_prototype_per_class,c)
        # b,num_h*num_w*k,k*num_proto
        proto_dist = torch.cdist(normalize_class_local_center_copy,prototype_expand)
        # b,num_h*num_w*k
        # 找到每个样本中距离最近的原型索引
        nearest_indices = torch.argmin(proto_dist, dim=-1)


        num_total_patch = self.num_patch*self.num_classes

        if(self.training):
            #print("there")
            for i in range(batch_size):
                for j in range(num_total_patch):
                    nearest_center = normalize_class_local_center_copy[i,j]
                    nearest_prototype_idx = nearest_indices[i][j]
                    new_proto_vec = self.momentum_update(self.prototype.data[nearest_prototype_idx], nearest_center)
                    self.prototype.data[nearest_prototype_idx] = new_proto_vec






        #batch_mean_class_local_center = torch.mean(normalize_class_local_center_copy, dim=0)

        #self.prototype.data = self.momentum_update(self.prototype.data, batch_mean_class_local_center)
        # b,k*num_proto,c
        prototype_expand = self.prototype.unsqueeze(0).expand(batch_size,self.num_classes*self.num_prototype_per_class,c)



        # b,c,h',w'
        normalize_feats = feats
        # b,h'*w',c
        normalize_feats = normalize_feats.view(batch_size,c,-1).permute(0,2,1).contiguous()
        # b,k*num_proto,h'*w'
        distance_l2 = torch.cdist(prototype_expand,normalize_feats) 

        # b*km*hw
        #p2c_sim_map = 1.0 / (1.0 + 2.0*distance_l2) 
        p2c_sim_map = 1.0 / (1.0 + 2.0*distance_l2)

        # batch_size*km*h*w
        p2c_sim_map = p2c_sim_map.view(batch_size,self.num_classes * self.num_prototype_per_class, h_head, w_head)
        
        #p2c_sim_map = self.batch_norm(p2c_sim_map)
        p2c_sim_map = self.layer_norm_last(p2c_sim_map)
        

        pred_dsn = F.interpolate(pred_dsn, size=(h_org, w_org), mode="bilinear", align_corners=False)

        temp = p2c_sim_map.view(batch_size,self.num_prototype_per_class,self.num_classes, h_head, w_head) 
        pred, _ = torch.max(temp,dim = 1)
        pred = F.interpolate(pred, size=(h_org, w_org), mode="bilinear", align_corners=False)
        
        if self.training:
            return pred, prototype_expand, pred_dsn, p2c_sim_map,distance_l2,self.prototype
        else:
            return pred, prototype_expand, p2c_sim_map,distance_l2,self.prototype