from lib.model.PositionEncoder import get_embedder
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import copy

from .BaseNet import BaseNet
from .SurfaceClassifier import SurfaceClassifier
from .HGFilters import *
from .ResNet3d import ResNet3d
from ..net_util import init_net, check_tensor
from ..sample_util import *
from ..mesh_util import *
from .attention import Encoder

class DMCNet(BaseNet):
    '''
    HG Pair network uses Hourglass stacks as the image filter.
    It does the following:
        1. Compute image feature stacks and store it in self.im_feat_list
            self.im_feat_list[-1] is the last stack (output stack)
        2. Calculate calibration
        3. If training, it index on every intermediate stacks,
            If testing, it index on the last stack.
        4. Classification.
        5. During training, error is calculated on all stacks.
    '''

    def __init__(self,
                 opt,
                 projection_mode='orthogonal',
                 error_term=nn.MSELoss(),
                 ):
        super(DMCNet, self).__init__(
            projection_mode=projection_mode,
            error_term=error_term)
        self.error_func = error_term
        self.name = 'dmcnet'

        self.opt = opt
        self.num_views = self.opt.num_views

        self.image_filter = HGFilter(opt)

        fine_opt = copy.deepcopy(opt)
        fine_opt.num_stack = opt.fine_num_stack
        fine_opt.num_hourglass = opt.fine_num_hourglass
        fine_opt.hourglass_dim = opt.fine_hourglass_dim
        self.normal_filter = HGFilter(fine_opt)
        
        self.normal_filter_hr = HGFilter(fine_opt, base=16, downsample=0)
        self.color_filter = HGFilter(fine_opt, base=16, downsample=0)
    
        
        self.conv_filter3d = ResNet3d()
        self.pos_embedding, _ = get_embedder(10)

        self.surface_classifier = SurfaceClassifier(
            filter_channels=self.opt.mlp_dim,
            num_views=self.opt.num_views,  # self.opt.num_views,
            no_residual=self.opt.no_residual,
            last_op=nn.Sigmoid())

        self.fine_sc = SurfaceClassifier(
            filter_channels=self.opt.fine_mlp_dim,
            num_views=self.opt.num_views,  # self.opt.num_views,
            no_residual=self.opt.no_residual,
            last_op=nn.Sigmoid())
        
        self.feature_fusion = Encoder(n_layers=2, n_head=8, d_model=256+64, d_v=32, d_k=32, d_inner=256)
        self.feature_fusion_n = Encoder(n_layers=1, n_head=8, d_model=256+64+32, d_v=32, d_k=32, d_inner=256)

        self.mv_feat_list = []
        self.mv_nfeat_list = []

        self.intermediate_preds_list = []
        self.fine_preds_list = []
        self.fine_sg_preds_list = []
        self.sg_preds_list = []

        init_net(self)
    
    def mask_init(self, mask, ero_mask):
        self.mask = mask.unsqueeze(0)
        self.ero_mask = ero_mask.unsqueeze(0)

    def norm_init(self, scale, center):
        self.scale = scale
        self.center = center

    def smpl_init(self, normal):
        self.normal_list = normal.unsqueeze(0)
    
    def filter3d(self, vox):
        '''
        Filter the input point cloud
        store all intermediate 3d features.
        :param feat_points: [B, 3, N] input feature cloud
        :param points: [B, 3, N] input points
        '''
        if self.opt.debug_3d:
            print(vox.shape)
            verts, faces, normals, values = measure.marching_cubes_lewiner(vox[0].squeeze(0).cpu().numpy(), 0.5)
            verts /= 128
            verts[:, 0] -= 0.5
            verts[:, 2] -= 0.5
            save_obj_mesh('show/conv3d.obj', verts, faces)
            # ex_points = extrinsic[:, 0, :, :] @ points
            # result = self.index_3d(vox, ex_points)
            # show_pts = ex_points[0].transpose(0, 1).cpu().numpy()
            # show_prob = result[0].transpose(0, 1).cpu().numpy()
            # save_samples_truncted_prob('show/conv3d.ply', show_pts, show_prob)
            exit(0)
        self.feature_3d = self.conv_filter3d(vox)
    
    def filter_normal(self, normals):
        self.normals = normals
        self.mv_nfeat_list = []
        for view in range(self.num_views):
            nm_feat_list, self.tmpx, self.normx = self.normal_filter_hr(normals[:, view, :, :, :]) # im_feat_list [Level, B, C, H, W]
            B, C, H, W = nm_feat_list[0].shape
            nm_feat_list = torch.cat(nm_feat_list, dim=0).reshape(-1, B, C, H, W)
            self.mv_nfeat_list.append(nm_feat_list.unsqueeze(0))
        self.mv_nfeat_list = torch.cat(self.mv_nfeat_list, dim=0) # [V, L, B, C, H, W]
        self.mv_nfeat_list = self.mv_nfeat_list.permute(1, 2, 0, 3, 4, 5) # [L, B, V, C, H, W]

    def filter2d(self, images):
        '''
        Filter the input images
        store all intermediate features.
        :param images: [B, views, C, H, W] input images
        '''
        self.images = images
        self.mv_feat_list = []
        for view in range(self.num_views):
            im_feat_list, self.tmpx, self.normx = self.image_filter(images[:, view, :, :, :]) # im_feat_list [Level, B, C, H, W]
            B, C, H, W = im_feat_list[0].shape
            im_feat_list = torch.cat(im_feat_list, dim=0).reshape(-1, B, C, H, W)
            self.mv_feat_list.append(im_feat_list.unsqueeze(0))
        self.mv_feat_list = torch.cat(self.mv_feat_list, dim=0) # [V, L, B, C, H, W]
        self.mv_feat_list = self.mv_feat_list.permute(1, 2, 0, 3, 4, 5) # [L, B, V, C, H, W]
        # If it is not in training, only produce the last im_feat
        if not self.training:
            self.mv_feat_list = self.mv_feat_list[-1:]
    
    def img_feature(self, points, feature):
        xy = points
        feature2d_list = []
        for level in range(feature.shape[0]):
            feature2d = []
            for view in range(self.num_views): 
                feature2d.append(self.index(feature[level, :, view, :, :, :], xy[:, view, :, :], size=self.opt.loadSize).unsqueeze(1))
            feature2d = torch.cat(feature2d, dim=1)
            feature2d_list.append(feature2d.unsqueeze(0))
        return torch.cat(feature2d_list, dim=0)

    def feature_2d_fusion(self, feature):
        # simply add
        # [L, B, V, C, N]
        return torch.mean(feature, dim=2)

    def index_3d(self, feature, points):
        sample_pts = points.clone()
        sample_pts[:, 0, :] *= 2
        sample_pts[:, 2, :] *= 2
        sample_pts[:, 1, :] = (sample_pts[:, 1, :] - 0.5) * 2
        return F.grid_sample(feature.permute(0, 1, 4, 3, 2), sample_pts.permute(0, 2, 1).unsqueeze(2).unsqueeze(2), mode='bilinear')
    
    def attention(self, feat, feature_fusion):
        att_feat = torch.zeros_like(feat[-1:])
        num_views = self.num_views
        
        for view in range(num_views):
            att_feat[-1, :, view] = feat[-1, :, view]
        att_feat = att_feat.permute(0, 1, 4, 2, 3).contiguous().reshape(-1, num_views, feat.shape[3])
        att_feat, = feature_fusion(att_feat)
        _, B, V, D, N = feat.shape
        att_feat = att_feat.reshape(-1, B, N, V, D).permute(0, 1, 3, 4, 2)
        return att_feat

    def query(self, points, calibs, extrinsic, transforms=None, labels=None):
        '''
        Given 3D points, query the network predictions for each point.
        Image features should be pre-computed before this call.
        store all intermediate features.
        query() function may behave differently during training/testing.
        :param points: [B, 3, N] world space coordinates of points
        :param calibs: [B, 3, 4] calibration matrices for each image
        :param transforms: Optional [B, 2, 3] image space coordinate transforms
        :param labels: Optional [B, Res, N] gt labeling
        :return: [B, Res, N] predictions for each point
        '''
        if labels is not None:
            self.labels = labels
        # 2d part
        xyz = []
        for view in range(self.num_views):
            xyz.append(self.projection(points, calibs[:, view, :, :], transforms).unsqueeze(1))
        xyz = torch.cat(xyz, dim=1)
        pts_xy = xyz[:, :, :2, :]
        
        pts_2d_feature = self.img_feature(pts_xy, self.mv_feat_list)
        
        if self.opt.fine_part:
            pts_normal_feature = self.img_feature(pts_xy, self.mv_nfeat_list)

        # 3d part
        ex_points = points.clone()

        ex_points -= self.center.reshape(ex_points.shape[0], 3, 1)
        ex_points *= self.scale.reshape(ex_points.shape[0], 1, 1).repeat(1, 3, 1)
        ex_points += ((torch.FloatTensor([-0.5, 0, -0.5]) + torch.FloatTensor([0.5, 1, 0.5])) / 2).reshape(1, 3, 1).to(ex_points.device)

        ex_points[:, 1, :] -= 0.5
        ex_points = extrinsic[:, 0, :, :] @ ex_points
        ex_points[:, 1, :] += 0.5

        points_feature = self.index_3d(self.feature_3d, ex_points).squeeze(3).squeeze(3)
        xyz_feat = max_min_norm(ex_points, [-0.5, 0, -0.5], [0.5, 1, 0.5])
        # if self.opt.wo_xyz:
        #     xyz_feat *= 0

        self.intermediate_preds_list = []
        self.fine_preds_list = []
        
        geo_feature = []
        for level in range(pts_2d_feature.shape[0]):
            g_feature = torch.cat([pts_2d_feature[level, :], points_feature.unsqueeze(1).repeat(1, pts_2d_feature.shape[2], 1, 1)], dim=2)
            geo_feature.append(g_feature.unsqueeze(0))
        geo_feature = torch.cat(geo_feature, dim=0)
        
        if self.opt.coarse_part:
            geo_feature = self.attention(geo_feature, self.feature_fusion)
            for level in range(geo_feature.shape[0]):
                # [B, Feat_i + Feat_3d + xyz, N]
                point_local_feat_list = [geo_feature[level, :, 0], xyz_feat]
                point_local_feat = torch.cat(point_local_feat_list, 1)

                # out of image plane is always set to 0
                pred = self.surface_classifier(point_local_feat)
                self.intermediate_preds_list.append(pred)
                
                if self.opt.preserve_single:
                    self.sg_preds_list = []
                    for view in range(1, self.num_views):
                        feat_list = [geo_feature[level, :, view], xyz_feat]
                        feat = torch.cat(feat_list, 1)
                        pred = self.surface_classifier(feat)
                        self.sg_preds_list.append(pred)
            self.preds = self.intermediate_preds_list[-1]
        if self.opt.fine_part:
            self.fine_sg_preds_list = []
            
            with torch.no_grad():
                geo_feature = self.attention(geo_feature, self.feature_fusion)
            geo_fine_feature = torch.cat([geo_feature, pts_normal_feature], dim=3)
            geo_fine_feature = self.attention(geo_fine_feature, self.feature_fusion_n)

            for level in range(geo_feature.shape[0]):
                # [B, Feat_i + Feat_3d + xyz, N]
                point_local_feat_list = [geo_fine_feature[level, :, 0], xyz_feat]
                point_local_feat = torch.cat(point_local_feat_list, 1)

                # out of image plane is always set to 0
                pred = self.fine_sc(point_local_feat)
                self.fine_preds_list.append(pred)
                
                if self.opt.preserve_single:
                    self.sg_preds_list = []
                    for view in range(1, self.num_views):
                        feat_list = [geo_fine_feature[level, :, view], xyz_feat]
                        feat = torch.cat(feat_list, 1)
                        pred = self.fine_sc(feat)
                        self.sg_preds_list.append(pred)
            self.preds = self.fine_preds_list[-1]

    def get_im_feat(self):
        '''
        Get the image filter
        :return: [B, C_feat, H, W] image feature after filtering
        '''
        return self.im_feat_list[-1]

    def get_error(self):
        '''
        Hourglass has its own intermediate supervision scheme
        '''
        error = 0
        error_fine = 0
        error_sg = 0
        if self.opt.fine_part:
            for preds in self.fine_preds_list:
                error_fine += self.error_func(preds, self.labels)
            error_fine /= len(self.fine_preds_list)
        if self.opt.coarse_part:
            for preds in self.intermediate_preds_list:
                error += self.error_func(preds, self.labels)
            error /= len(self.intermediate_preds_list)
        if self.opt.preserve_single:
            for preds in self.sg_preds_list:
                error_sg += self.error_func(preds, self.labels)
            for preds in self.fine_sg_preds_list:
                error_sg += self.error_func(preds, self.labels)
            if (len(self.sg_preds_list) + len(self.fine_sg_preds_list)) > 0:
                error_sg /= (len(self.sg_preds_list) + len(self.fine_sg_preds_list))
        return error + error_fine + error_sg * 0.3

    def forward(self, data):
        # Get data
        images, normals = data["image"], data["normal"]
        vox = data["vox"]
        smpl_normal = data["smpl_normal"]
        points, calibs, extrinsic = data["samples"], data["calib"], data["extrinsic"]
        labels = data["labels"]
        self.mask_init(data["mask"], data["ero_mask"])
        self.norm_init(data["scale"], data["center"])

        images = torch.cat([images, smpl_normal], dim=2)
        normals = torch.cat([normals, smpl_normal], dim=2)

        # encode
        if self.opt.fine_part:
            with torch.no_grad():
                self.smpl_init(smpl_normal)
                self.filter3d(vox)
                self.filter2d(images)
            self.filter_normal(normals)
        elif self.opt.coarse_part:
            self.smpl_init(smpl_normal)
            self.filter3d(vox)
            self.filter2d(images)

        # point query
        self.query(points=points, calibs=calibs, extrinsic=extrinsic, labels=labels)

        # get the prediction
        res = self.get_preds()

        # get the error
        error = self.get_error()

        return res, error