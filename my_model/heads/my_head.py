import math
import torch
import torch.nn as nn

from mmengine.structures import InstanceData
from torchvision.ops import nms  # 用 torchvision 的 nms，避免 mmcv.ops 编译问题

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.models.task_modules.prior_generators import MlvlPointGenerator
from mmdet.models.task_modules.samplers import PseudoSampler
from mmdet.models.task_modules.coders.distance_point_bbox_coder import DistancePointBBoxCoder
from mmdet.models.utils import multi_apply

from ..layers.custom_modules import Conv, DFL


def bbox_iou_xyxy(b1, b2, eps=1e-7):
    """IoU for xyxy boxes, b1/b2: (N,4)."""
    tl = torch.maximum(b1[:, :2], b2[:, :2])
    br = torch.minimum(b1[:, 2:], b2[:, 2:])
    wh = (br - tl).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    area1 = (b1[:, 2] - b1[:, 0]).clamp(min=0) * (b1[:, 3] - b1[:, 1]).clamp(min=0)
    area2 = (b2[:, 2] - b2[:, 0]).clamp(min=0) * (b2[:, 3] - b2[:, 1]).clamp(min=0)
    union = area1 + area2 - inter + eps
    return inter / union


@MODELS.register_module()
class LightDecoder(BaseDenseHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 reg_max=16,  # reg_max 是最大整数距离上界，bins = reg_max + 1
                 train_cfg=None,
                 test_cfg=None,
                 loss_cls=dict(
                     type='QualityFocalLoss',
                     use_sigmoid=True,
                     beta=2.0,
                     loss_weight=1.0),
                 loss_bbox=dict(type='Focaler_DIoU', loss_weight=2.0),
                 loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.num_classes = num_classes
        self.in_channels = in_channels

        self.reg_max = int(reg_max)
        self.reg_max_bins = self.reg_max + 1  # 关键：DFL bins

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # 3 层输出对应 strides
        self.prior_generator = MlvlPointGenerator([8, 16, 32], offset=0.0)

        # losses
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_dfl = MODELS.build(loss_dfl)

        # assigner / sampler
        if self.train_cfg is not None:
            self.assigner = TASK_UTILS.build(self.train_cfg.assigner)
            if self.train_cfg.get('sampler', None) is not None:
                self.sampler = TASK_UTILS.build(
                    self.train_cfg.sampler, default_args=dict(context=self)
                )
            else:
                self.sampler = PseudoSampler(context=self)

        self.bbox_coder = DistancePointBBoxCoder()

        self._init_layers()

    def _init_layers(self):
        self.stems = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_convs = nn.ModuleList()

        for c in self.in_channels:
            self.stems.append(
                nn.Sequential(
                    Conv(c, c, 3, g=max(1, c // 16)),
                    Conv(c, c, 3, g=max(1, c // 16))
                )
            )
            # 关键：输出通道必须是 4*(reg_max+1)
            self.reg_convs.append(nn.Conv2d(c, 4 * self.reg_max_bins, 1))
            self.cls_convs.append(nn.Conv2d(c, self.num_classes, 1))

        # 推理用（训练 DFL loss 不靠它）
        self.dfl = DFL(self.reg_max_bins) if self.reg_max_bins > 1 else nn.Identity()

    def init_weights(self):
        for reg_conv, cls_conv, stride in zip(self.reg_convs, self.cls_convs, self.prior_generator.strides):
            stride_val = stride[0] if isinstance(stride, (tuple, list)) else stride
            reg_conv.bias.data[:] = 1.0
            cls_conv.bias.data[:self.num_classes] = math.log(5 / self.num_classes / (640 / stride_val) ** 2)

    def forward(self, feats):
        # 注意：第二个输出 bbox_preds 这里仍返回 None（list[None]），但我们会重写 predict_by_feat 忽略它
        return multi_apply(self.forward_single, feats, self.stems, self.reg_convs, self.cls_convs)

    def forward_single(self, x, stem, reg_conv, cls_conv):
        x = stem(x)
        reg_dist_pred = reg_conv(x)  # (B, 4*(bins), H, W)
        cls_score = cls_conv(x)      # (B, C, H, W)
        return cls_score, None, reg_dist_pred

    def loss_by_feat(self,
                     cls_scores,
                     bbox_preds,
                     reg_dist_preds,
                     batch_gt_instances,
                     batch_img_metas,
                     batch_gt_instances_ignore=None):

        # --------- unpack gt ---------
        gt_bboxes = []
        gt_labels = []
        for inst in batch_gt_instances:
            b = inst.bboxes.tensor if hasattr(inst.bboxes, 'tensor') else inst.bboxes
            l = inst.labels
            gt_bboxes.append(b)
            gt_labels.append(l)

        num_imgs = len(batch_img_metas)
        featmap_sizes = [feat.size()[-2:] for feat in cls_scores]

        # priors: list[level] -> (n,4) [x,y,stride_w,stride_h]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes, device=cls_scores[0].device, with_stride=True
        )
        flatten_priors = torch.cat(mlvl_priors, dim=0)  # (N,4)
        num_priors = flatten_priors.size(0)

        # flatten preds
        flatten_cls_preds = torch.cat([
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes)
            for cls_pred in cls_scores
        ], dim=1)  # (B,N,C)

        flatten_reg_dist_preds = torch.cat([
            reg_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4 * self.reg_max_bins)
            for reg_pred in reg_dist_preds
        ], dim=1)  # (B,N,4*bins)

        # decode bbox (pixels)
        points = flatten_priors[:, :2]   # (N,2)
        strides = flatten_priors[:, 2:3] # (N,1)

        proj = torch.arange(self.reg_max_bins, device=flatten_reg_dist_preds.device, dtype=torch.float32)
        dist = flatten_reg_dist_preds.reshape(num_imgs, -1, 4, self.reg_max_bins).softmax(dim=3).matmul(proj)  # (B,N,4)
        dist = dist * strides.unsqueeze(0)  # 转像素

        flatten_bbox_preds = self.bbox_coder.decode(
            points.unsqueeze(0).expand(num_imgs, -1, 2).reshape(-1, 2),
            dist.reshape(-1, 4)
        ).reshape(num_imgs, -1, 4)  # (B,N,4)

        # targets per image
        priors_list = [flatten_priors for _ in range(num_imgs)]
        (labels_list, label_w_list, bbox_t_list, bbox_w_list,
         pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single,
            flatten_cls_preds.detach(),
            priors_list,
            flatten_bbox_preds.detach(),
            gt_bboxes,
            gt_labels
        )

        labels = torch.cat(labels_list, dim=0)          # (B*N,)
        label_weights = torch.cat(label_w_list, dim=0)  # (B*N,)
        bbox_targets = torch.cat(bbox_t_list, dim=0)    # (B*N,4)
        bbox_weights = torch.cat(bbox_w_list, dim=0)    # (B*N,)

        # global pos indices with offset
        pos_inds = []
        for i, p in enumerate(pos_inds_list):
            pos_inds.append(p + i * num_priors)
        pos_inds = torch.cat(pos_inds, dim=0) if len(pos_inds) else labels.new_zeros((0,), dtype=torch.long)

        # flatten to (B*N,...)
        cls_pred_flat = flatten_cls_preds.reshape(-1, self.num_classes)                 # (B*N,C)
        bbox_pred_flat = flatten_bbox_preds.reshape(-1, 4)                              # (B*N,4)
        dist_logits_flat = flatten_reg_dist_preds.reshape(-1, 4, self.reg_max_bins)     # (B*N,4,bins)

        # priors expanded to (B*N,4)
        priors_flat = flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1).reshape(-1, 4) # (B*N,4)
        points_flat = priors_flat[:, :2]
        strides_flat = priors_flat[:, 2:3].clamp(min=1.0)

        # 1) QFL
        quality = cls_pred_flat.new_zeros(labels.shape[0])
        if pos_inds.numel() > 0:
            iou = bbox_iou_xyxy(bbox_pred_flat[pos_inds].detach(), bbox_targets[pos_inds])
            quality[pos_inds] = iou

        avg_factor = max(float(pos_inds.numel()), 1.0)
        loss_cls = self.loss_cls(
            cls_pred_flat,
            (labels, quality),
            weight=label_weights,
            avg_factor=avg_factor
        )

        # 2) bbox loss
        if pos_inds.numel() == 0:
            loss_bbox = bbox_pred_flat.sum() * 0.0
        else:
            loss_bbox = self.loss_bbox(
                bbox_pred_flat[pos_inds],
                bbox_targets[pos_inds],
                weight=bbox_weights[pos_inds]
            )

        # 3) DFL loss (target 必须是 1D: (Npos*4,))
        if pos_inds.numel() == 0:
            loss_dfl = dist_logits_flat.sum() * 0.0
        else:
            pos_points = points_flat[pos_inds]             # (Npos,2)
            pos_strides = strides_flat[pos_inds]           # (Npos,1)
            pos_gt = bbox_targets[pos_inds]                # (Npos,4) xyxy

            l = (pos_points[:, 0] - pos_gt[:, 0]) / pos_strides[:, 0]
            t = (pos_points[:, 1] - pos_gt[:, 1]) / pos_strides[:, 0]
            r = (pos_gt[:, 2] - pos_points[:, 0]) / pos_strides[:, 0]
            b = (pos_gt[:, 3] - pos_points[:, 1]) / pos_strides[:, 0]

            dist_target = torch.stack([l, t, r, b], dim=1)  # (Npos,4)
            dist_target = dist_target.clamp(min=0.0, max=float(self.reg_max) - 1e-6)

            pred_dfl = dist_logits_flat[pos_inds].reshape(-1, self.reg_max_bins)  # (Npos*4,bins)
            target_dfl = dist_target.reshape(-1)                                  # (Npos*4,)

            w = bbox_weights[pos_inds].reshape(-1, 1).expand(-1, 4).reshape(-1)   # (Npos*4,)
            loss_dfl = self.loss_dfl(pred_dfl, target_dfl, weight=w)

        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox, loss_dfl=loss_dfl)

    # ============================
    # 关键：重写 predict_by_feat()
    # ============================
    def predict_by_feat(self,
                        cls_scores,
                        bbox_preds,
                        reg_dist_preds,
                        batch_img_metas,
                        cfg=None,
                        rescale=False,
                        **kwargs):
        """
        解决验证时报 'NoneType is not subscriptable'：
        BaseDenseHead 默认会用 bbox_preds 做解码，但你 bbox_preds 是 None。
        这里我们完全忽略 bbox_preds，用 reg_dist_preds 解码出 bbox，再 NMS。
        """
        if cfg is None:
            cfg = self.test_cfg

        score_thr = float(cfg.get('score_thr', 0.05))
        nms_pre = int(cfg.get('nms_pre', -1))
        max_per_img = int(cfg.get('max_per_img', 100))
        iou_thr = float(cfg.get('nms', {}).get('iou_threshold', 0.6))

        num_levels = len(cls_scores)
        assert num_levels == len(reg_dist_preds)

        featmap_sizes = [feat.size()[-2:] for feat in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes, device=cls_scores[0].device, with_stride=True
        )

        proj = torch.arange(self.reg_max_bins, device=cls_scores[0].device, dtype=torch.float32)

        results_list = []
        batch_size = cls_scores[0].shape[0]

        for img_id in range(batch_size):
            all_bboxes = []
            all_scores = []
            all_labels = []

            for lvl in range(num_levels):
                priors = mlvl_priors[lvl]            # (n,4)
                points = priors[:, :2]               # (n,2)
                strides = priors[:, 2:3].clamp(min=1.0)

                # cls
                cls = cls_scores[lvl][img_id].permute(1, 2, 0).reshape(-1, self.num_classes).sigmoid()  # (n,C)

                # reg logits -> dist -> bbox
                reg = reg_dist_preds[lvl][img_id].permute(1, 2, 0).reshape(-1, 4, self.reg_max_bins)    # (n,4,bins)
                dist = reg.softmax(dim=2).matmul(proj) * strides                                         # (n,4) pixels
                bboxes = self.bbox_coder.decode(points, dist)                                             # (n,4)

                # 单标签：取 max class
                scores, labels = cls.max(dim=1)  # (n,), (n,)

                all_bboxes.append(bboxes)
                all_scores.append(scores)
                all_labels.append(labels)

            bboxes = torch.cat(all_bboxes, dim=0)
            scores = torch.cat(all_scores, dim=0)
            labels = torch.cat(all_labels, dim=0)

            # nms_pre：先按分数取 TopK
            if nms_pre > 0 and scores.numel() > nms_pre:
                topk_scores, topk_inds = scores.topk(nms_pre)
                bboxes = bboxes[topk_inds]
                labels = labels[topk_inds]
                scores = topk_scores

            # score_thr
            keep = scores > score_thr
            bboxes = bboxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            if bboxes.numel() == 0:
                results = InstanceData()
                results.bboxes = bboxes.new_zeros((0, 4))
                results.scores = scores.new_zeros((0,))
                results.labels = labels.new_zeros((0,), dtype=torch.long)
                results_list.append(results)
                continue

            # NMS（这里做 class-agnostic；你是 1 类完全够用）
            keep_inds = nms(bboxes, scores, iou_thr)
            if keep_inds.numel() > max_per_img:
                keep_inds = keep_inds[:max_per_img]

            bboxes = bboxes[keep_inds]
            scores = scores[keep_inds]
            labels = labels[keep_inds]

            # rescale 回原图
            img_meta = batch_img_metas[img_id]
            if rescale and ('scale_factor' in img_meta) and (img_meta['scale_factor'] is not None):
                sf = img_meta['scale_factor']
                sf = bboxes.new_tensor(sf)

                # 兼容不同格式
                if sf.numel() == 1:
                    # 单个标量：扩展到 4
                    sf = sf.repeat(4)
                elif sf.numel() == 2:
                    # (w_scale, h_scale) -> (w_scale, h_scale, w_scale, h_scale)
                    sf = sf.repeat(2)
                elif sf.numel() == 4:
                    # 已经是 (w,h,w,h)
                    pass
                else:
                    raise ValueError(f'Unexpected scale_factor shape: {sf.shape}, value: {img_meta["scale_factor"]}')

                bboxes = bboxes / sf

            # clip
            if 'ori_shape' in img_meta and img_meta['ori_shape'] is not None:
                h, w = img_meta['ori_shape'][:2]
            else:
                h, w = img_meta['img_shape'][:2]
            bboxes[:, 0::2].clamp_(min=0, max=w)
            bboxes[:, 1::2].clamp_(min=0, max=h)

            results = InstanceData()
            results.bboxes = bboxes
            results.scores = scores
            results.labels = labels
            results_list.append(results)

        return results_list

    def _get_target_single(self, cls_preds, priors, decoded_bboxes, gt_bboxes, gt_labels):
        """
        cls_preds: (N,C)
        priors: (N,4) [x,y,stride_w,stride_h]
        decoded_bboxes: (N,4) xyxy
        gt_bboxes: (G,4) xyxy
        gt_labels: (G,)
        """
        num_priors = priors.size(0)
        num_gts = gt_bboxes.size(0)

        if num_gts == 0:
            return (torch.zeros((num_priors,), dtype=torch.long, device=priors.device),
                    torch.ones((num_priors,), dtype=torch.float32, device=priors.device),
                    torch.zeros((num_priors, 4), dtype=torch.float32, device=priors.device),
                    torch.zeros((num_priors,), dtype=torch.float32, device=priors.device),
                    torch.zeros(0, dtype=torch.long, device=priors.device),
                    torch.arange(num_priors, dtype=torch.long, device=priors.device))

        gt_instances = InstanceData()
        gt_instances.bboxes = gt_bboxes
        gt_instances.labels = gt_labels

        pred_instances = InstanceData()
        pred_instances.bboxes = decoded_bboxes
        pred_instances.priors = priors
        pred_instances.scores = cls_preds.sigmoid()

        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            gt_instances_ignore=None
        )

        sampling_result = self.sampler.sample(
            assign_result=assign_result,
            pred_instances=pred_instances,
            gt_instances=gt_instances
        )

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        labels = priors.new_full((num_priors,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = priors.new_ones(num_priors)

        bbox_targets = priors.new_zeros((num_priors, 4))
        bbox_weights = priors.new_zeros(num_priors)
        bbox_weights[pos_inds] = 1.0

        if pos_inds.numel() > 0:
            bbox_targets[pos_inds, :] = sampling_result.pos_gt_bboxes

        return labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds
