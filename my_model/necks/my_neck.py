import torch
import torch.nn as nn
from mmdet.registry import MODELS
from mmengine.model import BaseModule
from ..layers.custom_modules import Conv, MFFM, FSSBlock, DySample


@MODELS.register_module()
class MFFMEncoder(BaseModule):
    def __init__(self,
                 in_channels=(64, 128, 256),
                 out_channels=256,        # 兼容 config 里的 out_channels
                 head_channel=None,       # 保留你原来的写法（可选）
                 nc=80,                   # 暂时不用也没关系
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)

        c3, c4, c5 = in_channels
        ch = out_channels if head_channel is None else head_channel

        # P5 / P4 / P3 通道对齐到 ch
        self.conv24 = Conv(c5, ch, 1, 1)      # P5 -> ch
        self.conv23 = Conv(c4, ch, 1, 1)      # P4 -> ch
        self.conv22 = Conv(c3, ch, 1, 1)      # P3 -> ch

        # 下采样/上采样
        self.conv25 = Conv(ch, ch, 3, 2)
        self.conv29 = Conv(ch, ch, 3, 2)

        self.dysample28 = DySample(ch, 2, 'lp')
        self.dysample32 = DySample(ch, 2, 'lp')

        # 关键：MFFM 输入通道必须等于 cat 后的通道数
        # 2 路融合 => 2*ch；3 路融合 => 3*ch
        self.mfm26 = MFFM(2 * ch, ch)         # [x25, c_p5]
        self.fss27 = FSSBlock(ch, ch, n=3)

        self.mfm30 = MFFM(3 * ch, ch)         # [x28, x29, c_p4]
        self.fss31 = FSSBlock(ch, ch, n=3)

        self.mfm34 = MFFM(2 * ch, ch)         # [x32, c_p3]
        self.fss35 = FSSBlock(ch, ch, n=3)

    def forward(self, inputs):
        # inputs: (P3, P4, P5) from backbone
        p3, p4, p5 = inputs

        c_p3 = self.conv22(p3)
        c_p4 = self.conv23(p4)
        c_p5 = self.conv24(p5)

        # High level (P5)
        x25 = self.conv25(c_p4)
        x26 = self.mfm26([x25, c_p5])
        x27 = self.fss27(x26)                # P5_out (ch)

        # Mid level (P4)
        x28 = self.dysample28(x27)
        x29 = self.conv29(c_p3)
        x30 = self.mfm30([x28, x29, c_p4])
        x31 = self.fss31(x30)                # P4_out (ch)

        # Low level (P3)
        x32 = self.dysample32(x31)
        x34 = self.mfm34([x32, c_p3])
        x35 = self.fss35(x34)                # P3_out (ch)

        # 返回给 Head：P3, P4, P5
        return (x35, x31, x27)
