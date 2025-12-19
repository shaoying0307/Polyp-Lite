import torch
import torch.nn as nn
from mmdet.registry import MODELS
from mmengine.model import BaseModule
from ..layers.custom_modules import HGStem, HGBlock, FSSBlock, DKConv, LGFM, Conv

@MODELS.register_module()
class DualFusionNet(BaseModule):
    def __init__(self, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        # 定义通道数 (根据 YAML 计算)
        # P2/4: 32, P3/8: 64, P4/16: 128, P5/32: 256 (假设各层通道数)
        
        # 0. Stem
        self.layer0 = HGStem(3, 16, 32) # [-1, 1, HGStem, [16, 32]]
        
        # Branch 1 (Upper Path in YAML)
        self.layer1 = FSSBlock(32, 32, n=2, shortcut=False, e=0.25) # [0, 2, C3k2_KW]
        self.layer2 = DKConv(32, 64, k=3, s=2) # [-1, 1, KWConv]
        self.layer3 = FSSBlock(64, 64, n=2, shortcut=False, e=0.25) 
        
        # Branch 2 (HGBlock Path)
        self.layer4 = HGBlock(32, 16, 32, k=3)    # c1=32, cm=16, c2=32
        self.layer5 = Conv(32, 64, k=3, s=2, g=1, act=False) # [-1, 1, DWConv...] 注意这里用 Conv 模拟 DWConv 如果参数一致
        self.layer6 = HGBlock(64, 32, 64, k=3)    # c1=64, cm=32, c2=64
        
        # Merge P3
        self.layer7 = LGFM(64, 64) # [[3, 6], 1, HAFB, [64, True]] (假设输出64)
        
        # P4 Part
        self.layer8 = DKConv(64, 128, k=3, s=2)
        self.layer9 = FSSBlock(128, 128, n=2, shortcut=True)
        
        self.layer10 = Conv(64, 128, k=3, s=2, g=1, act=False) # DWConv
        self.layer11 = HGBlock(128, 64, 128, k=5, light=True, shortcut=False)
        self.layer12 = HGBlock(128, 64, 128, k=5, light=True, shortcut=True)
        self.layer13 = HGBlock(128, 64, 128, k=5, light=True, shortcut=True)
        
        # Merge P4
        self.layer14 = LGFM(128, 128) # [[9, 13], 1, HAFB]
        
        # P5 Part
        self.layer15 = DKConv(128, 256, k=3, s=2)
        self.layer16 = FSSBlock(256, 256, n=2, shortcut=True)
        
        self.layer17 = Conv(128, 256, k=3, s=2, g=1, act=False)
        self.layer18 = HGBlock(256, 128, 256, k=5, light=True, shortcut=False)
        
        # Merge P5
        self.layer19 = LGFM(256, 256) # [[16, 18], 1, HAFB]
        
        # SPP & PSA
        from ..layers.custom_modules import SPPF, C2PSA # 假设你有 C2PSA
        self.layer20 = SPPF(256, 256, k=5)
        self.layer21 = C2PSA(256, 256) # 需要实现 C2PSA

    def forward(self, x):
        # 模拟 YAML 的执行流
        x0 = self.layer0(x) # P2
        
        # P3 Branching
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        
        x4 = self.layer4(x0)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)
        
        x7 = self.layer7([x3, x6]) # P3 Output
        
        # P4 Branching
        x8 = self.layer8(x7)
        x9 = self.layer9(x8)
        
        x10 = self.layer10(x7)
        x11 = self.layer11(x10)
        x12 = self.layer12(x11)
        x13 = self.layer13(x12)
        
        x14 = self.layer14([x9, x13]) # P4 Output
        
        # P5 Branching
        x15 = self.layer15(x14)
        x16 = self.layer16(x15)
        
        x17 = self.layer17(x14)
        x18 = self.layer18(x17)
        
        x19 = self.layer19([x16, x18])
        x20 = self.layer20(x19)
        x21 = self.layer21(x20) # P5 Output
        
        # 返回 MFFM Encoder 需要的特征层: P3, P4, P5
        return (x7, x14, x21)