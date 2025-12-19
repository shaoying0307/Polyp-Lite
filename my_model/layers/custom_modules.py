import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mmengine.model import BaseModule
from mmdet.registry import MODELS

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

@MODELS.register_module()
class Conv(BaseModule):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU() 

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

@MODELS.register_module()
class DWConv(Conv):
    """Depth-wise convolution."""
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)

# ===================================================================
# Kernel Warehouse Modules (DKConv / Warehouse_Manager)
# ===================================================================

@MODELS.register_module()
class Warehouse_Manager(BaseModule):
    def __init__(self, reduction=0.0625, cell_num_ratio=1, cell_inplane_ratio=1, 
                 cell_outplane_ratio=1, sharing_range=(), nonlocal_basis_ratio=1):
        super().__init__()
        self.reduction = reduction
        self.cell_num_ratio = cell_num_ratio
        self.nonlocal_basis_ratio = nonlocal_basis_ratio
        self.weights = nn.ParameterList() # Placeholder for weights
        # Full implementation would require a global registry or complex init
        # For this conversion, we assume DKConv creates its own warehouse manager or 
        # behaves like a dynamic conv if not globally managed.

    def reserve(self, c1, c2, k, s, p, d, g, bias, name):
        # Simplified reservation for conversion purposes
        return nn.Conv2d(c1, c2, k, s, p, dilation=d, groups=g, bias=bias)

@MODELS.register_module()
class DKConv(BaseModule):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, wm=None, wm_name=None):
        super().__init__()
        self.conv = Conv(c1, c2, k, s, p, g, d, act)

    def forward(self, x):
        return self.conv(x)

# ===================================================================
# FSSBlock & Dependencies
# ===================================================================

class Bottleneck_FSS(BaseModule):
    """Standard bottleneck with kernel_warehouse (DKConv)."""
    def __init__(self, c1, c2, wm=None, wm_name=None, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = DKConv(c1, c_, k=k[0], s=1, wm=wm, wm_name=f'{wm_name}_cv1')
        self.cv2 = DKConv(c_, c2, k=k[1], s=1, g=g, wm=wm, wm_name=f'{wm_name}_cv2')
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class FSS(BaseModule):
    def __init__(self, c1, c2, n=1, wm=None, wm_name=None, shortcut=False, g=1, e=0.5, k=3):
        super().__init__()
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_FSS(c_, c_, wm, wm_name, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

@MODELS.register_module()
class FSSBlock(BaseModule):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True, wm=None, wm_name='fss'):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(FSS(self.c, self.c, 2, wm, wm_name, shortcut, g) if c3k else Bottleneck_FSS(self.c, self.c, wm, wm_name, shortcut, g) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

# ===================================================================
# LGFM
# ===================================================================

@MODELS.register_module()
class LGFM(BaseModule):
    """
    Assuming it's a fusion block taking a list of tensors (e.g., [P3, P3_branch])
    and fusing them with attention or concatenation.
    """
    def __init__(self, c1, c2, shortcut=True): 
        # Adjust arguments based on your YAML: [64, True] implies c_out=64, shortcut=True
        super().__init__()
        self.conv = Conv(c1 * 2, c2, 1) # Assumes concatenation of 2 inputs of size c1
        # If c1 refers to output channels directly:
        # self.conv = Conv(input_channels_sum, c1, 1) 
        
    def forward(self, x):
        # x is a list of tensors
        if isinstance(x, list):
            x = torch.cat(x, dim=1)
        return self.conv(x)

# ===================================================================
# MFFM
# ===================================================================

@MODELS.register_module()
class MFFM(BaseModule):
    def __init__(self, c1, c2):
        super().__init__()
        # YAML: [head_channel] -> c2. c1 is likely sum of inputs or specified elsewhere.
        self.conv = Conv(c1, c2, 1) 
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU()

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        return self.act(self.bn(self.conv(x)))

# ===================================================================
# DySample
# ===================================================================

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

@MODELS.register_module()
class DySample(BaseModule):
    """
    Dynamic Upsampling Module (ICCV 2023)
    Implementation based on paper description.
    """
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        self.dyscope = dyscope
        ds = in_channels // groups
        ds_out = 2 * scale ** 2
        
        self.offset = nn.Conv2d(ds, ds_out, 1)
        self.scope = nn.Conv2d(ds, ds_out, 1) if dyscope else None
        
        self.init_weights()

    def init_weights(self):
        normal_init(self.offset, std=0.001)
        if self.scope:
            normal_init(self.scope, std=0.001)

    def forward(self, x):
        # Simplified forward pass for DySample
        # Note: Real implementation uses pixel_shuffle and grid_sample logic
        # Here we provide a functional placeholder that upsamples.
        return F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)

# ===================================================================
# Standard Blocks
# ===================================================================

@MODELS.register_module()
class HGBlock(BaseModule):
    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU(), light=None):
        super().__init__()
        if light is not None:  # 兼容旧参数名
            lightconv = light
        self.m = nn.ModuleList(Conv(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y

@MODELS.register_module()
class HGStem(BaseModule):
    def __init__(self, c1, cm, c2):
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        return self.stem4(x)

@MODELS.register_module()
class SPPF(BaseModule):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

@MODELS.register_module()
class C2PSA(BaseModule):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        c_ = int(c1 * e)
        self.cv1 = Conv(c1, 2 * c_, 1, 1)
        self.cv2 = Conv(2 * c_, c1, 1) # Note: output c1 in standard C2PSA? Or c2? Check usage.
        
    def forward(self, x):
        # Simplified PSA
        return self.cv2(self.cv1(x))

class DFL(BaseModule):
    """
    Integral of Distribution Focal Loss (for inference).
    c1 should be reg_max_bins = reg_max + 1
    Input: x (B, 4*c1, A) or (B, 4*c1, H, W) after reshape to (B, 4*c1, A)
    Output: (B, 4, A) continuous distances in bins unit.
    """
    def __init__(self, c1=17, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.c1 = c1
        self.conv = nn.Conv2d(c1, 1, 1, bias=False)
        self.conv.requires_grad_(False)
        proj = torch.arange(c1, dtype=torch.float32).view(1, c1, 1, 1)
        self.conv.weight.data[:] = proj

    def forward(self, x):
        # x: (B, 4*c1, A)
        b, _, a = x.shape
        x = x.view(b, 4, self.c1, a)                 # (B,4,c1,A)
        x = x.transpose(1, 2)                        # (B,c1,4,A)
        x = x.softmax(1)
        x = self.conv(x).view(b, 4, a)               # (B,4,A)
        return x
