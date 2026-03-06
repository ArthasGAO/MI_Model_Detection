# --- FitNet Helper Class ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from KnowledgeDistillation.base_distiller import Distiller


class ConvReg(nn.Module):
    """1x1 Convolution to align Student channel dimensions to Teacher channel dimensions."""
    def __init__(self, s_channels, t_channels):
        super(ConvReg, self).__init__()
        self.conv = nn.Conv2d(s_channels, t_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(t_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


# --- FitNet Distiller Class ---
class FitNet(Distiller):
    """FitNets: Hints for Thin Deep Nets"""
    def __init__(self, student, teacher, ce_weight=1.0, feat_weight=100.0, hint_layer=2, input_size=(32, 32)):
        super(FitNet, self).__init__(student, teacher)
        self.ce_weight = ce_weight
        self.feat_weight = feat_weight
        self.hint_layer = hint_layer

        # Dynamically determine the channel sizes using a dummy tensor
        # This replaces the need for the external get_feat_shapes function

        self.student.eval()

        device = next(teacher.parameters()).device
        dummy_img = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
        
        with torch.no_grad():
            _, feat_s = self.student(dummy_img)
            _, feat_t = self.teacher(dummy_img)
            
        s_channels = feat_s["feats"][self.hint_layer].shape[1]
        t_channels = feat_t["feats"][self.hint_layer].shape[1]

        self.conv_reg = ConvReg(s_channels, t_channels).to(device)

    def get_learnable_parameters(self):
        # Must include the conv_reg parameters so the optimizer updates them!
        return super().get_learnable_parameters() + list(self.conv_reg.parameters())

    def get_extra_parameters(self):
        num_p = sum(p.numel() for p in self.conv_reg.parameters())
        return num_p

    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            _, feature_teacher = self.teacher(image)

        loss_ce = self.ce_weight * F.cross_entropy(logits_student, target)
        
        # Project student features to match teacher features
        f_s = self.conv_reg(feature_student["feats"][self.hint_layer])
        
        loss_feat = self.feat_weight * F.mse_loss(
            f_s, feature_teacher["feats"][self.hint_layer]
        )
        
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_feat,
        }
        return logits_student, losses_dict