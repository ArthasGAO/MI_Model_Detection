import torch
import torch.nn as nn
import torch.nn.functional as F

from KnowledgeDistillation.base_distiller import Distiller

def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="batchmean")
    loss_kd *= temperature**2
    return loss_kd


# ==========================================
# 3. The KD Wrapper Class
# ==========================================
class KD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, temperature=4.0, ce_weight=1.0, kd_weight=1.0):
        super(KD, self).__init__(student, teacher)
        # Replaced the strict 'cfg' requirement with standard arguments 
        # for easier integration into your custom framework.
        self.temperature = temperature
        self.ce_weight = ce_weight
        self.kd_weight = kd_weight

    def forward_train(self, image, target, **kwargs):
        # 1. Get Student Logits
        logits_student, _ = self.student(image)
        
        # 2. Get Teacher Logits (safely without gradients)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # 3. Calculate Losses
        loss_ce = self.ce_weight * F.cross_entropy(logits_student, target)
        loss_kd = self.kd_weight * kd_loss(logits_student, logits_teacher, self.temperature)
        
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        
        return logits_student, losses_dict