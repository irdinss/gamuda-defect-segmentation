import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)

        targets_one_hot = F.one_hot(
            targets,
            num_classes=logits.shape[1]
        )

        targets_one_hot = (
            targets_one_hot
            .permute(0, 3, 1, 2)
            .float()
        )

        intersection = (probs * targets_one_hot).sum(
            dim=(2, 3)
        )

        union = (
            probs.sum(dim=(2, 3))
            + targets_one_hot.sum(dim=(2, 3))
        )

        dice = (
            2 * intersection + self.smooth
        ) / (
            union + self.smooth
        )

        return 1 - dice.mean()
    
class CombinedSegmentationLoss(nn.Module):

    def __init__(self, class_weights):
        super().__init__()

        self.dice = DiceLoss()

        self.ce = nn.CrossEntropyLoss(
            weight=torch.tensor(
                class_weights,
                dtype=torch.float32
            )
        )

    def forward(self, logits, targets):

        dice_loss = self.dice(
            logits,
            targets
        )

        ce_loss = self.ce(
            logits,
            targets
        )

        return dice_loss + ce_loss