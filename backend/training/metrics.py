import torch

def compute_per_class_iou(
    predictions,
    targets,
    num_classes,
):
    ious = []

    for class_id in range(num_classes):

        pred_mask = (
            predictions == class_id
        )

        target_mask = (
            targets == class_id
        )

        intersection = (
            pred_mask & target_mask
        ).sum().item()

        union = (
            pred_mask | target_mask
        ).sum().item()

        if union == 0:
            iou = float("nan")
        else:
            iou = (
                intersection / union
            )

        ious.append(iou)

    return ious

# Compute mean IoU across all classes, ignoring NaN values
def compute_miou(
    predictions,
    targets,
    num_classes,
):
    per_class_iou = (
        compute_per_class_iou(
            predictions,
            targets,
            num_classes,
        )
    )

    valid_ious = [
        iou
        for iou in per_class_iou
        if not torch.isnan(
            torch.tensor(iou)
        )
    ]

    if len(valid_ious) == 0:
        return 0.0

    return sum(valid_ious) / len(valid_ious)