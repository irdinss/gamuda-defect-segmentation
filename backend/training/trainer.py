import torch
import torch.nn.functional as F
from tqdm import tqdm

from backend.training.metrics import (
    compute_miou,
)

class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        train_loader,
        valid_loader,
        device,
        num_classes=5,
    ):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.device = device
        self.num_classes = num_classes

    def train_epoch(self):
        self.model.train()

        total_loss = 0.0

        for images, masks in tqdm(self.train_loader):

            images = images.to(
                self.device
            )

            masks = masks.to(
                self.device
            )

            outputs = self.model(
                pixel_values=images
            )

            logits = outputs.logits

            logits = F.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            loss = self.criterion(
                logits,
                masks,
            )

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

        return (
            total_loss
            / len(self.train_loader)
        )
    
    
    @torch.no_grad()
    def validate_epoch(self):

        self.model.eval()

        total_loss = 0.0

        total_miou = 0.0

        for images, masks in self.valid_loader:

            images = images.to(
                self.device
            )

            masks = masks.to(
                self.device
            )

            outputs = self.model(
                pixel_values=images
            )

            logits = outputs.logits

            logits = F.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            loss = self.criterion(
                logits,
                masks,
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            miou = compute_miou(
                predictions.cpu(),
                masks.cpu(),
                num_classes=self.num_classes,
            )

            total_loss += loss.item()

            total_miou += miou

        return (
            total_loss
            / len(self.valid_loader),
            total_miou
            / len(self.valid_loader),
        )
    
