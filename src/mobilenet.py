from pytorch_lightning import LightningModule
import torch
import torchvision.models as models
from torch.optim import Adam
import torch.nn as nn
from batch_all_triplet_loss import BatchAllTripletLoss
from triplet_loss import TripletLoss
import torch.nn.functional as F

class MobileNet(LightningModule):
    def __init__(self, embedding_dim=None, learning_rate = 1e-3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.net = models.mobilenet_v3_small(weights=None)
        if embedding_dim is not None:
            self.linear = nn.Linear(1000, embedding_dim)
            
        # self.loss_fn = BatchAllTripletLoss()
        self.loss_fn = TripletLoss(None)
        self.lr = learning_rate

    def forward(self, x):
        x = self.net(torch.cat([x, x, x], 1))
        if self.embedding_dim is not None:
            x = nn.functional.relu(x)
            x = self.linear(x)

        return F.normalize(x, p=2, dim=-1)

    def training_step(self, batch, batch_idx):
        x, y = batch

        embedding = self.forward(x)
        loss = self.loss_fn(embedding, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch

        embedding = self.forward(x)
        loss = self.loss_fn(embedding, y)
        
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.net.parameters(), lr=self.lr)
        return optimizer