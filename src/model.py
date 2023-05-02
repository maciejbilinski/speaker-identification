from pytorch_lightning import LightningModule
import torchvision.models as models
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

class EmbeddingModel(LightningModule):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.net = models.mobilenet_v3_small(weight=None, num_of_classes=embedding_dim)
        self.net.features = nn.Sequential(nn.Conv2d(1, 3, kernel_size=1), *list(self.net.features))
        self.net.classifier = nn.Sequential(*list(self.net.classifier), nn.Linear(1000, embedding_dim))

    def forward(self, x):
        x = self.net(x)
        return x

    def training_step(self, batch, batch_idx):
        anchor_spectrogram, positive_spectrogram, negative_spectrogram = batch
        anchor_embedding = self(anchor_spectrogram)
        positive_embedding = self(positive_spectrogram)
        negative_embedding = self(negative_spectrogram)
        loss = F.triplet_margin_loss(anchor_embedding, positive_embedding, negative_embedding, margin=1.0)
        self.log("my_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.net.parameters(), lr=0.001)
        return optimizer