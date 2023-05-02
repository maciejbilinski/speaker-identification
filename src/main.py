from pytorch_lightning import Trainer
import torch
from data_module import AudioDataModule
from model import EmbeddingModel

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium' | 'high')
    
    dm = AudioDataModule(batch_size=32)
    model = EmbeddingModel(embedding_dim=256)
    trainer = Trainer(max_epochs=10)
    trainer.fit(model, dm)