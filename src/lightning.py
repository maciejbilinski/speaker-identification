import time
from pytorch_lightning import Trainer, callbacks
import torch
from common_voice_data_module import CommonVoiceModule
from mobilenet import MobileNet

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    
    dm = CommonVoiceModule()
    model = MobileNet()
    trainer = Trainer(
        max_epochs=100,
        callbacks=[
            callbacks.ModelCheckpoint(
                dirpath=f'checkpoints_{time.time():.0f}',
                filename='model-{epoch:02d}-{val_loss:.2f}',
                save_top_k=3,
                monitor='val_loss',
                mode='min'
            ),
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                mode='min'
            ),
        ]
    )
    trainer.fit(model, dm)