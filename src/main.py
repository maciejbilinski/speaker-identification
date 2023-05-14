import os
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision.models as models
from common_voice_data_module import CommonVoiceModule

from mobilenet import MobileNet


if __name__ == "__main__":
    num_of_workers = 0

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_float32_matmul_precision('high')
    else:
        device = torch.device("cpu")

    language = 'en'
    embedding_dim = 256

    batch_size = 128
    num_epochs = 100
    lr = 1e-4
    
    patience = 3
    best_val_loss = float('inf')
    early_stopping_counter = 0

    model = MobileNet(learning_rate=lr)
    model.net = model.net.to(device)
    model.loss_fn.device = device
    data = CommonVoiceModule(batch_size=batch_size, language=language)

    writer = SummaryWriter()
    pbar = tqdm()
    checkpoint_dir = f'checkpoints_{time.time()}'
    os.makedirs(checkpoint_dir)

    optimizer = model.configure_optimizers()

    for epoch in range(num_epochs):
        pbar.reset()
        pbar.total = len(data.train_dataset)
        model.train()

        train_loss = 0.0
        num_train_batches = 0

        train_dataloader = data.train_dataloader()

        for inputs, labels in train_dataloader:
            optimizer.zero_grad()

            inputs = inputs.to(device)
            labels = labels.to(device)

            loss = model.training_step((inputs, labels), None)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_train_batches += 1

            pbar.update(batch_size)
            pbar.set_description(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss / num_train_batches:.5f}")

        train_loss = train_loss / num_train_batches
        writer.add_scalar("Train/Loss", train_loss, epoch)

        model.eval()

        val_loss = 0.0
        num_val_batches = 0

        val_dataloader = data.val_dataloader()

        pbar.reset()
        pbar.total = len(data.val_dataset)
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                loss = model.validation_step((inputs, labels), None)

                val_loss += loss.item()
                num_val_batches += 1

                pbar.update(batch_size)
                pbar.set_description(f"Epoch {epoch + 1}/{num_epochs} | Val Loss: {val_loss / num_val_batches:.5f}")


        val_loss = val_loss / num_val_batches
        writer.add_scalar("Val/Loss", val_loss, epoch)

        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch+1}_{val_loss:.2f}.pt"))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            break

    writer.close()
    pbar.close()


