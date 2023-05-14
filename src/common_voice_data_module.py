from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Audio
from torchaudio.transforms import MelSpectrogram
import torch.nn.functional as F

class CommonVoiceModule(LightningDataModule):
    def __init__(self, batch_size = 128, language = 'pl', spectrogram_n_mels = 64, sampling_rate = 16000, num_workers = 4):
        super().__init__()

        self._spectrogram = MelSpectrogram(sample_rate=sampling_rate, n_mels=spectrogram_n_mels)

        self.train_dataset = load_dataset("mozilla-foundation/common_voice_13_0", language, split="train").select_columns(['audio', 'client_id']).cast_column('audio', Audio(
            sampling_rate=sampling_rate
        )).with_format('torch')

        self.val_dataset = load_dataset("mozilla-foundation/common_voice_13_0", language, split="validation").select_columns(['audio', 'client_id']).cast_column('audio', Audio(
            sampling_rate=sampling_rate
        )).with_format('torch')

        self._batch_size = batch_size
        self._num_workers = num_workers

    def _collate(self, batch):
        spectrograms = [self._spectrogram(item['audio']['array'].view(1, -1)) for item in batch]
        max_len = max([item.shape[2] for item in spectrograms])
        resized = [F.pad(item, (0, max_len-item.shape[2]), 'constant', 0) for item in spectrograms]

        labels = [item['client_id'] for item in batch]
        label_map = {label: idx for idx, label in enumerate(set(labels))}
        labels = [label_map[label] for label in labels]
        return torch.stack(resized), torch.tensor(labels)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self._batch_size, shuffle=True, collate_fn=self._collate, num_workers=self._num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self._batch_size, shuffle=False, collate_fn=self._collate, num_workers=self._num_workers)
