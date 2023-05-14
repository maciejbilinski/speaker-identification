from datasets import load_dataset, Audio
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram
from torch.nn.functional import pad
import torch
import time
from tqdm import tqdm
import numpy as np

spectrogram = MelSpectrogram(sample_rate=16000, n_mels=64)
def collate(batch):
    spectrograms = [spectrogram(item['audio']['array'].view(1, -1)) for item in batch]
    max_len = max([item.shape[2] for item in spectrograms])
    resized = [pad(item, (0, max_len-item.shape[2]), 'constant', 0) for item in spectrograms]

    labels = [item['client_id'] for item in batch]
    label_map = {label: idx for idx, label in enumerate(set(labels))}
    labels = [label_map[label] for label in labels]
    return torch.stack(resized), torch.tensor(labels)

cv_13 = load_dataset("mozilla-foundation/common_voice_13_0", "pl", split="train").select_columns(['audio', 'client_id']).cast_column('audio', Audio(
    sampling_rate=16000
)).with_format('torch')
batch_size = 128
dataloader = DataLoader(cv_13, batch_size=batch_size, shuffle=True, collate_fn=collate)

progress = tqdm(total = cv_13.num_rows)
for x, y in dataloader:
    progress.update(batch_size)
