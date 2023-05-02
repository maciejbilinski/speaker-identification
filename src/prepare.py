import os
import pandas as pd
import torchaudio
import torch
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import Dataset

if __name__ == "__main__":
    tsv_file = 'train.tsv'
    df = pd.read_csv(tsv_file, delimiter='\t')
    output = 'prepared_data'
    samples = pd.DataFrame({
        'client_id': df['client_id'],
        'path': df['path'].apply(lambda x: os.path.join('clips', x)),
        'filename': df['path'].apply(lambda x: x[:-4]),
    })
    audio_transforms = MelSpectrogram(n_mels=64)
    max_size = 0
    for index, sample in samples.iterrows():
        wf, _ = torchaudio.load(sample.path)
        spectrogram = audio_transforms(wf)
        if spectrogram.shape[2] > max_size:
            max_size = spectrogram.shape[2]
            print(f'New max_size={max_size}')
    os.makedirs(output, exist_ok=True)

    prepared_samples = pd.DataFrame({'client_id': [], 'path': []})
    for index, sample in samples.iterrows():
        wf, _ = torchaudio.load(sample.path)
        spectrogram = audio_transforms(wf)
        spectrogram = torch.nn.functional.pad(spectrogram, (0, max_size-spectrogram.shape[2]), 'constant', 0)
        prepared_samples.loc[len(prepared_samples)] = {
            'client_id': sample.client_id,
            'path': sample.filename + '.pt'
        }
        torch.save(spectrogram, os.path.join(output, sample.filename + '.pt'))
    prepared_samples.to_csv('prepared_train.csv', index=False)


        


