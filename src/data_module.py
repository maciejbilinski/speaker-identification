from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from test_dataset import AudioTestDataset

from train_dataset import AudioTrainDataset

class AudioDataModule(LightningDataModule):
    def __init__(self, batch_size, train_csv_file='prepared_train.csv', test_tsv_file='test.tsv', audio_dir='clips', spect_dir='prepared_data'):
        super().__init__()
        self.train_dataset = AudioTrainDataset(train_csv_file, spect_dir)
        self.test_dataset = AudioTestDataset(test_tsv_file, audio_dir)
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=4)
