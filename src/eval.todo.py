from model import EmbeddingModel
import os
import torch
import numpy as np
import pandas as pd
from test_dataset import AudioTestDataset
if __name__ == "__main__":
    N = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = AudioTestDataset('test.tsv', 'clips')
    model = EmbeddingModel(embedding_dim=256)
    model.load_from_checkpoint(os.path.join('lightning_logs', 'version_18', 'checkpoints', 'epoch=9-step=5310.ckpt'))
    model = model.to(device)
    max = -np.inf

    embeddings = []
    with torch.no_grad():
        rand = torch.randint(0, dataset.__len__(), (N,), device=device)
        for element in rand:
            anchor, positive, negative = dataset.__getitem__(element.item())
            embeddings.append(
                {
                    'client_id': anchor[0],
                    'embedding': model(anchor[1].to(device).unsqueeze(0)).cpu().numpy()
                }
            )
            embeddings.append(
                {
                    'client_id': positive[0],
                    'embedding': model(positive[1].to(device).unsqueeze(0)).cpu().numpy()
                }
            )
            embeddings.append(
                {
                    'client_id': negative[0],
                    'embedding': model(negative[1].to(device).unsqueeze(0)).cpu().numpy()
                }
            )
    pd.DataFrame(embeddings).to_pickle('evaluated.pkl')

