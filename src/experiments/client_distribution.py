from common_voice_data_module import CommonVoiceModule
import numpy as np

dm = CommonVoiceModule(num_workers = 0, language = 'pl')

train = []
val = []
for _, labels in dm.train_dataloader():
    train.append(len(set(labels)))


for _, labels in dm.val_dataloader():
    val.append(len(set(labels)))

print(train)
print(val)
print(np.mean(train))
print(np.mean(val))