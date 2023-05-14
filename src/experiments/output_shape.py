from common_voice_data_module import CommonVoiceModule
from mobilenet import MobileNet
import torch
from torch.optim import Adam

print(MobileNet()(torch.rand((128, 1, 64, 2000))).shape)
print(MobileNet(embedding_dim=256)(torch.rand((1, 1, 64, 2000))).shape)
print(MobileNet().training_step((torch.rand((128, 1, 64, 2000)), torch.randint(0, 128, (128, ))), None))

net = MobileNet()
data = CommonVoiceModule(num_workers=0)
with torch.no_grad():
    for batch in data.train_dataloader():
        print(net.training_step(batch, None))

device = torch.device('cuda')
net = MobileNet().to(device)
data = CommonVoiceModule(num_workers=0)
optimizer = Adam(net.parameters(), lr=1e-3)

for x, y in data.train_dataloader():
    optimizer.zero_grad()

    x = x.to(device)
    y = y.to(device)
    loss = net.training_step((x, y), None)
    print(loss)
    loss.backward()
    optimizer.step()
    