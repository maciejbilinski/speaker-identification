from batch_all_triplet_loss import BatchAllTripletLoss
import torch

loss_fn = BatchAllTripletLoss()

# Zero triplets
print(loss_fn(torch.tensor([
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [1, 1, 1, 1],
]), torch.tensor([
    0,
    0,
    0
])))

# Negative better than positive
print(loss_fn(torch.tensor([
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [1, 1, 1, 1],
]), torch.tensor([
    0,
    1,
    0
])))

#  Positive better than negative
print(loss_fn(torch.tensor([
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [1, 1, 1, 1],
]), torch.tensor([
    0,
    0,
    1
])))

#  Best positive
print(loss_fn(torch.tensor([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [1, 1, 1, 1],
]), torch.tensor([
    0,
    0,
    1
])))

# More triplets
print(loss_fn(torch.tensor([
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [1, 1, 1, 1],
    [1, 1, 1, -1],
]), torch.tensor([
    0,
    0,
    1,
    1
])))