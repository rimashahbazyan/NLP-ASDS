import torch
from datasets import load_dataset
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import functional as F


import dataset
import model

net = model.Net3()
data = load_dataset("glue", "sst2")
train_sentences = data['train']['sentence']
vocab, freq = dataset.SentenceDataset.load_vocab('./vocab.csv')

training_data = dataset.SentenceDataset(data['train']['sentence'],
                                        data['train']['label'],
                                        128,
                                        vocab)
val_data = dataset.SentenceDataset(data['validation']['sentence'],
                                   data['validation']['label'],
                                   128,
                                   vocab)
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=64, shuffle=True)



criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(100):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, d in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, lens, labels = d
        inputs = inputs.type(torch.int32)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:  # print every 10 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
            running_loss = 0.0

    inputs, lens, labels = next(iter(val_dataloader))
    inputs = inputs.type(torch.int32)

    with torch.no_grad():
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        print(f"EPOCH1: Val LOSS - {loss}")

        torch.save({
            'epoch':                epoch,
            'model_state_dict':     net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss':                 loss,
        }, f'./results/task_d/{epoch}.pt')

