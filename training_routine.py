import torch
import data_loader
import numpy as np
from models import tucker
from tqdm import tqdm

BATCH_SIZE = 8

def unzip(x):
    x = zip(*x)
    return list(x)

def _train_step(model, data_loader, batch_loader, optimizer, desc=None):
    loss = torch.nn.BCELoss()
    for subject_index, relation_index in tqdm(batch_loader, desc=desc):
        optimizer.zero_grad()
        output = model(
            subject_index=subject_index,
            relation_index=relation_index
        )
        target = data_loader.get_y(
            subject_idxs=subject_index,
            relation_idxs=relation_index
        )
        loss_val = loss(output, target=target)
        loss_val.backward()
        optimizer.step()
    
def test(model, data_loader, batch_loader):

    total_predictions, correct_predictions = 0, 0
    for subject_index, relation_index in tqdm(batch_loader, 'Testing'):
        output = model(
            subject_index=subject_index, 
            relation_index=relation_index
        )
        output = torch.round(output)
        target = data_loader.get_y(
            subject_idxs=subject_index,
            relation_idxs=relation_index
        )
        _correct_predictions = (output == target).count_nonzero()
        _correct_predictions = int(_correct_predictions)
        correct_predictions += _correct_predictions
        total_predictions += len(torch.reshape(output, [-1]))
    return correct_predictions / total_predictions

def train(model, data_loader, epochs, lr, lr_decay):
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=lr
    )
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer,
        gamma=lr_decay
    )
    sl = data_loader.get_1_to_n_train_data()[0]
    batch_loader = torch.utils.data.DataLoader(list(sl.keys()), batch_size=BATCH_SIZE)
    for epoch in range(epochs):
        _train_step(
            model=model,
            data_loader=data_loader,
            batch_loader=batch_loader,
            optimizer=optimizer,
            desc='Epoch {}'.format(epoch)
        )
        lr_scheduler.step()
        train_accuracy = test(
            model=model,
            data_loader=data_loader,
            batch_loader=batch_loader
        )
        print('Train Accuracy: {}'.format(train_accuracy))

dl = data_loader.DataLoader('FB15k')
model = tucker.TuckER(
    len(dl.entities),
    len(dl.relations),
    np.random.normal(size=[3, 5, 3])
)

train(model, data_loader=dl, epochs=2, lr=0.0001, lr_decay=0.99)
