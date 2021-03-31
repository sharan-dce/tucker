import torch
import data_loader
import models
from tqdm import tqdm

def _train_step(model, data_loader, batch_loader, optimizer, desc=None):
    loss = torch.nn.BCELoss()
    for subject_index, relation_index in tqdm(batch_loader, desc=desc):
        optimizer.zero_grad()
        output = model(
            subject_index=subject_index,
            relation_index=relation_index
        )
        target = data_loader.get_y(
            subject_index=subject_index,
            relation_index=relation_index
        )
        loss_val = loss(output=output, target=target)
        loss_val.backward()
        optimizer.step()
    
def test(model, data_loader, batch_loader):
    total_predictions, correct_predictions = 0, 0
    for subject_index, relation_index in batch_loader:
        output = model(
            subject_index=subject_index, 
            relation_index=relation_index
        )
        output = torch.round(output)
        target = data_loader.get_y(
            subject_index=subject_index,
            relation_index=relation_index
        )
        _correct_predictions = (output == target).count_nonzero()
        _correct_predictions = int(_correct_predictions)
        correct_predictions += _correct_predictions
        total_predictions += len(output.squeeze())
    return correct_predictions / total_predictions
        
def train(model, data_loader, batch_loader, epochs, lr, lr_decay):
    optimizer = torch.optim.Adam(
        params=model.params,
        lr=lr
    )
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer,
        gamma=lr_decay
    )
    for epoch in range(epochs):
        _train_step(
            model=model,
            data_loader=data_loader,
            batch_loader=batch_loader,
            optimizer=optimizer,
            desc='Epoch {}'.format(epoch)
        )
        lr_scheduler.step()
        test_accuracy = test(
            model=model,
            data_loader=data_loader,
            batch_loader=batch_loader
        )
        print('Test Accuracy: {}'.format(test_accuracy))
