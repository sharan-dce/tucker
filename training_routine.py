from typing import List, Tuple
import torch
import data_loader
from data_loader import DataLoader
import numpy as np
from models import tucker
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def unzip(x):
    x = zip(*x)
    return list(x)


def generate_positive_objects_for_triple(
        dl: DataLoader,
        s_idx: int,
        r_idx: int,
        o_idx: int) -> torch.LongTensor:
    '''
    Given a fact (s, r, o), return all the objects o' such that the triple (s,
    r, o') is in the training set along o itself
    
    Returns a boolean vector of length `len(dl.entities)` with 1s where the
    positive objects are.
    '''
    positive = set()

    for dset in ['train', 'valid', 'test']:
        for o in dl.sr_pairs[dset][(s_idx, r_idx)]:
            positive.add(o)

    positive.add(o_idx)

    result = torch.zeros(len(dl.entities)).type(torch.BoolTensor).to(device)
    result.scatter_(0, torch.LongTensor(list(positive)).to(device), 1)
    return result


def generate_positive_objects(
        dl: DataLoader,
        s_idxs: torch.Tensor,
        r_idxs: torch.Tensor,
        o_idxs: torch.Tensor) -> List[torch.LongTensor]:
    '''
    Given a tensor of facts (s, r, o), for each fact return all the objects o'
    such that (s, r, o') is present in the training dataset (i.e. positive
    facts) along with o itself
    '''
    result = []

    for i in range(len(s_idxs)):
        s, r, o = s_idxs[i].item(), r_idxs[i].item(), o_idxs[i].item()

        result.append(generate_positive_objects_for_triple(dl, s, r, o))

    return torch.stack(result)


def measure_performance(
        model: tucker.TuckER,
        dl: DataLoader,
        batch_size: int,
        ks: List[int] = [1, 3, 10]) -> Tuple[int, dict]:
    '''
    Measure the performance of a model by computing a mean reciprocal rank and
    hits@k for each k in `ks`
    '''
    model.eval()
    mrr = 0
    test_facts = dl.get_all_facts('test')
    hits_k = {k: 0 for k in ks}
    batch_test_loader = torch.utils.data.DataLoader(test_facts, batch_size=batch_size)

    for s, r, o in tqdm(batch_test_loader, 'Measuring performance'):
        output = model(s.to(device), r.to(device))
        positives = generate_positive_objects(dl, s, r, o)

        ranks = (((~positives) * output) >= torch.gather(output, 1, o.unsqueeze(1).to(device))).sum(dim=1) + 1
        mrr += (1/ranks).sum().item()

        for k in hits_k.keys():
            hits_k[k] += (ranks <= k).sum().item()

    # normalise
    mrr /= len(test_facts)

    for k in hits_k.keys():
        hits_k[k] /= len(test_facts)

    model.train()
    return mrr, hits_k


def _train_step(
        model: tucker.TuckER, 
        data_loader, 
        batch_loader, 
        optimizer, 
        label_smoothing_rate: float, 
        desc: str=None):
    loss = torch.nn.BCELoss()
    loss_avg = 0
    for subject_index, relation_index in tqdm(batch_loader, desc=desc):
        optimizer.zero_grad()
        output = model(
            subject_index.to(device),
            relation_index.to(device)
        )
        target = data_loader.get_y(
            subject_idxs=subject_index,
            relation_idxs=relation_index
        ).to(device)
        target = (1.0 - label_smoothing_rate) * target + (1.0 / target.size(1))
        loss_val = loss(output, target=target)
        loss_val.backward()
        loss_avg += loss_val.item()
        optimizer.step()
    print('Loss Val:', loss_avg)


def test(model, data_loader, batch_loader):
    model.eval()
    total_predictions, correct_predictions = 0, 0
    for subject_index, relation_index in tqdm(batch_loader, 'Testing'):
        output = model(
            subject_index.to(device), 
            relation_index.to(device)
        )
        output = torch.round(output)
        target = data_loader.get_y(
            subject_idxs=subject_index,
            relation_idxs=relation_index
        ).to(device)
        _correct_predictions = (output == target).count_nonzero()
        _correct_predictions = int(_correct_predictions)
        correct_predictions += _correct_predictions
        total_predictions += len(torch.reshape(output, [-1]))
    model.train()
    return correct_predictions / total_predictions


def train(
        model: tucker.TuckER, 
        data_loader, 
        epochs: int, 
        lr: float, 
        lr_decay: float, 
        batch_size: int,
        label_smoothing_rate: float,
        weight_decay: float):
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer,
        gamma=lr_decay
    )
    sl = data_loader.sr_pairs['train']
    batch_loader = torch.utils.data.DataLoader(list(sl.keys()), batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        _train_step(
            model=model,
            data_loader=data_loader,
            batch_loader=batch_loader,
            optimizer=optimizer,
            label_smoothing_rate=label_smoothing_rate,
            desc='Epoch {}'.format(epoch)
        )
        lr_scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(measure_performance(model, data_loader, batch_size))


if __name__ == '__main__':
    dl = data_loader.DataLoader('data/FB15k')
    model = tucker.TuckER(
        len(dl.entities),
        len(dl.relations),
        np.random.normal(size=[200, 30, 200])
    ).to(device)

    train(
        model,
        data_loader=dl,
        epochs=2,
        lr=0.0001,
        lr_decay=0.99,
        batch_size=4,
        label_smoothing_rate=0.1,
        weight_decay=0
    )
