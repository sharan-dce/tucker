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


def generate_negative_facts(
        dl: DataLoader,
        s_idx: int,
        r_idx: int,
        o_idx: int) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    '''
    Given a fact (s, r, o), return all the triples (s', r, o) and (s, r, o')
    that are not present in the dataset (negative facts) along with the
    original true fact
    '''
    sr_negative_facts = []
    ro_negative_facts = []

    for e in dl.entities:
        e_idx = dl.entity_to_idx[e]

        if e_idx not in dl.sr_pairs[(s_idx, r_idx)]:
            sr_negative_facts.append((s_idx, r_idx, e_idx))
        if e_idx not in dl.ro_pairs[(r_idx, o_idx)]:
            ro_negative_facts.append((e_idx, r_idx, o_idx))

    return sr_negative_facts, ro_negative_facts


def measure_performance(model: tucker.TuckER, dl: DataLoader, ks: List[int] = [1, 3, 10]) -> Tuple[int, dict]:
    '''
    Measure the performance of a model by computing a mean reciprocal rank and
    hits@k for each k in `ks`
    '''
    mrr = 0
    test_facts = dl.get_all_facts('test')
    hits_k = {k: 0 for k in ks}

    for s, r, o in test_facts:
        output = model(torch.LongTensor([s]), torch.LongTensor([r]))[0]

        rank = 1
        negatives, _ = generate_negative_facts(dl, s, r, o)

        for _, _, negative_o in negatives:
            if output[negative_o] > output[o]:
                rank += 1

        mrr += 1/rank

        for k in hits_k.keys():
            if rank <= k:
                hits_k[k] += 1

    # normalise
    mrr /= len(test_facts)

    for k in hits_k.keys():
        hits_k[k] /= len(test_facts)

    return mrr, hits_k


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
        ).to(device)
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
        ).to(device)
        _correct_predictions = (output == target).count_nonzero()
        _correct_predictions = int(_correct_predictions)
        correct_predictions += _correct_predictions
        total_predictions += len(torch.reshape(output, [-1]))
    return correct_predictions / total_predictions


def train(model, data_loader, epochs, lr, lr_decay, batch_size):
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=lr
    )
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer,
        gamma=lr_decay
    )
    sl = data_loader.get_1_to_n_train_data()[0]
    batch_loader = torch.utils.data.DataLoader(list(sl.keys()), batch_size=batch_size)
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


if __name__ == '__main__':
    dl = data_loader.DataLoader('FB15k')
    model = tucker.TuckER(
        len(dl.entities),
        len(dl.relations),
        np.random.normal(size=[200, 30, 200])
    )

    train(model, data_loader=dl, epochs=2, lr=0.0001, lr_decay=0.99)
