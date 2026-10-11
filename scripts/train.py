"""Train CNN / BiLSTM for RQ1 and RQ2; save actual runs and predictions."""
import argparse
import copy
import csv
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['Claude', 'GPT', 'Llama']
CONFIG = {'embedding_dim': 64, 'hidden_dim': 64, 'cnn_filters': 64,
          'kernel_sizes': [3, 4, 5], 'dropout': .3, 'vocab_limit': 12000,
          'prompt_tokens': 96, 'output_tokens': 192, 'batch_size': 48,
          'learning_rate': .001, 'weight_decay': .0001, 'max_epochs': 15,
          'patience': 4, 'gradient_clip': 1.0}


def tokens(text):
    # Preserve punctuation and newline as potentially informative style features.
    return re.findall(r'\n|[A-Za-z]+(?:\x27[A-Za-z]+)?|\d+|[^\w\s]', text.lower())


def sequence(row, mode):
    p = tokens(row['LLM_Input'])[:CONFIG['prompt_tokens']]
    o = tokens(row['LLM_output'])[:CONFIG['output_tokens']]
    if mode == 'input':
        return p
    if mode == 'output':
        return o
    return p + ['<sep>'] + o


def vocabulary(rows, mode):
    counts = Counter(t for row in rows for t in sequence(row, mode))
    vocab = {'<pad>': 0, '<unk>': 1, '<sep>': 2}
    for token, _ in counts.most_common(CONFIG['vocab_limit'] - 3):
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode(rows, mode, vocab):
    limit = {'input': 96, 'output': 192, 'both': 289}[mode]
    x = torch.zeros((len(rows), limit), dtype=torch.long)
    lengths = []
    for i, row in enumerate(rows):
        ids = [vocab.get(t, 1) for t in sequence(row, mode)] or [1]
        x[i, :len(ids)] = torch.tensor(ids)
        lengths.append(len(ids))
    y = torch.tensor([LABELS.index(row['LLM_name']) for row in rows])
    return TensorDataset(x, torch.tensor(lengths), y)


class TextCNN(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.embedding = nn.Embedding(size, 64, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(64, 64, k) for k in [3, 4, 5]])
        self.dropout = nn.Dropout(.3)
        self.classifier = nn.Linear(192, 3)

    def forward(self, x, lengths):
        emb = self.embedding(x).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            values = torch.relu(conv(emb))
            # Permit windows touching short-text padding, exclude padding-only windows.
            mask = torch.arange(values.shape[2], device=x.device)[None, :] >= lengths[:, None]
            values = values.masked_fill(mask[:, None, :], -1e9)
            pooled.append(values.max(dim=2).values)
        return self.classifier(self.dropout(torch.cat(pooled, dim=1)))


class TextLSTM(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.embedding = nn.Embedding(size, 64, padding_idx=0)
        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(.3)
        self.classifier = nn.Linear(128, 3)

    def forward(self, x, lengths):
        emb = self.embedding(x)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return self.classifier(self.dropout(torch.cat([hidden[-2], hidden[-1]], dim=1)))


def predict(model, dataset):
    model.eval()
    logits, truth = [], []
    with torch.no_grad():
        for x, lengths, y in DataLoader(dataset, batch_size=96):
            logits.extend(model(x, lengths).tolist())
            truth.extend(y.tolist())
    predicted = np.asarray(logits).argmax(axis=1).tolist()
    return truth, predicted, logits


def metrics(truth, predicted):
    return {'accuracy': accuracy_score(truth, predicted),
            'macro_f1': f1_score(truth, predicted, average='macro', zero_division=0),
            'confusion_matrix': confusion_matrix(truth, predicted, labels=range(3)).tolist(),
            'per_class': classification_report(truth, predicted, labels=range(3),
                                               target_names=LABELS, output_dict=True, zero_division=0)}


def run(rows, arch, mode, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    splits = {s: [r for r in rows if r['split'] == s] for s in ['train', 'validation', 'test']}
    vocab = vocabulary(splits['train'], mode)
    datasets = {s: encode(r, mode, vocab) for s, r in splits.items()}
    model = (TextCNN if arch == 'CNN' else TextLSTM)(len(vocab))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(datasets['train'], batch_size=48, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    best, best_state, best_epoch, stale, history = -1, None, 0, 0, []
    start = time.perf_counter()
    for epoch in range(1, CONFIG['max_epochs'] + 1):
        model.train()
        loss_sum = 0
        for x, lengths, y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x, lengths), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            loss_sum += loss.item() * len(y)
        truth, pred, _ = predict(model, datasets['validation'])
        val = metrics(truth, pred)
        history.append({'epoch': epoch, 'train_loss': loss_sum / len(datasets['train']),
                        'val_accuracy': val['accuracy'], 'val_macro_f1': val['macro_f1']})
        if val['macro_f1'] > best + 1e-8:
            best, best_state, best_epoch, stale = val['macro_f1'], copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        print(f'{arch}/{mode}/{seed} epoch {epoch}: loss={history[-1]["train_loss"]:.3f} '
              f'val_f1={val["macro_f1"]:.3f}', flush=True)
        if stale >= CONFIG['patience']:
            break
    model.load_state_dict(best_state)
    truth, pred, logits = predict(model, datasets['test'])
    result = {'architecture': arch, 'mode': mode, 'seed': seed, 'best_epoch': best_epoch,
              'best_val_macro_f1': best, 'vocab_size': len(vocab),
              'parameters': sum(p.numel() for p in model.parameters()),
              'elapsed_seconds': time.perf_counter() - start, 'history': history,
              'test': metrics(truth, pred)}
    out = ROOT / 'artifacts/runs'
    out.mkdir(parents=True, exist_ok=True)
    name = f'{arch}_{mode}_{seed}'
    (out / f'{name}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    checkpoints = ROOT / 'artifacts/checkpoints'
    checkpoints.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': best_state, 'vocab': vocab, 'architecture': arch,
                'mode': mode, 'seed': seed, 'config': CONFIG}, checkpoints / f'{name}.pt')
    with (out / f'{name}_predictions.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['prompt_id', 'truth', 'prediction', 'logit_Claude', 'logit_GPT', 'logit_Llama'])
        writer.writerows([r['prompt_id'], LABELS[t], LABELS[p], *z]
                        for r, t, p, z in zip(splits['test'], truth, pred, logits))
    print(f'FINISHED {name}: test_accuracy={result["test"]["accuracy"]:.4f} '
          f'test_f1={result["test"]["macro_f1"]:.4f} seconds={result["elapsed_seconds"]:.1f}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[448, 449, 450])
    parser.add_argument('--architectures', nargs='+', default=['CNN', 'LSTM'])
    parser.add_argument('--modes', nargs='+', default=['input', 'output', 'both'])
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    CONFIG['max_epochs'] = args.epochs
    torch.set_num_threads(args.threads)
    rows = [json.loads(line) for line in (ROOT / 'data/processed/examples.jsonl').read_text(encoding='utf-8').splitlines()]
    # Check invariants before expensive training; never split model responses independently.
    prompt_splits = {}
    for row in rows:
        prompt_splits.setdefault(row['prompt_id'], set()).add(row['split'])
    assert all(len(s) == 1 for s in prompt_splits.values()), 'Prompt leakage'
    for split in ['train', 'validation', 'test']:
        counts = Counter(r['LLM_name'] for r in rows if r['split'] == split)
        assert len(counts) == 3 and len(set(counts.values())) == 1, 'Unbalanced split'
    print({'torch': torch.__version__, 'device': 'cpu', 'threads': args.threads, 'config': CONFIG}, flush=True)
    for seed in args.seeds:
        for arch in args.architectures:
            for mode in args.modes:
                path = ROOT / f'artifacts/runs/{arch}_{mode}_{seed}.json'
                if path.exists() and not args.force:
                    print(f'Skipping completed {path.name}', flush=True)
                    continue
                run(rows, arch, mode, seed)


if __name__ == '__main__':
    main()
