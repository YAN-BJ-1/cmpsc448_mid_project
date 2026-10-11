"""Check dataset, split, vocabulary and architecture invariants."""
import json
import csv
from collections import Counter, defaultdict

import torch

from train import ROOT, LABELS, TextCNN, TextLSTM, encode, vocabulary, metrics


def main():
    rows = [json.loads(l) for l in (ROOT / 'data/processed/examples.jsonl').read_text(encoding='utf-8').splitlines()]
    groups = defaultdict(list)
    for row in rows:
        groups[row['prompt_id']].append(row)
    for group in groups.values():
        assert len(group) == 3 and {r['LLM_name'] for r in group} == set(LABELS)
        assert len({r['LLM_Input'] for r in group}) == 1
        assert len({r['split'] for r in group}) == 1
    seen_output_splits = defaultdict(set)
    for row in rows:
        seen_output_splits[' '.join(row['LLM_output'].split())].add(row['split'])
    assert all(len(s) == 1 for s in seen_output_splits.values()), 'Exact response leakage'
    train = [r for r in rows if r['split'] == 'train']
    for mode in ['input', 'output', 'both']:
        vocab = vocabulary(train, mode)
        assert vocab['<pad>'] == 0 and vocab['<unk>'] == 1 and vocab['<sep>'] == 2
        synthetic = dict(rows[0], LLM_Input='zzunseenwordtokenzz', LLM_output='q')
        x, lengths, y = encode([synthetic], mode, vocab).tensors
        assert x.max() < len(vocab) and lengths.min() > 0
        if mode == 'input':
            assert x[0, 0] == 1, 'Unknown token not mapped to UNK'
        for model_type in [TextCNN, TextLSTM]:
            model = model_type(len(vocab))
            model.eval()
            logits = model(x, lengths)
            assert logits.shape == (1, 3) and torch.isfinite(logits).all()
            if model_type is TextLSTM:
                # Packed LSTM ignores extra right padding.
                extra = torch.cat([x, torch.zeros((1, 10), dtype=torch.long)], dim=1)
                assert torch.allclose(logits, model(extra, lengths), atol=1e-6)
    print(f'PASS: {len(rows)} rows / {len(groups)} balanced prompt groups; disjoint prompts and exact '
          'outputs; training-only vocabulary; short texts and packed LSTM padding verified.')
    print('Split counts:', dict(Counter(r['split'] for r in rows)))
    test = [r for r in rows if r['split'] == 'test']
    completed = list((ROOT / 'artifacts/runs').glob('*.json'))
    for path in completed:
        result = json.loads(path.read_text())
        with path.with_name(path.stem + '_predictions.csv').open(encoding='utf-8') as f:
            predictions = list(csv.DictReader(f))
        assert [(r['prompt_id'], r['LLM_name']) for r in test] == [(r['prompt_id'], r['truth']) for r in predictions]
        recomputed = metrics([LABELS.index(r['truth']) for r in predictions],
                             [LABELS.index(r['prediction']) for r in predictions])
        for key in ['accuracy', 'macro_f1']:
            assert abs(recomputed[key] - result['test'][key]) < 1e-12
        assert recomputed['confusion_matrix'] == result['test']['confusion_matrix']
        selected = result['history'][result['best_epoch'] - 1]['val_macro_f1']
        assert selected == max(h['val_macro_f1'] for h in result['history'])
        if result['mode'] == 'input':
            assert abs(result['test']['accuracy'] - 1/3) < 1e-12
    print(f'PASS: recomputed metrics, prompt alignment and validation checkpoint choice for {len(completed)} saved runs.')


if __name__ == '__main__':
    main()
