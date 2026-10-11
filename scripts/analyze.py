"""Aggregate repeated runs; RQ4 feature baselines and held-out perturbations."""
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train import ROOT, LABELS, TextCNN, TextLSTM, encode, metrics, predict, tokens

FEATURE_NAMES = ['words', 'characters', 'newlines', 'list_lines', 'lexical_diversity',
                 'mean_word_length', 'punctuation_fraction']
IDENTITY = re.compile(r'\b(?:gpt(?:-?\d+(?:\.\d+)?)?|chatgpt|openai|claude|anthropic|llama|meta)\b', re.I)


def features(text):
    words = re.findall(r'[A-Za-z]+', text.lower())
    return [len(words), len(text), text.count('\n'),
            len(re.findall(r'^\s*(?:[-*]|\d+[.)])\s+', text, re.M)),
            len(set(words)) / max(len(words), 1),
            sum(map(len, words)) / max(len(words), 1),
            len(re.findall(r'[^\w\s]', text)) / max(len(text), 1)]


def load_predictions(name):
    with (ROOT / f'artifacts/runs/{name}_predictions.csv').open(encoding='utf-8') as f:
        return list(csv.DictReader(f))


def paired_ci(first, second):
    a, b = load_predictions(first), load_predictions(second)
    assert [(r['prompt_id'], r['truth']) for r in a] == [(r['prompt_id'], r['truth']) for r in b]
    difference = np.array([int(y['truth'] == y['prediction']) - int(x['truth'] == x['prediction'])
                           for x, y in zip(a, b)]).reshape(-1, 3).mean(axis=1)
    rng = np.random.default_rng(448)
    bootstrap = difference[rng.integers(0, len(difference), size=(5000, len(difference)))].mean(axis=1)
    return {'first': first, 'second': second, 'difference': float(difference.mean()),
            'ci95': np.quantile(bootstrap, [.025, .975]).tolist(),
            'unit': 'prompt group; second minus first; 5000 bootstrap samples'}


def main():
    torch.set_num_threads(4)
    rows = [json.loads(l) for l in (ROOT / 'data/processed/examples.jsonl').read_text(encoding='utf-8').splitlines()]
    train = [r for r in rows if r['split'] == 'train']
    test = [r for r in rows if r['split'] == 'test']
    out = ROOT / 'artifacts'
    figs = out / 'figures'
    figs.mkdir(parents=True, exist_ok=True)
    runs = [json.loads(p.read_text()) for p in sorted((out / 'runs').glob('*.json'))]
    if '--plots-only' in sys.argv:
        saved = json.loads((out / 'analysis.json').read_text())
        render_figures(rows, runs, saved['summary'], saved['baselines'], figs)
        print('Redrew figures from saved results; metrics unchanged.')
        return
    summary = []
    for arch in ['CNN', 'LSTM']:
        for mode in ['input', 'output', 'both']:
            group = [r for r in runs if r['architecture'] == arch and r['mode'] == mode]
            assert len(group) == 3, f'Missing completed runs: {arch}/{mode}'
            item = {'architecture': arch, 'mode': mode, 'seeds': [r['seed'] for r in group]}
            for key in ['accuracy', 'macro_f1']:
                values = [r['test'][key] for r in group]
                item[key + '_mean'] = float(np.mean(values))
                item[key + '_std'] = float(np.std(values, ddof=1))
            summary.append(item)
    with (out / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    ytrain = [LABELS.index(r['LLM_name']) for r in train]
    ytest = [LABELS.index(r['LLM_name']) for r in test]
    baselines = {}
    for name, selection in [('length_only', [0]), ('structural_lexical', list(range(7)))]:
        xtrain = np.array([features(r['LLM_output']) for r in train])[:, selection]
        xtest = np.array([features(r['LLM_output']) for r in test])[:, selection]
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000, random_state=448))
        clf.fit(xtrain, ytrain)
        baselines[name] = metrics(ytest, clf.predict(xtest))
    tfidf = make_pipeline(TfidfVectorizer(max_features=12000, ngram_range=(1, 2), min_df=2),
                          LogisticRegression(C=1, max_iter=2000, random_state=448))
    tfidf.fit([r['LLM_output'] for r in train], ytrain)
    baselines['tfidf_logistic'] = metrics(ytest, tfidf.predict([r['LLM_output'] for r in test]))
    vectorizer, clf = tfidf.steps[0][1], tfidf.steps[1][1]
    names = vectorizer.get_feature_names_out()
    salient = {label: names[np.argsort(clf.coef_[i])[-12:][::-1]].tolist() for i, label in enumerate(LABELS)}
    stats = {}
    for label in LABELS:
        group = [r for r in rows if r['LLM_name'] == label]
        x = np.array([features(r['LLM_output']) for r in group])
        stats[label] = {n: {'mean': float(x[:, i].mean()), 'median': float(np.median(x[:, i]))}
                        for i, n in enumerate(FEATURE_NAMES)}
        stats[label]['identity_mentions'] = sum(bool(IDENTITY.search(r['LLM_output'])) for r in group)
        stats[label]['truncated_output_fraction'] = sum(len(tokens(r['LLM_output'])) > 192 for r in group) / len(group)
    perturbations = {}
    for arch in ['CNN', 'LSTM']:
        ckpt = torch.load(out / f'checkpoints/{arch}_output_448.pt', weights_only=False, map_location='cpu')
        model = (TextCNN if arch == 'CNN' else TextLSTM)(len(ckpt['vocab']))
        model.load_state_dict(ckpt['state_dict'])
        perturbations[arch] = {}
        for name in ['original', 'remove_identity', 'words_only', 'first_64_tokens']:
            modified = []
            for row in test:
                r = dict(row)
                if name == 'remove_identity':
                    r['LLM_output'] = IDENTITY.sub('', row['LLM_output'])
                elif name == 'words_only':
                    r['LLM_output'] = ' '.join(re.findall(r'[A-Za-z]+(?:\x27[A-Za-z]+)?|\d+', row['LLM_output']))
                modified.append(r)
            dataset = encode(modified, 'output', ckpt['vocab'])
            if name == 'first_64_tokens':
                x, lengths, y = dataset.tensors
                dataset = torch.utils.data.TensorDataset(x[:, :64], lengths.clamp(max=64), y)
            truth, pred, _ = predict(model, dataset)
            perturbations[arch][name] = metrics(truth, pred)
    ci = [paired_ci(f'{arch}_output_448', f'{arch}_both_448') for arch in ['CNN', 'LSTM']]
    ci.append(paired_ci('LSTM_output_448', 'CNN_output_448'))
    analysis = {'summary': summary, 'baselines': baselines, 'feature_statistics': stats,
                'tfidf_positive_terms': salient, 'test_perturbations_seed448': perturbations,
                'paired_accuracy_bootstrap': ci,
                'total_training_seconds': sum(r['elapsed_seconds'] for r in runs),
                'notes': 'Perturbations applied only at test time; distribution shifts, not retrained causal ablations.'}
    (out / 'analysis.json').write_text(json.dumps(analysis, indent=2), encoding='utf-8')
    render_figures(rows, runs, summary, baselines, figs)
    print(json.dumps({'summary': summary, 'baseline_accuracy': {k: v['accuracy'] for k, v in baselines.items()},
                      'perturbation_accuracy': {a: {k: v['accuracy'] for k, v in z.items()} for a, z in perturbations.items()},
                      'paired_ci': ci}, indent=2))


def render_figures(rows, runs, summary, baselines, figs):
    for filename in ['times.ttf', 'timesbd.ttf', 'timesi.ttf', 'timesbi.ttf']:
        font = Path('C:/Windows/Fonts') / filename
        if font.exists():
            font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                         'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'text.color': '#382F42', 'axes.labelcolor': '#382F42',
                         'xtick.color': '#382F42', 'ytick.color': '#382F42'})
    fig, ax = plt.subplots(figsize=(7.4, 3.6), layout='constrained')
    x = np.arange(3)
    for j, arch in enumerate(['CNN', 'LSTM']):
        group = [r for r in summary if r['architecture'] == arch]
        ax.bar(x + (j-.5)*.32, [r['accuracy_mean'] for r in group], .32,
               yerr=[r['accuracy_std'] for r in group], label=arch, capsize=4,
               color=['#B49ACC', '#DFD1EB'][j], edgecolor='#8E75A5', linewidth=.6)
    ax.axhline(1/3, ls='--', color='#887B94', label='Chance (33.3%)')
    ax.set(xticks=x, xticklabels=['Prompt only', 'Response only', 'Prompt + response'],
           ylim=(0, 1), ylabel='Test accuracy', title='RQ1/RQ2: mean accuracy across three training seeds')
    ax.legend(loc='upper left', fontsize=8)
    fig.savefig(figs / 'main_results.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), layout='constrained')
    for ax, arch in zip(axes, ['CNN', 'LSTM']):
        run = next(r for r in runs if r['architecture'] == arch and r['mode'] == 'output' and r['seed'] == 448)
        matrix = np.array(run['test']['confusion_matrix'])
        cmap = LinearSegmentedColormap.from_list('lavender', ['#FCFAFE', '#E3D6EF', '#AD8BC7'])
        ax.imshow(matrix, cmap=cmap, vmin=0, vmax=121)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(matrix[i, j]), ha='center', va='center',
                        color='#34263E')
        ax.set(xticks=range(3), yticks=range(3), xticklabels=LABELS, yticklabels=LABELS,
               xlabel='Predicted family', ylabel='True family', title=f'{arch}: response only, seed 448')
    fig.savefig(figs / 'confusion_matrices.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), layout='constrained')
    wordcounts = [[features(r['LLM_output'])[0] for r in rows if r['LLM_name'] == label] for label in LABELS]
    axes[0].boxplot(wordcounts, tick_labels=LABELS, showfliers=False, patch_artist=True,
                    boxprops={'facecolor': '#E7DCF0', 'edgecolor': '#8E75A5'},
                    medianprops={'color': '#72558C'}, whiskerprops={'color': '#8E75A5'},
                    capprops={'color': '#8E75A5'})
    axes[0].set(ylabel='English word count', title='Response length (all 804 groups)')
    axes[1].bar(['Length', '7 features', 'TF-IDF'], [baselines[n]['accuracy'] for n in baselines],
                color='#C5ACD8', edgecolor='#8E75A5', linewidth=.6)
    axes[1].axhline(1/3, ls='--', color='#887B94')
    axes[1].set(ylim=(0, 1), ylabel='Test accuracy', title='Diagnostic baselines')
    fig.savefig(figs / 'feature_analysis.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
