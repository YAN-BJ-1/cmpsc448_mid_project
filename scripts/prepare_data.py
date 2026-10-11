"""Download pinned AlpacaEval responses and build a prompt-grouped dataset."""
import hashlib
import json
import random
import re
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {'GPT': 'gpt4', 'Claude': 'claude-2', 'Llama': 'llama-2-70b-chat-hf'}


def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'CMPSC448-course-project'})
    with urllib.request.urlopen(req, timeout=90) as response:
        return response.read()


def normalize(text):
    return re.sub(r'\s+', ' ', text).strip()


def main():
    raw = ROOT / 'data/raw'
    processed = ROOT / 'data/processed'
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / 'data/manifest.json'
    expected_hashes = {}
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        revision = saved['revision']
        expected_hashes = {f['model']: f['sha256'] for f in saved['files']}
    else:
        revision = json.loads(fetch('https://api.github.com/repos/tatsu-lab/alpaca_eval/commits/main'))['sha']
    manifest = {'repository': 'https://github.com/tatsu-lab/alpaca_eval', 'revision': revision,
                'seed': 448, 'data_license': 'AlpacaEval dataset card: CC BY-NC 4.0; repository: Apache 2.0',
                'files': [], 'models': MODELS}
    indexed = {}
    for family, model in MODELS.items():
        url = f'https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/{revision}/results/{model}/model_outputs.json'
        path = raw / f'{model}.json'
        if not path.exists():
            path.write_bytes(fetch(url))
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if model in expected_hashes and digest != expected_hashes[model]:
            raise ValueError(f'SHA-256 mismatch for {model}; raw data changed')
        items = json.loads(content)
        by_prompt = {}
        for item in items:
            prompt = normalize(item['instruction'])
            if not prompt or not item.get('output', '').strip():
                continue
            if prompt in by_prompt:
                raise ValueError(f'Duplicate normalized prompt in {model}')
            by_prompt[prompt] = item
        indexed[family] = by_prompt
        manifest['files'].append({'family': family, 'model': model, 'url': url,
                                  'sha256': digest, 'rows': len(items)})
        print(f'{family}: {len(items)} downloaded / {len(by_prompt)} valid', flush=True)
    common = sorted(set.intersection(*(set(v) for v in indexed.values())))
    # Remove exact repeated outputs belonging to distinct prompt groups. Keep
    # same-prompt collisions: they are real ambiguous cases, not label errors.
    seen = set()
    kept = []
    removed = 0
    for prompt in common:
        outputs = {normalize(indexed[f][prompt]['output']) for f in MODELS}
        if outputs & seen:
            removed += 1
            continue
        seen.update(outputs)
        kept.append(prompt)
    random.Random(448).shuffle(kept)
    n = len(kept)
    cut1, cut2 = int(n * .70), int(n * .85)
    rows = []
    for i, prompt in enumerate(kept):
        split = 'train' if i < cut1 else 'validation' if i < cut2 else 'test'
        pid = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        for family, model in MODELS.items():
            item = indexed[family][prompt]
            rows.append({'prompt_id': pid, 'LLM_name': family, 'model_id': model,
                         'LLM_Input': item['instruction'], 'LLM_output': item['output'],
                         'source': item.get('dataset', 'unknown'), 'split': split})
    with (processed / 'examples.jsonl').open('w', encoding='utf-8') as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
    manifest.update({'common_prompts': len(common), 'retained_prompts': n,
                     'removed_cross_prompt_duplicate_groups': removed,
                     'rows': len(rows), 'split_rows': dict(Counter(r['split'] for r in rows)),
                     'source_rows': dict(Counter(r['source'] for r in rows))})
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
