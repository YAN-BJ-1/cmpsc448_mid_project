# Who Wrote It? Identifying LLMs from Their Responses

CMPSC 448 midterm project. Team leader: **Bingjie Yan**. Team member: **Siyuan Zhou**.

This project classifies public GPT-4, Claude 2 and Llama 2 70B Chat responses with a text CNN and a bidirectional LSTM. Required RQ1/RQ2 experiments compare prompt-only, response-only and prompt-plus-response inputs. RQ4 explores length, vocabulary and formatting signals.

## Reproduce

Python 3.9 or later is required. CPU execution is supported; no LLM API key is needed.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe scripts/prepare_data.py
.venv/Scripts/python.exe scripts/verify.py
.venv/Scripts/python.exe scripts/train.py
.venv/Scripts/python.exe scripts/analyze.py
.venv/Scripts/python.exe scripts/build_report.py
```

The downloader uses the pinned upstream revision in `data/manifest.json`. Downloaded files are checked against SHA-256 hashes. The same three seeds (448, 449, 450) are used for all six experimental settings, producing 18 runs. Completed runs are skipped; use `--force` to retrain. Default training is 15 epochs maximum, with validation macro-F1 checkpoint selection and patience 4. The test set is never used to choose checkpoints.

`requirements-reproduced.txt` records exact library versions for the saved runs; `artifacts/environment.json` records the training environment. Report generation used Python 3.12.14 / ReportLab 4.4.9; model training and analysis used Python 3.9.13. All scripts are compatible with Python 3.9+.

## Data and evaluation

Source: [Tatsu Lab AlpacaEval](https://github.com/tatsu-lab/alpaca_eval), specifically `results/gpt4`, `results/claude-2`, and `results/llama-2-70b-chat-hf`. This is a repurposing of released benchmark outputs for authorship classification, not an AlpacaEval preference-score experiment. We retain the 804 shared prompts, yielding 2,412 examples. Every prompt has one response from each family. Train/validation/test have 562/121/121 prompt groups (1,686/363/363 examples). Prompt groups are disjoint across splits. Vocabulary is learned from training data only.

The task is three-way **closed-set classification** of these particular model releases. It does not distinguish humans from AI or identify arbitrary new LLMs. Prompt-only accuracy is necessarily 1/3 on the balanced matched-prompt evaluation set. Source labels identify prompt collections, not verified task/domain annotations.

## Files

- `scripts/prepare_data.py`: pinned downloads, normalization, grouping, split creation.
- `scripts/train.py`: CNN/BiLSTM implementations, training, validation selection and test predictions.
- `scripts/verify.py`: data and model invariant checks.
- `scripts/analyze.py`: repeated-seed summaries, baselines, group bootstrap, RQ4 and figures.
- `scripts/build_report.py`: English PDF and editable Markdown report generation.
- `data/manifest.json`: upstream URLs, revision, checksums, split counts.
- `artifacts/runs/`: all run metrics and test prediction CSVs.
- `artifacts/analysis.json`, `artifacts/summary.csv`: derived results.
- `artifacts/figures/`: plots.
- `output/pdf/project_report.pdf`: report for review and submission.
- `output/project_report.md`: editable report text.

Raw and processed text and checkpoints are excluded from Git. The download script reconstructs them; the tracked manifest and predictions document what was used. This avoids republishing an entire third-party text collection.

## Attribution and scope

AlpacaEval: Xuechen Li, Tianyi Zhang, Yann Dubois, Rohan Taori, Ishaan Gulrajani, Carlos Guestrin, Percy Liang and Tatsunori B. Hashimoto (2023), *AlpacaEval: An Automatic Evaluator of Instruction-following Models*. [Repository](https://github.com/tatsu-lab/alpaca_eval).

The [dataset card](https://huggingface.co/datasets/tatsu-lab/alpaca_eval) lists CC BY-NC 4.0; the upstream repository uses Apache 2.0. We use the data for this noncommercial class project and retain source attribution. Underlying prompt collections include [Self-Instruct](https://github.com/yizhongw/self-instruct), [OpenAssistant](https://huggingface.co/datasets/OpenAssistant/oasst1), [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/), [Koala](https://bair.berkeley.edu/blog/2023/04/03/koala/), and [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf).

AI assistance was used in code, experiment design and report drafting. Team members must review, understand and verify the submission, and add their own contribution descriptions and personal reflections before handing it in.
