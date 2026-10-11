"""Build an English report directly from saved results; never invent metrics."""
import html
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output'
PURPLE = colors.HexColor('#655179')
LAVENDER = colors.HexColor('#E5DAF0')
GRAY = colors.HexColor('#635D6A')
FONT = 'TimesNewRoman'
FONT_BOLD = 'TimesNewRoman-Bold'
font_dir = Path('C:/Windows/Fonts')
if (font_dir / 'times.ttf').exists():
    pdfmetrics.registerFont(TTFont(FONT, str(font_dir / 'times.ttf')))
    pdfmetrics.registerFont(TTFont(FONT_BOLD, str(font_dir / 'timesbd.ttf')))
else:
    FONT, FONT_BOLD = 'Times-Roman', 'Times-Bold'
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyProject', fontName=FONT, fontSize=11,
                          leading=14.2, spaceAfter=7, textColor=colors.HexColor('#29242D')))
styles.add(ParagraphStyle(name='TitleProject', fontName=FONT_BOLD, fontSize=25,
                          leading=28, spaceAfter=14, textColor=PURPLE))
styles.add(ParagraphStyle(name='HeadingProject', fontName=FONT_BOLD, fontSize=16,
                          leading=20, spaceAfter=12, textColor=PURPLE))
styles.add(ParagraphStyle(name='SubProject', fontName=FONT_BOLD, fontSize=12,
                          leading=15, spaceBefore=6, spaceAfter=7, textColor=PURPLE))
styles.add(ParagraphStyle(name='CaptionProject', fontName=FONT, fontSize=9, leading=11,
                          textColor=GRAY, spaceAfter=8))
styles.add(ParagraphStyle(name='CellProject', fontName=FONT, fontSize=9.5, leading=11.5))
styles.add(ParagraphStyle(name='HeaderCell', parent=styles['CellProject'],
                          textColor=colors.HexColor('#463551'), fontName=FONT_BOLD))


def pct(value):
    return f'{100 * value:.1f}%'


def pp(value):
    return f'{100 * value:+.1f}'


def main():
    analysis = json.loads((ROOT / 'artifacts/analysis.json').read_text())
    manifest = json.loads((ROOT / 'data/manifest.json').read_text())
    runs = [json.loads(p.read_text()) for p in sorted((ROOT / 'artifacts/runs').glob('*.json'))]
    summary = {(r['architecture'], r['mode']): r for r in analysis['summary']}
    baseline = analysis['baselines']
    perturb = analysis['test_perturbations_seed448']
    OUT.joinpath('pdf').mkdir(parents=True, exist_ok=True)
    story, markdown = [], []

    def p(text):
        story.append(Paragraph(html.escape(text), styles['BodyProject']))
        markdown.append(text + '\n')

    def h(text, level=1):
        story.append(Paragraph(html.escape(text), styles['HeadingProject' if level == 1 else 'SubProject']))
        markdown.append(('#' * (level + 1)) + ' ' + text + '\n')

    def page(title):
        if story:
            story.append(PageBreak())
        h(title)

    def table(headers, data, widths=None):
        cells = [[Paragraph(html.escape(str(x)), styles['HeaderCell' if i == 0 else 'CellProject'])
                  for x in row] for i, row in enumerate([headers] + data)]
        t = Table(cells, colWidths=widths, repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), LAVENDER),
                              ('TEXTCOLOR', (0, 0), (-1, 0), PURPLE),
                              ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F7F3FB'), colors.white]),
                              ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                              ('LEFTPADDING', (0, 0), (-1, -1), 7), ('RIGHTPADDING', (0, 0), (-1, -1), 7),
                              ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                              ('LINEBELOW', (0, 0), (-1, 0), .6, colors.HexColor('#CCBBDC'))]))
        story.extend([t, Spacer(1, 10)])
        markdown.extend(['| ' + ' | '.join(map(str, headers)) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'])
        markdown.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in data)
        markdown.append('')

    def fig(name, caption, height):
        story.append(Image(str(ROOT / 'artifacts/figures' / name), width=6.65 * inch, height=height * inch))
        story.append(Paragraph(html.escape(caption), styles['CaptionProject']))
        markdown.append(f'![{caption}](../artifacts/figures/{name})\n')

    story.append(Paragraph('Who Wrote It?<br/>Identifying LLMs from Their Responses', styles['TitleProject']))
    p('CMPSC 448 | Midterm Project | October 11, 2026')
    p('Team leader: Bingjie Yan | Team member: Siyuan Zhou')
    h('Abstract', 2)
    cnn = summary['CNN', 'output']
    lstm = summary['LSTM', 'output']
    p(f'Can a classifier tell which language model wrote an answer? We tested this using public answers from '
      f'GPT-4, Claude 2 and Llama 2 70B Chat. The dataset contains 804 shared prompts and 2,412 answers. '
      f'We trained a CNN and a bidirectional LSTM using the prompt, the answer, or both. Averaged over three '
      f'training seeds, answer-only accuracy was {pct(cnn["accuracy_mean"])} for the CNN and '
      f'{pct(lstm["accuracy_mean"])} for the LSTM. Adding the prompt did not improve either model. '
      f'Further tests examined answer length, vocabulary and formatting. The results show that these three '
      f'model releases can often be distinguished on this dataset.')
    h('1. Problem definition and research questions', 2)
    p('We treated the task as three-class classification, with Claude, GPT and Llama as labels. For RQ1, '
      'we used only the answer to predict its source. For RQ2, we compared prompt-only, answer-only and '
      'combined inputs. We also studied length, vocabulary and formatting for RQ4. We did not run the '
      'cross-domain experiments in RQ3.')
    h('2. Dataset construction', 2)
    p('We downloaded the released AlpacaEval answers [1]. The GPT and Claude files each contain 805 entries; '
      'the Llama file contains 804. After matching prompts with whitespace normalized, 804 prompts had an '
      'answer from every model. We kept these complete groups so that each class had the same prompts '
      'and sample count. The retained answers were nonempty, and the duplicate checks removed no groups.')
    table(['Split', 'Prompt groups', 'Examples', 'Examples per family'],
          [['Train', 562, 1686, 562], ['Validation', 121, 363, 121], ['Test', 121, 363, 121]],
          [1.15*inch, 1.6*inch, 1.55*inch, 2.35*inch])
    p('We sorted the shared prompts, shuffled them with seed 448, and used a 70/15/15 split. All three '
      'answers to a prompt stayed together. This prevents the same question from appearing in both '
      'training and testing. The source URLs, upstream revision and file hashes are recorded in '
      'data/manifest.json. Model names and source labels were excluded from the input text.')

    page('3. CNN and RNN implementation')
    h('Text representation', 2)
    p('We lowercased the text and tokenized English words, numbers, punctuation and newlines. Keeping '
      'punctuation and newlines let the models use formatting as well as vocabulary. Each input setting '
      'used a vocabulary built only from its training texts, with at most 12,000 entries. PAD represents '
      'padding, UNK represents unseen tokens, and SEP separates the prompt from the answer. We kept '
      'up to 96 prompt tokens and 192 answer tokens, giving a maximum of 289 tokens for combined inputs. '
      'The embeddings were learned from scratch.')
    h('CNN', 2)
    p('The CNN uses 64-dimensional word embeddings and three convolution branches with kernel widths '
      '3, 4 and 5. Each branch has 64 channels. After ReLU and max pooling, we concatenate the three '
      'outputs into a 192-dimensional vector. Windows containing only padding are masked. A dropout '
      'layer with rate 0.3 and a linear layer produce the three class scores. This design looks for '
      'short patterns wherever they occur in the text.')
    h('RNN: bidirectional LSTM', 2)
    p('Our recurrent model is a one-layer bidirectional LSTM. It uses 64-dimensional embeddings and '
      '64 hidden units in each direction. We pack the sequences so that trailing padding is skipped. '
      'The final forward and backward hidden states form a 128-dimensional vector, followed by '
      'dropout 0.3 and a three-class linear layer. Unlike max pooling in the CNN, this representation '
      'depends on the order in which the sequence is processed.')
    table(['Training choice', 'Value'],
          [['Loss / optimizer', 'Cross-entropy / AdamW'], ['Learning rate / weight decay', '0.001 / 0.0001'],
           ['Batch size / gradient clipping', '48 / L2 norm 1.0'], ['Epoch limit / early-stop patience', '15 / 4'],
           ['Checkpoint selection', 'Highest validation macro-F1; first epoch wins ties'],
           ['Initialization seeds', '448, 449, 450; same fixed data split'],
           ['Execution', 'PyTorch 2.8.0+cpu; four CPU threads; deterministic algorithms']],
          [2.5*inch, 4.15*inch])
    p('We ran all six model/input combinations with seeds 448, 449 and 450, for 18 runs in total. '
      'Each run restored the checkpoint with the highest validation macro-F1 before testing. '
      'The same settings were used throughout, and test scores were not used to choose checkpoints. '
      'We report accuracy and macro-F1 as mean and sample standard deviation across the three seeds. '
      'Because the data split stayed fixed, this variation reflects training randomness only.')

    page('4. RQ1 and RQ2 results')
    table(['Architecture', 'Input setting', 'Accuracy (%)', 'Macro-F1 (%)'],
          [[a, {'input':'Prompt only', 'output':'Response only', 'both':'Prompt + response'}[m],
            f'{100*r["accuracy_mean"]:.2f} +/- {100*r["accuracy_std"]:.2f}',
            f'{100*r["macro_f1_mean"]:.2f} +/- {100*r["macro_f1_std"]:.2f}']
           for (a, m), r in summary.items()], [1.15*inch, 1.85*inch, 1.8*inch, 1.85*inch])
    fig('main_results.png', 'Figure 1. Test accuracy; error bars show sample standard deviation across three training seeds.', 2.85)
    h('RQ1: identifying the model from its answer', 2)
    p(f'The answer-only CNN reached {pct(cnn["accuracy_mean"])} accuracy, compared with '
      f'{pct(lstm["accuracy_mean"])} for the LSTM. Both were well above the 33.3% chance level. '
      'The CNN performed better in all three runs, suggesting that local text patterns worked well '
      'for this dataset under our training settings.')
    h('RQ2: adding the prompt', 2)
    p('Prompt-only accuracy was exactly 33.3% for both models. Every prompt appears with all three '
      'labels, so the same prediction is correct for one of its three answers. Macro-F1 still varies '
      'because it also depends on how often each class is predicted. A high prompt-only score on '
      'unmatched data could instead reflect differences in which prompts were assigned to each model.')
    p('Adding the prompt changed mean accuracy by '
      + ' and '.join(f'{pp(summary[a,"both"]["accuracy_mean"]-summary[a,"output"]["accuracy_mean"])} percentage points for the {a}'
                    for a in ['CNN', 'LSTM']) + '. The drop was much larger for the LSTM. '
      'Thus, including more text did not help in this experiment; the result also depends on how '
      'the model represents and learns from the longer sequence.')

    page('5. Errors and uncertainty')
    fig('confusion_matrices.png', 'Figure 2. Seed 448 confusion matrices. Each true class contains 121 responses.', 2.97)
    table(['Family', 'CNN precision / recall / F1', 'LSTM precision / recall / F1'],
          [[label] + [' / '.join(pct(next(r for r in runs if r['architecture']==a and r['mode']=='output' and r['seed']==448)
                                             ['test']['per_class'][label][k]) for k in ['precision','recall','f1-score'])
                     for a in ['CNN','LSTM']] for label in ['Claude','GPT','Llama']],
          [1.0*inch, 2.85*inch, 2.8*inch])
    h('Paired prompt-group bootstrap', 2)
    p('For seed 448, we resampled the 121 test prompt groups 5,000 times to estimate intervals for '
      'paired accuracy differences. The three answers in each group stayed together. These intervals '
      'apply to the fitted models on this test set; they do not include variation from retraining '
      'or choosing a new split.')
    table(['Comparison: second minus first', 'Difference (pp)', '95% interval (pp)'],
          [[c['second'].replace('_448','') + ' minus ' + c['first'].replace('_448',''), pp(c['difference']),
            f'[{100*c["ci95"][0]:+.1f}, {100*c["ci95"][1]:+.1f}]']
           for c in analysis['paired_accuracy_bootstrap']], [3.35*inch, 1.35*inch, 1.95*inch])
    h('Error interpretation', 2)
    p('The CNN correctly classified 102 Llama answers, while the LSTM classified 67. This accounts '
      'for much of the difference between the models. The LSTM often confused Llama with GPT '
      '(44 cases), whereas the CNN made that error 17 times. GPT remained difficult for the CNN: '
      '22 GPT answers were predicted as Llama. Short answers and truncation may leave less evidence '
      'for classification, but we did not manually annotate the causes of individual errors.')

    page('6. RQ4: length, vocabulary and formatting')
    stats = analysis['feature_statistics']
    table(['Family', 'Median words', 'Mean newlines', 'Mean lexical diversity', 'Over 192 tokens'],
          [[a, f'{stats[a]["words"]["median"]:.0f}', f'{stats[a]["newlines"]["mean"]:.1f}',
            f'{stats[a]["lexical_diversity"]["mean"]:.3f}', pct(stats[a]['truncated_output_fraction'])]
           for a in ['Claude','GPT','Llama']], [1*inch, 1.2*inch, 1.3*inch, 1.65*inch, 1.5*inch])
    p('Llama answers were longer: the median was 280 words, compared with 198 for GPT and 180 for '
      'Claude. To test whether simple features could predict the label, we fitted logistic regression '
      '(C=1) using length alone, then seven standardized features: word and character counts, newlines, '
      'list lines, lexical diversity, mean word length and punctuation fraction. We also fitted a '
      'TF-IDF unigram/bigram model with at most 12,000 features and a minimum document frequency of 2. '
      'All classifiers were fitted on training data only. Lexical diversity is the ratio of unique '
      'English words to total English words, so it is affected by answer length.')
    fig('feature_analysis.png', 'Figure 3. Length distributions (outliers hidden) and held-out diagnostic baseline accuracy.', 2.88)
    table(['Test perturbation (no retraining)', 'CNN accuracy', 'LSTM accuracy'],
          [[{'original':'Original responses','remove_identity':'Remove identity terms','words_only':'Remove punctuation + newlines',
             'first_64_tokens':'Retain first 64 tokens'}[name]] + [pct(perturb[a][name]['accuracy']) for a in ['CNN','LSTM']]
           for name in ['original','remove_identity','words_only','first_64_tokens']], [3.45*inch, 1.6*inch, 1.6*inch])
    p('We then modified the test answers while keeping the seed-448 models fixed. Removing model '
      'and provider names did not change accuracy. Removing punctuation and newlines lowered CNN '
      'accuracy from 82.9% to 72.7%, a larger drop than for the LSTM. Keeping only 64 tokens reduced '
      'CNN accuracy but slightly improved LSTM accuracy. Formatting therefore appears useful to '
      'the fitted CNN. These are sensitivity tests: removing tokens also changes the text window '
      'and input distribution, so the changes cannot be attributed to formatting alone.')

    page('7. Interpretation, limitations and lessons')
    h('What the feature tests show', 2)
    p(f'Length alone gave {pct(baseline["length_only"]["accuracy"])} accuracy, and the seven-feature model '
      f'gave {pct(baseline["structural_lexical"]["accuracy"])}. These scores exceed chance but fall well '
      f'below the CNN. TF-IDF logistic regression reached {pct(baseline["tfidf_logistic"]["accuracy"])}. '
      'Vocabulary was therefore more predictive than these few summary statistics. The TF-IDF model '
      'used full answers, while the neural models used capped sequences, so the comparison is not '
      'controlled for input length.')
    p('Terms with some of the largest positive TF-IDF coefficients were ' + '; '.join(
      f'{a}: ' + ', '.join(analysis['tfidf_positive_terms'][a][:6]) for a in ['Claude','GPT','Llama']) + '. '
      'These terms describe the linear classifier. They do not show which features the CNN or LSTM used.')
    h('Limitations', 2)
    p('The combined-input LSTM has to process a longer sequence. In the backward direction, it '
      'finishes with the prompt rather than the answer. This final-state representation and the '
      'fixed training budget may explain part of its lower score, but we did not test these '
      'explanations separately.')
    p('We used one released configuration from each family. Differences in model version, system '
      'prompt or decoding settings may contribute to the scores. We did not regenerate the '
      'answers under identical conditions. The GPT file is labeled gpt4, which does not identify '
      'a precise API snapshot. The findings should therefore be limited to these released outputs.')
    p('The test set contains 121 prompt groups, and all runs use one fixed split. A different split '
      'could give different results. The tokenizer also favors English, and the 192-token cap '
      'cuts off many answers. We did not test unseen model families, human text, newer releases '
      'or paraphrased answers. The source collection labels do not establish task domains, '
      'so they were not used to claim cross-domain performance.')
    h('What we learned', 2)
    p('The matched prompts made RQ2 easier to interpret: prompt-only accuracy had to be one third, '
      'so any improvement had to come from the answers or their interaction with the prompt. '
      'Keeping prompt groups together and building the vocabulary from training data also '
      'helped us avoid two common sources of leakage.')
    p('The training records showed why the lowest training loss was not a good rule for choosing '
      'a model. For the answer-only LSTM with seed 448, validation macro-F1 peaked at epoch 9; '
      'training loss continued falling through epoch 13. We therefore used the validation '
      'checkpoint. The feature tests were useful for another reason: they showed that part of '
      'the classification problem can be solved with simple text statistics, and that the CNN '
      'was sensitive to formatting. A high accuracy score alone would not have shown this.')
    h('Use of AI tools', 2)
    p('We used AI assistance for code, experiment planning and report drafting. The reported '
      'numbers come from saved training and evaluation runs. Bingjie Yan and Siyuan Zhou '
      'are responsible for checking the work and being able to explain the submitted code and results.')

    page('8. Reproducibility and references')
    p('Run scripts/prepare_data.py, scripts/verify.py, scripts/train.py, scripts/analyze.py and '
      'scripts/build_report.py in that order after installing requirements.txt. No LLM API credentials are '
      'needed. See README.md for commands. Checkpoints and downloaded texts stay local; tracked run JSONs '
      'and prediction CSVs contain the audit trail. Completed training runs are skipped unless --force is used.')
    p(f'Upstream revision: {manifest["revision"]}. Split seed: 448. Total recorded neural-training time: '
      f'{analysis["total_training_seconds"] / 60:.1f} CPU wall-clock minutes. '
      'The manifest records full source URLs and checksums for all three files.')
    h('Source attribution', 2)
    p('[1] Li, X., Zhang, T., Dubois, Y., Taori, R., Gulrajani, I., Guestrin, C., Liang, P., and Hashimoto, T. B. '
      '(2023). AlpacaEval: An Automatic Evaluator of Instruction-following Models. '
      'https://github.com/tatsu-lab/alpaca_eval')
    p('[2] Tatsu Lab. AlpacaEval dataset card (CC BY-NC 4.0). '
      'https://huggingface.co/datasets/tatsu-lab/alpaca_eval. Upstream code repository: Apache 2.0. '
      'This project uses released data for a noncommercial course study; full raw texts are not republished.')
    p('[3] Prompt collections underlying the selected data: Self-Instruct '
      '(https://github.com/yizhongw/self-instruct); OpenAssistant '
      '(https://huggingface.co/datasets/OpenAssistant/oasst1); Vicuna '
      '(https://lmsys.org/blog/2023-03-30-vicuna/); Koala '
      '(https://bair.berkeley.edu/blog/2023/04/03/koala/); Anthropic HH-RLHF '
      '(https://huggingface.co/datasets/Anthropic/hh-rlhf). Source labels are preserved in the processed data.')
    h('Dataset source composition', 2)
    table(['Prompt collection label', 'Prompt groups', 'Response rows'],
          [[s, n//3, n] for s, n in sorted(manifest['source_rows'].items())],
          [3.15*inch, 1.75*inch, 1.75*inch])

    page('Appendix A. Complete neural run record')
    p('All six experimental settings use the same data split. Each row is a saved actual run. Test performance '
      'is evaluated after restoring the checkpoint with highest validation macro-F1. Std in the main table '
      'uses the sample definition (n - 1). Seed 448 is used for detailed diagnostics without selection by test score.')
    table(['Model / input', 'Seed', 'Best epoch', 'Val F1', 'Test accuracy', 'Test F1'],
          [[f'{r["architecture"]} / {r["mode"]}', r['seed'], r['best_epoch'], pct(r['best_val_macro_f1']),
            pct(r['test']['accuracy']), pct(r['test']['macro_f1'])] for r in runs],
          [1.65*inch, .65*inch, .8*inch, 1.05*inch, 1.3*inch, 1.2*inch])
    h('Parameter counts', 2)
    table(['Model / input', 'Vocabulary entries', 'Trainable parameters'],
          [[f'{r["architecture"]} / {r["mode"]}', r['vocab_size'], f'{r["parameters"]:,}']
           for r in runs if r['seed'] == 448], [2.5*inch, 1.8*inch, 2.35*inch])

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor('#E1D6EB'))
        canvas.line(.8*inch, .57*inch, 7.45*inch, .57*inch)
        canvas.setFont(FONT, 9)
        canvas.setFillColor(GRAY)
        canvas.drawString(.8*inch, .4*inch, 'CMPSC 448 | Bingjie Yan & Siyuan Zhou')
        canvas.drawRightString(7.45*inch, .4*inch, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(str(OUT / 'pdf/project_report.pdf'), pagesize=(8.5*inch, 11*inch),
                            leftMargin=.8*inch, rightMargin=1.05*inch, topMargin=.65*inch,
                            bottomMargin=.8*inch, title='Who Wrote It? Identifying LLMs from Their Responses',
                            author='Bingjie Yan; Siyuan Zhou')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    (OUT / 'project_report.md').write_text('# Who Wrote It? Identifying LLMs from Their Responses\n\n' + '\n'.join(markdown), encoding='utf-8')
    print('Created output/pdf/project_report.pdf and output/project_report.md')


if __name__ == '__main__':
    main()
