# emotion-recognition-models

Multimodal emotion-recognition baselines — **text**, **speech**, and **facial
expression** — together with a reproducible evaluation suite. These models are the
perception layer behind a blended-care healthbot for clinical screening and remote
monitoring of chronic disease: each modality maps a patient signal onto a shared
clinical emotion vocabulary.

Every model is delivered as a self-contained, documented Jupyter notebook (built to
run on Google Colab) covering data loading, training, and evaluation.

---

## Repository layout

```
emotion-recognition-models/
├── text/
│   └── LSTM_TextSentimentAnalysis.ipynb     # BiLSTM + GloVe text emotion classifier
├── speech/
│   └── Speech_emotion_recognition.ipynb      # emotion2vec+ features → MLP
├── face/
│   ├── 01_train_vit_face.ipynb               # ViT-B/16 (face-pretrained) on FER+
│   ├── 02_train_convnext_large.ipynb         # ConvNeXt-large (IN-22k) on FER+
│   └── 03_ensemble_inference.ipynb           # soft-vote ensemble of the two
├── LICENSE                                   # Apache-2.0
└── README.md
```

---

## Models

### Text — Bidirectional LSTM

[`text/LSTM_TextSentimentAnalysis.ipynb`](text/LSTM_TextSentimentAnalysis.ipynb)

Classifies an English sentence into one of six emotions — **anger, fear, joy, love,
sadness, surprise**. A 3-layer bidirectional LSTM over fine-tuned 200-dimensional
GloVe embeddings, with dropout, L2 regularization, and balanced class weights to
handle the heavy class imbalance (`joy` has ~9× the samples of `surprise`).

- **Dataset:** [Emotions Dataset for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp) — ~20 000 tweets, pre-split train/val/test.
- **Pipeline:** EDA → preprocessing (lowercasing, stop-word removal with negation preserved, lemmatization) → tokenization & padding (max length from the p99 of token lengths) → GloVe → BiLSTM → systematic error analysis.
- **Results:** accuracy **0.930**, macro-F1 **0.889**, weighted-F1 **0.930**.

### Speech — emotion2vec+ → MLP

[`speech/Speech_emotion_recognition.ipynb`](speech/Speech_emotion_recognition.ipynb)

A Speech Emotion Recognition pipeline using
[**emotion2vec_plus_large**](https://github.com/modelscope/FunASR) (Alibaba's FunASR)
as a frozen feature extractor, followed by a custom PyTorch MLP classifier. Labels
are encoded as `{gender}_{emotion}` and the model is evaluated both globally and
**per-dataset** to expose where it struggles.

| Dataset | Samples | Gender | Emotions |
|---|---|---|---|
| RAVDESS | ~1 440 | M + F | 7 |
| CREMA-D | ~7 442 | M + F | 6 (no surprise) |
| TESS | ~2 800 | F only | 7 |
| SAVEE | ~480 | M only | 7 |

### Face — ViT + ConvNeXt ensemble on FER+

[`face/01_train_vit_face.ipynb`](face/01_train_vit_face.ipynb) ·
[`face/02_train_convnext_large.ipynb`](face/02_train_convnext_large.ipynb) ·
[`face/03_ensemble_inference.ipynb`](face/03_ensemble_inference.ipynb)

Two backbones are fine-tuned independently on the **FER+** relabeling of FER2013
(8 classes, soft labels from per-image annotator vote distributions; 28 493 train /
3 569 val / 3 563 test). They are chosen to be *architecturally and pretrain-domain
orthogonal*, so a soft-vote ensemble reduces error rather than merely averaging it.

Both share the same recipe: single 48→224 resize before augmentation
(RandAugment + ColorJitter + RandomErasing), weighted **soft cross-entropy** against
the vote distributions, two-stage fine-tuning (head-only then full with layer-wise
LR decay), EMA weight averaging, MixUp/CutMix with interpolated soft targets, and
test-time augmentation (flip + 5-crop + multi-scale, 16 passes/image).

| Model | Backbone | Pretrain | Params | Test acc | Macro-F1 |
|---|---|---|---|---|---|
| `01` ViT | ViT-B/16 ([dima806](https://huggingface.co/dima806/facial_emotions_image_detection)) | face emotion (AffectNet+FER) | ~86M | 0.848 | 0.696 |
| `02` ConvNeXt | [convnext-large-224-22k-1k](https://huggingface.co/facebook/convnext-large-224-22k-1k) | ImageNet-22k | 198M | 0.852 | 0.676 |
| `03` **soft-vote** | ViT + ConvNeXt | — | — | **0.863** | **0.731** |

The ensemble's +1.1 pp accuracy / +3.5 pp macro-F1 over the better single model
comes from genuine error decorrelation — per-class F1 improves most on the
rare/ambiguous classes (Disgust, Fear, Contempt). Notebook `01` produces a probability
cache consumed by `03`.

## Getting started

The three model notebooks are self-contained and intended for **Google Colab** (GPU
runtime); they install their own dependencies and mount Google Drive for the audio /
image datasets.

---

## License

Released under the [Apache License 2.0](LICENSE).
