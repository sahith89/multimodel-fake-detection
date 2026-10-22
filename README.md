# TweFuse-W Reliability Dashboard

Your contribution to the project — a Streamlit app that loads your trained
TweFuse-W checkpoint and shows, for any (text, image) post:

- **Predicted class** — one of the six Fakeddit categories (True, Satire,
  Misleading Content, False Connection, Manipulated Content, Imposter Content)
- **Confidence** of that prediction
- **αT / αV** — the textual/visual reliability weights from the adaptive
  gating network
- **Modality Conflict** (Low / Medium / High) — your proposed extension,
  comparing the *independent* text-only vs image-only predictions
- **Dynamic Dual-Answer Resolution** — When image and text are **not similar** (conflicting / out-of-context), the model outputs **two separate answers** (one for Text Data, one for Image Media) to pinpoint selective deception without forced fusion bias; when they are similar, it outputs a single unified consensus answer.

## Files

| File | Purpose |
|---|---|
| `model.py` | TweFuse-W architecture: frozen BERTweet + Swin encoders, projection heads, gating network, fusion, classifier, and the two extra unimodal heads used for modality-conflict detection |
| `utils.py` | Text/image preprocessing (matches the paper's pipeline: 80-token padding, 224×224 ImageNet-normalized images) and the Low/Medium/High conflict scoring logic |
| `app.py` | The Streamlit dashboard UI |
| `requirements.txt` | Dependencies |

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Put your trained checkpoint at `checkpoints/twefuse_w.pt` (or point the
sidebar's "Checkpoint path" field at wherever you saved it).

Run:

```bash
streamlit run app.py
```

Then in the sidebar: confirm the checkpoint path / encoder names, click
**Load model**, and on the main page paste a post's text, upload its image,
and click **Analyze post**.

## Expected checkpoint contract

`model.py` defines these module names — `load_state_dict` is called with
`strict=False`, so as long as your saved keys use (or can be renamed to)
these prefixes it will load correctly:

```
text_encoder.*              # BERTweet (frozen — usually not saved, that's fine)
image_encoder.*             # Swin Transformer (frozen — usually not saved, that's fine)
text_proj.net.0 / net.2     # 768 -> 768 -> 768
image_proj.net.0 / net.2    # 1024 -> 768 -> 768
gate.net.0 / net.2          # 1536 -> 128 -> 2
classifier.weight/bias      # 768 -> 6
text_only_classifier.*      # 768 -> 6   (used only for the conflict extension)
image_only_classifier.*     # 768 -> 6   (used only for the conflict extension)
```

`app.py`'s `load_model()` also unwraps common checkpoint conventions
automatically (`{"state_dict": ...}`, `{"model_state_dict": ...}`,
`{"model": ...}`).

**If your training run didn't produce `text_only_classifier` /
`image_only_classifier`:** these two heads only exist to power the
Modality Conflict feature. The app will still load and run the core
TweFuse-W prediction + αT/αV with a "missing keys" warning in the sidebar;
the conflict indicator just won't be meaningful until those heads have
weights. Two ways to get them:

1. **Reuse your ablation-study unimodal models** (the text-only and
   image-only configurations you already train for the ablation study) —
   load their final linear layer's weights into `text_only_classifier` /
   `image_only_classifier` after training, either by saving them into the
   same checkpoint file, or with a small script that copies those two
   layers' `state_dict` entries into the main checkpoint before loading.
2. **Fine-tune them jointly** with the rest of TweFuse-W by adding their
   cross-entropy losses (on `text_only_logits` / `image_only_logits`) to
   your training loop as auxiliary losses — cheap since they only sit on
   top of representations you're already computing.

If your own code uses different attribute names for any of these modules,
either rename the modules in `model.py` to match, or remap the checkpoint's
keys before calling `load_state_dict` in `app.py`.

## Modality Conflict thresholds

The Low/Medium/High conflict level (in `utils.modality_conflict_level`) is
based on Jensen–Shannon divergence between the text-only and image-only
class-probability distributions, plus whether their top predicted class
agrees. The default thresholds (JS ≥ 0.15 → Medium, JS ≥ 0.35 and both
branches individually confident → High) are a reasonable starting point —
the sidebar lets you adjust them live. Once you have real model outputs,
it's worth calibrating these against a labeled validation slice (e.g. plot
JS divergence for correctly vs. incorrectly fused-predicted samples and
pick thresholds that separate them) so you can justify the thresholds in
your report rather than leaving them as defaults.

## Notes

- Text/image preprocessing intentionally mirrors your preprocessing report:
  text is tokenized and padded/truncated to 80 tokens; images are resized to
  224×224 and normalized with ImageNet mean/std.
- Encoders are frozen at inference time (as in training) — only the
  projection heads, gate, and classifiers need to be loaded with your
  trained weights.
- First run will download `vinai/bertweet-base` and the Swin checkpoint
  from Hugging Face / timm if not already cached locally — make sure the
  machine you run this on has internet access (or pre-cache the weights).
