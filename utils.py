"""
Preprocessing and modality-conflict utilities for the Reliability Dashboard.

Text/image preprocessing here mirrors what's described in the project's
preprocessing report and objectives:
  - Text: tokenized with the pretrained encoder's tokenizer, padded/truncated
    to 80 tokens.
  - Image: resized to 224x224, normalized with ImageNet statistics.
"""

from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

MAX_TEXT_LEN = 80

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def preprocess_text(text: str, tokenizer, max_len: int = MAX_TEXT_LEN) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenize + pad/truncate a post's text, matching the paper's pipeline."""
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return encoded["input_ids"], encoded["attention_mask"]


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Resize/normalize a post's image, matching the paper's pipeline."""
    return IMAGE_TRANSFORM(image.convert("RGB")).unsqueeze(0)


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    JS divergence between two probability distributions, base-2, clipped to
    [0, 1]. Used to quantify how much the independent text-only and
    image-only predictions disagree.
    """
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        return float(np.sum(a * np.log2(a / b)))

    js = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return float(np.clip(js, 0.0, 1.0))


def modality_conflict_level(
    text_probs: np.ndarray,
    image_probs: np.ndarray,
    js_medium_threshold: float = 0.15,
    js_high_threshold: float = 0.35,
) -> Tuple[str, float, bool]:
    """
    Classify the disagreement between the independent text-only and
    image-only predictions into Low / Medium / High, as proposed in the
    project's Modality Conflict extension.

    Logic:
      - Same predicted class + low divergence          -> Low
      - Same predicted class + higher divergence,
        OR different predicted class + low/med conf gap -> Medium
      - Different predicted class + high divergence AND
        both branches are individually confident        -> High

    These thresholds are a reasonable starting point for the extension —
    tune js_medium_threshold / js_high_threshold against a labeled
    validation slice once you have real model outputs to calibrate on.

    Returns: (level, js_divergence, branches_disagree)
    """
    text_pred = int(np.argmax(text_probs))
    image_pred = int(np.argmax(image_probs))
    js = jensen_shannon_divergence(text_probs, image_probs)
    disagree = text_pred != image_pred

    text_conf = float(text_probs[text_pred])
    image_conf = float(image_probs[image_pred])
    min_conf = min(text_conf, image_conf)

    if not disagree:
        level = "Low" if js < js_medium_threshold else "Medium"
    else:
        if js >= js_high_threshold and min_conf >= 0.5:
            level = "High"
        else:
            level = "Medium"

    return level, js, disagree


def evaluate_modality_similarity(
    text_probs: np.ndarray,
    image_probs: np.ndarray,
    js_medium_threshold: float = 0.15,
    js_high_threshold: float = 0.35,
    decision_mode: str = "Disagreement or High Divergence",
) -> dict:
    """
    Evaluates whether the image and text modalities are similar or dissimilar.
    If dissimilar, the system triggers separate answers for image and text data.
    """
    level, js, disagree = modality_conflict_level(
        text_probs, image_probs, js_medium_threshold, js_high_threshold
    )
    t_pred = int(np.argmax(text_probs))
    i_pred = int(np.argmax(image_probs))
    t_conf = float(text_probs[t_pred])
    i_conf = float(image_probs[i_pred])
    similarity_score = float(np.clip(1.0 - js, 0.0, 1.0))

    if decision_mode == "Always Separate":
        is_dissimilar = True
    elif decision_mode == "Strict (Any Divergence JS ≥ Medium)":
        is_dissimilar = disagree or (js >= js_medium_threshold)
    else:  # Default: "Disagreement or High Divergence"
        is_dissimilar = disagree or (level == "High")

    return {
        "is_dissimilar": bool(is_dissimilar),
        "is_similar": bool(not is_dissimilar),
        "conflict_level": level,
        "js_divergence": js,
        "similarity_score": similarity_score,
        "branches_disagree": disagree,
        "text_pred": t_pred,
        "image_pred": i_pred,
        "text_conf": t_conf,
        "image_conf": i_conf,
    }

