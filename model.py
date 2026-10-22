"""
TweFuse-W model architecture.

Matches the project specification:
  - Text encoder: BERTweet (frozen), [CLS] -> 768-d
  - Image encoder: Swin Transformer (frozen), pooled -> 1024-d
  - Projection heads (not shared): text 768->768->768, image 1024->768->768
  - Gating network: concat(text, image) [1536] -> 128 -> 2 -> softmax  =>  alpha_T, alpha_V
  - Fusion: F_WS = alpha_T * T + alpha_V * V   (alpha_V = 1 - alpha_T)
  - Classifier: 768 -> 6 classes (softmax)

Modality-conflict extension (project's proposed contribution):
  - text_only_classifier and image_only_classifier give independent
    predictions from each modality's projected representation, so the
    dashboard can compare them and flag disagreement between modalities
    (separately from what the fusion gate itself trusts).

NOTE ON CHECKPOINTS
--------------------
This file defines the module names (text_encoder, image_encoder, text_proj,
image_proj, gate, classifier, text_only_classifier, image_only_classifier).
`load_state_dict(..., strict=False)` is used in app.py so that a checkpoint
missing the two conflict heads (e.g. if you only trained the core TweFuse-W
model and trained unimodal ablation heads separately) will still load the
core model correctly — you'll just see a "missing keys" warning in the
sidebar for the parts that weren't in your checkpoint.

If your own training code used different module/attribute names, either
rename the modules below to match, or remap keys before calling
`load_state_dict` in app.py's `load_model()`.
"""

import torch
import torch.nn as nn
from transformers import AutoModel

try:
    import timm
except ImportError as e:
    raise ImportError(
        "timm is required for the Swin Transformer image encoder. "
        "Install with: pip install timm"
    ) from e


class ProjectionHead(nn.Module):
    """Two-layer MLP projection head: in_dim -> hidden_dim -> out_dim."""

    def __init__(self, in_dim: int, hidden_dim: int = 768, out_dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatingNetwork(nn.Module):
    """Sample-conditioned gate producing (alpha_T, alpha_V), alpha_V = 1 - alpha_T."""

    def __init__(self, in_dim: int = 1536, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, text_feat: torch.Tensor, image_feat: torch.Tensor):
        combined = torch.cat([text_feat, image_feat], dim=-1)
        logits = self.net(combined)
        weights = torch.softmax(logits, dim=-1)
        alpha_t = weights[:, 0]
        alpha_v = weights[:, 1]
        return alpha_t, alpha_v


class TweFuseW(nn.Module):
    """
    Adaptive multimodal fake-news classifier (TweFuse-W) with a
    modality-conflict extension for the reliability dashboard.
    """

    CLASS_NAMES = [
        "True",
        "Satire",
        "Misleading Content",
        "False Connection",
        "Manipulated Content",
        "Imposter Content",
    ]

    def __init__(
        self,
        num_classes: int = 6,
        bertweet_name: str = "vinai/bertweet-base",
        swin_name: str = "swin_base_patch4_window7_224",
        freeze_encoders: bool = True,
    ):
        super().__init__()

        # --- Frozen pretrained encoders ---
        self.text_encoder = AutoModel.from_pretrained(bertweet_name)
        self.image_encoder = timm.create_model(swin_name, pretrained=True, num_classes=0)  # pooled features

        if freeze_encoders:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            for p in self.image_encoder.parameters():
                p.requires_grad = False

        text_in_dim = self.text_encoder.config.hidden_size          # 768 for BERTweet-base
        image_in_dim = self.image_encoder.num_features               # 1024 for swin_base

        # --- Trainable components ---
        self.text_proj = ProjectionHead(text_in_dim, 768, 768)
        self.image_proj = ProjectionHead(image_in_dim, 768, 768)
        self.gate = GatingNetwork(in_dim=768 * 2, hidden_dim=128)
        self.classifier = nn.Linear(768, num_classes)

        # Independent unimodal heads used only for the modality-conflict
        # extension (comparing text-only vs image-only predictions).
        self.text_only_classifier = nn.Linear(768, num_classes)
        self.image_only_classifier = nn.Linear(768, num_classes)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = out.last_hidden_state[:, 0, :]
        return self.text_proj(cls_token)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pooled = self.image_encoder(pixel_values)
        return self.image_proj(pooled)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, pixel_values: torch.Tensor):
        text_feat = self.encode_text(input_ids, attention_mask)
        image_feat = self.encode_image(pixel_values)

        alpha_t, alpha_v = self.gate(text_feat, image_feat)
        fused = alpha_t.unsqueeze(-1) * text_feat + alpha_v.unsqueeze(-1) * image_feat

        logits = self.classifier(fused)
        text_only_logits = self.text_only_classifier(text_feat)
        image_only_logits = self.image_only_classifier(image_feat)

        return {
            "logits": logits,
            "alpha_t": alpha_t,
            "alpha_v": alpha_v,
            "text_only_logits": text_only_logits,
            "image_only_logits": image_only_logits,
            "text_feat": text_feat,
            "image_feat": image_feat,
        }
