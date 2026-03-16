"""Sensitive-label helpers adapted from the STAQ notebook workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from staq.core.clip_features import compute_similarity_scores, encode_images

DEFAULT_SENSITIVE_PATTERNS = [
    "person",
    "human",
    "rider",
    "driver",
    "passenger",
    "pilot",
    "copilot",
    "flight attendant",
]


def build_sensitive_index(concepts: list[str], patterns: list[str] | None = None) -> torch.Tensor:
    patterns = DEFAULT_SENSITIVE_PATTERNS if patterns is None else patterns
    return torch.tensor(
        [idx for idx, concept in enumerate(concepts) if any(pattern in concept.lower() for pattern in patterns)],
        dtype=torch.long,
    )


@torch.no_grad()
def compute_s_from_image_features(
    image_features: torch.Tensor,
    logit_scale,
    dictionary: torch.Tensor,
    sens_idx: torch.Tensor,
    tau: float = 0.7,
    topk: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    sims = compute_similarity_scores(
        image_features=image_features,
        dictionary=dictionary,
        logit_scale=logit_scale,
    )
    sens_scores = sims[:, sens_idx.to(sims.device)]
    k = min(topk, sens_scores.size(1))
    s_soft = sens_scores.topk(k=k, dim=1).values.mean(dim=1)
    s_hard = (s_soft >= tau).float()
    return s_soft, s_hard


@torch.no_grad()
def compute_s_batch(
    images: torch.Tensor,
    model_clip,
    dictionary: torch.Tensor,
    sens_idx: torch.Tensor,
    clip_device: torch.device,
    tau: float = 0.7,
    topk: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_features = encode_images(model_clip=model_clip, images=images, device=clip_device)
    return compute_s_from_image_features(
        image_features=image_features,
        logit_scale=model_clip.logit_scale.exp(),
        dictionary=dictionary,
        sens_idx=sens_idx,
        tau=tau,
        topk=topk,
    )


def build_sensitive_labels(
    loader,
    model_clip,
    dictionary: torch.Tensor,
    sens_idx: torch.Tensor,
    clip_device: torch.device,
    tau: float = 0.7,
    topk: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    soft_chunks = []
    hard_chunks = []
    for images, _ in loader:
        s_soft, s_hard = compute_s_batch(
            images=images,
            model_clip=model_clip,
            dictionary=dictionary,
            sens_idx=sens_idx,
            clip_device=clip_device,
            tau=tau,
            topk=topk,
        )
        soft_chunks.append(s_soft.cpu())
        hard_chunks.append(s_hard.cpu())
    return torch.cat(soft_chunks).numpy(), torch.cat(hard_chunks).numpy()


def save_sensitive_labels(output_dir: str | Path, train_soft, train_hard, test_soft, test_hard) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "s_soft_train.npy", train_soft)
    np.save(output_dir / "s_hard_train.npy", train_hard)
    np.save(output_dir / "s_soft_test.npy", test_soft)
    np.save(output_dir / "s_hard_test.npy", test_hard)


def load_sensitive_labels(output_dir: str | Path) -> dict[str, np.ndarray]:
    output_dir = Path(output_dir)
    return {
        "s_soft_train": np.load(output_dir / "s_soft_train.npy"),
        "s_hard_train": np.load(output_dir / "s_hard_train.npy"),
        "s_soft_test": np.load(output_dir / "s_soft_test.npy"),
        "s_hard_test": np.load(output_dir / "s_hard_test.npy"),
    }
