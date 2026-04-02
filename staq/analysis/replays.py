"""Replay sampling helpers for sample-level STAQ analysis."""

from __future__ import annotations

import numpy as np
import torch
from tqdm.auto import tqdm

from staq.analysis.rollouts import (
    build_random_initial_history,
    first_divergence_step,
    rollout_until_confidence,
)


def format_history(history_idx: list[int], answers_row: torch.Tensor, concepts: list[str], max_items: int = 4) -> list[str]:
    parts = []
    for idx in history_idx[:max_items]:
        answer_name = "yes" if float(answers_row[idx].item()) > 0 else "no"
        parts.append(f"{concepts[idx]}={answer_name}")
    if len(history_idx) > max_items:
        parts.append("...")
    return parts


def _build_record(
    sample_idx: int,
    trial: int,
    label_idx: int,
    label_name: str,
    init_history_idx: list[int],
    answers_row: torch.Tensor,
    baseline_stop: dict,
    staq_stop: dict,
    concepts: list[str],
) -> dict:
    both_correct = bool(
        (baseline_stop["final_pred_idx"] == label_idx) and (staq_stop["final_pred_idx"] == label_idx)
    )
    return {
        "sample_idx": int(sample_idx),
        "trial": int(trial),
        "label_idx": int(label_idx),
        "label_name": label_name,
        "initial_history": format_history(init_history_idx, answers_row, concepts, max_items=max(1, len(init_history_idx))),
        "initial_history_size": int(len(init_history_idx)),
        "baseline": baseline_stop,
        "staq": staq_stop,
        "both_correct": both_correct,
        "sensitive_gap": int(baseline_stop["sensitive_steps"] - staq_stop["sensitive_steps"]),
        "first_divergence_step": first_divergence_step(baseline_stop["sequence"], staq_stop["sequence"]),
    }


def sample_intuition_replays(
    dataset,
    answer_builder,
    baseline_bundle: dict,
    staq_bundle: dict,
    concepts: list[str],
    sensitive_mask: torch.Tensor,
    class_names: list[str],
    num_cases: int = 8,
    pool_size: int = 400,
    num_trials: int = 2,
    random_seed: int = 0,
    min_history: int = 1,
    max_history: int = 2,
    history_mode: str = "non_sensitive",
    require_nontrivial: bool = True,
    prefer_divergent: bool = True,
    prefer_baseline_sensitive: bool = True,
) -> list[dict]:
    rng = np.random.default_rng(random_seed)
    sample_indices = rng.permutation(len(dataset))[: min(pool_size, len(dataset))]
    sensitive_indices = (sensitive_mask > 0.5).nonzero(as_tuple=False).flatten().cpu()
    records = []

    with torch.no_grad():
        for sample_idx in tqdm(sample_indices, desc="Sampling intuition replays"):
            image, label_idx = dataset[int(sample_idx)]
            answers = answer_builder(image.unsqueeze(0))
            answers_row = answers[0]
            label_idx = int(label_idx)
            label_name = class_names[label_idx]

            for trial in range(num_trials):
                init_mask, _, init_history_idx = build_random_initial_history(
                    answers=answers,
                    sample_idx=int(sample_idx),
                    trial=trial,
                    min_history=min_history,
                    max_history=max_history,
                    mode=history_mode,
                    sensitive_indices=sensitive_indices,
                )
                baseline_stop = rollout_until_confidence(
                    bundle=baseline_bundle,
                    answers_row=answers_row,
                    init_mask=init_mask,
                    concepts=concepts,
                    sensitive_mask=sensitive_mask,
                    class_names=class_names,
                )
                staq_stop = rollout_until_confidence(
                    bundle=staq_bundle,
                    answers_row=answers_row,
                    init_mask=init_mask,
                    concepts=concepts,
                    sensitive_mask=sensitive_mask,
                    class_names=class_names,
                )
                if require_nontrivial and baseline_stop["queries_asked"] == 0 and staq_stop["queries_asked"] == 0:
                    continue
                records.append(
                    _build_record(
                        sample_idx=int(sample_idx),
                        trial=trial,
                        label_idx=label_idx,
                        label_name=label_name,
                        init_history_idx=init_history_idx,
                        answers_row=answers_row,
                        baseline_stop=baseline_stop,
                        staq_stop=staq_stop,
                        concepts=concepts,
                    )
                )

    def _intuition_sort_key(row: dict):
        divergence = row["first_divergence_step"]
        baseline_sensitive = row["baseline"]["sensitive_steps"] > 0
        staq_avoids_sensitive = row["baseline"]["sensitive_steps"] > row["staq"]["sensitive_steps"]
        return (
            1 if (prefer_baseline_sensitive and baseline_sensitive) else 0,
            1 if staq_avoids_sensitive else 0,
            1 if row["both_correct"] else 0,
            1 if (prefer_divergent and divergence is not None) else 0,
            abs(row["baseline"]["queries_asked"] - row["staq"]["queries_asked"]),
            abs(row["sensitive_gap"]),
            row["baseline"]["sensitive_steps"],
            max(row["baseline"]["queries_asked"], row["staq"]["queries_asked"]),
            -(divergence if divergence is not None else 999),
        )

    baseline_sensitive_gap = [
        row for row in records if row["baseline"]["sensitive_steps"] > 0 and row["sensitive_gap"] > 0
    ]
    baseline_sensitive_records = [row for row in records if row["baseline"]["sensitive_steps"] > 0]

    if prefer_baseline_sensitive and baseline_sensitive_gap:
        candidate_pool = baseline_sensitive_gap
    elif prefer_baseline_sensitive and baseline_sensitive_records:
        candidate_pool = baseline_sensitive_records
    else:
        candidate_pool = records

    candidate_pool = sorted(candidate_pool, key=_intuition_sort_key, reverse=True)
    selected = []
    seen = set()
    for row in candidate_pool:
        if row["sample_idx"] in seen:
            continue
        selected.append(row)
        seen.add(row["sample_idx"])
        if len(selected) >= num_cases:
            break
    return selected
