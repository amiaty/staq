"""Replay sampling and confidence-stop contrast mining adapted from staq.ipynb."""

from __future__ import annotations

from dataclasses import dataclass

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


def _late_sensitive_step(stop_result: dict, max_steps: int) -> int:
    first_sensitive = stop_result["first_sensitive_step"]
    return (max_steps + 1) if first_sensitive is None else int(first_sensitive)


def _build_record(
    sample_idx: int,
    trial: int,
    label_idx: int,
    label_name: str,
    init_history_idx: list[int],
    answers_row: torch.Tensor,
    baseline_stop: dict,
    fair_stop: dict,
    concepts: list[str],
) -> dict:
    return {
        "sample_idx": int(sample_idx),
        "trial": int(trial),
        "label_idx": int(label_idx),
        "label_name": label_name,
        "initial_history": format_history(init_history_idx, answers_row, concepts, max_items=max(1, len(init_history_idx))),
        "initial_history_size": int(len(init_history_idx)),
        "baseline": baseline_stop,
        "fair": fair_stop,
        "baseline_correct": bool(baseline_stop["final_pred_idx"] == label_idx),
        "fair_correct": bool(fair_stop["final_pred_idx"] == label_idx),
        "both_correct": bool(
            (baseline_stop["final_pred_idx"] == label_idx) and (fair_stop["final_pred_idx"] == label_idx)
        ),
        "sensitive_gap": int(baseline_stop["sensitive_steps"] - fair_stop["sensitive_steps"]),
        "delay_gap": int(_late_sensitive_step(fair_stop, 999) - _late_sensitive_step(baseline_stop, 999)),
        "first_divergence_step": first_divergence_step(baseline_stop["sequence"], fair_stop["sequence"]),
    }


def _record_sort_key(row: dict):
    return (
        row["both_correct"],
        row["fair"]["sensitive_steps"] == 0,
        row["sensitive_gap"],
        row["delay_gap"],
        row["baseline"]["sensitive_steps"],
        row["baseline"]["queries_asked"],
    )


def sample_intuition_replays(
    dataset,
    answer_builder,
    baseline_bundle: dict,
    fair_bundle: dict,
    concepts: list[str],
    sensitive_mask: torch.Tensor,
    class_names: list[str],
    num_cases: int = 8,
    pool_size: int = 400,
    num_trials: int = 2,
    random_seed: int = 0,
    min_history: int = 1,
    max_history: int = 2,
    history_mode: str = "random",
    require_nontrivial: bool = True,
    prefer_divergent: bool = True,
) -> list[dict]:
    rng = np.random.default_rng(random_seed)
    sample_indices = rng.permutation(len(dataset))[: min(pool_size, len(dataset))]
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
                    sensitive_indices=(sensitive_mask > 0.5).nonzero(as_tuple=False).flatten().cpu(),
                )
                baseline_stop = rollout_until_confidence(
                    bundle=baseline_bundle,
                    answers_row=answers_row,
                    init_mask=init_mask,
                    concepts=concepts,
                    sensitive_mask=sensitive_mask,
                    class_names=class_names,
                )
                fair_stop = rollout_until_confidence(
                    bundle=fair_bundle,
                    answers_row=answers_row,
                    init_mask=init_mask,
                    concepts=concepts,
                    sensitive_mask=sensitive_mask,
                    class_names=class_names,
                )
                if require_nontrivial and baseline_stop["queries_asked"] == 0 and fair_stop["queries_asked"] == 0:
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
                        fair_stop=fair_stop,
                        concepts=concepts,
                    )
                )

    def _intuition_sort_key(row: dict):
        divergence = row["first_divergence_step"]
        return (
            1 if (prefer_divergent and divergence is not None) else 0,
            1 if row["both_correct"] else 0,
            abs(row["baseline"]["queries_asked"] - row["fair"]["queries_asked"]),
            abs(row["sensitive_gap"]),
            max(row["baseline"]["queries_asked"], row["fair"]["queries_asked"]),
            -(divergence if divergence is not None else 999),
        )

    records = sorted(records, key=_intuition_sort_key, reverse=True)
    selected = []
    seen = set()
    for row in records:
        if row["sample_idx"] in seen:
            continue
        selected.append(row)
        seen.add(row["sample_idx"])
        if len(selected) >= num_cases:
            break
    return selected


def mine_confidence_stop_contrasts(
    loader,
    answer_builder,
    baseline_bundle: dict,
    fair_bundle: dict,
    concepts: list[str],
    sensitive_mask: torch.Tensor,
    class_names: list[str],
    threshold: float,
    max_steps: int,
    min_history: int = 1,
    max_history: int = 2,
    history_mode: str = "random",
    max_search_samples: int | None = None,
    num_trials: int = 4,
    require_both_correct: bool = True,
) -> dict:
    strict_candidates = []
    delay_candidates = []
    baseline_sensitive_examples = []
    stats = {
        "tested_states": 0,
        "both_stop_immediately": 0,
        "baseline_any_sensitive": 0,
        "fair_any_sensitive": 0,
        "gap_positive": 0,
        "both_correct_gap_positive": 0,
        "strict_zero_sensitive_fair": 0,
        "delay_positive": 0,
        "both_correct_delay_positive": 0,
        "baseline_sensitive_both_correct": 0,
    }

    seen = 0
    sensitive_indices = (sensitive_mask > 0.5).nonzero(as_tuple=False).flatten().cpu()

    for images, labels in tqdm(loader, desc="Mining confidence-stop contrasts"):
        if max_search_samples is not None and seen >= max_search_samples:
            break
        if max_search_samples is not None:
            remaining = max_search_samples - seen
            if remaining <= 0:
                break
            if images.size(0) > remaining:
                images = images[:remaining]
                labels = labels[:remaining]

        with torch.no_grad():
            answers_batch = answer_builder(images)
            for row_idx in range(answers_batch.size(0)):
                sample_idx = seen + row_idx
                label_idx = int(labels[row_idx].item())
                label_name = class_names[label_idx]
                answers_row = answers_batch[row_idx]

                for trial in range(num_trials):
                    init_mask, _, init_history_idx = build_random_initial_history(
                        answers=answers_row.unsqueeze(0),
                        sample_idx=sample_idx,
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
                        threshold=threshold,
                        max_steps=max_steps,
                    )
                    fair_stop = rollout_until_confidence(
                        bundle=fair_bundle,
                        answers_row=answers_row,
                        init_mask=init_mask,
                        concepts=concepts,
                        sensitive_mask=sensitive_mask,
                        class_names=class_names,
                        threshold=threshold,
                        max_steps=max_steps,
                    )

                    baseline_correct = baseline_stop["final_pred_idx"] == label_idx
                    fair_correct = fair_stop["final_pred_idx"] == label_idx
                    both_correct = baseline_correct and fair_correct
                    sensitive_gap = int(baseline_stop["sensitive_steps"] - fair_stop["sensitive_steps"])
                    delay_gap = int(_late_sensitive_step(fair_stop, max_steps) - _late_sensitive_step(baseline_stop, max_steps))

                    stats["tested_states"] += 1
                    if baseline_stop["queries_asked"] == 0 and fair_stop["queries_asked"] == 0:
                        stats["both_stop_immediately"] += 1
                    if baseline_stop["sensitive_steps"] > 0:
                        stats["baseline_any_sensitive"] += 1
                    if fair_stop["sensitive_steps"] > 0:
                        stats["fair_any_sensitive"] += 1
                    if sensitive_gap > 0:
                        stats["gap_positive"] += 1
                    if sensitive_gap > 0 and both_correct:
                        stats["both_correct_gap_positive"] += 1
                    if sensitive_gap > 0 and both_correct and fair_stop["sensitive_steps"] == 0:
                        stats["strict_zero_sensitive_fair"] += 1
                    if baseline_stop["sensitive_steps"] > 0 and delay_gap > 0:
                        stats["delay_positive"] += 1
                    if baseline_stop["sensitive_steps"] > 0 and delay_gap > 0 and both_correct:
                        stats["both_correct_delay_positive"] += 1
                    if baseline_stop["sensitive_steps"] > 0 and both_correct:
                        stats["baseline_sensitive_both_correct"] += 1

                    record = _build_record(
                        sample_idx=sample_idx,
                        trial=trial,
                        label_idx=label_idx,
                        label_name=label_name,
                        init_history_idx=init_history_idx,
                        answers_row=answers_row,
                        baseline_stop=baseline_stop,
                        fair_stop=fair_stop,
                        concepts=concepts,
                    )
                    record["delay_gap"] = delay_gap

                    if baseline_stop["sensitive_steps"] > 0:
                        baseline_sensitive_examples.append(record)
                    if baseline_stop["sensitive_steps"] > 0 and delay_gap > 0 and ((not require_both_correct) or both_correct):
                        delay_candidates.append(record)
                    if sensitive_gap > 0 and ((not require_both_correct) or both_correct):
                        strict_candidates.append(record)

        seen += images.size(0)

    strict_candidates = sorted(strict_candidates, key=_record_sort_key, reverse=True)
    delay_candidates = sorted(delay_candidates, key=_record_sort_key, reverse=True)
    baseline_sensitive_examples = sorted(baseline_sensitive_examples, key=_record_sort_key, reverse=True)

    strict_zero_sensitive = [row for row in strict_candidates if row["both_correct"] and row["fair"]["sensitive_steps"] == 0]
    both_correct_gap = [row for row in strict_candidates if row["both_correct"]]
    both_correct_delay = [row for row in delay_candidates if row["both_correct"]]
    baseline_sensitive_both_correct = [row for row in baseline_sensitive_examples if row["both_correct"]]

    bucket_map = {
        "strict_zero_sensitive_fair": strict_zero_sensitive,
        "both_correct_gap_positive": both_correct_gap,
        "all_gap_positive": strict_candidates,
        "both_correct_delay_positive": both_correct_delay,
        "all_delay_positive": delay_candidates,
        "both_correct_baseline_sensitive": baseline_sensitive_both_correct,
        "all_baseline_sensitive": baseline_sensitive_examples,
    }
    bucket_counts = {name: len(rows) for name, rows in bucket_map.items()}
    selection_priority = [
        "strict_zero_sensitive_fair",
        "both_correct_gap_positive",
        "all_gap_positive",
        "both_correct_delay_positive",
        "all_delay_positive",
        "both_correct_baseline_sensitive",
        "all_baseline_sensitive",
    ]
    plot_priority = [
        "strict_zero_sensitive_fair",
        "both_correct_gap_positive",
        "all_gap_positive",
        "both_correct_delay_positive",
        "all_delay_positive",
    ]
    selected_bucket = next((name for name in selection_priority if bucket_counts[name] > 0), "none")
    plot_bucket = next((name for name in plot_priority if bucket_counts[name] > 0), "none")

    return {
        "stats": stats,
        "bucket_map": bucket_map,
        "bucket_counts": bucket_counts,
        "selected_bucket": selected_bucket,
        "plot_bucket": plot_bucket,
        "selected_candidates": [] if selected_bucket == "none" else bucket_map[selected_bucket],
        "plot_candidates": [] if plot_bucket == "none" else bucket_map[plot_bucket],
    }
