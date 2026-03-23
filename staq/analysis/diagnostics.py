"""Small diagnostic helpers for probing sensitive concept behavior."""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from tqdm.auto import tqdm

from staq.analysis.rollouts import build_random_initial_history, rollout_until_confidence


def probe_topk_sensitive_queries(
    dataset,
    answer_builder,
    bundle: dict,
    concepts: list[str],
    sensitive_mask: torch.Tensor,
    class_names: list[str],
    matched_sensitive_concepts: list[str],
    pool_size: int = 400,
    num_trials: int = 3,
    min_history: int = 0,
    max_history: int = 1,
    history_mode: str = "non_sensitive",
    topk_steps: int = 3,
    random_seed: int = 0,
    label_tag: str = "S",
) -> dict:
    rng = np.random.default_rng(random_seed)
    sample_indices = rng.permutation(len(dataset))[: min(pool_size, len(dataset))]
    sensitive_indices = (sensitive_mask > 0.5).nonzero(as_tuple=False).flatten().cpu()

    hit_states = 0
    sensitive_query_total = 0
    total_states = 0
    position_hits = Counter()
    sensitive_concept_hits = Counter()
    overall_topk_concepts = Counter()
    class_hit_states = Counter()
    examples = []

    with torch.no_grad():
        for sample_idx in tqdm(sample_indices, desc="Baseline sensitive probe"):
            image, label_idx = dataset[int(sample_idx)]
            answers = answer_builder(image.unsqueeze(0))
            answers_row = answers[0]
            label_name = class_names[int(label_idx)]

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
                rollout = rollout_until_confidence(
                    bundle=bundle,
                    answers_row=answers_row,
                    init_mask=init_mask,
                    concepts=concepts,
                    sensitive_mask=sensitive_mask,
                    class_names=class_names,
                    threshold=2.0,
                    max_steps=topk_steps,
                )
                topk = rollout["sequence"][:topk_steps]
                total_states += 1
                has_sensitive = False

                for item in topk:
                    overall_topk_concepts[item["concept"]] += 1
                    if item["sensitive"]:
                        has_sensitive = True
                        sensitive_query_total += 1
                        position_hits[item["step"]] += 1
                        sensitive_concept_hits[item["concept"]] += 1

                if has_sensitive:
                    hit_states += 1
                    class_hit_states[label_name] += 1
                    if len(examples) < 5:
                        examples.append(
                            {
                                "sample_idx": int(sample_idx),
                                "label": label_name,
                                "initial_history": [
                                    f"{concepts[idx]}={'yes' if float(answers_row[idx].item()) > 0 else 'no'}"
                                    for idx in init_history_idx
                                ],
                                "topk": [
                                    f"{item['step']}. {item['concept']}"
                                    + (f" [{label_tag}]" if item["sensitive"] else "")
                                    for item in topk
                                ],
                            }
                        )

    summary = {
        "matched_sensitive_concepts": matched_sensitive_concepts,
        "pool_size": int(len(sample_indices)),
        "num_trials": int(num_trials),
        "history_mode": history_mode,
        "history_range": [int(min_history), int(max_history)],
        "forced_topk_steps": int(topk_steps),
        "tested_states": int(total_states),
        "states_with_sensitive_in_topk": int(hit_states),
        "sensitive_in_topk_rate": 0.0 if total_states == 0 else hit_states / total_states,
        "avg_sensitive_queries_in_topk": 0.0 if total_states == 0 else sensitive_query_total / total_states,
        "position_hits": dict(sorted(position_hits.items())),
        "top_sensitive_concepts": sensitive_concept_hits.most_common(10),
        "top_overall_concepts_in_topk": overall_topk_concepts.most_common(15),
        "class_hit_states": class_hit_states.most_common(),
    }
    return {"summary": summary, "examples": examples}
