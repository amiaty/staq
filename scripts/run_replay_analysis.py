"""Run tiny-start replay analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from staq.analysis import plot_rollout_comparisons, sample_intuition_replays
from staq.config import Cifar10StaqConfig, default_paths
from staq.core import (
    build_concept_dictionary,
    concept_answers_batch,
    load_clip_model,
    load_concept_qa_checkpoint,
    load_concepts,
    load_run_bundle,
    make_sensitive_mask,
)
from staq.data import get_cifar10_datasets, get_raw_cifar10_dataset
from staq.sensitive_labels import build_cifar10_sensitive_match


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept-qa-checkpoint", type=str, default=None)
    parser.add_argument("--baseline-checkpoint", type=str, default=None)
    parser.add_argument("--staq-checkpoint", type=str, default=None)
    parser.add_argument("--min-history", type=int, default=1)
    parser.add_argument("--max-history", type=int, default=2)
    parser.add_argument("--num-cases", type=int, default=6)
    parser.add_argument("--pool-size", type=int, default=400)
    parser.add_argument("--num-trials", type=int, default=2)
    parser.add_argument("--output-name", type=str, default="cifar10_replay")
    parser.add_argument("--figure-ext", type=str, default=".svg")
    args = parser.parse_args()

    paths = default_paths(repo_root=REPO_ROOT)
    paths.ensure_artifact_dirs()
    config = Cifar10StaqConfig()
    device = config.device

    model_clip, preprocess = load_clip_model(config.clip_model_name, device=device)
    concepts = load_concepts(paths.concept_file)
    dictionary = build_concept_dictionary(model_clip=model_clip, concepts=concepts, device=device)
    sens_idx = build_cifar10_sensitive_match(concepts).indices
    sensitive_mask = make_sensitive_mask(config.max_queries, sens_idx, device)

    qa_checkpoint = args.concept_qa_checkpoint
    if qa_checkpoint is None and paths.bootstrap_concept_qa_checkpoint.exists():
        qa_checkpoint = str(paths.bootstrap_concept_qa_checkpoint)
    if qa_checkpoint is None:
        raise FileNotFoundError(
            "Provide --concept-qa-checkpoint or place a bootstrap Concept-QA checkpoint in artifacts/checkpoints/bootstrap."
        )
    answering_model = load_concept_qa_checkpoint(qa_checkpoint, device=device)

    baseline_checkpoint = args.baseline_checkpoint or str(paths.checkpoints_root / "baseline_best.pt")
    staq_checkpoint = args.staq_checkpoint or str(paths.checkpoints_root / "lam_0.40_best.pt")
    baseline_bundle = load_run_bundle(
        baseline_checkpoint,
        device=device,
        max_queries=config.max_queries,
        num_classes=config.num_classes,
    )
    staq_bundle = load_run_bundle(
        staq_checkpoint,
        device=device,
        max_queries=config.max_queries,
        num_classes=config.num_classes,
    )

    class_names = get_raw_cifar10_dataset(paths.data_root, train=False).classes
    raw_test_ds = get_raw_cifar10_dataset(paths.data_root, train=False)
    _, test_ds = get_cifar10_datasets(transform=preprocess, root=paths.data_root)

    def answer_builder(images):
        return concept_answers_batch(
            images=images,
            model_clip=model_clip,
            dictionary=dictionary,
            answering_model=answering_model,
            clip_device=device,
            train_device=device,
            threshold=config.threshold_for_binarization,
        )

    selected = sample_intuition_replays(
        dataset=test_ds,
        answer_builder=answer_builder,
        baseline_bundle=baseline_bundle,
        staq_bundle=staq_bundle,
        concepts=concepts,
        sensitive_mask=sensitive_mask,
        class_names=class_names,
        num_cases=args.num_cases,
        pool_size=args.pool_size,
        num_trials=args.num_trials,
        min_history=args.min_history,
        max_history=args.max_history,
    )
    result = {"selected": selected}
    fig_path = None
    if selected:
        figure_ext = args.figure_ext if args.figure_ext.startswith(".") else f".{args.figure_ext}"
        fig_path = plot_rollout_comparisons(
            records=selected,
            raw_dataset=raw_test_ds,
            output_path=paths.figures_root / f"{args.output_name}_intuition{figure_ext}",
            title_prefix="tiny-start replay",
        )

    result_path = paths.runs_root / f"{args.output_name}.json"
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(f"Saved analysis data: {result_path}")
    if fig_path is None:
        print("No replay figure was created because no replay records were selected.")
    else:
        print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
