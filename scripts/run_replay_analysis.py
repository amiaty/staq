"""Run tiny-start replay or confidence-stop contrast mining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from staq.analysis import mine_confidence_stop_contrasts, plot_rollout_comparisons, sample_intuition_replays
from staq.config import Cifar10StaqConfig, default_paths
from staq.core import (
    build_concept_dictionary,
    concept_answers_batch,
    load_clip_model,
    load_concept_qa_checkpoint,
    load_concepts,
    load_vip_bundle,
    make_sensitive_mask,
)
from staq.data import get_cifar10_datasets, get_raw_cifar10_dataset
from staq.sensitive_labels import build_sensitive_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept-qa-checkpoint", type=str, default=None)
    parser.add_argument("--baseline-checkpoint", type=str, default=None)
    parser.add_argument("--fair-checkpoint", type=str, default=None)
    parser.add_argument("--mode", choices=["intuition", "contrast"], default="intuition")
    parser.add_argument("--min-history", type=int, default=1)
    parser.add_argument("--max-history", type=int, default=2)
    parser.add_argument("--num-cases", type=int, default=6)
    parser.add_argument("--pool-size", type=int, default=400)
    parser.add_argument("--num-trials", type=int, default=2)
    parser.add_argument("--max-search-samples", type=int, default=None)
    parser.add_argument("--output-name", type=str, default="cifar10_analysis")
    args = parser.parse_args()

    paths = default_paths(repo_root=REPO_ROOT)
    paths.ensure_artifact_dirs()
    config = Cifar10StaqConfig()
    device = config.device

    model_clip, preprocess = load_clip_model(config.clip_model_name, device=device)
    concepts = load_concepts(paths.concept_file)
    dictionary = build_concept_dictionary(model_clip=model_clip, concepts=concepts, device=device)
    sens_idx = build_sensitive_index(concepts)
    sensitive_mask = make_sensitive_mask(config.max_queries, sens_idx, device)

    qa_checkpoint = args.concept_qa_checkpoint
    if qa_checkpoint is None and paths.bootstrap_concept_qa_checkpoint.exists():
        qa_checkpoint = str(paths.bootstrap_concept_qa_checkpoint)
    if qa_checkpoint is None:
        raise FileNotFoundError(
            "Provide --concept-qa-checkpoint or place a bootstrap Concept-QA checkpoint in artifacts/checkpoints/bootstrap."
        )
    answering_model = load_concept_qa_checkpoint(qa_checkpoint, device=device)

    baseline_checkpoint = args.baseline_checkpoint or str(paths.checkpoints_root / "baseline_vip_best.pt")
    fair_checkpoint = args.fair_checkpoint or str(paths.checkpoints_root / "lam_0.60_best.pt")
    baseline_bundle = load_vip_bundle(baseline_checkpoint, device=device, max_queries=config.max_queries, num_classes=config.num_classes)
    fair_bundle = load_vip_bundle(fair_checkpoint, device=device, max_queries=config.max_queries, num_classes=config.num_classes)

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

    if args.mode == "intuition":
        selected = sample_intuition_replays(
            dataset=test_ds,
            answer_builder=answer_builder,
            baseline_bundle=baseline_bundle,
            fair_bundle=fair_bundle,
            concepts=concepts,
            sensitive_mask=sensitive_mask,
            class_names=class_names,
            num_cases=args.num_cases,
            pool_size=args.pool_size,
            num_trials=args.num_trials,
            min_history=args.min_history,
            max_history=args.max_history,
        )
        result = {"mode": "intuition", "selected": selected}
        fig_path = plot_rollout_comparisons(
            records=selected,
            raw_dataset=raw_test_ds,
            output_path=paths.figures_root / f"{args.output_name}_intuition.png",
            title_prefix="tiny-start replay",
        )
    else:
        test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
        result = mine_confidence_stop_contrasts(
            loader=test_loader,
            answer_builder=answer_builder,
            baseline_bundle=baseline_bundle,
            fair_bundle=fair_bundle,
            concepts=concepts,
            sensitive_mask=sensitive_mask,
            class_names=class_names,
            threshold=config.confidence_threshold,
            max_steps=config.confidence_max_steps,
            min_history=args.min_history,
            max_history=args.max_history,
            max_search_samples=args.max_search_samples,
            num_trials=args.num_trials,
        )
        plot_records = result["plot_candidates"] if result["plot_candidates"] else result["selected_candidates"]
        plot_warning = None
        fig_path = None
        if not result["plot_candidates"] and result["selected_candidates"]:
            plot_warning = "Showing weak fallback examples only."
        if plot_records:
            fig_path = plot_rollout_comparisons(
                records=plot_records,
                raw_dataset=raw_test_ds,
                output_path=paths.figures_root / f"{args.output_name}_contrast.png",
                title_prefix="confidence-stop contrast",
                bucket_name=result["plot_bucket"] if result["plot_candidates"] else result["selected_bucket"],
                warning=plot_warning,
            )
        result = {
            "mode": "contrast",
            "stats": result["stats"],
            "bucket_counts": result["bucket_counts"],
            "selected_bucket": result["selected_bucket"],
            "plot_bucket": result["plot_bucket"],
            "selected_candidates": result["selected_candidates"],
        }

    result_path = paths.runs_root / f"{args.output_name}_{args.mode}.json"
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(f"Saved analysis data: {result_path}")
    if fig_path is None:
        print("No contrast figure was created because the miner found no plotable candidates.")
    else:
        print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
