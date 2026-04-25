"""Train STAQ from the clean repo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from staq.config import Cifar10StaqConfig, default_paths
from staq.core import (
    build_concept_dictionary,
    load_clip_model,
    load_concept_qa_checkpoint,
    load_concepts,
    save_bundle_checkpoint,
)
from staq.data import get_cifar10_loaders
from staq.sensitive_labels import build_cifar10_sensitive_match
from staq.training import HistorySamplingConfig, build_staq_models, fit_staq, seed_everything


def main():
    parser = argparse.ArgumentParser()
    default_config = Cifar10StaqConfig()
    parser.add_argument("--concept-qa-checkpoint", type=str, default=None)
    parser.add_argument("--actor-checkpoint", type=str, default=None)
    parser.add_argument("--classifier-checkpoint", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lambda-adv", type=float, default=0.6)
    parser.add_argument("--alpha-sens", type=float, default=0.0)
    parser.add_argument("--min-history", type=int, default=default_config.min_history)
    parser.add_argument("--max-history", type=int, default=default_config.max_history)
    parser.add_argument(
        "--allow-sensitive-history",
        action="store_true",
        help="If set, random initial history sampling may include sensitive concepts.",
    )
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="lam_0.60")
    args = parser.parse_args()

    paths = default_paths(repo_root=REPO_ROOT)
    paths.ensure_artifact_dirs()
    config = Cifar10StaqConfig(
        default_train_epochs=args.epochs,
        lambda_adv=args.lambda_adv,
        alpha_sens=args.alpha_sens,
        min_history=args.min_history,
        max_history=args.max_history,
        non_sensitive_history_only=not args.allow_sensitive_history,
    )
    seed_everything(config.random_seed)

    device = config.device
    model_clip, preprocess = load_clip_model(config.clip_model_name, device=device)
    concepts = load_concepts(paths.concept_file)
    dictionary = build_concept_dictionary(model_clip=model_clip, concepts=concepts, device=device)
    sens_idx = build_cifar10_sensitive_match(concepts).indices

    batch_size = args.batch_size or config.batch_size
    train_loader, test_loader = get_cifar10_loaders(
        transform=preprocess,
        root=paths.data_root,
        batch_size=batch_size,
        num_workers=config.num_workers,
    )

    qa_checkpoint = args.concept_qa_checkpoint
    if qa_checkpoint is None and paths.bootstrap_concept_qa_checkpoint.exists():
        qa_checkpoint = str(paths.bootstrap_concept_qa_checkpoint)
    if qa_checkpoint is None:
        raise FileNotFoundError(
            "Provide --concept-qa-checkpoint or place a bootstrap Concept-QA checkpoint in artifacts/models/bootstrap."
        )
    answering_model = load_concept_qa_checkpoint(qa_checkpoint, device=device)

    actor_checkpoint = args.actor_checkpoint
    classifier_checkpoint = args.classifier_checkpoint
    if actor_checkpoint is None and paths.bootstrap_actor_checkpoint.exists():
        actor_checkpoint = str(paths.bootstrap_actor_checkpoint)
    if classifier_checkpoint is None and paths.bootstrap_classifier_checkpoint.exists():
        classifier_checkpoint = str(paths.bootstrap_classifier_checkpoint)

    actor, classifier, s_head = build_staq_models(
        max_queries=config.max_queries,
        num_classes=config.num_classes,
        device=device,
        actor_eps=config.actor_eps,
        actor_checkpoint=actor_checkpoint,
        classifier_checkpoint=classifier_checkpoint,
    )
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(classifier.parameters()) + list(s_head.parameters()),
        lr=config.learning_rate,
    )
    history_config = HistorySamplingConfig(
        min_history=config.min_history,
        max_history=config.max_history,
        non_sensitive_only=config.non_sensitive_history_only,
    )

    history, best = fit_staq(
        actor=actor,
        classifier=classifier,
        s_head=s_head,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        model_clip=model_clip,
        dictionary=dictionary,
        answering_model=answering_model,
        sens_idx=sens_idx,
        history_config=history_config,
        clip_device=device,
        train_device=device,
        threshold_for_binarization=config.threshold_for_binarization,
        lambda_adv=args.lambda_adv,
        alpha_sens=args.alpha_sens,
        sensitive_tau=config.sensitive_tau,
        sensitive_topk=config.sensitive_topk,
        num_epochs=args.epochs,
        max_train_batches=args.max_train_batches,
        max_test_batches=args.max_test_batches,
    )

    checkpoint_path = paths.checkpoints_root / f"{args.run_name}_best.pt"
    save_bundle_checkpoint(
        checkpoint_path=checkpoint_path,
        metadata={
            "run_name": args.run_name,
            "lambda_adv": args.lambda_adv,
            "alpha_sens": args.alpha_sens,
            "best_test_acc": best["test_acc"],
            "best_epoch": best["epoch"],
            "history_config": {
                "min_history": history_config.min_history,
                "max_history": history_config.max_history,
                "non_sensitive_only": history_config.non_sensitive_only,
            },
            "actor_state_dict": best["actor_state_dict"],
            "classifier_state_dict": best["classifier_state_dict"],
            "s_head_state_dict": best["s_head_state_dict"],
        },
    )

    history_path = paths.runs_root / f"{args.run_name}_history.json"
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    print(f"Saved best checkpoint: {checkpoint_path}")
    print(f"Saved history: {history_path}")


if __name__ == "__main__":
    main()
