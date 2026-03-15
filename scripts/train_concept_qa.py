"""Train the CIFAR-10 Concept-QA model in the clean repo."""

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
from staq.core import build_concept_dictionary, load_clip_model, load_concepts
from staq.data import get_cifar10_loaders
from staq.models import ConceptNet2
from staq.training import fit_concept_qa
from staq.training.concept_qa import load_gpt_answers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-name", type=str, default="concept_qa_cifar10")
    args = parser.parse_args()

    paths = default_paths(repo_root=REPO_ROOT)
    paths.ensure_artifact_dirs()
    config = Cifar10StaqConfig(default_train_epochs=args.epochs)
    device = config.device

    model_clip, preprocess = load_clip_model(config.clip_model_name, device=device)
    concepts = load_concepts(paths.concept_file)
    dictionary = build_concept_dictionary(model_clip=model_clip, concepts=concepts, device=device)

    batch_size = args.batch_size or config.batch_size
    train_loader, test_loader = get_cifar10_loaders(
        transform=preprocess,
        root=paths.data_root,
        batch_size=batch_size,
        num_workers=config.num_workers,
    )
    gpt_answers = load_gpt_answers(paths.gpt_answers_file)

    model = ConceptNet2().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    history = fit_concept_qa(
        model=model,
        train_loader=train_loader,
        eval_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        model_clip=model_clip,
        dictionary=dictionary,
        gpt_answers=gpt_answers,
        clip_device=device,
        train_device=device,
    )

    checkpoint_path = paths.checkpoints_root / f"{args.output_name}.pt"
    torch.save(model.state_dict(), checkpoint_path)

    history_path = paths.runs_root / f"{args.output_name}_history.json"
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved history: {history_path}")


if __name__ == "__main__":
    main()
