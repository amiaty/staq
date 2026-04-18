"""Concept-QA training helpers adapted from the original VIP paper training code."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from staq.core.clip_features import concept_qa_batch_inputs


def load_gpt_answers(path: str | Path) -> np.ndarray:
    return np.load(path)


def _unpack_concept_qa_batch(batch):
    if len(batch) == 2:
        images, labels = batch
        return images, labels, None
    if len(batch) == 3:
        images, labels, concept_targets = batch
        return images, labels, concept_targets
    raise ValueError("Concept-QA batches must contain (images, labels) or (images, labels, concept_targets)")


def prepare_concept_targets(labels: torch.Tensor, gpt_answers: np.ndarray, positive_depends: bool = True) -> torch.Tensor:
    query_answers = gpt_answers[labels.cpu().numpy()]
    query_answers = np.where(query_answers == -1, np.zeros(query_answers.shape), query_answers)
    depends_value = np.ones(query_answers.shape) if positive_depends else np.zeros(query_answers.shape)
    query_answers = np.where(query_answers == 2, depends_value, query_answers)
    return torch.tensor(query_answers)


def resolve_concept_targets(
    labels: torch.Tensor,
    gpt_answers: np.ndarray | None = None,
    batch_targets: torch.Tensor | None = None,
    positive_depends: bool = True,
) -> torch.Tensor:
    if batch_targets is not None:
        return batch_targets.float()
    if gpt_answers is None:
        raise ValueError("gpt_answers are required when concept targets are not provided by the batch")
    return prepare_concept_targets(labels=labels, gpt_answers=gpt_answers, positive_depends=positive_depends).float()


def concept_qa_loss(logits: torch.Tensor, query_answers: torch.Tensor, clip_scores: torch.Tensor) -> torch.Tensor:
    log_positive = torch.log(torch.sigmoid(logits))
    log_negative = torch.log(1 - torch.sigmoid(logits))
    loss = log_positive * (query_answers * clip_scores) + log_negative * (
        (1 - query_answers) + query_answers * (1 - clip_scores)
    )
    return -loss.sum() / torch.numel(loss)


def train_concept_qa_epoch(
    model,
    loader,
    optimizer,
    model_clip,
    dictionary: torch.Tensor,
    gpt_answers: np.ndarray | None,
    clip_device: torch.device,
    train_device: torch.device,
    positive_depends: bool = True,
    max_batches: int | None = None,
):
    model.train()
    sum_loss = 0.0
    n_batches = 0
    for batch_index, batch in enumerate(tqdm(loader, desc="Concept-QA train", leave=False)):
        if max_batches is not None and batch_index >= max_batches:
            break
        images, labels, batch_targets = _unpack_concept_qa_batch(batch)
        targets = resolve_concept_targets(
            labels=labels,
            gpt_answers=gpt_answers,
            batch_targets=batch_targets,
            positive_depends=positive_depends,
        ).to(train_device)
        inputs, clip_scores = concept_qa_batch_inputs(
            model_clip=model_clip,
            images=images,
            dictionary=dictionary,
            device=clip_device,
        )
        logits = model(inputs.to(train_device)).view_as(clip_scores).float()
        loss = concept_qa_loss(logits=logits, query_answers=targets.float(), clip_scores=clip_scores.to(train_device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += float(loss.item())
        n_batches += 1

    return {"loss": sum_loss / max(n_batches, 1)}


@torch.no_grad()
def evaluate_concept_qa(
    model,
    loader,
    model_clip,
    dictionary: torch.Tensor,
    gpt_answers: np.ndarray | None,
    clip_device: torch.device,
    train_device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    sum_accuracy = 0.0
    n_batches = 0
    for batch_index, batch in enumerate(tqdm(loader, desc="Concept-QA eval", leave=False)):
        if max_batches is not None and batch_index >= max_batches:
            break
        images, labels, batch_targets = _unpack_concept_qa_batch(batch)
        targets = resolve_concept_targets(
            labels=labels,
            gpt_answers=gpt_answers,
            batch_targets=batch_targets,
            positive_depends=False,
        ).to(train_device)
        inputs, _ = concept_qa_batch_inputs(
            model_clip=model_clip,
            images=images,
            dictionary=dictionary,
            device=clip_device,
        )
        logits = model(inputs.to(train_device)).view(targets.size(0), targets.size(1))
        preds = torch.where(logits > 0, torch.ones_like(logits), torch.zeros_like(logits))
        sum_accuracy += float((preds == targets).float().mean().item())
        n_batches += 1
    return {"accuracy": sum_accuracy / max(n_batches, 1)}


def fit_concept_qa(
    model,
    train_loader,
    eval_loader,
    optimizer,
    scheduler,
    num_epochs: int,
    model_clip,
    dictionary: torch.Tensor,
    clip_device: torch.device,
    train_device: torch.device,
    gpt_answers: np.ndarray | None = None,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
):
    history = []
    for epoch in tqdm(range(1, num_epochs + 1), desc="Concept-QA epochs"):
        train_metrics = train_concept_qa_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            model_clip=model_clip,
            dictionary=dictionary,
            gpt_answers=gpt_answers,
            clip_device=clip_device,
            train_device=train_device,
            max_batches=max_train_batches,
        )
        eval_metrics = evaluate_concept_qa(
            model=model,
            loader=eval_loader,
            model_clip=model_clip,
            dictionary=dictionary,
            gpt_answers=gpt_answers,
            clip_device=clip_device,
            train_device=train_device,
            max_batches=max_eval_batches,
        )
        scheduler.step()
        history.append({"epoch": epoch, **train_metrics, **eval_metrics})
    return history
