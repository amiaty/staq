"""Concept-QA training helpers adapted from the original VIP paper training code."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from staq.core.clip_features import concept_qa_batch_inputs


def load_gpt_answers(path: str | Path) -> np.ndarray:
    return np.load(path)


def prepare_concept_targets(labels: torch.Tensor, gpt_answers: np.ndarray, positive_depends: bool = True) -> torch.Tensor:
    query_answers = gpt_answers[labels.cpu().numpy()]
    query_answers = np.where(query_answers == -1, np.zeros(query_answers.shape), query_answers)
    depends_value = np.ones(query_answers.shape) if positive_depends else np.zeros(query_answers.shape)
    query_answers = np.where(query_answers == 2, depends_value, query_answers)
    return torch.tensor(query_answers)


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
    gpt_answers: np.ndarray,
    clip_device: torch.device,
    train_device: torch.device,
    positive_depends: bool = True,
):
    model.train()
    sum_loss = 0.0
    n_batches = 0
    for images, labels in tqdm(loader, desc="Concept-QA train", leave=False):
        targets = prepare_concept_targets(labels=labels, gpt_answers=gpt_answers, positive_depends=positive_depends).to(train_device)
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
    gpt_answers: np.ndarray,
    clip_device: torch.device,
    train_device: torch.device,
) -> dict[str, float]:
    model.eval()
    sum_accuracy = 0.0
    n_batches = 0
    for images, labels in tqdm(loader, desc="Concept-QA eval", leave=False):
        targets = prepare_concept_targets(labels=labels, gpt_answers=gpt_answers, positive_depends=False).to(train_device)
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
    gpt_answers: np.ndarray,
    clip_device: torch.device,
    train_device: torch.device,
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
        )
        eval_metrics = evaluate_concept_qa(
            model=model,
            loader=eval_loader,
            model_clip=model_clip,
            dictionary=dictionary,
            gpt_answers=gpt_answers,
            clip_device=clip_device,
            train_device=train_device,
        )
        scheduler.step()
        history.append({"epoch": epoch, **train_metrics, **eval_metrics})
    return history
