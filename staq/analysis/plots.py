"""Plot helpers for training curves and rollout comparisons."""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np

from staq.analysis.rollouts import format_confidence_path, format_stop_sequence


def plot_training_curves(history_by_run: dict[str, list[dict]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for run_name, rows in history_by_run.items():
        epochs = [row["epoch"] for row in rows]
        acc = [row["test_acc"] for row in rows]
        sens = [row["test_sens_q_rate"] for row in rows]
        axes[0].plot(epochs, acc, marker="o", linewidth=2, label=run_name)
        axes[1].plot(epochs, sens, marker="o", linewidth=2, label=run_name)

    axes[0].set_title("Test accuracy by epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(fontsize=9)

    axes[1].set_title("Sensitive query rate by epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Sensitive query rate")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _wrap_block(label: str, row: dict, key: str, wrap_width: int = 64, seq_items: int = 6, conf_items: int = 8) -> str:
    stop = row[key]
    lines = [
        label,
        (
            f"q={stop['queries_asked']} | sens={stop['sensitive_steps']} | "
            f"first S={stop['first_sensitive_step']} | stop={stop['final_confidence']:.2f} | "
            f"pred={stop['final_pred_name']}"
        ),
        textwrap.fill(
            f"conf path: {format_confidence_path(stop['states'], max_items=conf_items)}",
            width=wrap_width,
            subsequent_indent="    ",
        ),
        textwrap.fill(
            f"path: {format_stop_sequence(stop['sequence'], max_items=seq_items)}",
            width=wrap_width,
            subsequent_indent="    ",
        ),
    ]
    return "\n".join(lines)


def plot_rollout_comparisons(
    records: list[dict],
    raw_dataset,
    output_path: str | Path,
    title_prefix: str,
    bucket_name: str | None = None,
    warning: str | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError("records must not be empty")

    fig, axes = plt.subplots(
        len(records),
        3,
        figsize=(22, max(4.8, 4.5 * len(records))),
        gridspec_kw={"width_ratios": [1.0, 1.55, 1.55]},
    )
    if len(records) == 1:
        axes = np.array([axes])

    for (ax_img, ax_base, ax_staq), row in zip(axes, records):
        image, _ = raw_dataset[row["sample_idx"]]
        ax_img.imshow(image)
        ax_img.axis("off")
        ax_img.set_title(
            f"{title_prefix} | idx={row['sample_idx']} | true={row['label_name']} | "
            f"gap={row['sensitive_gap']} | div={row['first_divergence_step']}",
            fontsize=10,
        )

        for ax in (ax_base, ax_staq):
            ax.axis("off")

        start_text = ", ".join(row["initial_history"]) if row["initial_history"] else "(empty)"
        meta_parts = [
            f"start ({row['initial_history_size']}): {start_text}",
            f"both correct: {row['both_correct']}",
            f"divergence step: {row['first_divergence_step']}",
        ]
        if bucket_name is not None:
            meta_parts.insert(0, f"bucket: {bucket_name}")
        if warning is not None:
            meta_parts.append(warning)
        meta_text = textwrap.fill(" | ".join(meta_parts), width=56)

        ax_base.text(
            0.0,
            1.0,
            meta_text + "\n\n" + _wrap_block("baseline", row, "baseline"),
            fontsize=9.5,
            va="top",
            ha="left",
            linespacing=1.35,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#fff5f5", edgecolor="crimson", alpha=0.95),
        )
        ax_staq.text(
            0.0,
            1.0,
            _wrap_block("STAQ", row, "staq"),
            fontsize=9.5,
            va="top",
            ha="left",
            linespacing=1.35,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#f5fff5", edgecolor="darkgreen", alpha=0.95),
        )

    plt.subplots_adjust(wspace=0.10, hspace=0.42)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path
