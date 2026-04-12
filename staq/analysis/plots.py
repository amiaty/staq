"""Plot helpers for rollout comparisons and fixed-history summaries."""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np

from staq.analysis.rollouts import format_confidence_path, format_stop_sequence


def plot_fixed_history_eval_summary(
    summary_rows: list[dict],
    output_path: str | Path,
    hparam_name: str = "lambda_adv",
    hparam_label: str | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not summary_rows:
        raise ValueError("summary_rows must not be empty")

    hparam_label = hparam_label or hparam_name.replace("_", " ")
    rows = sorted(
        summary_rows,
        key=lambda row: (float("-inf") if row[hparam_name] is None else row[hparam_name], row["run_name"]),
    )
    x = [row[hparam_name] for row in rows]
    mean_acc = [row["mean_acc"] for row in rows]
    std_acc = [row["std_acc"] for row in rows]
    mean_sens = [row["mean_sensitive_query_rate"] for row in rows]
    std_sens = [row["std_sensitive_query_rate"] for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    axes[0].errorbar(x, mean_acc, yerr=std_acc, marker="o", linewidth=2, capsize=4, color="#1f77b4")
    axes[0].set_title("Fixed-history accuracy")
    axes[0].set_xlabel(hparam_label)
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(alpha=0.25)

    axes[1].errorbar(x, mean_sens, yerr=std_sens, marker="o", linewidth=2, capsize=4, color="#d62728")
    axes[1].set_title("Fixed-history sensitive query rate")
    axes[1].set_xlabel(hparam_label)
    axes[1].set_ylabel("Sensitive query rate")
    axes[1].grid(alpha=0.25)

    axes[2].errorbar(
        mean_sens,
        mean_acc,
        xerr=std_sens,
        yerr=std_acc,
        fmt="o",
        linewidth=1.5,
        capsize=4,
        color="#2e8b57",
    )
    axes[2].set_title("Fixed-history trade-off")
    axes[2].set_xlabel("Sensitive query rate")
    axes[2].set_ylabel("Accuracy")
    axes[2].grid(alpha=0.25)
    for row in rows:
        axes[2].annotate(
            f"{row[hparam_name]:.2f}",
            (row["mean_sensitive_query_rate"], row["mean_acc"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _wrap_block(label: str, row: dict, key: str, wrap_width: int = 64, seq_items: int = 6, conf_items: int = 8) -> str:
    stop = row[key]
    first_sensitive = "none" if stop["first_sensitive_step"] is None else str(stop["first_sensitive_step"])
    lines = [
        label,
        (
            f"q={stop['queries_asked']} | sens={stop['sensitive_steps']} | "
            f"first sensitive={first_sensitive} | stop={stop['final_confidence']:.2f} | "
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
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path
