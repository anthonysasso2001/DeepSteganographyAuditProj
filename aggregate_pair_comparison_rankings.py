#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REQUIRED_COLUMNS = {
    "run_data_dir",
    "method",
    "pair_name",
    "score_gap",
    "flagged_gap",
}
METRICS = ("score_gap", "flagged_gap")
RANKING_OBJECTIVE = "stealth_minimize_gap"


def discover_batch_pair_csvs(data_root, glob_pattern, explicit_csvs):
    if explicit_csvs:
        resolved = [Path(p).resolve() for p in explicit_csvs]
        missing = [str(p) for p in resolved if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing CSV files provided by --batch-pair-csv:\n" +
                "\n".join(missing)
            )
        return resolved

    root = Path(data_root).resolve()
    csvs = sorted(root.glob(glob_pattern))
    if not csvs:
        raise FileNotFoundError(
            f"No files matched pattern '{glob_pattern}' under {root}."
        )
    return csvs


def parse_run_identity(run_data_dir):
    parts = Path(str(run_data_dir).replace("\\", "/")).parts
    ablation_id = "unknown"
    experiment = "unknown"

    for part in reversed(parts):
        if part.isdigit():
            ablation_id = part
            break

    if ablation_id != "unknown":
        idx = list(parts).index(ablation_id)
        if idx > 0:
            experiment = parts[idx - 1]
    elif parts:
        experiment = parts[-1]

    ablation_key = f"{experiment}/{ablation_id}"
    return experiment, ablation_id, ablation_key


def weighted_mean(values, weights):
    vals = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)

    mask = np.isfinite(vals) & np.isfinite(w) & (w >= 0)
    if not np.any(mask):
        return np.nan

    vals = vals[mask]
    w = w[mask]
    total_w = float(np.sum(w))
    if total_w <= 0:
        return np.nan

    return float(np.sum(vals * w) / total_w)


def zscore_or_zero(series):
    s = pd.to_numeric(series, errors="coerce")
    std = float(s.std(ddof=0)) if s.notna().any() else np.nan
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.zeros(len(series), dtype=float), index=series.index)
    mean = float(s.mean())
    return (s - mean) / std


def load_and_prepare(csv_paths):
    frames = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        missing = sorted(REQUIRED_COLUMNS - set(df.columns))
        if missing:
            raise ValueError(
                f"CSV missing required columns ({', '.join(missing)}): {csv_path}"
            )

        out = df.copy()
        out["source_csv"] = str(csv_path)

        if "n_matched_images" not in out.columns:
            out["n_matched_images"] = 1

        identities = out["run_data_dir"].apply(parse_run_identity)
        out[["experiment", "ablation_id", "ablation_key"]] = pd.DataFrame(
            identities.tolist(), index=out.index
        )

        out["n_matched_images"] = pd.to_numeric(
            out["n_matched_images"], errors="coerce"
        ).fillna(1.0)

        for metric in METRICS:
            out[metric] = pd.to_numeric(out[metric], errors="coerce")

        frames.append(out)

    if not frames:
        raise ValueError("No data loaded from CSV files.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(
        subset=["method", "pair_name", "ablation_key"], how="any")
    return combined


def aggregate_cells(df):
    rows = []
    group_cols = ["pair_name", "ablation_key", "method"]

    for key, group in df.groupby(group_cols, dropna=False):
        pair_name, ablation_key, method = key
        row = {
            "pair_name": pair_name,
            "ablation_key": ablation_key,
            "method": method,
            "n_rows": int(len(group)),
            "n_total_matched_images": float(group["n_matched_images"].sum()),
        }
        for metric in METRICS:
            row[metric] = weighted_mean(
                group[metric], group["n_matched_images"])
        rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values(["pair_name", "ablation_key", "method"]).reset_index(drop=True)


def summarize_rank_entities(df, entity_col):
    rows = []
    for pair_name, group in df.groupby("pair_name", dropna=False):
        for entity_value, sub in group.groupby(entity_col, dropna=False):
            row = {
                "pair_name": pair_name,
                entity_col: entity_value,
                "n_cells": int(len(sub)),
                "n_total_matched_images": float(sub["n_total_matched_images"].sum()),
                "mean_score_gap": weighted_mean(sub["score_gap"], sub["n_total_matched_images"]),
                "mean_flagged_gap": weighted_mean(sub["flagged_gap"], sub["n_total_matched_images"]),
            }
            rows.append(row)

    return pd.DataFrame(rows)


def apply_stealth_rankings(summary_df):
    if summary_df.empty:
        return summary_df

    ranked_parts = []
    for pair_name, group in summary_df.groupby("pair_name", dropna=False):
        g = group.copy()
        g["score_cost_z"] = zscore_or_zero(g["mean_score_gap"])
        g["flagged_cost_z"] = zscore_or_zero(g["mean_flagged_gap"])
        g["composite_cost_z"] = g["score_cost_z"] + g["flagged_cost_z"]

        # Stealth-first objective: lower role_b-role_a gaps are better.
        g["rank_score_gap"] = g["mean_score_gap"].rank(
            ascending=True, method="dense")
        g["rank_flagged_gap"] = g["mean_flagged_gap"].rank(
            ascending=True, method="dense")
        g["rank_composite"] = g["composite_cost_z"].rank(
            ascending=True, method="dense")
        g["ranking_objective"] = RANKING_OBJECTIVE
        ranked_parts.append(g)

    return pd.concat(ranked_parts, ignore_index=True)


def build_method_rankings(df):
    out = summarize_rank_entities(df, entity_col="method")
    out = apply_stealth_rankings(out)
    return out.sort_values(["pair_name", "rank_composite", "method"]).reset_index(drop=True)


def build_ablation_rankings(df):
    out = summarize_rank_entities(df, entity_col="ablation_key")
    out = apply_stealth_rankings(out)
    return out.sort_values(["pair_name", "rank_composite", "ablation_key"]).reset_index(drop=True)


def build_top_k_summary(rank_df, entity_col, entity_type, top_k=3):
    if rank_df.empty:
        return pd.DataFrame(
            columns=[
                "pair_name",
                "entity_type",
                entity_col,
                "rank_composite",
                "mean_score_gap",
                "mean_flagged_gap",
                "n_total_matched_images",
                "ranking_objective",
            ]
        )

    rows = []
    for pair_name, group in rank_df.groupby("pair_name", dropna=False):
        top = group.sort_values(
            ["rank_composite", entity_col]).head(top_k).copy()
        top.insert(1, "entity_type", entity_type)
        rows.append(top)

    out = pd.concat(rows, ignore_index=True)
    keep_cols = [
        "pair_name",
        "entity_type",
        entity_col,
        "rank_composite",
        "mean_score_gap",
        "mean_flagged_gap",
        "n_total_matched_images",
        "ranking_objective",
    ]
    return out[keep_cols]


def add_heatmap_trace(fig, cell_df, pair_name, metric, row, col, x_field, y_field, show_scale):
    subset = cell_df[cell_df["pair_name"] == pair_name]
    pivot = subset.pivot(index=y_field, columns=x_field, values=metric)

    if pivot.empty:
        fig.add_trace(
            go.Heatmap(z=[[np.nan]], x=["N/A"], y=["N/A"],
                       colorscale="RdBu", zmid=0),
            row=row,
            col=col,
        )
        return

    fig.add_trace(
        go.Heatmap(
            z=pivot.to_numpy(dtype=float),
            x=[str(c) for c in pivot.columns],
            y=[str(i) for i in pivot.index],
            colorscale="RdBu",
            zmid=0,
            colorbar_title=f"{metric}",
            showscale=show_scale,
        ),
        row=row,
        col=col,
    )


def save_figure(fig, out_pdf_path):
    try:
        fig.write_image(str(out_pdf_path), format="pdf")
        print(f"Saved plot: {out_pdf_path}")
    except Exception as exc:
        fallback = out_pdf_path.with_suffix(".html")
        fig.write_html(str(fallback), include_plotlyjs="cdn")
        print(
            f"Warning: could not save PDF ({exc}). Saved HTML instead: {fallback}")


def plot_cross_comparison(cell_df, output_dir):
    pairs = sorted(cell_df["pair_name"].dropna().unique().tolist())
    if not pairs:
        print("No pair_name values available for plotting.")
        return

    n_pairs = len(pairs)
    n_methods = int(cell_df["method"].nunique()) if "method" in cell_df else 0
    n_ablations = int(cell_df["ablation_key"].nunique()
                      ) if "ablation_key" in cell_df else 0

    # Size plots dynamically from dataset cardinality to reduce axis/label overlap.
    width_methods_view = max(1800, 320 + (n_methods * 180) + (2 * 460))
    width_ablations_view = max(1800, 320 + (n_ablations * 180) + (2 * 460))
    per_pair_height = max(520, 220 + max(n_methods, n_ablations) * 26)
    full_height = max(700, per_pair_height * n_pairs)

    subplot_titles_1 = []
    for pair in pairs:
        subplot_titles_1.append(f"{pair} | score_gap")
        subplot_titles_1.append(f"{pair} | flagged_gap")

    fig_1 = make_subplots(
        rows=len(pairs),
        cols=2,
        subplot_titles=subplot_titles_1,
        horizontal_spacing=0.14,
        vertical_spacing=0.14,
    )

    for r, pair_name in enumerate(pairs, start=1):
        add_heatmap_trace(
            fig_1,
            cell_df,
            pair_name,
            "score_gap",
            row=r,
            col=1,
            x_field="method",
            y_field="ablation_key",
            show_scale=(r == 1),
        )
        add_heatmap_trace(
            fig_1,
            cell_df,
            pair_name,
            "flagged_gap",
            row=r,
            col=2,
            x_field="method",
            y_field="ablation_key",
            show_scale=False,
        )

    fig_1.update_layout(
        title="Cross-Comparison: Ablations by Embed Method (lower gap is better)",
        width=width_methods_view,
        height=full_height,
        margin={"l": 170, "r": 90, "t": 100, "b": 190},
        template="plotly_white",
    )
    fig_1.update_xaxes(title_text="Embed Method",
                       tickangle=-45, automargin=True)
    fig_1.update_yaxes(title_text="Ablation", automargin=True)
    save_figure(fig_1, output_dir / "cross_compare_ablations_by_method.pdf")

    subplot_titles_2 = []
    for pair in pairs:
        subplot_titles_2.append(f"{pair} | score_gap")
        subplot_titles_2.append(f"{pair} | flagged_gap")

    fig_2 = make_subplots(
        rows=len(pairs),
        cols=2,
        subplot_titles=subplot_titles_2,
        horizontal_spacing=0.14,
        vertical_spacing=0.14,
    )

    for r, pair_name in enumerate(pairs, start=1):
        add_heatmap_trace(
            fig_2,
            cell_df,
            pair_name,
            "score_gap",
            row=r,
            col=1,
            x_field="ablation_key",
            y_field="method",
            show_scale=(r == 1),
        )
        add_heatmap_trace(
            fig_2,
            cell_df,
            pair_name,
            "flagged_gap",
            row=r,
            col=2,
            x_field="ablation_key",
            y_field="method",
            show_scale=False,
        )

    fig_2.update_layout(
        title="Cross-Comparison: Embed Methods by Ablation (lower gap is better)",
        width=width_ablations_view,
        height=full_height,
        margin={"l": 170, "r": 90, "t": 100, "b": 190},
        template="plotly_white",
    )
    fig_2.update_xaxes(title_text="Ablation", tickangle=-45, automargin=True)
    fig_2.update_yaxes(title_text="Embed Method", automargin=True)
    save_figure(fig_2, output_dir / "cross_compare_methods_by_ablation.pdf")


def _add_bar_or_placeholder(fig, row, col, x_vals, y_vals, name, color, title):
    if len(x_vals) == 0:
        fig.add_trace(
            go.Bar(x=["N/A"], y=[0.0],
                   marker_color="#cccccc", showlegend=False),
            row=row,
            col=col,
        )
        fig.update_yaxes(title_text=title, row=row, col=col)
        return

    fig.add_trace(
        go.Bar(x=x_vals, y=y_vals, name=name,
               marker_color=color, showlegend=False),
        row=row,
        col=col,
    )
    fig.update_yaxes(title_text=title, row=row, col=col)


def plot_pairwise_metric_bars_by_method(cell_df, output_dir):
    if cell_df is None or cell_df.empty:
        print("No cell rows available for method metric bar plots.")
        return

    pair_names = sorted(cell_df["pair_name"].dropna().unique().tolist())
    pair_a = pair_names[0] if pair_names else None
    pair_b = pair_names[1] if len(pair_names) > 1 else None

    method_metric = (
        cell_df.groupby(["pair_name", "method"], dropna=False)[
            ["score_gap", "flagged_gap"]]
        .mean()
        .reset_index()
    )

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            f"{pair_a} | score_gap" if pair_a else "Pair: N/A | score_gap",
            f"{pair_a} | flagged_gap" if pair_a else "Pair: N/A | flagged_gap",
            f"{pair_b} | score_gap" if pair_b else "Pair: N/A | score_gap",
            f"{pair_b} | flagged_gap" if pair_b else "Pair: N/A | flagged_gap",
        ),
        horizontal_spacing=0.12,
        vertical_spacing=0.22,
    )

    if pair_a:
        sub_a = method_metric[method_metric["pair_name"] == pair_a].sort_values(
            ["score_gap", "method"]
        )
        _add_bar_or_placeholder(
            fig,
            row=1,
            col=1,
            x_vals=sub_a["method"].astype(str).tolist(),
            y_vals=sub_a["score_gap"].astype(float).tolist(),
            name=pair_a,
            color="#1f77b4",
            title="score_gap",
        )
        sub_a_flag = method_metric[method_metric["pair_name"] == pair_a].sort_values(
            ["flagged_gap", "method"]
        )
        _add_bar_or_placeholder(
            fig,
            row=1,
            col=2,
            x_vals=sub_a_flag["method"].astype(str).tolist(),
            y_vals=sub_a_flag["flagged_gap"].astype(float).tolist(),
            name=pair_a,
            color="#ff7f0e",
            title="flagged_gap",
        )
    else:
        _add_bar_or_placeholder(
            fig,
            row=1,
            col=1,
            x_vals=[],
            y_vals=[],
            name="N/A",
            color="#1f77b4",
            title="score_gap",
        )
        _add_bar_or_placeholder(
            fig,
            row=1,
            col=2,
            x_vals=[],
            y_vals=[],
            name="N/A",
            color="#ff7f0e",
            title="flagged_gap",
        )

    if pair_b:
        sub_b = method_metric[method_metric["pair_name"] == pair_b].sort_values(
            ["score_gap", "method"]
        )
        _add_bar_or_placeholder(
            fig,
            row=2,
            col=1,
            x_vals=sub_b["method"].astype(str).tolist(),
            y_vals=sub_b["score_gap"].astype(float).tolist(),
            name=pair_b,
            color="#2ca02c",
            title="score_gap",
        )
        sub_b_flag = method_metric[method_metric["pair_name"] == pair_b].sort_values(
            ["flagged_gap", "method"]
        )
        _add_bar_or_placeholder(
            fig,
            row=2,
            col=2,
            x_vals=sub_b_flag["method"].astype(str).tolist(),
            y_vals=sub_b_flag["flagged_gap"].astype(float).tolist(),
            name=pair_b,
            color="#d62728",
            title="flagged_gap",
        )
    else:
        _add_bar_or_placeholder(
            fig,
            row=2,
            col=1,
            x_vals=[],
            y_vals=[],
            name="N/A",
            color="#2ca02c",
            title="score_gap",
        )
        _add_bar_or_placeholder(
            fig,
            row=2,
            col=2,
            x_vals=[],
            y_vals=[],
            name="N/A",
            color="#d62728",
            title="flagged_gap",
        )

    fig.update_layout(
        title="Pairwise Metric Bars by Embed Method",
        width=max(1900, 520 + 200 * max(1, int(cell_df["method"].nunique()))),
        height=1200,
        margin={"l": 120, "r": 70, "t": 110, "b": 220},
        template="plotly_white",
        showlegend=False,
    )
    fig.update_xaxes(tickangle=-45, automargin=True)
    fig.update_yaxes(automargin=True)

    save_figure(fig, output_dir / "pairwise_metrics_by_method.pdf")


def plot_pairwise_metric_bars_by_ablation(cell_df, output_dir):
    if cell_df is None or cell_df.empty:
        print("No cell rows available for ablation metric bar plots.")
        return

    pair_names = sorted(cell_df["pair_name"].dropna().unique().tolist())
    pair_a = pair_names[0] if pair_names else None
    pair_b = pair_names[1] if len(pair_names) > 1 else None

    ablation_metric = (
        cell_df.groupby(["pair_name", "ablation_key"], dropna=False)[
            ["score_gap", "flagged_gap"]]
        .mean()
        .reset_index()
    )

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            f"{pair_a} | score_gap" if pair_a else "Pair: N/A | score_gap",
            f"{pair_a} | flagged_gap" if pair_a else "Pair: N/A | flagged_gap",
            f"{pair_b} | score_gap" if pair_b else "Pair: N/A | score_gap",
            f"{pair_b} | flagged_gap" if pair_b else "Pair: N/A | flagged_gap",
        ),
        horizontal_spacing=0.12,
        vertical_spacing=0.22,
    )

    if pair_a:
        sub_a = ablation_metric[ablation_metric["pair_name"] == pair_a].sort_values(
            ["score_gap", "ablation_key"]
        )
        _add_bar_or_placeholder(
            fig,
            row=1,
            col=1,
            x_vals=sub_a["ablation_key"].astype(str).tolist(),
            y_vals=sub_a["score_gap"].astype(float).tolist(),
            name=pair_a,
            color="#1f77b4",
            title="score_gap",
        )
        sub_a_flag = ablation_metric[ablation_metric["pair_name"] == pair_a].sort_values(
            ["flagged_gap", "ablation_key"]
        )
        _add_bar_or_placeholder(
            fig,
            row=1,
            col=2,
            x_vals=sub_a_flag["ablation_key"].astype(str).tolist(),
            y_vals=sub_a_flag["flagged_gap"].astype(float).tolist(),
            name=pair_a,
            color="#ff7f0e",
            title="flagged_gap",
        )
    else:
        _add_bar_or_placeholder(
            fig,
            row=1,
            col=1,
            x_vals=[],
            y_vals=[],
            name="N/A",
            color="#1f77b4",
            title="score_gap",
        )
        _add_bar_or_placeholder(
            fig,
            row=1,
            col=2,
            x_vals=[],
            y_vals=[],
            name="N/A",
            color="#ff7f0e",
            title="flagged_gap",
        )

    if pair_b:
        sub_b = ablation_metric[ablation_metric["pair_name"] == pair_b].sort_values(
            ["score_gap", "ablation_key"]
        )
        _add_bar_or_placeholder(
            fig,
            row=2,
            col=1,
            x_vals=sub_b["ablation_key"].astype(str).tolist(),
            y_vals=sub_b["score_gap"].astype(float).tolist(),
            name=pair_b,
            color="#2ca02c",
            title="score_gap",
        )
        sub_b_flag = ablation_metric[ablation_metric["pair_name"] == pair_b].sort_values(
            ["flagged_gap", "ablation_key"]
        )
        _add_bar_or_placeholder(
            fig,
            row=2,
            col=2,
            x_vals=sub_b_flag["ablation_key"].astype(str).tolist(),
            y_vals=sub_b_flag["flagged_gap"].astype(float).tolist(),
            name=pair_b,
            color="#d62728",
            title="flagged_gap",
        )
    else:
        _add_bar_or_placeholder(
            fig,
            row=2,
            col=1,
            x_vals=[],
            y_vals=[],
            name="N/A",
            color="#2ca02c",
            title="score_gap",
        )
        _add_bar_or_placeholder(
            fig,
            row=2,
            col=2,
            x_vals=[],
            y_vals=[],
            name="N/A",
            color="#d62728",
            title="flagged_gap",
        )

    fig.update_layout(
        title="Pairwise Metric Bars by Ablation",
        width=max(1900, 520 + 180 *
                  max(1, int(cell_df["ablation_key"].nunique()))),
        height=1200,
        margin={"l": 120, "r": 70, "t": 110, "b": 220},
        template="plotly_white",
        showlegend=False,
    )
    fig.update_xaxes(tickangle=-45, automargin=True)
    fig.update_yaxes(automargin=True)

    save_figure(fig, output_dir / "pairwise_metrics_by_ablation.pdf")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate Aletheia batch pair comparison CSVs and generate rankings + cross-comparison plots."
        )
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Root path to scan for batch pair comparison CSVs (default: data).",
    )
    parser.add_argument(
        "--glob-pattern",
        default="**/aletheia_batch_pair_comparison.csv",
        help="Glob pattern under --data-root for CSV discovery.",
    )
    parser.add_argument(
        "--batch-pair-csv",
        action="append",
        default=None,
        help="Explicit CSV path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for aggregated CSVs/plots.",
    )
    parser.add_argument(
        "--pair-name",
        action="append",
        default=None,
        help="Optional pair_name filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plot generation.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top K rows per pair_name to export in compact summary CSVs (default: 3).",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1.")

    csv_paths = discover_batch_pair_csvs(
        data_root=args.data_root,
        glob_pattern=args.glob_pattern,
        explicit_csvs=args.batch_pair_csv,
    )
    print(f"Discovered {len(csv_paths)} batch pair comparison CSV file(s).")

    combined = load_and_prepare(csv_paths)

    if args.pair_name:
        keep = set(args.pair_name)
        combined = combined[combined["pair_name"].isin(keep)].copy()
        if combined.empty:
            raise ValueError(
                "After applying --pair-name filter, there are no rows left to process."
            )

    output_dir = Path(args.output_dir) if args.output_dir else Path(
        args.data_root) / "aggregated_pair_rankings"
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_csv = output_dir / "combined_batch_pair_comparison.csv"
    combined.to_csv(combined_csv, index=False)

    cell_df = aggregate_cells(combined)
    cell_csv = output_dir / "cross_ablation_method_cells.csv"
    cell_df.to_csv(cell_csv, index=False)

    method_rank_df = build_method_rankings(cell_df)
    method_rank_csv = output_dir / "method_ranking_across_ablations.csv"
    method_rank_df.to_csv(method_rank_csv, index=False)

    ablation_rank_df = build_ablation_rankings(cell_df)
    ablation_rank_csv = output_dir / "ablation_ranking_across_methods.csv"
    ablation_rank_df.to_csv(ablation_rank_csv, index=False)

    top_methods_df = build_top_k_summary(
        method_rank_df,
        entity_col="method",
        entity_type="method",
        top_k=args.top_k,
    )
    top_methods_csv = output_dir / "top_methods_per_pair.csv"
    top_methods_df.to_csv(top_methods_csv, index=False)

    top_ablations_df = build_top_k_summary(
        ablation_rank_df,
        entity_col="ablation_key",
        entity_type="ablation",
        top_k=args.top_k,
    )
    top_ablations_csv = output_dir / "top_ablations_per_pair.csv"
    top_ablations_df.to_csv(top_ablations_csv, index=False)

    compact_summary_df = pd.concat(
        [top_methods_df, top_ablations_df], ignore_index=True)
    compact_summary_csv = output_dir / "top_rankings_compact_summary.csv"
    compact_summary_df.to_csv(compact_summary_csv, index=False)

    print(f"Saved: {combined_csv}")
    print(f"Saved: {cell_csv}")
    print(f"Saved: {method_rank_csv}")
    print(f"Saved: {ablation_rank_csv}")
    print(f"Saved: {top_methods_csv}")
    print(f"Saved: {top_ablations_csv}")
    print(f"Saved: {compact_summary_csv}")

    if not args.no_plot:
        plot_cross_comparison(cell_df, output_dir)
        plot_pairwise_metric_bars_by_method(cell_df, output_dir)
        plot_pairwise_metric_bars_by_ablation(cell_df, output_dir)


if __name__ == "__main__":
    main()
