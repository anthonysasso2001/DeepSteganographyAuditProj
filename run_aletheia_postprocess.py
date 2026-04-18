#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _safe_stem(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(name).stem)


def _paired_stats(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {
            "n_matched": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "sem": np.nan,
            "ci95": np.nan,
            "n_positive": 0,
            "n_negative": 0,
            "n_zero": 0,
            "sign_consistency": np.nan,
        }

    mean = float(np.mean(arr))
    median = float(np.median(arr))
    if n > 1:
        std = float(np.std(arr, ddof=1))
        sem = float(std / np.sqrt(n))
        ci95 = float(1.96 * sem)
    else:
        std = 0.0
        sem = np.nan
        ci95 = np.nan

    n_positive = int(np.sum(arr > 0))
    n_negative = int(np.sum(arr < 0))
    n_zero = int(np.sum(arr == 0))
    non_zero = n_positive + n_negative
    sign_consistency = float(
        max(n_positive, n_negative) / non_zero) if non_zero else np.nan

    return {
        "n_matched": n,
        "mean": mean,
        "median": median,
        "std": std,
        "sem": sem,
        "ci95": ci95,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_zero": n_zero,
        "sign_consistency": sign_consistency,
    }


def _run_cmd(cmd, cwd, timeout_sec):
    start = time.time()
    completed = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    elapsed = time.time() - start
    return completed, elapsed


def _role_prefixed_df(parsed_df, role):
    if parsed_df is None or parsed_df.empty:
        return pd.DataFrame(columns=["image_key", f"{role}_image_name"])

    out = parsed_df.copy()
    out["image_name"] = out["image_name"].astype(str)
    out["image_key"] = out["image_name"].str.lower()

    rename_map = {c: f"{role}_{c}" for c in out.columns if c != "image_key"}
    return out.rename(columns=rename_map)


def _run_pair_tool_audits(py_exec, repo_root, timeout_sec, logs_dir, method, pair_name, role_a, role_b, role_paths, image_names, include_dct):
    rows = []
    for image_name in image_names:
        img_a = os.path.join(role_paths[role_a], image_name)
        img_b = os.path.join(role_paths[role_b], image_name)
        if not (os.path.isfile(img_a) and os.path.isfile(img_b)):
            continue

        tools_to_run = ["print-diffs"]
        if include_dct and Path(img_a).suffix.lower() in {".jpg", ".jpeg"} and Path(img_b).suffix.lower() in {".jpg", ".jpeg"}:
            tools_to_run.append("print-dct-diffs")

        safe_img = _safe_stem(image_name)
        for tool_name in tools_to_run:
            cmd = [py_exec, "aletheia/aletheia.py", tool_name, img_a, img_b]
            completed, elapsed = _run_cmd(
                cmd, cwd=repo_root, timeout_sec=timeout_sec)

            stdout_log = os.path.join(
                logs_dir,
                f"{method}_{pair_name}_{safe_img}_{tool_name}_stdout.txt",
            )
            stderr_log = os.path.join(
                logs_dir,
                f"{method}_{pair_name}_{safe_img}_{tool_name}_stderr.txt",
            )
            with open(stdout_log, "w", encoding="utf-8") as f:
                f.write(completed.stdout or "")
            with open(stderr_log, "w", encoding="utf-8") as f:
                f.write(completed.stderr or "")

            rows.append(
                {
                    "method": method,
                    "pair_name": pair_name,
                    "image_name": image_name,
                    "tool": tool_name,
                    "role_a": role_a,
                    "role_b": role_b,
                    "role_a_path": img_a,
                    "role_b_path": img_b,
                    "return_code": int(completed.returncode),
                    "elapsed_sec": float(elapsed),
                    "stdout_log": stdout_log,
                    "stderr_log": stderr_log,
                }
            )

    return rows


def parse_aletheia_auto_stdout(stdout_text):
    """Parse `aletheia.py auto` output into per-image score rows."""
    rows = []
    image_ext_re = re.compile(
        r"\.(png|jpg|jpeg|bmp|tif|tiff)$", flags=re.IGNORECASE)
    pair_re = re.compile(
        r"(\[)?([0-9]*\.?[0-9]+)(\])?\s*\(([0-9]*\.?[0-9]+)\)")
    bracket_single_re = re.compile(r"\[([0-9]*\.?[0-9]+)\]")
    plain_single_re = re.compile(
        r"(?<!\[)(?<![A-Za-z])([0-9]*\.?[0-9]+)(?!\s*\()")

    for raw_line in stdout_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("-"):
            continue

        parts = line.split()
        if not parts:
            continue

        image_name = parts[0]
        if not image_ext_re.search(image_name):
            continue

        payload = line[len(image_name):].strip()
        scores = []
        confs = []
        flagged = 0

        for match in pair_re.finditer(payload):
            score = float(match.group(2))
            conf = float(match.group(4))
            scores.append(score)
            confs.append(conf)
            if match.group(1) == "[" and match.group(3) == "]":
                flagged += 1

        cleaned_payload = pair_re.sub(" ", payload)
        for match in bracket_single_re.finditer(cleaned_payload):
            scores.append(float(match.group(1)))
            flagged += 1

        cleaned_payload = bracket_single_re.sub(" ", cleaned_payload)
        for match in plain_single_re.finditer(cleaned_payload):
            scores.append(float(match.group(1)))

        if not scores:
            continue

        rows.append(
            {
                "image_name": image_name,
                "n_scores": len(scores),
                "mean_score": float(np.mean(scores)),
                "max_score": float(np.max(scores)),
                "mean_conf": float(np.mean(confs)) if confs else np.nan,
                "n_flagged": int(flagged),
                "flagged_ratio": float(flagged / len(scores)),
            }
        )

    return pd.DataFrame(rows)


def run_aletheia_on_results_raw(
    run_data_dir,
    repo_root=".",
    python_executable=None,
    timeout_sec=None,
    aletheia_dev=None,
    require_gpu=False,
    run_tool_audits=False,
    tool_audit_limit=25,
    include_dct_tool=False,
):
    """Run Aletheia on role folders in results_raw and write CSV outputs.

    Comparison is performed image-by-image by matching image_name across pair roles.
    Optional tool audits run Aletheia tool commands per matched image pair.
    """
    results_raw_dir = os.path.join(run_data_dir, "results_raw")
    if not os.path.isdir(results_raw_dir):
        raise FileNotFoundError(
            f"results_raw not found: {results_raw_dir}. Run notebook evaluation first."
        )

    py_exec = python_executable or sys.executable
    pair_specs = [
        ("cover_vs_stego", "cover", "stego"),
        ("pre_secret_vs_secret", "pre_secret", "secret"),
    ]

    out_dir = os.path.join(run_data_dir, "aletheia_outputs")
    logs_dir = os.path.join(out_dir, "logs")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    run_rows = []
    image_level_rows = []
    compare_rows = []
    tool_rows = []

    method_dirs = [
        d
        for d in sorted(os.listdir(results_raw_dir))
        if os.path.isdir(os.path.join(results_raw_dir, d))
    ]

    for method in method_dirs:
        method_dir = os.path.join(results_raw_dir, method)

        for pair_name, role_a, role_b in pair_specs:
            role_paths = {
                role_a: os.path.join(method_dir, role_a),
                role_b: os.path.join(method_dir, role_b),
            }

            if not all(os.path.isdir(path) for path in role_paths.values()):
                continue

            pair_summary = {"method": method, "pair_name": pair_name}
            all_ok = True
            role_parsed = {}

            for role, image_dir in role_paths.items():
                cmd = [py_exec, "aletheia/aletheia.py", "auto", image_dir]
                if aletheia_dev is not None:
                    cmd.append(str(aletheia_dev))
                completed, elapsed = _run_cmd(
                    cmd, cwd=repo_root, timeout_sec=timeout_sec)

                stdout_log = os.path.join(
                    logs_dir, f"{method}_{pair_name}_{role}_stdout.txt")
                stderr_log = os.path.join(
                    logs_dir, f"{method}_{pair_name}_{role}_stderr.txt")
                with open(stdout_log, "w", encoding="utf-8") as f:
                    f.write(completed.stdout or "")
                with open(stderr_log, "w", encoding="utf-8") as f:
                    f.write(completed.stderr or "")

                parsed = parse_aletheia_auto_stdout(completed.stdout or "")
                role_parsed[role] = parsed

                stdout_text = completed.stdout or ""
                used_cpu = (
                    "Running with CPU" in stdout_text
                    or "'dev' not provided, using: CPU" in stdout_text
                )

                if completed.returncode != 0:
                    all_ok = False

                if require_gpu and used_cpu:
                    all_ok = False

                error_head = ""
                if completed.returncode != 0 and completed.stderr:
                    error_head = completed.stderr.strip().splitlines()[-1]
                elif require_gpu and used_cpu:
                    error_head = "GPU required but Aletheia used CPU"

                run_row = {
                    "method": method,
                    "pair_name": pair_name,
                    "role": role,
                    "image_dir": image_dir,
                    "requested_dev": str(aletheia_dev) if aletheia_dev is not None else "CPU(default)",
                    "used_cpu": bool(used_cpu),
                    "return_code": int(completed.returncode),
                    "elapsed_sec": float(elapsed),
                    "n_images_parsed": int(len(parsed)),
                    "mean_image_score": float(parsed["mean_score"].mean()) if not parsed.empty else np.nan,
                    "max_image_score": float(parsed["max_score"].max()) if not parsed.empty else np.nan,
                    "mean_flagged_ratio": float(parsed["flagged_ratio"].mean()) if not parsed.empty else np.nan,
                    "mean_confidence": float(parsed["mean_conf"].mean()) if not parsed.empty else np.nan,
                    "error_summary": error_head,
                    "stdout_log": stdout_log,
                    "stderr_log": stderr_log,
                }
                run_rows.append(run_row)

                pair_summary[f"{role}_score"] = run_row["mean_image_score"]
                pair_summary[f"{role}_flagged_ratio"] = run_row["mean_flagged_ratio"]

            # Match rows by image name (correlated image-by-image comparison).
            a_df = _role_prefixed_df(role_parsed.get(role_a), role_a)
            b_df = _role_prefixed_df(role_parsed.get(role_b), role_b)
            matched = a_df.merge(b_df, on="image_key", how="inner")

            if not matched.empty:
                matched["image_name"] = matched[f"{role_b}_image_name"].fillna(
                    matched[f"{role_a}_image_name"]
                )
                matched["method"] = method
                matched["pair_name"] = pair_name
                matched["role_a"] = role_a
                matched["role_b"] = role_b

                matched["score_delta"] = matched[f"{role_b}_mean_score"] - \
                    matched[f"{role_a}_mean_score"]
                matched["max_score_delta"] = matched[f"{role_b}_max_score"] - \
                    matched[f"{role_a}_max_score"]
                matched["flagged_ratio_delta"] = matched[f"{role_b}_flagged_ratio"] - \
                    matched[f"{role_a}_flagged_ratio"]
                matched["confidence_delta"] = matched[f"{role_b}_mean_conf"] - \
                    matched[f"{role_a}_mean_conf"]

                pair_conf = np.nanmean(
                    np.vstack(
                        [
                            matched[f"{role_a}_mean_conf"].to_numpy(
                                dtype=float),
                            matched[f"{role_b}_mean_conf"].to_numpy(
                                dtype=float),
                        ]
                    ),
                    axis=0,
                )
                matched["pair_mean_conf"] = pair_conf
                matched["score_delta_conf_weighted"] = matched["score_delta"] * \
                    matched["pair_mean_conf"]

                image_level_rows.append(
                    matched[
                        [
                            "method",
                            "pair_name",
                            "role_a",
                            "role_b",
                            "image_name",
                            f"{role_a}_mean_score",
                            f"{role_b}_mean_score",
                            f"{role_a}_max_score",
                            f"{role_b}_max_score",
                            f"{role_a}_flagged_ratio",
                            f"{role_b}_flagged_ratio",
                            f"{role_a}_mean_conf",
                            f"{role_b}_mean_conf",
                            "score_delta",
                            "max_score_delta",
                            "flagged_ratio_delta",
                            "confidence_delta",
                            "pair_mean_conf",
                            "score_delta_conf_weighted",
                        ]
                    ]
                )

                score_stats = _paired_stats(matched["score_delta"].to_numpy())
                flag_stats = _paired_stats(
                    matched["flagged_ratio_delta"].to_numpy())
                confw_stats = _paired_stats(
                    matched["score_delta_conf_weighted"].to_numpy())

                pair_summary.update(
                    {
                        "n_images_role_a": int(len(a_df)),
                        "n_images_role_b": int(len(b_df)),
                        "n_matched_images": int(score_stats["n_matched"]),
                        "score_gap": score_stats["mean"],
                        "score_gap_median": score_stats["median"],
                        "score_gap_std": score_stats["std"],
                        "score_gap_sem": score_stats["sem"],
                        "score_gap_ci95": score_stats["ci95"],
                        "score_gap_sign_consistency": score_stats["sign_consistency"],
                        "flagged_gap": flag_stats["mean"],
                        "flagged_gap_median": flag_stats["median"],
                        "flagged_gap_std": flag_stats["std"],
                        "flagged_gap_sem": flag_stats["sem"],
                        "flagged_gap_ci95": flag_stats["ci95"],
                        "score_gap_conf_weighted": confw_stats["mean"],
                        "score_gap_conf_weighted_ci95": confw_stats["ci95"],
                    }
                )

                if run_tool_audits:
                    audit_names = list(matched["image_name"].astype(
                        str).head(max(1, int(tool_audit_limit))))
                    tool_rows.extend(
                        _run_pair_tool_audits(
                            py_exec=py_exec,
                            repo_root=repo_root,
                            timeout_sec=timeout_sec,
                            logs_dir=logs_dir,
                            method=method,
                            pair_name=pair_name,
                            role_a=role_a,
                            role_b=role_b,
                            role_paths=role_paths,
                            image_names=audit_names,
                            include_dct=include_dct_tool,
                        )
                    )
            else:
                pair_summary.update(
                    {
                        "n_images_role_a": int(len(a_df)),
                        "n_images_role_b": int(len(b_df)),
                        "n_matched_images": 0,
                        "score_gap": np.nan,
                        "flagged_gap": np.nan,
                    }
                )

            if all_ok:
                compare_rows.append(pair_summary)

    df_runs = pd.DataFrame(run_rows)
    df_image_level = pd.concat(
        image_level_rows, ignore_index=True) if image_level_rows else pd.DataFrame()
    df_compare = pd.DataFrame(compare_rows)
    df_tool_audits = pd.DataFrame(tool_rows)

    runs_csv = os.path.join(out_dir, "aletheia_raw_runs.csv")
    image_level_csv = os.path.join(out_dir, "aletheia_pair_image_level.csv")
    compare_csv = os.path.join(out_dir, "aletheia_pair_comparison.csv")
    tool_audit_csv = os.path.join(out_dir, "aletheia_tool_audits.csv")
    df_runs.to_csv(runs_csv, index=False)
    df_image_level.to_csv(image_level_csv, index=False)
    df_compare.to_csv(compare_csv, index=False)
    if not df_tool_audits.empty:
        df_tool_audits.to_csv(tool_audit_csv, index=False)

    return df_runs, df_image_level, df_compare, df_tool_audits, out_dir


def plot_aletheia_pair_comparison(df_compare, save_path=None):
    if df_compare is None or df_compare.empty:
        print("No successful Aletheia pair comparisons to plot.")
        return

    pair_names = sorted(df_compare["pair_name"].dropna().unique().tolist())
    pair_a = pair_names[0] if pair_names else None
    pair_b = pair_names[1] if len(pair_names) > 1 else None

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            f"{pair_a} | Score Gap" if pair_a else "Pair: N/A | Score Gap",
            f"{pair_a} | Flagged Gap" if pair_a else "Pair: N/A | Flagged Gap",
            f"{pair_b} | Score Gap" if pair_b else "Pair: N/A | Score Gap",
            f"{pair_b} | Flagged Gap" if pair_b else "Pair: N/A | Flagged Gap",
        ),
        horizontal_spacing=0.12,
        vertical_spacing=0.22,
    )

    if pair_a:
        sub_a = df_compare[df_compare["pair_name"] == pair_a]
        labels_a = sub_a["method"].astype(str).tolist()
        fig.add_trace(
            go.Bar(x=labels_a, y=sub_a["score_gap"],
                   name=pair_a, marker_color="#1f77b4", showlegend=False),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(x=labels_a, y=sub_a["flagged_gap"],
                   name=pair_a, marker_color="#ff7f0e", showlegend=False),
            row=1,
            col=2,
        )
        fig.update_yaxes(title_text="Score Gap", row=1, col=1)
        fig.update_yaxes(title_text="Flagged Gap", row=1, col=2)
    else:
        fig.add_trace(
            go.Bar(x=["N/A"], y=[0.0],
                   marker_color="#cccccc", showlegend=False),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(x=["N/A"], y=[0.0],
                   marker_color="#cccccc", showlegend=False),
            row=1,
            col=2,
        )
        fig.update_yaxes(title_text="Score Gap", row=1, col=1)
        fig.update_yaxes(title_text="Flagged Gap", row=1, col=2)

    if pair_b:
        sub_b = df_compare[df_compare["pair_name"] == pair_b]
        labels_b = sub_b["method"].astype(str).tolist()
        fig.add_trace(
            go.Bar(x=labels_b, y=sub_b["score_gap"],
                   name=pair_b, marker_color="#2ca02c", showlegend=False),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Bar(x=labels_b, y=sub_b["flagged_gap"],
                   name=pair_b, marker_color="#d62728", showlegend=False),
            row=2,
            col=2,
        )
        fig.update_yaxes(title_text="Score Gap", row=2, col=1)
        fig.update_yaxes(title_text="Flagged Gap", row=2, col=2)
    else:
        fig.add_trace(
            go.Bar(x=["N/A"], y=[0.0],
                   marker_color="#cccccc", showlegend=False),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Bar(x=["N/A"], y=[0.0],
                   marker_color="#cccccc", showlegend=False),
            row=2,
            col=2,
        )
        fig.update_yaxes(title_text="Score Gap", row=2, col=1)
        fig.update_yaxes(title_text="Flagged Gap", row=2, col=2)

    fig.update_xaxes(tickangle=-45, automargin=True)
    fig.update_layout(
        width=max(1400, 500 + 150 *
                  max(1, int(df_compare["method"].nunique()))),
        height=1000,
        template="plotly_white",
        title="Aletheia Pairwise Comparison by Method",
        showlegend=False,
        margin={"l": 100, "r": 70, "t": 100, "b": 200},
    )

    if save_path:
        fig.write_image(save_path, format="pdf")


def _parse_index_range(index_range):
    if index_range is None:
        return None

    m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", index_range)
    if not m:
        raise ValueError(
            f"Invalid --run-index-range '{index_range}'. Use format START-END, e.g. 0-6."
        )

    start = int(m.group(1))
    end = int(m.group(2))
    if end < start:
        raise ValueError(
            f"Invalid --run-index-range '{index_range}': END must be >= START."
        )

    return {str(i) for i in range(start, end + 1)}


def _discover_run_dirs(run_data_dir=None, run_data_root=None, run_index_range=None, scan_all_numbered=False):
    if run_data_dir:
        if not os.path.isdir(run_data_dir):
            raise ValueError(
                f"--run-data-dir does not exist or is not a directory: {run_data_dir}")
        return [os.path.normpath(run_data_dir)]

    if not run_data_root:
        raise ValueError(
            "Provide either --run-data-dir for one run, or --run-data-root for batch mode.")

    if not os.path.isdir(run_data_root):
        raise ValueError(
            f"--run-data-root does not exist or is not a directory: {run_data_root}")

    numbered = [
        d for d in sorted(os.listdir(run_data_root))
        if d.isdigit() and os.path.isdir(os.path.join(run_data_root, d))
    ]
    if not numbered:
        raise ValueError(
            f"No numbered run folders found under: {run_data_root}")

    allowed = _parse_index_range(run_index_range)
    if allowed is not None:
        selected = [d for d in numbered if d in allowed]
    else:
        # Default batch behavior uses all numbered folders; explicit flag is kept for readability.
        selected = numbered if scan_all_numbered or True else []

    if not selected:
        msg = f"No runs selected under {run_data_root}."
        if run_index_range is not None:
            msg += f" Requested range: {run_index_range}."
        raise ValueError(msg)

    return [os.path.join(run_data_root, d) for d in selected]


def _apply_research_full_defaults(args):
    if not args.research_full:
        return args

    # Keep explicit user-provided run target if present; otherwise default to baseline 0-6.
    if not args.run_data_dir and not args.run_data_root:
        args.run_data_root = os.path.join("data", "baseline")
        if args.run_index_range is None:
            args.run_index_range = "0-6"

    if args.aletheia_dev is None:
        args.aletheia_dev = "0"

    args.require_gpu = True
    args.run_tool_audits = True
    args.tool_audit_include_dct = True
    args.no_plot = False

    # If still default, promote to effectively all matched pairs.
    if args.tool_audit_limit == 25:
        args.tool_audit_limit = 1000000

    return args


def _load_aletheia_csvs(run_data_dir):
    """Load existing Aletheia CSV outputs from a run directory."""
    out_dir = os.path.join(run_data_dir, "aletheia_outputs")

    df_runs = pd.DataFrame()
    df_image_level = pd.DataFrame()
    df_compare = pd.DataFrame()
    df_tool_audits = pd.DataFrame()

    runs_csv = os.path.join(out_dir, "aletheia_raw_runs.csv")
    if os.path.exists(runs_csv):
        df_runs = pd.read_csv(runs_csv)

    image_csv = os.path.join(out_dir, "aletheia_pair_image_level.csv")
    if os.path.exists(image_csv):
        df_image_level = pd.read_csv(image_csv)

    compare_csv = os.path.join(out_dir, "aletheia_pair_comparison.csv")
    if os.path.exists(compare_csv):
        df_compare = pd.read_csv(compare_csv)

    tool_csv = os.path.join(out_dir, "aletheia_tool_audits.csv")
    if os.path.exists(tool_csv):
        df_tool_audits = pd.read_csv(tool_csv)

    return df_runs, df_image_level, df_compare, df_tool_audits, out_dir


def _process_run(run_dir, args, plot_only=False):
    """Process a single run: either run Aletheia or load existing CSVs."""
    print(f"\n=== Processing run: {run_dir} ===")

    if plot_only:
        print("(plot-only mode: loading existing Aletheia outputs)")
        df_runs, df_image_level, df_compare, df_tool_audits, out_dir = _load_aletheia_csvs(
            run_dir)
        if df_compare.empty:
            print(
                f"Warning: No aletheia_pair_comparison.csv found in {out_dir}")
            return None
    else:
        df_runs, df_image_level, df_compare, df_tool_audits, out_dir = run_aletheia_on_results_raw(
            run_data_dir=run_dir,
            repo_root=args.repo_root,
            python_executable=args.python_executable,
            timeout_sec=args.timeout_sec,
            aletheia_dev=args.aletheia_dev,
            require_gpu=args.require_gpu,
            run_tool_audits=args.run_tool_audits,
            tool_audit_limit=args.tool_audit_limit,
            include_dct_tool=args.tool_audit_include_dct,
        )

        print(
            f"Saved raw run CSV: {os.path.join(out_dir, 'aletheia_raw_runs.csv')}")
        print(
            f"Saved image-level matched CSV: {os.path.join(out_dir, 'aletheia_pair_image_level.csv')}")
        print(
            f"Saved pair comparison CSV: {os.path.join(out_dir, 'aletheia_pair_comparison.csv')}")
        if not df_tool_audits.empty:
            print(
                f"Saved tool audit CSV: {os.path.join(out_dir, 'aletheia_tool_audits.csv')}")

        if not df_image_level.empty:
            print(
                "Matched image-level comparison complete "
                f"({len(df_image_level)} correlated rows)."
            )

    if not plot_only:
        failures = df_runs[df_runs["return_code"] != 0]
        if args.require_gpu:
            gpu_failures = df_runs[df_runs["used_cpu"] == True]
            if not gpu_failures.empty:
                print("\nGPU enforcement warnings (showing first 10 CPU fallbacks):")
                print(gpu_failures[["method", "pair_name", "role", "requested_dev", "used_cpu", "error_summary"]]
                      .head(10)
                      .to_string(index=False))

        if not failures.empty:
            print("\nAletheia failures detected (showing first 10):")
            print(failures[["method", "pair_name", "role", "return_code",
                  "error_summary"]].head(10).to_string(index=False))

    if not args.no_plot:
        plot_path = os.path.join(out_dir, "aletheia_pair_comparison.pdf")
        plot_aletheia_pair_comparison(df_compare, save_path=plot_path)
        if not df_compare.empty:
            print(f"Saved comparison plot: {plot_path}")

    run_label = os.path.normpath(run_dir)
    summary_row = {
        "run_data_dir": run_label,
        "n_role_runs": int(len(df_runs)),
        "n_pair_rows": int(len(df_compare)),
        "n_image_rows": int(len(df_image_level)),
        "n_tool_rows": int(len(df_tool_audits)),
    }

    if not plot_only:
        failures = df_runs[df_runs["return_code"] != 0]
        summary_row["n_failures"] = int(len(failures))
        summary_row["n_cpu_fallbacks"] = int(
            len(df_runs[df_runs["used_cpu"] == True])) if "used_cpu" in df_runs else 0

    return {
        "summary": summary_row,
        "df_runs": df_runs,
        "df_image_level": df_image_level,
        "df_compare": df_compare,
        "df_tool_audits": df_tool_audits,
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run Aletheia postprocess on notebook results_raw outputs.")
    parser.add_argument(
        "--run-data-dir",
        default=None,
        help="Run folder containing results_raw, e.g. data/baseline/0",
    )
    parser.add_argument(
        "--run-data-root",
        default=None,
        help="Batch root containing numbered run folders, e.g. data/baseline",
    )
    parser.add_argument(
        "--run-index-range",
        default=None,
        help="Optional batch filter for numbered run folders, format START-END, e.g. 0-6",
    )
    parser.add_argument(
        "--scan-all-numbered",
        action="store_true",
        help="In batch mode, include all numbered run folders under --run-data-root.",
    )
    parser.add_argument(
        "--research-full",
        action="store_true",
        help=(
            "Preset for full research run: defaults to data/baseline 0-6 (if no run path is set), "
            "forces GPU dev=0 with --require-gpu, enables all tool audits with DCT, "
            "raises tool-audit-limit, and saves a per-ablation comparison plot."
        ),
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repo root used as cwd when invoking aletheia/aletheia.py",
    )
    parser.add_argument(
        "--python-executable",
        default=None,
        help="Python executable for launching Aletheia (default: current interpreter)",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=None,
        help="Optional subprocess timeout in seconds",
    )
    parser.add_argument(
        "--aletheia-dev",
        default=None,
        help="Device passed to `aletheia.py auto` as [dev], e.g. `0` for first GPU or `CPU`.",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail pair summaries when Aletheia reports CPU usage.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip PDF/interactive plot generation",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Load existing Aletheia CSVs and regenerate plots (skip Aletheia execution)",
    )
    parser.add_argument(
        "--run-tool-audits",
        action="store_true",
        help="Run per-image Aletheia tool audits (print-diffs; optional print-dct-diffs) on matched pairs.",
    )
    parser.add_argument(
        "--tool-audit-limit",
        type=int,
        default=25,
        help="Max matched image pairs to audit per (method, pair).",
    )
    parser.add_argument(
        "--tool-audit-include-dct",
        action="store_true",
        help="Also run print-dct-diffs for matched JPEG pairs during tool audits.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    args = _apply_research_full_defaults(args)
    run_dirs = _discover_run_dirs(
        run_data_dir=args.run_data_dir,
        run_data_root=args.run_data_root,
        run_index_range=args.run_index_range,
        scan_all_numbered=args.scan_all_numbered,
    )

    if args.research_full:
        print(
            "Research preset active: "
            f"dev={args.aletheia_dev}, require_gpu={args.require_gpu}, "
            f"tool_audits={args.run_tool_audits}, dct={args.tool_audit_include_dct}, "
            f"tool_audit_limit={args.tool_audit_limit}, no_plot={args.no_plot}."
        )

    all_runs = []
    all_image = []
    all_compare = []
    all_tools = []
    summary_rows = []

    for run_dir in run_dirs:
        result = _process_run(run_dir, args, plot_only=args.plot_only)
        if result is None:
            continue

        summary_rows.append(result["summary"])

        if not result["df_runs"].empty:
            tmp = result["df_runs"].copy()
            tmp.insert(0, "run_data_dir", os.path.normpath(run_dir))
            all_runs.append(tmp)
        if not result["df_image_level"].empty:
            tmp = result["df_image_level"].copy()
            tmp.insert(0, "run_data_dir", os.path.normpath(run_dir))
            all_image.append(tmp)
        if not result["df_compare"].empty:
            tmp = result["df_compare"].copy()
            tmp.insert(0, "run_data_dir", os.path.normpath(run_dir))
            all_compare.append(tmp)
        if not result["df_tool_audits"].empty:
            tmp = result["df_tool_audits"].copy()
            tmp.insert(0, "run_data_dir", os.path.normpath(run_dir))
            all_tools.append(tmp)

    if len(run_dirs) > 1:
        batch_out_dir = args.run_data_root or os.path.dirname(run_dirs[0])
        os.makedirs(batch_out_dir, exist_ok=True)

        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(os.path.join(
            batch_out_dir, "aletheia_batch_summary.csv"), index=False)

        if all_runs:
            pd.concat(all_runs, ignore_index=True).to_csv(
                os.path.join(batch_out_dir,
                             "aletheia_batch_raw_runs.csv"),
                index=False,
            )
        if all_image:
            pd.concat(all_image, ignore_index=True).to_csv(
                os.path.join(batch_out_dir,
                             "aletheia_batch_image_level.csv"),
                index=False,
            )
        if all_compare:
            pd.concat(all_compare, ignore_index=True).to_csv(
                os.path.join(batch_out_dir,
                             "aletheia_batch_pair_comparison.csv"),
                index=False,
            )
        if all_tools:
            pd.concat(all_tools, ignore_index=True).to_csv(
                os.path.join(batch_out_dir,
                             "aletheia_batch_tool_audits.csv"),
                index=False,
            )

        print("\n=== Batch outputs ===")
        print(os.path.join(batch_out_dir, "aletheia_batch_summary.csv"))
        if all_runs:
            print(os.path.join(batch_out_dir, "aletheia_batch_raw_runs.csv"))
        if all_image:
            print(os.path.join(batch_out_dir, "aletheia_batch_image_level.csv"))
        if all_compare:
            print(os.path.join(batch_out_dir,
                  "aletheia_batch_pair_comparison.csv"))
        if all_tools:
            print(os.path.join(batch_out_dir, "aletheia_batch_tool_audits.csv"))


if __name__ == "__main__":
    main()
