#!/usr/bin/env python3
import argparse
import json
import os

DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _parse_csv(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_device(device):
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _resolve_dtype(dtype_name, device):
    import torch

    if device == "cpu":
        return torch.float32

    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp32":
        return torch.float32
    raise ValueError("Unsupported dtype: {}".format(dtype_name))


def _configure_tokenizer(tokenizer):
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
    return tokenizer


def _load_model_and_tokenizer(model_name, device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
        )
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )
    model.to(device)
    model.eval()
    tokenizer = _configure_tokenizer(AutoTokenizer.from_pretrained(model_name))
    return model, tokenizer


def _run_eval_tasks(model, tokenizer, tasks, limit, max_length, stride, normalize_choice_scores):
    from src.eval import evaluate_multiple_choice, evaluate_perplexity

    results = []
    for task in tasks:
        if task in ("wikitext103", "c4_en"):
            results.append(
                evaluate_perplexity(
                    model,
                    tokenizer,
                    task_name=task,
                    limit=limit,
                    max_length=max_length,
                    stride=stride,
                )
            )
        else:
            results.append(
                evaluate_multiple_choice(
                    model,
                    tokenizer,
                    task_name=task,
                    limit=limit,
                    max_length=max_length,
                    normalize_choice_scores=normalize_choice_scores,
                )
            )
    return results


def _run_latency_grid(model, batch_sizes, prefill_lengths, decode_steps, warmup_steps, benchmark_steps):
    from src.eval import benchmark_model_latency

    results = []
    for batch_size in batch_sizes:
        for prefill_length in prefill_lengths:
            results.append(
                benchmark_model_latency(
                    model,
                    batch_size=batch_size,
                    prefill_length=prefill_length,
                    decode_steps=decode_steps,
                    warmup_steps=warmup_steps,
                    benchmark_steps=benchmark_steps,
                )
            )
    return results


def _list_to_table(wandb, rows):
    if not rows:
        return None

    columns = sorted({key for row in rows for key in row.keys()})
    data = [[row.get(column) for column in columns] for row in rows]
    return wandb.Table(columns=columns, data=data)


def _sanitize_key(value):
    return "".join(
        character if character.isalnum() else "_"
        for character in str(value)
    ).strip("_")


def _row_has_metric(rows, metric_name):
    return any(row.get(metric_name) is not None for row in rows)


def _comparison_key(metric_name, row):
    return "{}_bs{}_pl{}".format(
        metric_name,
        row.get("batch_size"),
        row.get("prefill_length"),
    )


def _build_summary_metrics(report):
    summary = {}

    dense_layer_rows = report.get("dense", {}).get("layer_benchmark", [])
    dense_latency = report.get("dense", {}).get("latency", [])
    kronecker_latency = report.get("kronecker", {}).get("latency", [])
    kronecker_stats = report.get("kronecker", {}).get("stats", {})

    summary.update(kronecker_stats)
    summary["dense_layer_benchmark_count"] = len(dense_layer_rows)
    summary["dense_latency_grid_points"] = len(dense_latency)
    summary["kronecker_latency_grid_points"] = len(kronecker_latency)

    if dense_layer_rows:
        layer_speedups = [
            row["speedup"]
            for row in dense_layer_rows
            if row.get("speedup") is not None
        ]
        compression_ratios = [
            row["compression_ratio"]
            for row in dense_layer_rows
            if row.get("compression_ratio") is not None
        ]
        output_errors = [
            row["relative_output_error"]
            for row in dense_layer_rows
            if row.get("relative_output_error") is not None
        ]

        if layer_speedups:
            summary["layer_speedup_mean"] = sum(layer_speedups) / len(layer_speedups)
            summary["layer_speedup_best"] = max(layer_speedups)
        if compression_ratios:
            summary["layer_compression_ratio_mean"] = (
                sum(compression_ratios) / len(compression_ratios)
            )
        if output_errors:
            summary["layer_relative_output_error_max"] = max(output_errors)

    dense_latency_map = {
        _comparison_key("latency", row): row
        for row in dense_latency
    }
    kronecker_latency_map = {
        _comparison_key("latency", row): row
        for row in kronecker_latency
    }

    for key, dense_row in dense_latency_map.items():
        if key not in kronecker_latency_map:
            continue

        kronecker_row = kronecker_latency_map[key]
        for metric_name in ("prefill_mean_ms", "decode_mean_ms"):
            dense_metric = dense_row.get(metric_name)
            kronecker_metric = kronecker_row.get(metric_name)
            if dense_metric and kronecker_metric:
                summary["{}_speedup".format(key.replace("latency", metric_name))] = (
                    dense_metric / kronecker_metric
                )

    latency_rows = _build_latency_comparison_rows(report)
    prefill_speedups = [
        row["prefill_speedup"]
        for row in latency_rows
        if row.get("prefill_speedup") is not None
    ]
    decode_speedups = [
        row["decode_speedup"]
        for row in latency_rows
        if row.get("decode_speedup") is not None
    ]
    if prefill_speedups:
        summary["prefill_speedup_mean"] = sum(prefill_speedups) / len(prefill_speedups)
        summary["prefill_speedup_best"] = max(prefill_speedups)
    if decode_speedups:
        summary["decode_speedup_mean"] = sum(decode_speedups) / len(decode_speedups)
        summary["decode_speedup_best"] = max(decode_speedups)

    for section_name in ("dense", "kronecker"):
        evaluation_rows = report.get(section_name, {}).get("evaluation", [])
        for row in evaluation_rows:
            task_name = row.get("task")
            if "accuracy" in row:
                summary["{}_{}_accuracy".format(section_name, task_name)] = row["accuracy"]
            if "perplexity" in row:
                summary["{}_{}_perplexity".format(section_name, task_name)] = row["perplexity"]

    return summary


def _build_latency_comparison_rows(report):
    dense_latency = report.get("dense", {}).get("latency", [])
    kronecker_latency = report.get("kronecker", {}).get("latency", [])
    dense_latency_map = {
        _comparison_key("latency", row): row
        for row in dense_latency
    }
    kronecker_latency_map = {
        _comparison_key("latency", row): row
        for row in kronecker_latency
    }

    rows = []
    for key in sorted(set(dense_latency_map) | set(kronecker_latency_map)):
        dense_row = dense_latency_map.get(key, {})
        kronecker_row = kronecker_latency_map.get(key, {})
        batch_size = dense_row.get("batch_size", kronecker_row.get("batch_size"))
        prefill_length = dense_row.get(
            "prefill_length",
            kronecker_row.get("prefill_length"),
        )
        decode_steps = dense_row.get("decode_steps", kronecker_row.get("decode_steps"))

        row = {
            "grid_label": "bs{}_pl{}".format(batch_size, prefill_length),
            "batch_size": batch_size,
            "prefill_length": prefill_length,
            "decode_steps": decode_steps,
            "dense_prefill_mean_ms": dense_row.get("prefill_mean_ms"),
            "kronecker_prefill_mean_ms": kronecker_row.get("prefill_mean_ms"),
            "dense_decode_mean_ms": dense_row.get("decode_mean_ms"),
            "kronecker_decode_mean_ms": kronecker_row.get("decode_mean_ms"),
            "dense_prefill_tokens_per_second": dense_row.get("prefill_tokens_per_second"),
            "kronecker_prefill_tokens_per_second": kronecker_row.get("prefill_tokens_per_second"),
            "dense_decode_tokens_per_second": dense_row.get("decode_tokens_per_second"),
            "kronecker_decode_tokens_per_second": kronecker_row.get("decode_tokens_per_second"),
        }

        dense_prefill_mean = row["dense_prefill_mean_ms"]
        kronecker_prefill_mean = row["kronecker_prefill_mean_ms"]
        dense_decode_mean = row["dense_decode_mean_ms"]
        kronecker_decode_mean = row["kronecker_decode_mean_ms"]

        if dense_prefill_mean and kronecker_prefill_mean:
            row["prefill_speedup"] = dense_prefill_mean / kronecker_prefill_mean
        if dense_decode_mean and kronecker_decode_mean:
            row["decode_speedup"] = dense_decode_mean / kronecker_decode_mean

        rows.append(row)

    return rows


def _build_evaluation_comparison_rows(report):
    dense_evaluation = report.get("dense", {}).get("evaluation", [])
    kronecker_evaluation = report.get("kronecker", {}).get("evaluation", [])
    dense_evaluation_map = {
        row.get("task"): row
        for row in dense_evaluation
    }
    kronecker_evaluation_map = {
        row.get("task"): row
        for row in kronecker_evaluation
    }

    rows = []
    for task_name in sorted(set(dense_evaluation_map) | set(kronecker_evaluation_map)):
        dense_row = dense_evaluation_map.get(task_name, {})
        kronecker_row = kronecker_evaluation_map.get(task_name, {})
        kind = dense_row.get("kind", kronecker_row.get("kind"))
        metric_name = None
        dense_value = None
        kronecker_value = None

        if "accuracy" in dense_row or "accuracy" in kronecker_row:
            metric_name = "accuracy"
            dense_value = dense_row.get("accuracy")
            kronecker_value = kronecker_row.get("accuracy")
        elif "perplexity" in dense_row or "perplexity" in kronecker_row:
            metric_name = "perplexity"
            dense_value = dense_row.get("perplexity")
            kronecker_value = kronecker_row.get("perplexity")

        row = {
            "task": task_name,
            "kind": kind,
            "metric": metric_name,
            "dense_samples": dense_row.get("samples"),
            "kronecker_samples": kronecker_row.get("samples"),
            "dense_value": dense_value,
            "kronecker_value": kronecker_value,
        }

        if dense_value is not None and kronecker_value is not None:
            row["kronecker_minus_dense"] = kronecker_value - dense_value
            if dense_value != 0:
                row["relative_change_pct"] = (
                    100.0 * (kronecker_value - dense_value) / abs(dense_value)
                )

        rows.append(row)

    return rows


def _build_metric_variant_rows(rows, metric_name):
    flattened_rows = []
    for row in rows:
        if row.get("metric") != metric_name:
            continue
        if row.get("dense_value") is not None:
            flattened_rows.append(
                {
                    "label": "{} (dense)".format(row["task"]),
                    "task": row["task"],
                    "variant": "dense",
                    "value": row["dense_value"],
                }
            )
        if row.get("kronecker_value") is not None:
            flattened_rows.append(
                {
                    "label": "{} (kronecker)".format(row["task"]),
                    "task": row["task"],
                    "variant": "kronecker",
                    "value": row["kronecker_value"],
                }
            )
    return flattened_rows


def _build_metric_delta_rows(rows, metric_name):
    delta_rows = []
    for row in rows:
        if row.get("metric") != metric_name:
            continue
        if row.get("kronecker_minus_dense") is None:
            continue
        delta_rows.append(
            {
                "task": row["task"],
                "delta": row["kronecker_minus_dense"],
            }
        )
    return delta_rows


def _build_latency_series(rows, metric_name, batch_size):
    batch_rows = [
        row
        for row in rows
        if row.get("batch_size") == batch_size
        and row.get("prefill_length") is not None
        and row.get("dense_{}".format(metric_name)) is not None
        and row.get("kronecker_{}".format(metric_name)) is not None
    ]
    batch_rows.sort(key=lambda row: row["prefill_length"])

    if not batch_rows:
        return None

    return {
        "x_values": [row["prefill_length"] for row in batch_rows],
        "dense_values": [row["dense_{}".format(metric_name)] for row in batch_rows],
        "kronecker_values": [
            row["kronecker_{}".format(metric_name)]
            for row in batch_rows
        ],
    }


def _build_wandb_visualizations(wandb, report):
    logs = {}

    dense_layer_rows = report.get("dense", {}).get("layer_benchmark", [])
    dense_layer_table = _list_to_table(wandb, dense_layer_rows)
    if dense_layer_table is not None:
        logs["tables/dense_layer_benchmark"] = dense_layer_table
        if _row_has_metric(dense_layer_rows, "speedup"):
            logs["charts/layer_speedup"] = wandb.plot.bar(
                dense_layer_table,
                "name",
                "speedup",
                title="Layer-Level Kronecker Speedup",
            )
        if _row_has_metric(dense_layer_rows, "compression_ratio"):
            logs["charts/layer_compression_ratio"] = wandb.plot.bar(
                dense_layer_table,
                "name",
                "compression_ratio",
                title="Layer Compression Ratio",
            )
        if _row_has_metric(dense_layer_rows, "compression_ratio") and _row_has_metric(
            dense_layer_rows,
            "speedup",
        ):
            logs["charts/layer_compression_vs_speedup"] = wandb.plot.scatter(
                dense_layer_table,
                "compression_ratio",
                "speedup",
                title="Compression Ratio vs Layer Speedup",
            )
        if _row_has_metric(dense_layer_rows, "relative_output_error") and _row_has_metric(
            dense_layer_rows,
            "speedup",
        ):
            logs["charts/layer_output_error_vs_speedup"] = wandb.plot.scatter(
                dense_layer_table,
                "relative_output_error",
                "speedup",
                title="Output Error vs Layer Speedup",
            )

    dense_latency_rows = report.get("dense", {}).get("latency", [])
    dense_latency_table = _list_to_table(wandb, dense_latency_rows)
    if dense_latency_table is not None:
        logs["tables/dense_latency"] = dense_latency_table

    kronecker_latency_rows = report.get("kronecker", {}).get("latency", [])
    kronecker_latency_table = _list_to_table(wandb, kronecker_latency_rows)
    if kronecker_latency_table is not None:
        logs["tables/kronecker_latency"] = kronecker_latency_table

    latency_comparison_rows = _build_latency_comparison_rows(report)
    latency_comparison_table = _list_to_table(wandb, latency_comparison_rows)
    if latency_comparison_table is not None:
        logs["tables/latency_comparison"] = latency_comparison_table
        if _row_has_metric(latency_comparison_rows, "prefill_speedup"):
            logs["charts/prefill_speedup"] = wandb.plot.bar(
                latency_comparison_table,
                "grid_label",
                "prefill_speedup",
                title="Prefill Speedup by Grid Point",
            )
        if _row_has_metric(latency_comparison_rows, "decode_speedup"):
            logs["charts/decode_speedup"] = wandb.plot.bar(
                latency_comparison_table,
                "grid_label",
                "decode_speedup",
                title="Decode Speedup by Grid Point",
            )

        batch_sizes = sorted(
            {
                row["batch_size"]
                for row in latency_comparison_rows
                if row.get("batch_size") is not None
            }
        )
        latency_metrics = (
            ("prefill_mean_ms", "Prefill Latency (ms)"),
            ("decode_mean_ms", "Decode Latency (ms)"),
            ("prefill_tokens_per_second", "Prefill Throughput (tokens/s)"),
            ("decode_tokens_per_second", "Decode Throughput (tokens/s)"),
        )
        for batch_size in batch_sizes:
            for metric_name, metric_title in latency_metrics:
                series = _build_latency_series(
                    latency_comparison_rows,
                    metric_name=metric_name,
                    batch_size=batch_size,
                )
                if series is None:
                    continue

                logs[
                    "charts/{}_bs{}".format(
                        _sanitize_key(metric_name),
                        _sanitize_key(batch_size),
                    )
                ] = wandb.plot.line_series(
                    xs=series["x_values"],
                    ys=[
                        series["dense_values"],
                        series["kronecker_values"],
                    ],
                    keys=["dense", "kronecker"],
                    title="{} vs Prefill Length (batch={})".format(
                        metric_title,
                        batch_size,
                    ),
                    xname="prefill_length",
                )

    dense_eval_rows = report.get("dense", {}).get("evaluation", [])
    dense_eval_table = _list_to_table(wandb, dense_eval_rows)
    if dense_eval_table is not None:
        logs["tables/dense_evaluation"] = dense_eval_table

    kronecker_eval_rows = report.get("kronecker", {}).get("evaluation", [])
    kronecker_eval_table = _list_to_table(wandb, kronecker_eval_rows)
    if kronecker_eval_table is not None:
        logs["tables/kronecker_evaluation"] = kronecker_eval_table

    evaluation_comparison_rows = _build_evaluation_comparison_rows(report)
    evaluation_comparison_table = _list_to_table(wandb, evaluation_comparison_rows)
    if evaluation_comparison_table is not None:
        logs["tables/evaluation_comparison"] = evaluation_comparison_table

        accuracy_rows = _build_metric_variant_rows(
            evaluation_comparison_rows,
            metric_name="accuracy",
        )
        accuracy_table = _list_to_table(wandb, accuracy_rows)
        if accuracy_table is not None:
            logs["tables/accuracy_comparison"] = accuracy_table
            logs["charts/accuracy_comparison"] = wandb.plot.bar(
                accuracy_table,
                "label",
                "value",
                title="Multiple-Choice Accuracy Comparison",
            )

        accuracy_delta_rows = _build_metric_delta_rows(
            evaluation_comparison_rows,
            metric_name="accuracy",
        )
        accuracy_delta_table = _list_to_table(wandb, accuracy_delta_rows)
        if accuracy_delta_table is not None:
            logs["tables/accuracy_delta"] = accuracy_delta_table
            logs["charts/accuracy_delta"] = wandb.plot.bar(
                accuracy_delta_table,
                "task",
                "delta",
                title="Accuracy Delta (Kronecker - Dense)",
            )

        perplexity_rows = _build_metric_variant_rows(
            evaluation_comparison_rows,
            metric_name="perplexity",
        )
        perplexity_table = _list_to_table(wandb, perplexity_rows)
        if perplexity_table is not None:
            logs["tables/perplexity_comparison"] = perplexity_table
            logs["charts/perplexity_comparison"] = wandb.plot.bar(
                perplexity_table,
                "label",
                "value",
                title="Perplexity Comparison",
            )

        perplexity_delta_rows = _build_metric_delta_rows(
            evaluation_comparison_rows,
            metric_name="perplexity",
        )
        perplexity_delta_table = _list_to_table(wandb, perplexity_delta_rows)
        if perplexity_delta_table is not None:
            logs["tables/perplexity_delta"] = perplexity_delta_table
            logs["charts/perplexity_delta"] = wandb.plot.bar(
                perplexity_delta_table,
                "task",
                "delta",
                title="Perplexity Delta (Kronecker - Dense)",
            )

    return logs


def _log_to_wandb(args, report):
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or None,
        group=args.wandb_group or None,
        tags=_parse_csv(args.wandb_tags),
        job_type="benchmark",
        mode=args.wandb_mode,
        config={
            "model_name": args.model_name,
            "device": args.device,
            "dtype": args.dtype,
            "target_modules": _parse_csv(args.target_modules),
            "max_candidate": args.max_candidate,
            "als_iter": args.als_iter,
            "factorization_objective": args.factorization_objective,
            "kronecker_implementation": args.kronecker_implementation,
            "tile_multiple": args.tile_multiple,
            "min_factor_size": args.min_factor_size,
            "tasks": _parse_csv(args.tasks),
            "decode_steps": args.decode_steps,
            "warmup_steps": args.warmup_steps,
            "benchmark_steps": args.benchmark_steps,
            "layer_seq_length": args.layer_seq_length,
            "layer_batch_size": args.layer_batch_size,
            "limit_layers": args.limit_layers,
            "eval_limit": args.eval_limit,
            "eval_max_length": args.eval_max_length,
            "eval_stride": args.eval_stride,
            "normalize_choice_scores": args.normalize_choice_scores,
            "prefill_lengths": _parse_csv(args.prefill_lengths),
            "batch_sizes": _parse_csv(args.batch_sizes),
            "skip_dense": args.skip_dense,
            "skip_layer_benchmark": args.skip_layer_benchmark,
            "skip_latency_benchmark": args.skip_latency_benchmark,
            "skip_eval": args.skip_eval,
            "wandb_mode": args.wandb_mode,
            "hostname": os.uname().nodename,
            "working_directory": os.getcwd(),
        },
    )

    run.summary.update(_build_summary_metrics(report))

    visualization_logs = _build_wandb_visualizations(wandb, report)
    if visualization_logs:
        wandb.log(visualization_logs)

    output_path = args.output or os.path.join(run.dir, "benchmark_report.json")
    with open(output_path, "w") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True))
        handle.write("\n")

    artifact = wandb.Artifact(
        name="benchmark-report-{}".format(run.id),
        type="benchmark-report",
    )
    artifact.add_file(output_path, name="benchmark_report.json")
    run.log_artifact(artifact)

    run.finish()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark dense vs Kronecker-approximated LLMs.",
    )
    parser.add_argument("--model-name", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--target-modules", default=",".join(DEFAULT_TARGET_MODULES))
    parser.add_argument("--max-candidate", type=int, default=32)
    parser.add_argument("--als-iter", type=int, default=10)
    parser.add_argument(
        "--factorization-objective",
        default="latency",
        choices=["compression", "balanced", "latency"],
    )
    parser.add_argument(
        "--kronecker-implementation",
        default="gemm",
        choices=["gemm", "einsum"],
    )
    parser.add_argument("--tile-multiple", type=int, default=16)
    parser.add_argument("--min-factor-size", type=int, default=32)
    parser.add_argument("--batch-sizes", default="1,4")
    parser.add_argument("--prefill-lengths", default="128,512")
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--benchmark-steps", type=int, default=20)
    parser.add_argument("--layer-seq-length", type=int, default=256)
    parser.add_argument("--layer-batch-size", type=int, default=4)
    parser.add_argument("--limit-layers", type=int, default=12)
    parser.add_argument("--tasks", default="wikitext103,c4_en,hellaswag,boolq,arc_easy,arc_challenge")
    parser.add_argument("--eval-limit", type=int, default=128)
    parser.add_argument("--eval-max-length", type=int, default=2048)
    parser.add_argument("--eval-stride", type=int, default=512)
    parser.add_argument("--normalize-choice-scores", action="store_true")
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-group", default="")
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument(
        "--wandb-mode",
        default="",
        choices=["", "online", "offline", "disabled"],
    )
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--skip-layer-benchmark", action="store_true")
    parser.add_argument("--skip-latency-benchmark", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--output", default="")
    return parser


def main():
    args = build_parser().parse_args()

    import torch

    from src.eval import benchmark_selected_linear_layers, get_supported_tasks
    from src.models.model_utils import (
        get_kronecker_stats,
        replace_all_linear_with_kronecker,
    )

    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype, device)
    target_modules = _parse_csv(args.target_modules)
    tasks = _parse_csv(args.tasks)
    batch_sizes = [int(value) for value in _parse_csv(args.batch_sizes)]
    prefill_lengths = [int(value) for value in _parse_csv(args.prefill_lengths)]
    unsupported_tasks = [task for task in tasks if task not in get_supported_tasks()]
    if unsupported_tasks:
        raise ValueError(
            "Unsupported tasks: {}. Supported tasks: {}".format(
                unsupported_tasks,
                get_supported_tasks(),
            )
        )

    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True

    report = {
        "model_name": args.model_name,
        "device": device,
        "dtype": str(dtype),
        "target_modules": target_modules,
        "factorization_objective": args.factorization_objective,
        "kronecker_implementation": args.kronecker_implementation,
        "tile_multiple": args.tile_multiple,
        "min_factor_size": args.min_factor_size,
    }

    model, tokenizer = _load_model_and_tokenizer(args.model_name, device, dtype)

    if not args.skip_dense:
        dense_report = {}
        if not args.skip_layer_benchmark:
            dense_report["layer_benchmark"] = benchmark_selected_linear_layers(
                model,
                target_modules=target_modules,
                batch_size=args.layer_batch_size,
                seq_length=args.layer_seq_length,
                max_candidate=args.max_candidate,
                als_iter=args.als_iter,
                factorization_objective=args.factorization_objective,
                tile_multiple=args.tile_multiple,
                min_factor_size=args.min_factor_size,
                implementation=args.kronecker_implementation,
                warmup_steps=args.warmup_steps,
                benchmark_steps=args.benchmark_steps,
                limit_layers=args.limit_layers,
            )

        if not args.skip_latency_benchmark:
            dense_report["latency"] = _run_latency_grid(
                model,
                batch_sizes=batch_sizes,
                prefill_lengths=prefill_lengths,
                decode_steps=args.decode_steps,
                warmup_steps=args.warmup_steps,
                benchmark_steps=args.benchmark_steps,
            )

        if not args.skip_eval:
            dense_report["evaluation"] = _run_eval_tasks(
                model,
                tokenizer,
                tasks=tasks,
                limit=args.eval_limit,
                max_length=args.eval_max_length,
                stride=args.eval_stride,
                normalize_choice_scores=args.normalize_choice_scores,
            )

        report["dense"] = dense_report

    replace_all_linear_with_kronecker(
        model,
        max_candidate=args.max_candidate,
        als_iter=args.als_iter,
        target_modules=target_modules,
        factorization_objective=args.factorization_objective,
        tile_multiple=args.tile_multiple,
        min_factor_size=args.min_factor_size,
        implementation=args.kronecker_implementation,
    )

    kronecker_report = {
        "stats": get_kronecker_stats(model),
    }

    if not args.skip_latency_benchmark:
        kronecker_report["latency"] = _run_latency_grid(
            model,
            batch_sizes=batch_sizes,
            prefill_lengths=prefill_lengths,
            decode_steps=args.decode_steps,
            warmup_steps=args.warmup_steps,
            benchmark_steps=args.benchmark_steps,
        )

    if not args.skip_eval:
        kronecker_report["evaluation"] = _run_eval_tasks(
            model,
            tokenizer,
            tasks=tasks,
            limit=args.eval_limit,
            max_length=args.eval_max_length,
            stride=args.eval_stride,
            normalize_choice_scores=args.normalize_choice_scores,
        )

    report["kronecker"] = kronecker_report

    output = json.dumps(report, indent=2, sort_keys=True)
    print(output)

    if args.output:
        with open(args.output, "w") as handle:
            handle.write(output)
            handle.write("\n")

    wandb_project = args.wandb_project or os.environ.get("WANDB_PROJECT", "")
    if wandb_project:
        args.wandb_project = wandb_project
        if not args.wandb_entity:
            args.wandb_entity = os.environ.get("WANDB_ENTITY", "")
        if not args.wandb_run_name:
            args.wandb_run_name = os.environ.get("WANDB_NAME", "")
        if not args.wandb_group:
            args.wandb_group = os.environ.get("WANDB_RUN_GROUP", "")
        if not args.wandb_tags:
            args.wandb_tags = os.environ.get("WANDB_TAGS", "")
        if not args.wandb_mode:
            args.wandb_mode = os.environ.get("WANDB_MODE", "online")
        _log_to_wandb(args, report)


if __name__ == "__main__":
    main()
