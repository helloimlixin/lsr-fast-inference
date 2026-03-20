import time

import torch
import torch.nn as nn
from datasets import load_dataset

from src.models.kronecker_linear import KroneckerLinear
from src.models.model_utils import (
    evaluate_factorization,
    try_replace_linear_with_kronecker,
)


TEXT_TASKS = {
    "wikitext103": {
        "dataset_name": "wikitext",
        "subset": "wikitext-103-raw-v1",
        "split": "validation",
        "kind": "text",
        "text_field": "text",
    },
    "c4_en": {
        "dataset_name": "allenai/c4",
        "subset": "en",
        "split": "validation",
        "kind": "text",
        "text_field": "text",
    },
}

MULTIPLE_CHOICE_TASKS = {
    "hellaswag": {
        "dataset_name": "Rowan/hellaswag",
        "split": "validation",
        "kind": "multiple_choice",
        "adapter": "hellaswag",
    },
    "boolq": {
        "dataset_name": "super_glue",
        "subset": "boolq",
        "split": "validation",
        "kind": "multiple_choice",
        "adapter": "boolq",
    },
    "arc_easy": {
        "dataset_name": "allenai/ai2_arc",
        "subset": "ARC-Easy",
        "split": "validation",
        "kind": "multiple_choice",
        "adapter": "arc",
    },
    "arc_challenge": {
        "dataset_name": "allenai/ai2_arc",
        "subset": "ARC-Challenge",
        "split": "validation",
        "kind": "multiple_choice",
        "adapter": "arc",
    },
}


def get_supported_tasks():
    return sorted(list(TEXT_TASKS) + list(MULTIPLE_CHOICE_TASKS))


def _load_dataset_split(task_spec, limit=None):
    subset = task_spec.get("subset")
    if subset is None:
        dataset = load_dataset(
            task_spec["dataset_name"],
            split=task_spec["split"],
        )
    else:
        dataset = load_dataset(
            task_spec["dataset_name"],
            subset,
            split=task_spec["split"],
        )

    if limit is not None:
        limit = min(limit, len(dataset))
        dataset = dataset.select(range(limit))

    return dataset


def _sync_device(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _benchmark_callable(fn, device, warmup_steps, benchmark_steps):
    for _ in range(warmup_steps):
        with torch.inference_mode():
            fn()
    _sync_device(device)

    durations = []
    for _ in range(benchmark_steps):
        start = time.perf_counter()
        with torch.inference_mode():
            fn()
        _sync_device(device)
        durations.append((time.perf_counter() - start) * 1000.0)

    mean_duration = sum(durations) / len(durations)
    return {
        "mean_ms": mean_duration,
        "min_ms": min(durations),
        "max_ms": max(durations),
    }


def _module_matches(name, target_modules):
    if not target_modules:
        return True
    return any(token in name for token in target_modules)


def benchmark_selected_linear_layers(
    model,
    target_modules,
    batch_size,
    seq_length,
    max_candidate,
    als_iter,
    factorization_objective,
    tile_multiple,
    min_factor_size,
    implementation,
    warmup_steps,
    benchmark_steps,
    limit_layers=None,
):
    results = []
    device = next(model.parameters()).device

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not _module_matches(name, target_modules):
            continue

        kron_layer = try_replace_linear_with_kronecker(
            module,
            max_candidate=max_candidate,
            als_iter=als_iter,
            factorization_objective=factorization_objective,
            tile_multiple=tile_multiple,
            min_factor_size=min_factor_size,
            implementation=implementation,
        )
        if not isinstance(kron_layer, KroneckerLinear):
            continue

        dtype = module.weight.dtype
        inputs = torch.randn(
            batch_size,
            seq_length,
            module.in_features,
            device=device,
            dtype=dtype,
        )

        module.eval()
        kron_layer.eval()

        with torch.inference_mode():
            dense_reference = module(inputs)
            factorization_stats = evaluate_factorization(
                module,
                kron_layer,
                test_input=inputs,
            )

        dense_timing = _benchmark_callable(
            lambda: module(inputs),
            device,
            warmup_steps,
            benchmark_steps,
        )
        kronecker_timing = _benchmark_callable(
            lambda: kron_layer(inputs),
            device,
            warmup_steps,
            benchmark_steps,
        )

        with torch.inference_mode():
            kronecker_reference = kron_layer(inputs)
        output_error = (
            torch.norm(dense_reference - kronecker_reference)
            / torch.norm(dense_reference).clamp_min(1e-8)
        ).item()

        results.append(
            {
                "name": name,
                "dense_mean_ms": dense_timing["mean_ms"],
                "kronecker_mean_ms": kronecker_timing["mean_ms"],
                "speedup": dense_timing["mean_ms"] / kronecker_timing["mean_ms"],
                "relative_weight_error": factorization_stats["relative_error"],
                "relative_output_error": output_error,
                "compression_ratio": factorization_stats["compression_ratio"],
                "implementation": implementation,
                "original_params": factorization_stats["original_params"],
                "factorized_params": factorization_stats["factorized_params"],
            }
        )

        if limit_layers is not None and len(results) >= limit_layers:
            break

    return results


def _make_random_input_ids(model, batch_size, seq_length, device):
    vocab_size = int(model.config.vocab_size)
    input_ids = torch.randint(
        0,
        vocab_size,
        (batch_size, seq_length),
        device=device,
    )

    bos_token_id = getattr(model.config, "bos_token_id", None)
    if bos_token_id is not None:
        input_ids[:, 0] = bos_token_id

    return input_ids


def benchmark_model_latency(
    model,
    batch_size,
    prefill_length,
    decode_steps,
    warmup_steps,
    benchmark_steps,
):
    device = next(model.parameters()).device
    input_ids = _make_random_input_ids(model, batch_size, prefill_length, device)
    decode_ids = _make_random_input_ids(model, batch_size, max(1, decode_steps), device)

    model.eval()

    def run_prefill():
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=True)

    def run_decode():
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
            for step in range(decode_steps):
                outputs = model(
                    input_ids=decode_ids[:, step : step + 1],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values

    prefill_timing = _benchmark_callable(
        run_prefill,
        device,
        warmup_steps,
        benchmark_steps,
    )
    decode_timing = _benchmark_callable(
        run_decode,
        device,
        warmup_steps,
        benchmark_steps,
    )

    decode_tokens = max(1, batch_size * max(1, decode_steps))
    prefill_tokens = max(1, batch_size * max(1, prefill_length))

    return {
        "batch_size": batch_size,
        "prefill_length": prefill_length,
        "decode_steps": decode_steps,
        "prefill_mean_ms": prefill_timing["mean_ms"],
        "decode_mean_ms": decode_timing["mean_ms"],
        "prefill_tokens_per_second": 1000.0 * prefill_tokens / prefill_timing["mean_ms"],
        "decode_tokens_per_second": 1000.0 * decode_tokens / decode_timing["mean_ms"],
    }


def _effective_max_length(model, tokenizer, requested_max_length):
    model_limit = getattr(model.config, "max_position_embeddings", requested_max_length)
    tokenizer_limit = getattr(tokenizer, "model_max_length", requested_max_length)
    if tokenizer_limit is None or tokenizer_limit > 100000:
        tokenizer_limit = requested_max_length
    return max(2, min(requested_max_length, model_limit, tokenizer_limit))


def _tokenize_text_corpus(tokenizer, texts):
    separator_ids = tokenizer.encode("\n\n", add_special_tokens=False)
    token_ids = []

    for text in texts:
        text_ids = tokenizer.encode(text, add_special_tokens=False)
        if not text_ids:
            continue

        if token_ids and separator_ids:
            token_ids.extend(separator_ids)
        token_ids.extend(text_ids)

    return token_ids


def evaluate_perplexity(
    model,
    tokenizer,
    task_name,
    limit,
    max_length,
    stride,
):
    task_spec = TEXT_TASKS[task_name]
    dataset = _load_dataset_split(task_spec, limit=limit)

    texts = []
    for row in dataset:
        text = row[task_spec["text_field"]].strip()
        if text:
            texts.append(text)

    if not texts:
        return {
            "task": task_name,
            "kind": "text",
            "samples": 0,
            "perplexity": float("nan"),
        }

    device = next(model.parameters()).device
    max_length = _effective_max_length(model, tokenizer, max_length)
    stride = min(max_length, stride)
    token_ids = _tokenize_text_corpus(tokenizer, texts)
    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device,
    )

    negative_log_likelihoods = []
    prev_end = 0

    for begin in range(0, input_ids.size(1), stride):
        end = min(begin + max_length, input_ids.size(1))
        target_length = end - prev_end
        window_ids = input_ids[:, begin:end]
        target_ids = window_ids.clone()
        target_ids[:, :-target_length] = -100

        with torch.inference_mode():
            outputs = model(window_ids, labels=target_ids)

        negative_log_likelihoods.append(outputs.loss * target_length)
        prev_end = end
        if end >= input_ids.size(1):
            break

    perplexity = torch.exp(torch.stack(negative_log_likelihoods).sum() / prev_end).item()
    return {
        "task": task_name,
        "kind": "text",
        "samples": len(texts),
        "tokens": int(input_ids.numel()),
        "perplexity": perplexity,
    }


def _prepare_prompt_choice(prompt, choice):
    prompt = prompt.rstrip()
    choice = choice.strip()
    if not choice:
        return prompt, choice
    if choice[0] in ".,;:!?)]}":
        return prompt, choice
    return prompt, " " + choice


def _build_multiple_choice_example(task_name, row):
    if task_name == "hellaswag":
        prompt = row.get("ctx") or "{} {}".format(row["ctx_a"], row["ctx_b"])
        label = int(row["label"])
        return {
            "prompt": prompt,
            "choices": row["endings"],
            "label": label,
        }

    if task_name == "boolq":
        prompt = "{}\nQuestion: {}\nAnswer:".format(
            row["passage"],
            row["question"],
        )
        return {
            "prompt": prompt,
            "choices": ["no", "yes"],
            "label": int(row["label"]),
        }

    if task_name in ("arc_easy", "arc_challenge"):
        choices = row["choices"]["text"]
        labels = row["choices"]["label"]
        answer_key = str(row["answerKey"])
        if answer_key not in labels:
            raise ValueError("Unsupported ARC answer key: {}".format(answer_key))

        prompt = "Question: {}\nAnswer:".format(row["question"])
        return {
            "prompt": prompt,
            "choices": choices,
            "label": labels.index(answer_key),
        }

    raise ValueError("Unsupported multiple-choice task: {}".format(task_name))


def _choice_loglikelihood(
    model,
    tokenizer,
    prompt,
    choice,
    max_length,
    normalize_choice_scores,
):
    prompt, choice = _prepare_prompt_choice(prompt, choice)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    choice_ids = tokenizer.encode(choice, add_special_tokens=False)
    if not choice_ids:
        return float("-inf")

    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        bos_token_id = tokenizer.eos_token_id
    if bos_token_id is None:
        bos_token_id = tokenizer.pad_token_id

    if not prompt_ids and bos_token_id is not None:
        prompt_ids = [bos_token_id]

    available_prompt = max_length - len(choice_ids)
    if available_prompt <= 0:
        choice_ids = choice_ids[-(max_length - 1) :]
        available_prompt = 1

    prompt_ids = prompt_ids[-available_prompt:]
    if not prompt_ids:
        if bos_token_id is None:
            prompt_ids = [choice_ids[0]]
            choice_ids = choice_ids[1:]
        else:
            prompt_ids = [bos_token_id]

    if not choice_ids:
        return float("-inf")

    input_ids = prompt_ids + choice_ids
    input_tensor = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=next(model.parameters()).device,
    )

    with torch.inference_mode():
        logits = model(input_ids=input_tensor).logits[:, :-1, :]
        log_probs = torch.log_softmax(logits, dim=-1)

    targets = input_tensor[:, 1:]
    target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    choice_mask = torch.zeros_like(targets, dtype=torch.bool)
    choice_mask[:, len(prompt_ids) - 1 :] = True
    score = target_log_probs.masked_select(choice_mask).sum().item()

    if normalize_choice_scores:
        score /= max(1, int(choice_mask.sum().item()))
    return score


def evaluate_multiple_choice(
    model,
    tokenizer,
    task_name,
    limit,
    max_length,
    normalize_choice_scores=False,
):
    task_spec = MULTIPLE_CHOICE_TASKS[task_name]
    dataset = _load_dataset_split(task_spec, limit=limit)
    max_length = _effective_max_length(model, tokenizer, max_length)

    correct = 0
    total = 0

    for row in dataset:
        example = _build_multiple_choice_example(task_name, row)
        scores = [
            _choice_loglikelihood(
                model,
                tokenizer,
                example["prompt"],
                choice,
                max_length=max_length,
                normalize_choice_scores=normalize_choice_scores,
            )
            for choice in example["choices"]
        ]

        prediction = max(range(len(scores)), key=lambda index: scores[index])
        correct += int(prediction == example["label"])
        total += 1

    accuracy = float(correct) / float(total) if total else float("nan")
    return {
        "task": task_name,
        "kind": "multiple_choice",
        "samples": total,
        "accuracy": accuracy,
    }
