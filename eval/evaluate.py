"""
Defense NLP Benchmarks - Evaluation Script

Runs benchmark evaluation tasks defined by YAML configuration files
against JSONL-formatted sample data. Computes standard classification
and generation metrics and outputs results as JSON.

Usage:
    python eval/evaluate.py --config benchmarks/threat-report-classification.yaml \
                            --data data/samples/threat-reports.jsonl \
                            --output results/output.json

Author: DLRA (Defense Language Research Agency)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict[str, Any]:
    """Load and validate a YAML benchmark configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    required_keys = ["task", "metrics", "evaluation"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Config missing required key: '{key}'")

    logger.info("Loaded config: %s (task: %s)", config_path, config["task"]["name"])
    return config


def load_data(data_path: str) -> list[dict[str, Any]]:
    """Load evaluation data from a JSONL file.

    Args:
        data_path: Path to the JSONL data file.

    Returns:
        List of parsed JSON records.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed line %d: %s", line_num, e)

    logger.info("Loaded %d records from %s", len(records), data_path)
    return records


def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    average: str = "weighted",
) -> dict[str, float]:
    """Compute standard classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        average: Averaging strategy for multi-class metrics.

    Returns:
        Dictionary with f1, precision, recall, and accuracy scores.
    """
    return {
        "f1": round(f1_score(y_true, y_pred, average=average, zero_division=0), 4),
        "precision": round(
            precision_score(y_true, y_pred, average=average, zero_division=0), 4
        ),
        "recall": round(
            recall_score(y_true, y_pred, average=average, zero_division=0), 4
        ),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
    }


def compute_ner_metrics(
    true_entities: list[list[dict]],
    pred_entities: list[list[dict]],
) -> dict[str, float]:
    """Compute entity-level NER metrics with strict span matching.

    Args:
        true_entities: List of ground-truth entity lists per sample.
        pred_entities: List of predicted entity lists per sample.

    Returns:
        Dictionary with entity-level precision, recall, and F1.
    """
    tp, fp, fn = 0, 0, 0

    for true_ents, pred_ents in zip(true_entities, pred_entities):
        true_spans = {(e["text"], e["type"], e["start"], e["end"]) for e in true_ents}
        pred_spans = {(e["text"], e["type"], e["start"], e["end"]) for e in pred_ents}

        tp += len(true_spans & pred_spans)
        fp += len(pred_spans - true_spans)
        fn += len(true_spans - pred_spans)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


def run_evaluation(
    config: dict[str, Any],
    data: list[dict[str, Any]],
    n_samples: Optional[int] = None,
    cv_folds: Optional[int] = None,
) -> dict[str, Any]:
    """Run the benchmark evaluation loop.

    This is a scaffold implementation that processes loaded data according
    to the task configuration. In production use, this function dispatches
    to model inference endpoints; here it operates on pre-annotated samples
    to validate the evaluation pipeline.

    Args:
        config: Parsed benchmark configuration.
        data: List of evaluation records.
        n_samples: Override for number of samples to evaluate.
        cv_folds: Override for number of cross-validation folds.

    Returns:
        Dictionary containing evaluation results and metadata.
    """
    task_name = config["task"]["name"]
    eval_config = config["evaluation"]

    n = n_samples or eval_config.get("n_samples", len(data))
    folds = cv_folds or int(str(eval_config.get("cross_validation", "5-fold")).split("-")[0])

    # Subsample if necessary
    if len(data) > n:
        rng = np.random.default_rng(seed=42)
        indices = rng.choice(len(data), size=n, replace=False)
        data = [data[i] for i in indices]
    elif len(data) < n:
        logger.warning(
            "Requested %d samples but only %d available. Using all.", n, len(data)
        )

    logger.info("Running evaluation: task=%s, samples=%d, folds=%d", task_name, len(data), folds)

    results: dict[str, Any] = {
        "task": task_name,
        "version": config["task"].get("version", "unknown"),
        "n_samples": len(data),
        "cv_folds": folds,
        "metrics": {},
    }

    if task_name == "threat-report-classification":
        labels = [record.get("category", "unknown") for record in data]
        # In scaffold mode, simulate predictions as ground truth
        # Replace with model inference in production
        predictions = labels.copy()
        metrics = compute_classification_metrics(labels, predictions)
        results["metrics"] = metrics

    elif task_name == "defense-named-entity-recognition":
        true_entities = [record.get("entities", []) for record in data]
        pred_entities = true_entities.copy()
        metrics = compute_ner_metrics(true_entities, pred_entities)
        results["metrics"] = metrics

    elif task_name == "maritime-text-analysis":
        labels = [record.get("label", "unknown") for record in data]
        predictions = labels.copy()
        metrics = compute_classification_metrics(labels, predictions)
        results["metrics"] = metrics

    else:
        logger.error("Unknown task: %s", task_name)
        results["error"] = f"Unknown task: {task_name}"

    return results


def save_results(results: dict[str, Any], output_path: str) -> None:
    """Save evaluation results to a JSON file.

    Args:
        results: Evaluation results dictionary.
        output_path: Path to write the JSON output.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Results saved to %s", output_path)


def main() -> None:
    """Main entry point for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Defense NLP Benchmarks - Evaluation Runner",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML benchmark configuration file",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to JSONL evaluation data file",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write JSON results (default: stdout)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Override number of evaluation samples",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=None,
        help="Override number of cross-validation folds",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run all benchmark tasks in the benchmarks/ directory",
    )

    args = parser.parse_args()

    if args.run_all:
        benchmarks_dir = Path("benchmarks")
        configs = sorted(benchmarks_dir.glob("*.yaml"))
        if not configs:
            logger.error("No YAML configs found in benchmarks/")
            sys.exit(1)
        for config_path in configs:
            logger.info("--- Running: %s ---", config_path.name)
            config = load_config(str(config_path))
            # Infer data path from task name
            task_name = config["task"]["name"]
            data_path = Path("data/samples") / f"{task_name.replace('-', '_')}.jsonl"
            if not data_path.exists():
                logger.warning("No data file for task %s, skipping", task_name)
                continue
            data = load_data(str(data_path))
            results = run_evaluation(config, data, args.n_samples, args.cv_folds)
            print(json.dumps(results, indent=2))
        return

    config = load_config(args.config)
    data = load_data(args.data)
    results = run_evaluation(config, data, args.n_samples, args.cv_folds)

    if args.output:
        save_results(results, args.output)
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
