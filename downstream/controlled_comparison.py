"""Figure 4 ASIS and CTRL checkpoint comparison on the native Basic Five."""

from decimal import Decimal, InvalidOperation

METRICS = {
    "ImageNet-1K": "top1_percent",
    "COCO": "bbox_ap_percent",
    "ADE20K": "miou_percent",
    "NYUv2": "rmse_meters",
    "SSv2": "top1_percent",
}
SETTINGS = ("asis", "ctrl_100", "ctrl_200", "ctrl_300")


def _score(value, task):
    if type(value) not in (int, float, str):
        raise ValueError("a measured primary score is required")
    try:
        score = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("invalid decimal score") from exc
    if not score.is_finite() or score < 0 or (task != "NYUv2" and score > 100):
        raise ValueError("invalid native primary score")
    return score


def compare_controlled(document):
    """Consume plotting-source records and derive statistics from scores alone."""
    records = document["records"]
    if document["cohort"] != 32 or not isinstance(records, list) or len(records) != 32:
        raise ValueError("Figure 4 requires 32 distinct paired methods")
    tasks = {task: {} for task in METRICS}
    for record in records:
        method = record["method"]
        if not isinstance(method, str) or not method.strip() or method in tasks["COCO"]:
            raise ValueError("each method must have a distinct name")
        rows = record["metrics"]
        if len(rows) != len(METRICS) or {row["task"] for row in rows} != set(METRICS):
            raise ValueError("each method requires all five native tasks")
        for row in rows:
            task = row["task"]
            if row.get("metric", METRICS[task]) != METRICS[task]:
                raise ValueError("declared metric differs from the native Basic Five metric")
            tasks[task][method] = {key: _score(row[key], task) for key in SETTINGS}
    report = {}
    for task, methods in tasks.items():
        lower = task == "NYUv2"
        results = {}
        counts = dict(improved=0, worse=0, equal=0)
        for name, scores in methods.items():
            ranks = {
                key: 1 + sum((other[key] < value if lower else other[key] > value)
                             for other in methods.values())
                for key, value in scores.items()
            }
            delta = scores["ctrl_300"] - scores["asis"]
            gain = -delta if lower else delta
            counts["improved" if gain > 0 else "worse" if gain < 0 else "equal"] += 1
            results[name] = {"scores": {key: str(value) for key, value in scores.items()},
                             "ranks": ranks, "delta": str(delta), "improvement": str(gain)}
        winners = [name for name, result in results.items() if result["ranks"]["ctrl_300"] == 1]
        report[task] = dict(metric=METRICS[task], methods=results, counts=counts, ctrl_winners=winners)
    return {"schema": "figure4_controlled_comparison_v1", "historical_run_verified": False,
            "tasks": report}
