"""Metric checks use synthetic judgments, never real benchmark labels."""

import json

import pytest

from evaluate import load_judgments, query_metrics, report_run


def test_missing_judgments_do_not_count_as_irrelevant():
    assert query_metrics(["a"], {}, ["a"]) is None
    assert query_metrics(["a"], {"a": {"relevance": 2}}, ["a", "b"]) is None


def test_precision_uses_five_slots_and_ndcg_uses_graded_pool_ideal():
    labels = {"a": {"relevance": 2}, "b": {"relevance": 1}, "c": {"relevance": 0}}
    top = query_metrics(["a", "b"], labels, list(labels))
    assert top["precision_at_5"] == 0.2
    assert top["pooled_ndcg_at_5"] == 1
    reversed_ = query_metrics(["b", "a"], labels, list(labels))
    assert 0 < reversed_["pooled_ndcg_at_5"] < 1


def test_zero_relevance_ndcg_is_undefined():
    assert query_metrics(["a"], {"a": {"relevance": 0}}, ["a"]) == {
        "precision_at_5": 0,
        "pooled_ndcg_at_5": None,
    }


def test_tone_needs_independent_mood_judgment():
    labels = {"a": {"relevance": 2}}
    assert query_metrics(["a"], labels, ["a"], has_tone=True) is None
    labels["a"]["mood"] = 0
    result = query_metrics(["a"], labels, ["a"], has_tone=True)
    assert result["precision_at_5"] == 0.2
    assert result["joint_precision_at_5"] == 0


def test_wrong_run_and_invalid_labels_rejected(tmp_path):
    p = tmp_path / "ratings.json"
    run = {"run_id": "correct", "queries": [{"pool": [{"item_id": "a"}]}]}
    for content in [
        {"run_id": "wrong", "judgments": {}},
        {"run_id": "correct", "judgments": {"unknown": {"relevance": 2}}},
        {"run_id": "correct", "judgments": {"a": {"relevance": True}}},
        {"run_id": "correct", "judgments": {"a": {"relevance": 3}}},
    ]:
        p.write_text(json.dumps(content))
        with pytest.raises(ValueError):
            load_judgments(p, run)


def test_report_uses_shared_complete_query_pools(tmp_path):
    run = {
        "run_id": "example",
        "k": 5,
        "systems": {"A": 0, "B": 0.5},
        "queries": [
            {
                "id": "q1",
                "split": "dev",
                "category": "Fiction",
                "tone": "All",
                "pool": [
                    {"item_id": "a", "category": "Fiction"},
                    {"item_id": "b", "category": "Fiction"},
                ],
                "rankings": {"A": ["a"], "B": ["b"]},
            },
        ],
    }
    partial = report_run(run, {"a": {"relevance": 2}}, "dev", tmp_path)
    assert partial["fully_judged_queries"] == 0
    assert partial["quality"]["A"]["precision_at_5"]["mean"] is None
    complete = report_run(
        run, {"a": {"relevance": 2}, "b": {"relevance": 0}}, "dev", tmp_path
    )
    assert complete["fully_judged_queries"] == 1
    assert complete["quality"]["A"]["precision_at_5"]["mean"] == 0.2
    assert complete["quality"]["B"]["precision_at_5"]["mean"] == 0
