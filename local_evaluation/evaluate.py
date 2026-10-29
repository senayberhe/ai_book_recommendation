"""Private, reproducible pooled relevance evaluation. See README.md for the protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
WEIGHTS = {"semantic_only": 0.0, "tone_025": 0.25, "tone_050": 0.5, "tone_075": 0.75}
K = 5


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def dcg(grades):
    return sum(
        (2**grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(grades)
    )


def query_metrics(ids, judgments, pool_ids, k=K, has_tone=False):
    """Require the entire shared pool judged; missing slots count against precision."""
    required = ("relevance", "mood") if has_tone else ("relevance",)
    if any(
        any(judgments.get(i, {}).get(field) not in (0, 1, 2) for field in required)
        for i in pool_ids
    ):
        return None
    grades = [judgments[i]["relevance"] for i in ids[:k]]
    ideal = sorted((judgments[i]["relevance"] for i in pool_ids), reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    result = {
        "precision_at_5": sum(g == 2 for g in grades) / k,
        # No judged relevant items: nDCG is undefined, not a synthetic zero.
        "pooled_ndcg_at_5": dcg(grades) / ideal_dcg if ideal_dcg else None,
    }
    if has_tone:
        result["mood_fit_at_5"] = sum(judgments[i]["mood"] / 2 for i in ids[:k]) / k
        result["joint_precision_at_5"] = (
            sum(
                judgments[i]["relevance"] == 2 and judgments[i]["mood"] >= 1
                for i in ids[:k]
            )
            / k
        )
    return result


def load_judgments(path, run):
    if not path:
        return {}
    payload = json.loads(Path(path).read_text())
    if payload.get("run_id") != run["run_id"]:
        raise ValueError("Judgments belong to another evaluation run")
    valid = {item["item_id"] for q in run["queries"] for item in q["pool"]}
    result = payload.get("judgments", {})
    if not isinstance(result, dict) or set(result) - valid:
        raise ValueError("Unknown judgment IDs")
    for fields in result.values():
        if not isinstance(fields, dict) or set(fields) - {"relevance", "mood"}:
            raise ValueError("Unexpected judgment fields")
        if any(type(v) is not int or v not in (0, 1, 2) for v in fields.values()):
            raise ValueError("Grades must be integers 0, 1 or 2")
    return result


def build_review(run, split, destination):
    # Do not put system names, ranking positions, emotion predictions, or test queries
    # in the development review. The report still retains a private mapping in run.json.
    review = {
        "run_id": run["run_id"],
        "split": split,
        "queries": [
            {k: q[k] for k in ("id", "query", "category", "tone", "pool")}
            for q in run["queries"]
            if q["split"] == split
        ],
    }
    template = Path(__file__).with_name("review_template.html").read_text()
    data = json.dumps(review, ensure_ascii=False).replace("<", "\\u003c")
    Path(destination).write_text(
        template.replace("__REVIEW_DATA__", data), encoding="utf-8"
    )


def prepare(args):
    import importlib.metadata
    import shutil
    import sqlite3

    import numpy as np
    from dotenv import dotenv_values
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings

    from book_recommender.data import load_books
    from book_recommender.recommender import BookRecommender

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run.json").exists():
        raise ValueError(
            "Run already exists. Use a fresh --out path; do not overwrite judgments"
        )
    queries = json.loads(Path(args.queries).read_text())
    if len({q["id"] for q in queries}) != len(queries):
        raise ValueError("Query IDs must be unique")
    source = ROOT / "chroma_db"
    # Snapshot SQLite using its backup API, then copy the vector files. Run while
    # the local index is idle. Never open or modify the production/local source index.
    snapshot = out / "index_snapshot"
    shutil.copytree(source, snapshot)
    with sqlite3.connect(f"file:{source / 'chroma.sqlite3'}?mode=ro", uri=True) as src:
        with sqlite3.connect(snapshot / "chroma.sqlite3") as dst:
            src.backup(dst)
    credentials = dotenv_values(ROOT / ".env")
    embedder = OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=credentials.get("OPENAI_API_KEY")
    )
    store = Chroma(persist_directory=str(snapshot), embedding_function=embedder)
    count = store._collection.count()
    if count == 0:
        raise ValueError(
            "The existing index is empty. No automatic reindexing is allowed"
        )
    catalogue = ROOT / "data/books_with_emotions.csv"
    books = load_books(catalogue)
    if not books["isbn13"].is_unique:
        raise ValueError("Catalogue ISBNs must be unique")
    catalogue_ids = set(books["isbn13"])
    indexed = store.get()["ids"]
    metadata = store.get(include=["metadatas"])["metadatas"]
    indexed_isbns = {str(m.get("isbn13")) for m in metadata if m}
    if indexed_isbns != catalogue_ids or len(indexed) != len(books):
        raise ValueError(
            "Index and catalogue ISBNs/counts disagree; rebuild explicitly before evaluation"
        )
    # Check indexed descriptions and categories, not just IDs, to catch stale content.
    records = store.get(include=["documents", "metadatas"])
    by_isbn = books.set_index("isbn13")
    for text, meta in zip(records["documents"], records["metadatas"], strict=True):
        row = by_isbn.loc[str(meta["isbn13"])]
        if (
            text != str(row["description"]).strip()
            or meta.get("category") != row["simple_categories"]
        ):
            raise ValueError("Index contains stale descriptions or categories")
    print(
        f"Validated {count} indexed books. Embedding {len(queries)} query descriptions only.",
        flush=True,
    )
    vectors = embedder.embed_documents([q["query"] for q in queries])
    np.save(out / "query_embeddings.npy", np.asarray(vectors))
    run = {
        "run_id": hashlib.sha256(
            (digest(catalogue) + json.dumps(queries) + json.dumps(WEIGHTS)).encode()
        ).hexdigest()[:16],
        "created_utc": datetime.now(UTC).isoformat(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "chromadb",
                "langchain-chroma",
                "langchain-openai",
                "numpy",
                "pandas",
            )
        },
        "k": K,
        "candidate_k": 50,
        "systems": WEIGHTS,
        "catalogue_sha256": digest(catalogue),
        "queries_sha256": digest(args.queries),
        "query_embedding_model": "text-embedding-3-small",
        "index_count": count,
        "index_metadata": store._collection.metadata,
        "index_sqlite_sha256": digest(snapshot / "chroma.sqlite3"),
        "engine_sha256": digest(ROOT / "book_recommender/recommender.py"),
        "queries": [],
    }
    write_json(out / "queries.json", queries)
    rng = random.Random(42)
    for q, vector in zip(queries, vectors, strict=True):
        kwargs = (
            {} if q["category"] == "All" else {"filter": {"category": q["category"]}}
        )
        hits = store.similarity_search_by_vector(vector, k=50, **kwargs)

        class CachedStore:
            def similarity_search(self, query, k, **kwargs):
                return hits[:k]

        rankings, pool = {}, {}
        for system, weight in WEIGHTS.items():
            results = BookRecommender(
                books,
                CachedStore(),
                initial_top_k=50,
                final_top_k=K,
                tone_weight=weight,
            ).recommend(q["query"], category=q["category"], tone=q["tone"])
            ids = []
            for _, row in results.iterrows():
                item_id = hashlib.sha256(
                    f"{q['id']}:{row['isbn13']}".encode()
                ).hexdigest()[:16]
                ids.append(item_id)
                pool[item_id] = {
                    "item_id": item_id,
                    "isbn13": str(row["isbn13"]),
                    "title": str(row["title"]),
                    "authors": str(row.get("authors", "")),
                    "description": str(row["description"]),
                    "category": str(row["simple_categories"]),
                }
            rankings[system] = ids
        pool_list = list(pool.values())
        rng.shuffle(pool_list)
        run["queries"].append({**q, "rankings": rankings, "pool": pool_list})
        print(q["id"], f"{len(pool)} unique books to judge", flush=True)
    run["run_id"] = hashlib.sha256(
        json.dumps(run, sort_keys=True).encode()
    ).hexdigest()[:16]
    write_json(out / "run.json", run)
    build_review(run, "dev", out / "review_dev.html")
    # Test review is created only by the explicit unlock-test command.
    report_run(run, {}, "dev", out)
    print(
        f"Open {out / 'review_dev.html'} in your browser. No relevance scores invented."
    )


def report_run(run, judgments, split, out):
    queries = [q for q in run["queries"] if q["split"] == split]
    rows = []
    complete = set()
    for q in queries:
        for system, ids in q["rankings"].items():
            metrics = query_metrics(
                ids,
                judgments,
                [i["item_id"] for i in q["pool"]],
                run["k"],
                q["tone"] != "All",
            )
            if metrics is not None:
                complete.add(q["id"])
                rows.append(
                    {
                        "query_id": q["id"],
                        "category": q["category"],
                        "system": system,
                        **metrics,
                    }
                )
    structural = {}
    for system in run["systems"]:
        returned, compliant, duplicates, empty = 0, 0, 0, 0
        for q in queries:
            ids = q["rankings"][system]
            items = {i["item_id"]: i for i in q["pool"]}
            returned += len(ids)
            compliant += sum(
                q["category"] == "All" or items[i]["category"] == q["category"]
                for i in ids
            )
            duplicates += len(ids) - len(set(ids))
            empty += not ids
        structural[system] = {
            "returned_books": returned,
            "category_compliance": compliant / returned if returned else None,
            "duplicate_count": duplicates,
            "empty_query_count": empty,
        }

    def aggregate(subset):
        result = {}
        for system in run["systems"]:
            r = [row for row in subset if row["system"] == system]
            values = {}
            for field in (
                "precision_at_5",
                "pooled_ndcg_at_5",
                "mood_fit_at_5",
                "joint_precision_at_5",
            ):
                valid = [row[field] for row in r if row.get(field) is not None]
                values[field] = {
                    "mean": sum(valid) / len(valid) if valid else None,
                    "queries": len(valid),
                }
            result[system] = values
        return result

    result = {
        "run_id": run["run_id"],
        "split": split,
        "queries": len(queries),
        "fully_judged_queries": len(complete),
        "structural": structural,
        "quality": aggregate(rows),
        "by_category": {
            cat: aggregate([r for r in rows if r["category"] == cat])
            for cat in sorted({q["category"] for q in queries})
        },
        "per_query": rows,
    }
    write_json(Path(out) / f"report_{split}.json", result)
    lines = [
        f"# Relevance evaluation — {split}",
        "",
        f"Run: `{run['run_id']}`",
        "",
        f"Fully judged query pools: **{len(complete)}/{len(queries)}**.",
        "",
        "All systems use the same completed queries. Partial pools are excluded.",
        "",
        "## Structural checks (not relevance measurements)",
        "",
        "| System | Returned | Category compliance | Duplicates | Empty queries |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, s in structural.items():
        compliance = (
            f"{s['category_compliance']:.1%}"
            if s["category_compliance"] is not None
            else "N/A"
        )
        lines.append(
            f"| {name} | {s['returned_books']} | {compliance} | {s['duplicate_count']} | {s['empty_query_count']} |"
        )
    lines += ["", "## Human relevance", ""]
    if not rows:
        lines += [
            "**Awaiting human judgments. There is no measured relevance result or winner yet.**"
        ]
    else:
        lines += [
            "| System | P@5 | Pooled nDCG@5 | Mood fit@5 | Joint P@5 |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, fields in result["quality"].items():
            values = [
                "N/A" if v["mean"] is None else f"{v['mean']:.3f} (n={v['queries']})"
                for v in fields.values()
            ]
            lines.append(f"| {name} | " + " | ".join(values) + " |")
    lines += [
        "",
        "P@5 counts relevance grade 2; denominator is always 5. nDCG uses graded",
        "gains 2^grade−1 and an ideal ranking from the judged pool, not the entire",
        "catalogue. Queries with zero ideal gain have undefined nDCG and are excluded",
        "from that mean. Mood fit averages mood/2 over 5 slots; joint P@5 requires",
        "relevance 2 and mood at least 1. Mood metrics apply only to tone queries.",
        "",
        "Category breakdown and per-query results are in the companion JSON.",
        "",
        "This small, single-reviewer benchmark is descriptive. Do not claim statistical",
        "significance or tune on held-out results. Inspect individual query failures.",
    ]
    Path(out, f"report_{split}.md").write_text("\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--queries", default=str(Path(__file__).with_name("queries.json")))
    p.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("--run", required=True)
    p.add_argument("--judgments")
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p = sub.add_parser("unlock-test")
    p.add_argument("--run", required=True)
    p.add_argument("--chosen-system", choices=list(WEIGHTS), required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
        return
    run = json.loads(Path(args.run).read_text())
    out = Path(args.run).parent
    if args.command == "unlock-test":
        path = out / "frozen_selection.json"
        if path.exists():
            raise ValueError(
                "Selection already frozen. Do not retune after seeing held-out judgments"
            )
        write_json(
            path,
            {
                "chosen_system": args.chosen_system,
                "frozen_utc": datetime.now(UTC).isoformat(),
                "run_id": run["run_id"],
            },
        )
        build_review(run, "test", out / "review_test.html")
        print(
            "Selection frozen. Open review_test.html; merge its export with dev only if needed."
        )
    else:
        if args.split == "test" and not (out / "frozen_selection.json").exists():
            raise ValueError(
                "Freeze a chosen system with unlock-test before evaluating held-out queries"
            )
        result = report_run(run, load_judgments(args.judgments, run), args.split, out)
        print(
            f"{result['fully_judged_queries']}/{result['queries']} fully judged queries. Report saved."
        )


if __name__ == "__main__":
    main()
