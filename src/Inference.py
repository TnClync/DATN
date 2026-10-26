
from pathlib import Path
import json
import re

import pandas as pd
import pyarrow.dataset as ds
import requests


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
ITEM_PROFILE_PATH = str(PROJECT_DIR / "data" / "Item_profile")

JEV_URL = "http://127.0.0.1:8017/v1/systemone"
HEALTH_URL = "http://127.0.0.1:8017/health"

OUTPUT_PATH = PROJECT_DIR / "outputs" / "jev_electronics_recommendations.csv"

# Số sản phẩm lấy để JevOS chấm điểm
N_CANDIDATES = 30

# Số sản phẩm muốn giữ lại sau khi sắp xếp điểm
TOP_N = 10

BATCH_SIZE = 4096
REQUEST_TIMEOUT = 120

# Sở thích ví dụ — hãy thay bằng nhu cầu thực tế
USER_FAV = {
    "categories": [],
    "topics": [],
    "brands": [],
    "max_price": 50.0,
}

USER_PROMPT = ""

CRITERIA = [
    "Not compatible",
    "Slightly compatible",
    "Moderately compatible",
    "Highly compatible",
    "Very highly compatible",
]

PRODUCT_COLUMNS = [
    "item_idx",
    "asin",
    "title",
    "brand",
    "category",
    "feature",
    "description",
    "price",
]


# ============================================================
# HELPERS
# ============================================================

def clean(value):
    """Convert common Parquet values into readable text."""
    if value is None:
        return ""

    if isinstance(value, (list, tuple, set)):
        return ", ".join(clean(v) for v in value if v is not None)

    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def category_text(value):
    """Normalize category values for category matching."""
    text = clean(value).lower()
    text = text.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


def is_electronics(value):
    """
    Match Electronics as a category token.

    Handles values such as:
      Electronics
      ['Electronics', 'Computers']
      Electronics, Computers
      ['Electronics', 'Computers & Accessories']

    It avoids matching unrelated words that merely contain
    the substring 'electronics'.
    """
    text = category_text(value)

    if not text:
        return False

    # Split category paths and common list representations.
    tokens = re.split(r"[,;/|>\[\]{}\"']+", text)
    tokens = [re.sub(r"\s+", " ", t).strip() for t in tokens]

    if any(token == "electronics" for token in tokens):
        return True

    # Support JSON-like category strings where Electronics
    # is a value alongside other category names.
    return bool(
        re.search(r"(?<![a-z0-9])electronics(?![a-z0-9])", text)
    )


# ============================================================
# JEVOS API
# ============================================================

def check_server():
    """Check whether the local JevOS server is responding."""
    response = requests.get(HEALTH_URL, timeout=5)
    response.raise_for_status()
    print("JevOS health:", response.text)


def build_product(row):
    """Keep product metadata meaningful for JevOS."""
    return {
        key: clean(row.get(key))
        for key in [
            "title",
            "brand",
            "category",
            "feature",
            "description",
            "price",
        ]
    }


def score_product(row, session):
    """Request one JevOS score and preserve the raw answer."""
    product = build_product(row)

    state = {
        "user_preferences": USER_FAV,
        "current_request": USER_PROMPT,
        "candidate_product": product,
    }

    payload = {
        "model": "jev-latest",
        "state": state,
        "questions": {
            "compatibility": {
                "type": "score",
                "instructions": (
                    "Evaluate how well the product matches the user's "
                    "stated preferences and current request. Use only "
                    "the supplied information. Do not assume missing "
                    "information is a match. Evaluate relevance and "
                    "compatibility, not overall product quality. "
                    "Use the provided criteria."
                ),
                "criteria": CRITERIA,
            }
        },
    }

    response = session.post(
        JEV_URL,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    body = response.json()
    answer = body.get("answers", {}).get("compatibility")

    if not isinstance(answer, dict):
        raise RuntimeError(
            "Unexpected JevOS response: "
            + json.dumps(body, ensure_ascii=False)
        )

    # Preserve the real response; do not assume the score is a percentage.
    result = row.to_dict()
    result["jev_answer"] = json.dumps(
        answer, ensure_ascii=False
    )
    result["jev_model"] = body.get("model")

    for key in ("score", "confidence"):
        if key in answer:
            result[f"jev_{key}"] = answer[key]

    if "score" not in answer:
        raise RuntimeError(
            "JevOS response does not contain 'score': "
            + json.dumps(body, ensure_ascii=False)
        )

    return result


# ============================================================
# LOAD ELECTRONICS CANDIDATES
# ============================================================

def load_electronics_candidates():
    """
    Read the Item_profile Parquet dataset incrementally.
    Stop after finding N_CANDIDATES Electronics products.
    """
    path = Path(ITEM_PROFILE_PATH)

    if not path.exists():
        raise FileNotFoundError(
            f"Item_profile path not found: {path}"
        )

    dataset = ds.dataset(str(path), format="parquet")
    available = set(dataset.schema.names)

    print("\nItem_profile schema:")
    print(dataset.schema)

    if "title" not in available:
        raise ValueError(
            f"Missing title column. Available: {sorted(available)}"
        )

    if "category" not in available:
        raise ValueError(
            "Item_profile has no category column."
        )

    columns = [
        col for col in PRODUCT_COLUMNS
        if col in available
    ]

    selected_frames = []
    selected_count = 0
    scanned_count = 0

    scanner = dataset.scanner(
        columns=columns,
        batch_size=BATCH_SIZE,
    )

    print("\nSearching for Electronics products...")

    for batch in scanner.to_batches():
        df = batch.to_pandas()
        scanned_count += len(df)

        # Only keep products with a title and matching category.
        mask = (
            df["title"].notna()
            & df["category"].apply(is_electronics)
        )

        matched = df.loc[mask].copy()

        if matched.empty:
            continue

        # Exclude duplicate item_idx values if available.
        if "item_idx" in matched.columns:
            matched = matched.drop_duplicates(
                subset=["item_idx"]
            )

        remaining = N_CANDIDATES - selected_count

        if remaining <= 0:
            break

        matched = matched.head(remaining)

        selected_frames.append(matched)
        selected_count += len(matched)

        print(
            f"Scanned {scanned_count:,} rows; "
            f"found {selected_count}/{N_CANDIDATES} candidates."
        )

        if selected_count >= N_CANDIDATES:
            break

    if not selected_frames:
        raise ValueError(
            "No Electronics products found. "
            "Inspect the real category values in Item_profile "
            "and adjust is_electronics() if necessary."
        )

    candidates = pd.concat(
        selected_frames,
        ignore_index=True,
    )

    print(f"\nLoaded {len(candidates)} Electronics candidates.")
    return candidates


# ============================================================
# MAIN
# ============================================================

def main():
    if N_CANDIDATES < 1 or TOP_N < 1:
        raise ValueError("N_CANDIDATES and TOP_N must be positive.")

    if TOP_N > N_CANDIDATES:
        print(
            "Warning: TOP_N exceeds N_CANDIDATES; "
            "the output can contain fewer than TOP_N products."
        )

    check_server()

    candidates = load_electronics_candidates()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []
    failures = 0

    with requests.Session() as session:
        total = len(candidates)

        for i, (_, row) in enumerate(
            candidates.iterrows(),
            start=1,
        ):
            title = clean(row.get("title"))

            try:
                result = score_product(row, session)
                results.append(result)

                print(
                    f"[{i}/{total}] OK | "
                    f"{title[:65]} | "
                    f"score={result.get('jev_score')}"
                )

            except Exception as exc:
                failures += 1
                print(
                    f"[{i}/{total}] ERROR | "
                    f"{title[:65]} | {exc}"
                )

    if not results:
        raise RuntimeError(
            "No products were scored successfully. "
            "Check the JevOS response and errors above."
        )

    output = pd.DataFrame(results)

    # Sort only by the actual numeric API score.
    # No conversion to percentages or assumed score range.
    if "jev_score" in output.columns:
        output["_numeric_score"] = pd.to_numeric(
            output["jev_score"],
            errors="coerce",
        )

        output = output.sort_values(
            "_numeric_score",
            ascending=False,
            na_position="last",
        )

        output = output.drop(
            columns=["_numeric_score"]
        )

    output = output.head(TOP_N).reset_index(drop=True)

    # Save only the final Top-N.
    output.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 70)
    print(f"Candidates: {len(candidates)}")
    print(f"Successfully scored: {len(results)}")
    print(f"Failed: {failures}")
    print(f"Saved Top-{len(output)}: {OUTPUT_PATH}")

    display_columns = [
        col for col in [
            "item_idx",
            "asin",
            "title",
            "brand",
            "category",
            "price",
            "jev_score",
            "jev_confidence",
        ]
        if col in output.columns
    ]

    print("\nTop recommendations:")
    print(
        output[display_columns].to_string(index=False)
    )


if __name__ == "__main__":
    main()
