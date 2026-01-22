#!/usr/bin/env python
"""
Score a parallel corpus file via bicleaner-service API.

Usage:
    python scripts/score_file.py input.tsv output.tsv
    python scripts/score_file.py input.tsv output.tsv --url http://localhost:8057

Input format (TSV):
    source_text<TAB>target_text

Output format (TSV):
    source_text<TAB>target_text<TAB>score
"""

import argparse
import sys
from pathlib import Path

try:
    import httpx
except ImportError:
    print("ERROR: httpx not installed. Run: pip install httpx")
    sys.exit(1)


def score_file(
    input_path: str,
    output_path: str,
    api_url: str = "http://localhost:8057",
    batch_size: int = 100,
) -> dict:
    """
    Score a parallel corpus file via API.

    Args:
        input_path: Path to input TSV (source<TAB>target)
        output_path: Path to output TSV (source<TAB>target<TAB>score)
        api_url: Bicleaner service URL
        batch_size: Batch size for API requests

    Returns:
        Statistics dict with counts and scores
    """
    # Read input file
    pairs = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                pairs.append((line_num, parts[0], parts[1]))
            else:
                print(f"WARNING: Line {line_num} has less than 2 columns, skipping")

    if not pairs:
        print("ERROR: No valid pairs found in input file")
        return {"error": "no_pairs"}

    print(f"Read {len(pairs)} pairs from {input_path}")

    # Score in batches
    all_results = {}
    client = httpx.Client(base_url=api_url, timeout=120.0)

    try:
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            requests = [
                {"id": str(line_num), "source": src, "target": tgt}
                for line_num, src, tgt in batch
            ]

            print(f"Scoring batch {i // batch_size + 1}/{(len(pairs) - 1) // batch_size + 1}...")

            response = client.post("/v1/score", json={"requests": requests})
            response.raise_for_status()
            data = response.json()

            for result in data["results"]:
                all_results[int(result["id"])] = result

    except httpx.HTTPError as e:
        print(f"ERROR: API request failed: {e}")
        return {"error": str(e)}
    finally:
        client.close()

    # Write output file
    stats = {"total": 0, "success": 0, "failed": 0, "scores": []}

    with open(output_path, "w", encoding="utf-8") as f:
        for line_num, src, tgt in pairs:
            result = all_results.get(line_num)
            if result and result["success"]:
                score = result["score"]
                stats["success"] += 1
                stats["scores"].append(score)
            else:
                score = "ERROR"
                stats["failed"] += 1

            f.write(f"{src}\t{tgt}\t{score}\n")
            stats["total"] += 1

    # Calculate statistics
    if stats["scores"]:
        stats["avg_score"] = sum(stats["scores"]) / len(stats["scores"])
        stats["min_score"] = min(stats["scores"])
        stats["max_score"] = max(stats["scores"])
        stats["high_quality"] = sum(1 for s in stats["scores"] if s >= 0.5)
        stats["low_quality"] = sum(1 for s in stats["scores"] if s < 0.5)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Score parallel corpus via bicleaner-service API"
    )
    parser.add_argument("input", help="Input TSV file (source<TAB>target)")
    parser.add_argument("output", help="Output TSV file (source<TAB>target<TAB>score)")
    parser.add_argument(
        "--url",
        default="http://localhost:8057",
        help="Bicleaner service URL (default: http://localhost:8057)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for API requests (default: 100)",
    )
    args = parser.parse_args()

    # Validate input
    if not Path(args.input).exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    # Check service health
    try:
        response = httpx.get(f"{args.url}/health", timeout=10.0)
        if response.status_code != 200:
            print(f"ERROR: Service not healthy at {args.url}")
            sys.exit(1)
        health = response.json()
        print(f"Service: {health['model']} ({health['model_path']})")
    except httpx.HTTPError as e:
        print(f"ERROR: Cannot connect to service at {args.url}: {e}")
        sys.exit(1)

    # Score file
    print(f"\nScoring {args.input} -> {args.output}")
    print("-" * 50)

    stats = score_file(args.input, args.output, args.url, args.batch_size)

    if "error" in stats:
        print(f"\nFailed: {stats['error']}")
        sys.exit(1)

    # Print statistics
    print("-" * 50)
    print(f"Results written to: {args.output}")
    print(f"\nStatistics:")
    print(f"  Total:        {stats['total']}")
    print(f"  Success:      {stats['success']}")
    print(f"  Failed:       {stats['failed']}")
    if stats.get("avg_score"):
        print(f"  Avg score:    {stats['avg_score']:.3f}")
        print(f"  Min score:    {stats['min_score']:.3f}")
        print(f"  Max score:    {stats['max_score']:.3f}")
        print(f"  High quality: {stats['high_quality']} (score >= 0.5)")
        print(f"  Low quality:  {stats['low_quality']} (score < 0.5)")


if __name__ == "__main__":
    main()
