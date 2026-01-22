#!/usr/bin/env python
"""
API tests for bicleaner-service.

Run with: pytest tests/test_api.py -v
Requires: bicleaner-service running on localhost:8057
"""

import os
import pytest
import httpx
import tempfile
from pathlib import Path

BASE_URL = os.getenv("BICLEANER_API_URL", "http://localhost:8057")


@pytest.fixture
def client():
    """HTTP client with timeout."""
    return httpx.Client(base_url=BASE_URL, timeout=60.0)


class TestHealthEndpoint:
    """Test /health endpoint."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_model_info(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert "model" in data
        assert "model_path" in data


class TestScoreEndpoint:
    """Test POST /v1/score endpoint."""

    def test_score_single_pair(self, client):
        """Score a single translation pair."""
        response = client.post(
            "/v1/score",
            json={
                "requests": [
                    {"id": "1", "source": "Hello world", "target": "Hola mundo"}
                ]
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["id"] == "1"
        assert data["results"][0]["success"] is True
        assert 0.0 <= data["results"][0]["score"] <= 1.0

    def test_score_batch(self, client):
        """Score multiple translation pairs."""
        response = client.post(
            "/v1/score",
            json={
                "requests": [
                    {"id": "1", "source": "Good morning", "target": "Buenos dias"},
                    {"id": "2", "source": "Thank you", "target": "Gracias"},
                    {"id": "3", "source": "How are you?", "target": "Como estas?"},
                ]
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 3
        assert data["metadata"]["count"] == 3
        for result in data["results"]:
            assert result["success"] is True
            assert 0.0 <= result["score"] <= 1.0

    def test_score_noise_detection(self, client):
        """Verify noise gets low score."""
        response = client.post(
            "/v1/score",
            json={
                "requests": [
                    {
                        "id": "good",
                        "source": "The weather is nice today",
                        "target": "El tiempo esta bien hoy",
                    },
                    {
                        "id": "noise",
                        "source": "The weather is nice today",
                        "target": "Random unrelated garbage text xyz123",
                    },
                ]
            },
        )
        assert response.status_code == 200
        data = response.json()

        results = {r["id"]: r for r in data["results"]}
        # Good translation should score higher than noise
        assert results["good"]["score"] > results["noise"]["score"]
        # Noise should have very low score
        assert results["noise"]["score"] < 0.3

    def test_score_empty_request(self, client):
        """Empty request returns 422 (validation error)."""
        response = client.post("/v1/score", json={"requests": []})
        # Empty list is rejected by validation
        assert response.status_code == 422

    def test_score_invalid_request(self, client):
        """Invalid request returns 422."""
        response = client.post("/v1/score", json={"invalid": "data"})
        assert response.status_code == 422


class TestFileScoring:
    """Test file-based scoring workflow."""

    def create_tsv_file(self, pairs: list[tuple[str, str]]) -> str:
        """Create a temp TSV file with source/target pairs."""
        fd, path = tempfile.mkstemp(suffix=".tsv")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for source, target in pairs:
                f.write(f"{source}\t{target}\n")
        return path

    def test_score_file_workflow(self, client):
        """
        End-to-end test: read file -> score via API -> write results.

        This simulates the workflow:
        1. Read TSV file with source/target pairs (no scores)
        2. Send to API for scoring
        3. Write results with scores
        """
        # Sample parallel corpus (no scores)
        pairs = [
            ("Hello world", "Hola mundo"),
            ("Good morning", "Buenos dias"),
            ("Thank you very much", "Muchas gracias"),
            ("The cat is on the table", "El gato esta sobre la mesa"),
            ("I love programming", "Me encanta programar"),
            ("Random text here", "Texto aleatorio sin relacion"),  # noise
        ]

        # Create input file
        input_file = self.create_tsv_file(pairs)

        try:
            # Read file and prepare API request
            requests = []
            with open(input_file, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        requests.append({
                            "id": str(i),
                            "source": parts[0],
                            "target": parts[1],
                        })

            # Score via API
            response = client.post("/v1/score", json={"requests": requests})
            assert response.status_code == 200
            data = response.json()

            # Verify results
            assert len(data["results"]) == len(pairs)

            # Create output with scores
            output_file = input_file.replace(".tsv", "_scored.tsv")
            results_map = {r["id"]: r for r in data["results"]}

            with open(output_file, "w", encoding="utf-8") as f:
                for i, (source, target) in enumerate(pairs, 1):
                    result = results_map[str(i)]
                    score = result["score"] if result["success"] else "ERROR"
                    f.write(f"{source}\t{target}\t{score}\n")

            # Verify output file
            with open(output_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                assert len(lines) == len(pairs)
                # Each line should have 3 columns (source, target, score)
                for line in lines:
                    parts = line.strip().split("\t")
                    assert len(parts) == 3
                    # Score should be a float or ERROR
                    try:
                        score = float(parts[2])
                        assert 0.0 <= score <= 1.0
                    except ValueError:
                        assert parts[2] == "ERROR"

            print(f"\nInput:  {input_file}")
            print(f"Output: {output_file}")
            print("\nScored results:")
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    print(f"  {line.strip()}")

        finally:
            # Cleanup
            Path(input_file).unlink(missing_ok=True)
            Path(input_file.replace(".tsv", "_scored.tsv")).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
