import argparse
from typing import Literal

import requests
from loguru import logger
from pydantic import BaseModel


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 0.9
    seed: int | None = None


class GenerateResponse(BaseModel):
    text: str
    token_ids: list[int]
    acceptance_rate: float
    num_accepted: int
    num_rejected: int
    total_tokens: int


class HealthResponse(BaseModel):
    status: Literal["healthy"]
    model_name: str
    graph_path: str
    draft_config: dict


class SpeculativeDecoderClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> HealthResponse:
        response = requests.get(f"{self.base_url}/health", timeout=5)
        response.raise_for_status()
        return HealthResponse(**response.json())

    def generate(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.9,
        seed: int | None = None,
    ) -> GenerateResponse:
        request = GenerateRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
        )
        response = requests.post(
            f"{self.base_url}/generate",
            json=request.model_dump(),
            timeout=120,
        )
        response.raise_for_status()
        return GenerateResponse(**response.json())


def main():
    parser = argparse.ArgumentParser(description="Speculative Decoder Client")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Server URL")
    parser.add_argument("--prompt", type=str, help="Prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens")
    parser.add_argument("--temperature", type=float, default=0.9, help="Temperature")
    parser.add_argument("--seed", type=int, help="Random seed")
    args = parser.parse_args()

    client = SpeculativeDecoderClient(base_url=args.url)

    logger.info("Checking server health...")
    try:
        health = client.health()
        logger.info("✓ Server healthy")
        logger.info(f"  Model: {health.model_name}")
        logger.info(f"  Draft config: {health.draft_config}")
    except requests.exceptions.RequestException as e:
        logger.error(f"✗ Health check failed: {e}")
        return

    prompts = (
        [args.prompt]
        if args.prompt
        else [
            "What is a force majeure clause?",
            "Explain indemnification clauses.",
            "What are liquidated damages?",
        ]
    )

    for i, prompt in enumerate(prompts, 1):
        logger.info(f"\n[{i}/{len(prompts)}] Prompt: {prompt}")

        try:
            result = client.generate(
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                seed=args.seed,
            )

            logger.info(f"Generated: {result.text}")
            logger.info(f"Acceptance: {result.acceptance_rate:.2%}")
            logger.info(
                f"Stats: {result.num_accepted} accepted, {result.num_rejected} rejected"
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"✗ Failed: {e}")


if __name__ == "__main__":
    main()
