#!/usr/bin/env python3
import os
import sys
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from speculant_graph import (
    SpeculativeDecoder,
    DraftConfig,
    VerifierConfig,
    GenerationConfig,
)
from models import (
    HealthResponse,
    GenerateRequest,
    GenerateResponse,
)


def init_decoder(args):
    graph_path = Path(args.graph_path).resolve()
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph not found: {graph_path}")

    verifier_config = VerifierConfig(
        model_name=args.model_name, hf_token=os.getenv("HF_TOKEN")
    )
    draft_config = DraftConfig(
        k=args.k,
        strategy=args.strategy,
        attentive_mix=args.attentive_mix,
        order_bias=args.order_bias,
        mix_temperature=args.mix_temperature,
        reliability_weight=args.reliability_weight,
        entropy_penalty=args.entropy_penalty,
    )

    decoder = SpeculativeDecoder(
        graph_path=str(graph_path),
        verifier_config=verifier_config,
        draft_config=draft_config,
    )
    return decoder, str(graph_path), args.model_name, draft_config


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting server...")
    try:
        decoder, gpath, mname, dconfig = init_decoder(app.state.args)
        app.state.decoder = decoder
        app.state.graph_path = gpath
        app.state.model_name = mname
        app.state.draft_config = dconfig
        logger.info("Model loaded!")
        logger.info(f"Ready: http://{app.state.args.host}:{app.state.args.port}")
    except Exception as e:
        logger.exception(f"Failed to initialize: {e}")
        sys.exit(1)

    yield
    logger.info("Shutting down...")
    app.state.decoder = None


app = FastAPI(title="Speculative Graph Decoder", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health():
    if app.state.decoder is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Not loaded")

    return HealthResponse(
        status="healthy",
        model_name=app.state.model_name,
        graph_path=app.state.graph_path,
        draft_config=app.state.draft_config.model_dump(),
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if app.state.decoder is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Not loaded")

    try:
        config = GenerationConfig(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            seed=request.seed,
        )
        result = app.state.decoder.generate(request.prompt, config)

        return GenerateResponse(
            text=result.text,
            token_ids=result.token_ids,
            acceptance_rate=result.acceptance_rate,
            num_accepted=result.num_accepted,
            num_rejected=result.num_rejected,
            total_tokens=result.total_tokens,
        )
    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(e))


def parse_args():
    parser = argparse.ArgumentParser(description="Speculative Decoder Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    parser.add_argument("--graph-path", required=True, help="Graph .pkl file")
    parser.add_argument("--model-name", default="openai/gpt-oss-20b")
    parser.add_argument("--k", type=int, default=8, help="Draft length")
    parser.add_argument("--strategy", choices=["greedy", "sampling"], default="greedy")
    parser.add_argument("--attentive-mix", action="store_true", default=True)
    parser.add_argument(
        "--no-attentive-mix", action="store_false", dest="attentive_mix"
    )
    parser.add_argument("--order-bias", type=float, default=1.0)
    parser.add_argument("--mix-temperature", type=float, default=1.0)
    parser.add_argument("--reliability-weight", type=float, default=1.0)
    parser.add_argument("--entropy-penalty", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    app.state.args = args

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
