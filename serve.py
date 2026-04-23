#!/usr/bin/env python3
"""HTTP service wrapping the restyle pipeline.

Loads a StyleTransfer pipeline once at startup and exposes a POST endpoint
that restyles one paragraph per request. The heavy work (model load, RAG,
verifier, etc.) happens during startup, not per request.

Usage:

    # From config.json (honors use_adapter / lora_adapters / models):
    python serve.py --author "H.P. Lovecraft"

    # Explicit fused model:
    python serve.py \\
        --model /workspace/models/lovecraft-awq \\
        --author "H.P. Lovecraft"

    # Explicit LoRA adapter:
    python serve.py \\
        --adapter lora_adapters/lovecraft_14b \\
        --author "H.P. Lovecraft"

Then:

    curl -X POST http://localhost:8000/restyle \\
        -H "Content-Type: application/json" \\
        -d '{"text": "The universe is vast and old."}'

Install deps (once):

    pip install fastapi uvicorn[standard]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


class RestyleRequest(BaseModel):
    text: str = Field(..., description="Paragraph to restyle.")
    previous: Optional[str] = Field(
        None, description="Previous output paragraph for continuity (optional)."
    )


class RestyleResponse(BaseModel):
    styled: str
    score: float
    input_words: int
    output_words: int


def _build_pipeline(args: argparse.Namespace):
    """Load config, resolve targets, and construct a StyleTransfer instance.

    Mirrors the init portion of restyle.transfer_file(), minus the file I/O
    and streaming output bits.
    """
    from src.config import load_config, LLMProviderConfig
    from src.generation.transfer import StyleTransfer, TransferConfig
    from src.llm.deepseek import DeepSeekProvider

    from restyle import _resolve_transfer_targets

    adapters, fused_models, fused_model_config = _resolve_transfer_targets(args)
    if not adapters and not fused_models:
        raise SystemExit(
            "No adapter or model resolved. Pass --adapter / --model or configure "
            "generation.lora_adapters / generation.models in config.json."
        )

    try:
        app_config = load_config(args.config)
    except FileNotFoundError:
        print(f"Warning: {args.config} not found, using defaults", file=sys.stderr)
        app_config = None

    effective_perspective = args.perspective
    if effective_perspective is None and app_config:
        effective_perspective = app_config.style.perspective
    if effective_perspective is None:
        effective_perspective = "preserve"

    if app_config:
        gen = app_config.generation
        cfg = TransferConfig(
            temperature=args.temperature,
            verify_semantic_fidelity=args.verify,
            perspective=effective_perspective,
            max_expansion_ratio=gen.max_expansion_ratio,
            target_expansion_ratio=gen.target_expansion_ratio,
            expand_for_texture=gen.expand_for_texture,
            skip_neutralization=gen.skip_neutralization,
            pass_headings_unchanged=gen.pass_headings_unchanged,
            min_paragraph_words=gen.min_paragraph_words,
            use_structural_rag=gen.use_structural_rag,
            use_structural_grafting=gen.use_structural_grafting,
            rag_sample_size=gen.rag_sample_size,
            apply_input_perturbation=gen.apply_input_perturbation,
            use_persona=gen.use_persona,
        )
    else:
        cfg = TransferConfig(
            temperature=args.temperature,
            verify_semantic_fidelity=args.verify,
            perspective=effective_perspective,
            use_structural_rag=True,
        )

    critic_provider = None
    if app_config and app_config.llm.providers.get("deepseek"):
        critic_provider = DeepSeekProvider(
            config=app_config.llm.get_provider_config("deepseek")
        )
    else:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if api_key:
            critic_provider = DeepSeekProvider(
                config=LLMProviderConfig(
                    api_key=api_key,
                    model="deepseek-chat",
                    base_url="https://api.deepseek.com",
                )
            )

    author = args.author
    if not author and fused_model_config and fused_model_config.author:
        author = fused_model_config.author
    if not author:
        raise SystemExit(
            "--author is required (or set 'author' on the fused model entry in config.json)."
        )

    print(f"Loading pipeline for {author!r}...", file=sys.stderr)
    transfer = StyleTransfer(
        adapter_path=None,
        author_name=author,
        critic_provider=critic_provider,
        config=cfg,
        adapters=adapters or None,
        fused_models=fused_models or None,
    )
    print("Pipeline ready.", file=sys.stderr)
    return transfer, author, fused_models, adapters


WARMUP_TEXT = (
    "The universe is vast and old, and we are but brief flickers of awareness "
    "adrift upon its indifferent tides."
)


def _build_app(args: argparse.Namespace):
    import secrets
    import time
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, HTTPException

    transfer, author, fused_models, adapters = _build_pipeline(args)
    api_token = args.api_token or os.environ.get("API_TOKEN") or ""
    started_at = time.time()

    def _check_token(token: Optional[str]) -> None:
        """Constant-time token comparison. No-op when no token is configured."""
        if not api_token:
            return
        if not token or not secrets.compare_digest(token, api_token):
            raise HTTPException(status_code=401, detail="invalid or missing token")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Force the lazy generator to materialize its weights and run through
        # the full pipeline once, so the first real /restyle request doesn't
        # pay the 30–60s model-load + CUDA-graph-compile cost.
        if args.skip_warmup:
            print("Skipping warmup (--skip-warmup).", file=sys.stderr)
            app.state.ready = True
        else:
            print("Warming up pipeline (first load compiles CUDA graphs)...", file=sys.stderr)
            try:
                styled, _ = transfer.transfer_paragraph(WARMUP_TEXT)
                print(
                    f"Warmup complete. Sample output: {styled[:80]!r}",
                    file=sys.stderr,
                )
                app.state.ready = True
            except Exception as e:
                # Log and keep the server up so /ready can signal failure and
                # /info still responds — easier to debug than a crashed process.
                print(f"Warmup failed: {type(e).__name__}: {e}", file=sys.stderr)
                app.state.ready = False
                app.state.warmup_error = f"{type(e).__name__}: {e}"
        yield

    app = FastAPI(
        title="Text Style Transfer",
        description=f"Restyle paragraphs in the voice of {author}.",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health():
        # Liveness: the process is up. Does not imply the model is ready.
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        # Readiness: warmup has completed successfully. Use this for load
        # balancers / autoscalers so traffic only hits warm pods.
        if getattr(app.state, "ready", False):
            return {"status": "ready"}
        detail = getattr(app.state, "warmup_error", "warming up")
        raise HTTPException(status_code=503, detail=detail)

    @app.get("/info")
    def info():
        return {
            "author": author,
            "fused_models": fused_models,
            "adapters": [
                {"path": a.path, "scale": a.scale, "checkpoint": a.checkpoint}
                for a in (adapters or [])
            ],
            "verify_entailment": transfer.config.verify_semantic_fidelity,
            "expand_for_texture": transfer.config.expand_for_texture,
            "perspective": transfer.config.perspective,
            "ready": getattr(app.state, "ready", False),
        }

    @app.get("/api/status")
    def api_status(token: Optional[str] = None):
        """Token-gated status endpoint (matches GET /api/status?token=...)."""
        _check_token(token)
        warmup_error = getattr(app.state, "warmup_error", None)
        return {
            "status": "ready" if getattr(app.state, "ready", False) else "starting",
            "author": author,
            "fused_models": fused_models,
            "adapters": [
                {"path": a.path, "scale": a.scale, "checkpoint": a.checkpoint}
                for a in (adapters or [])
            ],
            "uptime_seconds": int(time.time() - started_at),
            "warmup_error": warmup_error,
        }

    @app.post("/restyle", response_model=RestyleResponse)
    def restyle(req: RestyleRequest):
        text = req.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="'text' must be non-empty")
        try:
            styled, score = transfer.transfer_paragraph(text, previous=req.previous)
        except Exception as e:
            # Surface the error to the caller but keep the service alive.
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
        return RestyleResponse(
            styled=styled,
            score=float(score),
            input_words=len(text.split()),
            output_words=len(styled.split()),
        )

    return app


def _make_parser() -> argparse.ArgumentParser:
    # Match restyle.py's arg shape closely so _resolve_transfer_targets works.
    p = argparse.ArgumentParser(description="HTTP server for the restyle pipeline.")
    p.add_argument(
        "--adapter", "--adapters", dest="adapters", action="append",
        help="LoRA adapter path (repeatable; supports 'path:scale').",
    )
    p.add_argument(
        "--model", dest="model", action="append",
        help="Fused model path (repeatable).",
    )
    p.add_argument("--author", default=None, help="Author name.")
    p.add_argument("--config", default="config.json", help="Path to config.json.")
    p.add_argument("--checkpoint", default=None, help="Adapter checkpoint subfolder.")
    p.add_argument("--lora-scale", dest="lora_scale", type=float, default=None)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument(
        "--perspective",
        choices=["preserve", "first_person_singular", "first_person_plural",
                 "third_person", "author_voice_third_person"],
        default=None,
    )
    p.add_argument("--no-verify", dest="verify", action="store_false", default=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--log-level", default="info")
    p.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip the startup warmup pass. First real request will then pay "
        "the model-load + CUDA-graph-compile cost (~30-60s).",
    )
    p.add_argument(
        "--api-token",
        default=None,
        help="Token required for /api/status?token=... (or set via API_TOKEN "
        "env var). When unset, /api/status is open.",
    )
    return p


def main() -> None:
    args = _make_parser().parse_args()

    import uvicorn

    app = _build_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
