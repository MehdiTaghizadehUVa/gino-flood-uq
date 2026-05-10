"""Worker entrypoint for the FGN serving deployment.

For v1 Docker Compose this process validates and preloads the production model
bundle so deployment fails fast if checkpoints, normalizers, or domain assets are
wrong. Queue backend integration can then call the same ProductionFGNInferenceService
instance from a Celery task module without duplicating model loading.
"""

from __future__ import annotations

import os
import sys
import time

from neuralop.flood.serving.inference import ProductionFGNInferenceService
from neuralop.flood.serving.model_bundle import load_model_bundle


def main() -> int:
    bundle_path = os.environ.get("FGN_MODEL_BUNDLE_PATH")
    if not bundle_path:
        print("FGN_MODEL_BUNDLE_PATH is not configured; worker cannot start.", file=sys.stderr)
        return 2
    bundle = load_model_bundle(bundle_path)
    service = ProductionFGNInferenceService(
        bundle,
        device=os.environ.get("FGN_DEVICE", "cuda:0"),
        member_chunk_size=int(os.environ.get("FGN_MEMBER_CHUNK_SIZE", "4")),
    )
    if os.environ.get("FGN_PRELOAD_MODELS", "1").strip().lower() not in {"0", "false", "no"}:
        service._ensure_loaded()  # startup health check; keeps models cached in this worker process
    print(
        "Validated and preloaded model bundle "
        f"{bundle.bundle_id} with {bundle.total_members} members on {service.device_name}.",
        flush=True,
    )
    if os.environ.get("FGN_WORKER_STAY_ALIVE", "0").strip().lower() in {"1", "true", "yes"}:
        while True:
            time.sleep(60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
