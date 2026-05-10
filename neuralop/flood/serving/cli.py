"""CLI helpers for flood serving deployment."""

from __future__ import annotations

import argparse
import json

from neuralop.flood.serving.model_bundle import load_model_bundle


def validate_bundle_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a coastal FGN serving model bundle.")
    parser.add_argument("bundle", help="Path to model bundle JSON/YAML manifest.")
    parser.add_argument("--skip-path-checks", action="store_true", help="Validate schema/scientific contract only.")
    args = parser.parse_args(argv)
    bundle = load_model_bundle(args.bundle, validate_paths=not args.skip_path_checks)
    print(json.dumps(bundle.public_metadata(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(validate_bundle_main())
