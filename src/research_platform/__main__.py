from __future__ import annotations

import argparse
import json

from src.research_platform.service import ResearchPlatformService


def main() -> None:
    parser = argparse.ArgumentParser(description="Digital Twin research utilities")
    parser.add_argument("command", choices=["analytics", "benchmark"])
    args = parser.parse_args()
    service = ResearchPlatformService()
    records = service.list_recordings()
    if args.command == "analytics":
        print(json.dumps(service.analytics(records), indent=2))
    else:
        recommendations = [item for record in records for item in record.get("snapshot", {}).get("data", {}).get("recommendations", [])]
        print(json.dumps(service.benchmark(recommendations), indent=2))


if __name__ == "__main__":
    main()
