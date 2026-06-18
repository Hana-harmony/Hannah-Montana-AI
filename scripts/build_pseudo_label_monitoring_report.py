import json
from pathlib import Path
from typing import Any

from hannah_montana_ai.training.pseudo_label_monitor import (
    build_pseudo_label_monitoring_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEAK_DISTILLATION_REPORT_PATH = PROJECT_ROOT / "reports/weak-distillation-report.json"
MODEL_RELEASE_REPORT_PATH = PROJECT_ROOT / "reports/model-release-report.json"
PSEUDO_LABEL_MONITORING_REPORT_PATH = (
    PROJECT_ROOT / "reports/pseudo-label-promotion-monitoring.json"
)


def main() -> None:
    report = build_pseudo_label_monitoring_report(
        _read_json(WEAK_DISTILLATION_REPORT_PATH),
        _read_json(MODEL_RELEASE_REPORT_PATH),
    )
    PSEUDO_LABEL_MONITORING_REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
