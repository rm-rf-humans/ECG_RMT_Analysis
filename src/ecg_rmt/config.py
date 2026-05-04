from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    data_root: Path
    output_dir: Path

    @property
    def metrics_dir(self) -> Path:
        return self.output_dir / "metrics"

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def rmt_dir(self) -> Path:
        return self.output_dir / "rmt"

    @property
    def models_dir(self) -> Path:
        return self.output_dir / "models"

    def make_dirs(self) -> None:
        for path in [
            self.metrics_dir,
            self.figures_dir,
            self.rmt_dir,
            self.models_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


NUMERIC_METADATA = ["age", "height", "weight"]
CATEGORICAL_METADATA = ["sex", "nurse", "site", "device"]
TARGET_NORMAL = "normal"
TARGET_ABNORMAL = "diagnostic_abnormality"
