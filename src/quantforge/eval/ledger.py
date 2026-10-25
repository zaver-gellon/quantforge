"""Persistent record of every variant that was evaluated.

Re-running the same hypothesis (study, family, name, parameters, sample start)
updates the metrics and does not increment the trial count. A new parameter
combination does. Deflated Sharpe should use ``n_trials`` from this ledger,
including trials recorded by earlier studies.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, default=_json_default, separators=(",", ":"))


def _json_default(value):
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    raise TypeError(f"not JSON serializable: {type(value)}")


class TrialLedger:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.trials: dict[str, dict] = {}
        if self.path.exists():
            payload = json.loads(self.path.read_text())
            for trial in payload.get("trials", []):
                self.trials[trial["trial_id"]] = trial

    @staticmethod
    def trial_id(study: str, family: str, name: str, params: dict, sample_start: str) -> str:
        blob = _canonical(
            {
                "study": study,
                "family": family,
                "name": name,
                "params": params,
                "sample_start": sample_start,
            }
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def record(
        self,
        *,
        study: str,
        family: str,
        name: str,
        params: dict,
        sample_start: str,
        sample_end: str,
        metrics: dict,
        counts_for_deflation: bool = True,
    ) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        trial_id = self.trial_id(study, family, name, params, sample_start)
        clean_metrics = json.loads(json.dumps(metrics, default=_json_default))
        if trial_id in self.trials:
            trial = self.trials[trial_id]
            trial["metrics"] = clean_metrics
            trial["sample_end"] = sample_end
            trial["updated_at"] = now
            trial["counts_for_deflation"] = counts_for_deflation
        else:
            self.trials[trial_id] = {
                "trial_id": trial_id,
                "study": study,
                "family": family,
                "name": name,
                "params": json.loads(json.dumps(params, default=_json_default)),
                "sample_start": sample_start,
                "sample_end": sample_end,
                "metrics": clean_metrics,
                "counts_for_deflation": counts_for_deflation,
                "recorded_at": now,
                "updated_at": now,
            }
        return trial_id

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "trials": sorted(self.trials.values(), key=lambda trial: trial["trial_id"]),
        }
        text = json.dumps(payload, indent=2, default=_json_default) + "\n"
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(text)
        tmp.replace(self.path)

    def counting(self) -> list[dict]:
        return [trial for trial in self.trials.values() if trial.get("counts_for_deflation", True)]

    def n_trials(self) -> int:
        return len(self.counting())

    def sharpe_dispersion(self) -> float:
        values = []
        for trial in self.counting():
            sharpe = trial.get("metrics", {}).get("sharpe_per_period")
            if sharpe is None:
                continue
            sharpe = float(sharpe)
            if np.isfinite(sharpe):
                values.append(sharpe)
        if len(values) < 2:
            return 0.0
        return float(np.std(np.asarray(values, dtype=float), ddof=1))
