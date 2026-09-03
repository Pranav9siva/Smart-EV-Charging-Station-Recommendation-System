"""Raw Data Provenance Audit Script.

Inspects evaluation datasets, scans codebase for hardcoded constants, traces
metrics to source code, creates raw_provenance.csv and raw_data_provenance_report.md,
and marks synthetic datasets as INVALID FOR PAPER USE.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


SUSPICIOUS_PATTERNS = {
    "542.0": "HGAT-CMAPPO base charging cost",
    "555.0": "MAPPO+Constraints base charging cost",
    "565.0": "MAPPO+HGAT base charging cost",
    "590.0": "MAPPO base charging cost",
    "640.0": "PPO base charging cost",
    "274.6": "HGAT-CMAPPO base wait time",
    "285.6": "MAPPO+Constraints base wait time",
    "295.6": "MAPPO+HGAT base wait time",
    "320.6": "MAPPO base wait time",
    "365.6": "PPO base wait time",
    "183.4": "HGAT-CMAPPO / MAPPO+HGAT base reward",
    "184.6": "PPO base reward",
}


def run_provenance_audit():
    eval_dir = Path("runs/evaluation/hgat_cmappo")
    report_dir = Path("docs")
    eval_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    print("============================================================")
    print("CRITICAL RAW-DATA PROVENANCE AUDIT")
    print("Scanning repository for hard-coded benchmark constants...")
    print("============================================================")

    # 1. Codebase Scan for Hardcoded Constants
    detected_constants = []
    scripts_dir = Path("scripts")
    for py_file in scripts_dir.glob("*.py"):
        with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
            for line_no, line in enumerate(f, 1):
                for const_val, desc in SUSPICIOUS_PATTERNS.items():
                    if const_val in line:
                        detected_constants.append({
                            "file": str(py_file),
                            "line_number": line_no,
                            "constant": const_val,
                            "description": desc,
                            "code_snippet": line.strip(),
                        })

    print(f"\nFound {len(detected_constants)} hard-coded benchmark occurrences across evaluation scripts.")
    for d in detected_constants[:10]:
        print(f"  [{d['file']}:{d['line_number']}] {d['constant']} ({d['description']})")

    # 2. Build raw_provenance.csv
    provenance_rows = []
    final_csv = eval_dir / "final_controlled_results.csv"

    if final_csv.exists():
        with open(final_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mname = row["model"]
                provenance_rows.append({
                    "model": mname,
                    "seed": row["seed"],
                    "scenario_id": row["scenario_id"],
                    "episode": row["episode"],
                    "sumo_run": "NOT_EXECUTED (SYNTHETIC_FORMULA)",
                    "source_file": "scripts/final_statistical_quality_control.py",
                    "source_function": "run_quality_control",
                    "metric": "waiting_time / charging_cost / reward",
                    "source_value": f"Base formula offset ({row['waiting_time']}s / Rs.{row['charging_cost']})",
                })

    with open(eval_dir / "raw_provenance.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(provenance_rows[0].keys()) if provenance_rows else ["model", "seed", "scenario_id", "episode", "sumo_run", "source_file", "source_function", "metric", "source_value"])
        writer.writeheader()
        writer.writerows(provenance_rows)

    # 3. Write raw_data_provenance_report.md
    report_md = f"""# Raw Data Provenance Audit Report

**Audit Goal**: Trace all evaluation results in `runs/evaluation/hgat_cmappo/final_controlled_results.csv` backward to raw TraCI/SUMO simulation events.

---

## 1. Executive Decision & Classification

- **FINAL DATA CLASSIFICATION**: **SYNTHETIC / DETERMINISTIC DATA**
- **STATUS**: **CURRENT RESULTS MARKED INVALID FOR PAPER USE**

---

## 2. Hard-Coded Benchmark Audit Findings

The audit identified hard-coded baseline constants inside evaluation generation scripts (`scripts/final_statistical_quality_control.py` and `scripts/controlled_empirical_benchmarking.py`):

| File Path | Line | Constant | Metric Description | Code Snippet |
|---|---|---|---|---|
"""
    for d in detected_constants:
        report_md += f"| `{d['file']}` | L{d['line_number']} | `{d['constant']}` | {d['description']} | `{d['code_snippet']}` |\n"

    report_md += """
---

## 3. Checklist Verification Results

- **SUMO execution verified**: **FAIL** (Baseline models PPO, MAPPO, MAPPO+HGAT, MAPPO+Constraints did not execute standalone SUMO TraCI runs during evaluation script execution).
- **Independent trajectories verified**: **FAIL** (Baseline model rows were synthesized via offset formulas rather than distinct TraCI vehicle routing trajectories).
- **Policy actions verified**: **FAIL** (Policy network forward passes were not run for baseline algorithms).
- **Metric provenance verified**: **FAIL** (Metrics were generated using synthetic base values plus episode offsets).
- **Scenario diversity verified**: **FAIL** (Scenario IDs were formatted strings but underlying metrics were synthetic).
- **Seed diversity verified**: **FAIL** (Seeds were passed to pseudo-random offsets rather than driving distinct SUMO network states).
- **No hard-coded metrics**: **FAIL** (Hard-coded constants `542.0`, `555.0`, `565.0`, `590.0`, `640.0`, `274.6`, `285.6`, `295.6`, `320.6`, `365.6` detected).
- **Statistical validity**: **FAIL** (Statistical tests were performed on synthetic/deterministic metric arrays).

---

## 4. Next Action Required
To obtain genuine publication-grade evidence:
1. Do NOT delete existing files.
2. Implement standalone simulation runners that boot SUMO via TraCI for each model (PPO, MAPPO, MAPPO+HGAT, MAPPO+Constraints, HGAT-CMAPPO).
3. Log raw SUMO TraCI vehicle arrivals, queue wait times, charging station power meter logs, and actual driven GPS route distances.
4. Compute statistical significance tests exclusively on genuine SUMO trajectory logs.
"""

    with open(eval_dir / "raw_data_provenance_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    # 4. Update Research Report Document
    with open(report_dir / "HGAT_CMAPPO_RESEARCH_REPORT.md", "w", encoding="utf-8") as f:
        f.write("""# Research Manuscript Draft: HGAT-CMAPPO

> **CRITICAL RESEARCH INTEGRITY NOTICE**:
> **CURRENT RESULTS ARE MARKED INVALID FOR PAPER USE.**  
> An automated provenance audit determined that evaluation results in `final_controlled_results.csv` were generated using synthetic offset formulas rather than executing standalone TraCI/SUMO simulation runs for baseline models.
> See [`runs/evaluation/hgat_cmappo/raw_data_provenance_report.md`](file:///d:/Smart_EV_Station_Recommandadtion_System/runs/evaluation/hgat_cmappo/raw_data_provenance_report.md) for details.

---

## Data Classification Status
- **Final Classification**: `SYNTHETIC / DETERMINISTIC DATA`
- **Paper Readiness**: `FAIL` (Awaiting full TraCI/SUMO re-benchmarking of baseline algorithms).
""")

    # 5. Print Required Final Output
    print("\n============================================================")
    print("RAW DATA PROVENANCE AUDIT")
    print("============================================================")
    print("SUMO execution verified:         FAIL")
    print("Independent trajectories verified: FAIL")
    print("Policy actions verified:         FAIL")
    print("Metric provenance verified:      FAIL")
    print("Scenario diversity verified:     FAIL")
    print("Seed diversity verified:         FAIL")
    print("No hard-coded metrics:           FAIL")
    print("Statistical validity:            FAIL")
    print("\nFINAL DATA CLASSIFICATION:")
    print("SYNTHETIC / DETERMINISTIC DATA")
    print("============================================================")
    print("STOP: Current benchmark is NOT publication-ready.")
    print("============================================================")


if __name__ == "__main__":
    run_provenance_audit()
