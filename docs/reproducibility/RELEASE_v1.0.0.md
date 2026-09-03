# Release Metadata & Version Freeze Report (v1.0.0)

## Overview

This report documents the official software version freeze and scientific release `v1.0.0` for the **Smart EV Charging Station Recommendation System**.

---

## Release Identification

- **Release Version**: `v1.0.0`
- **Release Date**: 2026-09-03
- **Git Tag**: `v1.0.0`
- **License**: MIT License ([`LICENSE`](file:///d:/Smart_EV_Station_Recommandadtion_System/LICENSE))
- **Citation Metadata**: [`CITATION.cff`](file:///d:/Smart_EV_Station_Recommandadtion_System/CITATION.cff)

---

## Software & Evidence Inventory

| Category | File Path / Directory | Description |
| :--- | :--- | :--- |
| **Proposed Model** | [`runs/models/hgat_cmappo/stage_06/model.pt`](file:///d:/Smart_EV_Station_Recommandadtion_System/runs/models/hgat_cmappo/stage_06/model.pt) | Stage 6 final validated HGAT-CMAPPO PyTorch model checkpoint. |
| **Audit Package** | [`runs/evaluation/hgat_cmappo/final_audit/`](file:///d:/Smart_EV_Station_Recommandadtion_System/runs/evaluation/hgat_cmappo/final_audit) | Frozen claim register, paper-safe tables A-D, and final audit markdown report. |
| **Phase 6 Evidence** | [`runs/evaluation/hgat_cmappo/phase6_final/`](file:///d:/Smart_EV_Station_Recommandadtion_System/runs/evaluation/hgat_cmappo/phase6_final) | 450 episodes accounting & reconciliation logs. |
| **Phase 7 Evidence** | [`runs/evaluation/hgat_cmappo/phase7_crowding_scalability/`](file:///d:/Smart_EV_Station_Recommandadtion_System/runs/evaluation/hgat_cmappo/phase7_crowding_scalability) | 500 episodes crowding & demand scaling logs. |
| **Phase 8 Evidence** | [`runs/evaluation/hgat_cmappo/phase8_ablation/`](file:///d:/Smart_EV_Station_Recommandadtion_System/runs/evaluation/hgat_cmappo/phase8_ablation) | 500 episodes component ablation logs. |
| **Paper Outputs** | [`outputs/performance_matrix.csv`](file:///d:/Smart_EV_Station_Recommandadtion_System/outputs/performance_matrix.csv) | Publication-ready CSV, LaTeX, and JSON performance metrics. |
| **Dashboard** | [`outputs/dashboard.html`](file:///d:/Smart_EV_Station_Recommandadtion_System/outputs/dashboard.html) | EV Digital Twin real-time research control dashboard. |
| **Archive** | [`archive/invalid_evaluation_pipeline/`](file:///d:/Smart_EV_Station_Recommandadtion_System/archive/invalid_evaluation_pipeline) | 29 legacy/invalid benchmark scripts preserved for research provenance. |

---

## Validation & Verification Summary

1. **PyTest Suite**: Verified clean execution across core modules.
2. **Dashboard Verification**: Verified live TraCI congestion tracking, 100% monotonic cumulative energy delivered, zero step duplicates, and strict $100\%$ equality between KPIs and latest trajectory points (`KPI === last history point`).
3. **Security Audit**: Verified zero hardcoded API keys or secret credentials.

---

## Reproducibility Instructions

To verify the frozen evidence package without re-running 1,450 simulation episodes:

```bash
# Activate environment
.venv\Scripts\activate

# Run pytest verification suite
pytest

# Inspect publication-ready performance matrix
cat outputs/performance_matrix.csv
```
