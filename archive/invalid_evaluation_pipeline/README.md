# Invalid / Legacy Evaluation Pipeline Archive

## Provenance & Scientific Notice

> [!IMPORTANT]
> **RESEARCH PROVENANCE NOTICE**: The scripts archived in this directory (`archive/invalid_evaluation_pipeline/`) represent early exploratory prototypes, legacy benchmark iterations, and invalid synthetic simulation pipelines used during preliminary software development.
>
> **THESE SCRIPTS ARE RETAINED EXCLUSIVELY FOR HISTORICAL RESEARCH PROVENANCE AND MUST NOT BE USED TO GENERATE REPORTED SCIENTIFIC BENCHMARKS OR FIGURES.**

---

## Canonical Active Pipeline

All scientific evidence, paper-safe benchmark figures, Phase 6 accounting, Phase 7 demand scaling, and Phase 8 component ablation results were produced strictly using the canonical pipeline:

1. **Execution Script**: [`run_project.py`](file:///d:/Smart_EV_Station_Recommandadtion_System/run_project.py)
2. **Evaluation Framework**: [`src/evaluation/`](file:///d:/Smart_EV_Station_Recommandadtion_System/src/evaluation)
3. **Frozen Audit Results**: [`runs/evaluation/hgat_cmappo/final_audit/`](file:///d:/Smart_EV_Station_Recommandadtion_System/runs/evaluation/hgat_cmappo/final_audit)
4. **Paper Outputs**: [`outputs/performance_matrix.csv`](file:///d:/Smart_EV_Station_Recommandadtion_System/outputs/performance_matrix.csv)

---

## Archived Scripts Inventory

- `behavioral_ab_evaluation.py`: Pre-audit synthetic behavioral comparison script (superseded by Phase 8 ablation).
- `behavioral_eval_ppo.py`: Legacy unconstrained PPO evaluation script.
- `compare_ppo_checkpoints.py`: Intermediate checkpoint evaluation script.
- `progressive_validation.py`: Exploratory verification script.
- `run_traci_recommend.py`, `run_traci_rl.py`: Early TraCI integration scripts before multi-agent CMAPPO harmonization.
