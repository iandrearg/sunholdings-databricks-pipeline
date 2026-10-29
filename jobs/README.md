# Phase 2 — Databricks Jobs & Workflow

## 1. Overview

This document describes the Databricks Job and workflow for the SunHoldings / Applebee's Phase 2 data pipeline. The workflow organizes the execution of the notebooks responsible for data engineering, data quality validation, and business analysis.

The objective is to make the pipeline easier to execute, monitor, troubleshoot, and maintain.

## 2. Workflow Objectives

The workflow is designed to:

- Process the source datasets through the Bronze, Silver, and Gold layers.
- Apply data transformations and prepare analytical tables.
- Validate data quality before relying on downstream outputs.
- Support business analysis of transaction risk indicators, locations, and employee activity.
- Provide a repeatable execution process.

## 3. Pipeline Components

| Component | Responsibility |
|---|---|
| `06. Architecture, Job & Engineering` | Documents the architecture and job engineering approach. |
| `07_Data_Transformations_Tables_TEAM` | Implements the Bronze, Silver, and Gold data transformation pipeline. |
| `08. Data Quality & Expectations` | Supports data quality checks and validation. |
| `09. Business Analysis & GitHub` | Documents the business analysis and repository deliverables. |

Notebook names and paths should match the files currently committed to the repository and workspace.

## 4. Execution Flow

The intended logical sequence is:

1. Confirm that the source datasets are accessible.
2. Run the data transformation pipeline in notebook 07.
3. Validate the resulting data using notebook 08.
4. Review the Gold tables and perform business analysis using notebook 09.

Notebook 06 provides the architecture and job implementation documentation.

**Important:** The sequence above describes the intended workflow. Configure task dependencies in Databricks Workflows if automated orchestration is required.

## 5. Data Processing Parameters

Notebook 07 supports a stage parameter with the following documented values:

| Parameter value | Purpose |
|---|---|
| `explore` | Explore the source data. |
| `bronze` | Execute Bronze-layer processing. |
| `silver` | Execute Silver-layer processing. |
| `gold` | Execute Gold-layer processing. |
| `validate` | Run the pipeline's validation stage. |
| `all` | Execute the complete pipeline sequence supported by the notebook. |

The notebook uses the `p2_team_` namespace for Phase 2 objects. Confirm the parameter name and default value in the current notebook before configuring Job parameters.

## 6. Expected Analytical Outputs

The Gold-layer outputs documented for Phase 2 include:

- `workspace.default.p2_team_gold_transaction_risk_flags`
- `workspace.default.p2_team_gold_daily_risk_metrics`
- `workspace.default.p2_team_gold_location_risk_comparison`
- `workspace.default.p2_team_gold_employee_risk_profile`

These tables support rule-based transaction risk analysis, daily risk monitoring, location comparisons, and employee-level analysis. Confirm that each table exists after a successful pipeline run.

## 7. Monitoring and Troubleshooting

When a run fails:

1. Open the job run details in Databricks.
2. Identify the failed task and inspect its notebook output and error message.
3. Verify that the source data is available and that the execution identity has the required permissions.
4. Check the notebook parameters and the availability of upstream tables.
5. Rerun the failed task or the workflow after correcting the issue.
6. Confirm that downstream tables and validation results are consistent.

## 8. Configuration Status

Record the actual configuration of the Databricks Job before marking it as production-ready.

- Job name: [Enter the configured job name]
- Workspace: [Enter the workspace]
- Tasks and dependencies: [Document the configured tasks]
- Schedule or trigger: [Document the configured schedule, or state that runs are manual]
- Compute configuration: [Document the configured compute]
- Notifications and retries: [Document if configured]
- Latest successful run: [Enter the verified run date or status]

Do not mark scheduling, automatic retries, notifications, or successful execution as implemented unless verified in Databricks.

## 9. Operational Notes

The workflow should be tested from the source-data stage through the validation and analysis stages. A successful notebook execution alone does not guarantee that all downstream outputs are correct; confirm the expected tables and data quality results.

Update this document whenever notebook paths, parameters, dependencies, output tables, or Job configuration change.
