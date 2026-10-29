# Phase 2 — Databricks Pipeline Deliverables

## 1. Overview

Phase 2 extends the exploratory work completed in Phase 1 by developing an end-to-end data pipeline in Databricks for Applebee's transaction data.

The project focuses on transforming raw operational records into structured analytical tables that support transaction monitoring, data quality validation, and business analysis.

The solution follows the Medallion architecture: Bronze, Silver, and Gold.

## 2. Business Objective

The main objective is to build a reproducible and maintainable pipeline that processes transaction data and enables the analysis of discounts, voids, payment-related signals, and unusual transaction patterns.

The resulting analytical tables support operational monitoring and help prioritize records for further investigation.

Risk indicators are signals for review and must not be interpreted as proof of fraud or misconduct.

## 3. Source Data

The pipeline uses five source datasets:

- **Checks:** transaction and check-level records.
- **Employees:** employee-related information.
- **Items:** item and product information.
- **Locations:** restaurant location information.
- **Revenue Centers:** operational revenue-center information.

The source data is read from the configured Databricks Volume. The current pipeline configuration uses:

`/Volumes/workspace/default/applebees_datachefs_raw/OneDrive_1_5-10-2026/`

The source path must exist and be accessible in the target Databricks environment.

## 4. Pipeline Architecture

### Bronze — Raw Ingestion

The Bronze layer ingests the five source datasets and preserves ingestion metadata, including source-file information and ingestion timestamps.

Tables use the `p2_team_bronze_` prefix.

### Silver — Cleaning and Preparation

The Silver layer prepares transaction and event data for analysis, including checks, items sold, discounts, payments, and voids.

Tables use the `p2_team_silver_` prefix.

### Gold — Analytical Outputs

The Gold layer produces four analytical tables:

- `p2_team_gold_transaction_risk_flags`: transaction-level risk indicators and scores.
- `p2_team_gold_daily_risk_metrics`: daily metrics by location.
- `p2_team_gold_location_risk_comparison`: comparisons by location and revenue center.
- `p2_team_gold_employee_risk_profile`: aggregated employee-level metrics.

Tables are stored in the `workspace.default` schema.

## 5. Notebook Inventory

### `06. Architecture, Job & Engineering`

Documents the Databricks environment, existing tables, technical architecture, and engineering considerations for the pipeline.

### `07_Data_Transformations_Tables_TEAM`

Implements the Bronze, Silver, and Gold processing stages. It supports the `explore`, `bronze`, `silver`, `gold`, `validate`, and `all` stages through a notebook parameter.

The integration version writes to the `p2_team_` namespace to separate its outputs from the original `p2_` tables.

### `08. Data Quality & Expectations`

Documents the quality-validation approach for the Bronze and Silver layers, including table accessibility, key integrity, duplicate checks, event consistency, financial reconciliation, and anomaly-flag consistency.

The validation notebook must be executed and its actual code and results reviewed before the quality gate is considered complete.

### `09. Business Analysis & GitHub`

Analyzes the Gold outputs using executive KPIs, daily trends, location comparisons, employee-level metrics, and transaction risk indicators.

The analysis is intended to connect the processed data with meaningful operational insights and business value.

## 6. Risk Indicators and Assumptions

The transformation logic includes rule-based indicators for:

- Fully voided checks.
- High discounts relative to gross check sales.
- Checks with item lines but no payment records.
- Discounts recorded at both check and item levels.

The current transformation configuration uses the following parameters:

- High-discount threshold: 50%.
- Minimum checks for location/revenue-center comparison: 100.
- Minimum sold item lines for employee comparisons: 100.
- Location/revenue-center outlier threshold: 2 standard deviations.
- Employee outlier threshold: 3 standard deviations on the high side.

These values are configuration assumptions and should be reviewed in context. A risk score is a count of rule-based signals, not a validated fraud probability.

## 7. Data Quality and Reliability

The quality-validation approach addresses table availability, required identifiers, uniqueness, reconciliation between Bronze and Silver, event integrity, amount consistency, temporal validity, and anomaly-flag consistency.

Quality validation should be completed before business findings are treated as final.

The documented manual validation results and the current executable notebook must be checked for consistency. Native Lakeflow expectations should only be claimed if they are actually implemented and demonstrated.

## 8. Execution Order

The intended processing sequence is:

1. Inspect the environment and confirm source access.
2. Run notebook 07 with `stage = "bronze"`.
3. Run notebook 07 with `stage = "silver"`.
4. Execute the data quality validation notebook and review its output.
5. Run notebook 07 with `stage = "gold"` after the required quality checks pass.
6. Execute notebook 09 to analyze the Gold tables.
7. Review the results, update the findings documentation, and prepare the final presentation.

Alternatively, notebook 07 supports `stage = "all"` for a complete pipeline run. Use this option only when the required prerequisites and validation sequence have been confirmed.

## 9. Business Analysis

The final analysis focuses on:

- The number of checks evaluated and the proportion with risk signals.
- Discount and void amounts relative to gross sales.
- Daily changes in transaction metrics.
- Differences across locations and revenue centers.
- Aggregated employee-level patterns.
- The distribution and frequency of transaction risk indicators.

All conclusions must be supported by actual query results and interpreted alongside data-quality limitations.

## 10. Repository and Reproducibility

The project repository contains notebooks, documentation, and supporting materials for understanding the pipeline and its analytical outputs.

Before final submission:

- Confirm that notebook names and paths match the repository.
- Document the actual source path and required permissions.
- Verify the execution order and notebook parameters.
- Record the final quality-validation results.
- Include reproducible business findings and relevant visualizations.
- Remove credentials and avoid publishing sensitive or identifiable operational data.

