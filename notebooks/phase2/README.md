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

