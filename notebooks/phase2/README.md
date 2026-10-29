# Phase 2: Applebee’s Data Engineering and Business Analysis

## Project Overview

This project develops a data pipeline to transform Applebee’s operational data into reliable, analysis-ready datasets. The pipeline follows the Medallion Architecture, organizing data into Bronze, Silver, and Gold layers to support data quality validation, business analysis, and risk monitoring.

## Business Objective

The objective is to analyze transaction patterns, discounts, voids, payment records, locations, revenue centers, and employee activity to identify unusual patterns and provide actionable business insights. The analysis supports operational monitoring and helps prioritize transactions and business areas for further review.

## Data Sources

The project uses the following Applebee’s datasets:

- Checks
- Employees
- Items
- Locations
- Revenue Centers

The source files are stored in the configured Databricks Volume.

## Data Architecture

The pipeline uses the Medallion Architecture:

- **Bronze:** Ingests raw source data into Databricks tables.
- **Silver:** Cleans, standardizes, and prepares data for analysis.
- **Gold:** Produces curated datasets for business metrics, transaction risk analysis, location comparisons, and employee profiles.

The project uses the `p2_team_` prefix for its pipeline tables.

## Notebook Organization

| Notebook | Responsibility |
|---|---|
| `06. Architecture, Job & Engineering` | Documents the pipeline architecture, engineering approach, and job configuration. |
| `07_Data_Transformations_Tables_TEAM` | Implements data transformations and creates the Bronze, Silver, and Gold tables. |
| `08. Data Quality & Expectations` | Supports data quality checks and validation of pipeline outputs. |
| `09. Business Analysis & GitHub` | Analyzes business metrics, transaction risk indicators, locations, revenue centers, and employee activity. |

## Gold Tables

The Business Analysis notebook uses the following Gold tables:

- `p2_team_gold_transaction_risk_flags`
- `p2_team_gold_daily_risk_metrics`
- `p2_team_gold_location_risk_comparison`
- `p2_team_gold_employee_risk_profile`

These tables provide transaction-level risk indicators, daily metrics, location comparisons, and employee-level profiles.

## Transaction Risk Indicators

The analysis evaluates four transaction-level indicators:

- **Fully Voided Checks:** Checks with no item lines and at least one void.
- **High Discounts:** Checks where total discounts are at least 50% of gross check sales.
- **Missing Payment Records:** Checks with item lines but no corresponding payment records.
- **Discounts at Both Levels:** Checks with discounts recorded at both the check and item levels.

The `risk_score` represents the number of these indicators associated with a transaction. It is a screening measure and does not represent the probability of fraud.

## Business Analysis and Visualizations

The Business Analysis notebook includes:

- Key performance indicators and flagged transaction summaries.
- Distribution of risk scores and risk indicators.
- Daily and weekday transaction patterns.
- Location-level risk rankings and comparisons.
- Location scatter plots and risk concentration analysis.
- Revenue center comparisons.
- Location and revenue center outlier analysis.
- Employee-level transaction metrics and outlier analysis.
- An automated summary of the observed results.

Minimum transaction thresholds are applied to selected location and revenue center comparisons to reduce the influence of low-volume groups.

## Data Quality and Validation

Data quality checks are included in the pipeline workflow to support consistency and reliability across the data layers. Validation results should be reviewed before downstream business analysis is considered complete.

The Business Analysis notebook depends on the expected Gold tables being available and populated. The analysis should be executed after the transformation and validation stages have completed successfully.

## Execution Workflow

The pipeline supports the following stages:

- `explore`
- `bronze`
- `silver`
- `gold`
- `validate`
- `all`

The recommended workflow is:

1. Review the source data and configuration.
2. Run the required Bronze ingestion stage.
3. Execute the Silver transformations.
4. Generate the Gold analytical tables.
5. Run data quality validation.
6. Execute the Business Analysis notebook.
7. Review the resulting metrics, visualizations, and summary.
8. Confirm that expected outputs are available and that no execution errors occurred.

## Outputs

The project produces structured Databricks tables for downstream analysis and visualizations for reviewing transaction patterns, discounts, voids, payment records, locations, revenue centers, and employee activity.

Charts may be saved to the configured Databricks Volume path when the destination is available and writable.

## Limitations

The risk indicators identify unusual transaction patterns for review and do not establish misconduct or fraud. Results depend on the completeness and accuracy of the source data, the validity of the transformations, and the successful execution of data quality checks.

Comparisons using minimum transaction thresholds exclude groups below the configured thresholds. Visualizations and summaries should be interpreted in the context of these filters and the available data.

## Reproducibility

For reproducible results, use the configured source data location, maintain the project table naming conventions, execute the notebooks in the documented order, and verify the outputs of each pipeline stage before proceeding to the next stage.

## Expected Outcome

The completed Phase 2 pipeline provides a structured and repeatable process for preparing Applebee’s operational data, validating transformed datasets, and generating business insights into transaction risk patterns and operational performance.
