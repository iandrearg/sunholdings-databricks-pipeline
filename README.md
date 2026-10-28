# SunHoldings — Databricks Data Pipeline

## Project Overview

This project builds on Phase 1 exploratory data analysis to develop a reproducible Databricks pipeline for SunHoldings' Applebee's transaction data.

The pipeline aims to ingest, transform, validate, and analyze transaction records to support operational monitoring and the investigation of potentially unusual transaction patterns.

## Business Objective

Transform raw transaction data into reliable analytical tables that support the analysis of discounts, voids, employee activity, and location-level patterns.

Potential anomalies are indicators for further investigation, not proof of fraud.

## Data Sources

The project uses five datasets from Phase 1:

- Checks
- Employees
- Items
- Locations
- Revenue Centers

Data access, permissions, and source-file availability must be confirmed before execution.

## Pipeline Architecture

The planned architecture follows the Medallion approach:

- **Bronze:** Raw ingestion and source traceability.
- **Silver:** Data cleaning, standardization, transformations, and quality validation.
- **Gold:** Analytical tables for business insights and transaction risk monitoring.

## Repository Structure

- `notebooks/phase1/`: Exploratory analysis notebooks.
- `notebooks/phase2/`: Pipeline implementation and analytical notebooks.
- `notebooks/tests/`: Pipeline tests and failure-handling evidence.
- `docs/`: Architecture, data dictionary, quality report, and business findings.
- `jobs/`: Databricks Job documentation and configuration, where available.

## Data Quality

The pipeline will evaluate missing values, duplicates, invalid records, data types, consistency, and other relevant quality rules.

## Business Analysis

The planned analysis includes discount rates, void rates, repeated void patterns, and transaction flags, where supported by the source data.

## Execution Instructions

Detailed setup, dependencies, permissions, configuration, and execution instructions will be documented after the pipeline is validated.

## Project Status

Phase 2 implementation and validation are in progress.

## Team

SunHoldings Databricks Project — Phase 2.
