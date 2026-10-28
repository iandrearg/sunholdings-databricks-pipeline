# Databricks notebook source
# MAGIC %md
# MAGIC Se revisa que cargen las carpetas con las bases de datos

# COMMAND ----------

base_path = "/Volumes/workspace/default/applebees_datachefs_raw/OneDrive_1_5-10-2026"

files = dbutils.fs.ls(base_path)

display(files)

# COMMAND ----------

# MAGIC %md
# MAGIC Inventario

# COMMAND ----------

#Revisamos que las bases de datos dentro de las carpetas si se puedan leer de manera correcta
dimensions_path = f"{base_path}/dimensions"
facts_path = f"{base_path}/facts"

print("DIMENSIONS")
dimensions_files = dbutils.fs.ls(dimensions_path)
display(dimensions_files)

print("FACTS")
facts_files = dbutils.fs.ls(facts_path)
display(facts_files)

# COMMAND ----------


def list_files_recursive(path):
    files = []

    for item in dbutils.fs.ls(path):
        if item.isDir():
            files.extend(list_files_recursive(item.path))
        else:
            files.append(item)

    return files

all_files = list_files_recursive(base_path)

print(f"Total files found: {len(all_files)}")

# Show file paths
file_inventory = [(f.name, f.path, f.size) for f in all_files]

inventory_df = spark.createDataFrame(
    file_inventory,
    ["file_name", "file_path", "size_bytes"]
)

display(inventory_df)

#Se confirma que todas las bases están y se puede leer. Son 115 días de información transaccional y 4 archivos dimensionales

# COMMAND ----------

# MAGIC %md 3. Understand schemas
# MAGIC

# COMMAND ----------

import json
from pyspark.sql.functions import col, count, countDistinct, when, sum as _sum, spark_partition_id, max as _max

# ---------------------------------------------------------------------------
# Paths (read-only -- no data is modified)
# ---------------------------------------------------------------------------
base_path = "/Volumes/workspace/default/applebees_datachefs_raw/OneDrive_1_5-10-2026"
dimensions_path = f"{base_path}/dimensions"
facts_checks_path = f"{base_path}/facts/checks"

# ---------------------------------------------------------------------------
# 1. Load the 115 daily checks JSON files as one logical dataset
# ---------------------------------------------------------------------------
checks_df = (
    spark.read
    .option("recursiveFileLookup", "true")
    .option("multiLine", True)
    .json(facts_checks_path)
)

n_partitions = checks_df.select(spark_partition_id().alias("pid")).distinct().count()
print(f"checks_df  -- files loaded: {n_partitions} partitions")
checks_row_count = checks_df.count()
print(f"checks_df  -- row count  : {checks_row_count:,}")

# ---------------------------------------------------------------------------
# 2. Load the four dimension datasets
# ---------------------------------------------------------------------------
dim_files = dbutils.fs.ls(dimensions_path)
print("\nDimension files found:")
for f in dim_files:
    print(f"  {f.name}  ({f.size:,} bytes)")

def load_dim(file_entry):
    p = file_entry.path
    if p.endswith(".json"):
        return spark.read.option("multiLine", True).json(p)
    elif p.endswith(".csv"):
        return spark.read.option("header", "true").option("inferSchema", "true").csv(p)
    elif p.endswith(".parquet") or p.endswith(".snappy.parquet"):
        return spark.read.parquet(p)
    else:
        return spark.read.option("multiLine", True).json(p)

dim_dfs = {}
for f in dim_files:
    key = f.name.split(".")[0].lower().replace(" ", "_")
    dim_dfs[key] = load_dim(f)

# ---------------------------------------------------------------------------
# 3. Inspect every logical dataset (checks + 4 dimensions)
# ---------------------------------------------------------------------------
all_datasets = {"checks": checks_df, **dim_dfs}

dataset_summaries = []  # collect for final summary

for name, df in all_datasets.items():
    n_rows = df.count()
    n_cols = len(df.columns)
    schema = df.schema

    print("\n" + "=" * 90)
    print(f"DATASET: {name.upper()}   |   rows = {n_rows:,}   |   columns = {n_cols}")
    print("=" * 90)

    # Column-level profiling
    distinct_counts = df.agg(*[countDistinct(c).alias(c) for c in df.columns]).collect()[0]
    null_counts = df.agg(*[_sum(when(col(c).isNull(), 1).otherwise(0)).alias(c) for c in df.columns]).collect()[0]

    header = f"{'column':<45}{'type':<30}{'distinct':>10}{'nulls':>10}"
    print(header)
    print("-" * len(header))
    for field in schema.fields:
        c = field.name
        print(f"{c:<45}{str(field.dataType):<30}{distinct_counts[c]:>10,}{null_counts[c]:>10,}")

    # Likely primary keys: unique + non-null
    pk_candidates = []
    for field in schema.fields:
        c = field.name
        if distinct_counts[c] == n_rows and null_counts[c] == 0:
            pk_candidates.append(c)
    print(f"\nLikely PK(s): {pk_candidates if pk_candidates else '(none single -- possible composite key)'}")

    # Likely foreign keys: column name contains 'id' or '_id' and is not the PK
    fk_candidates = []
    for field in schema.fields:
        c = field.name
        cl = c.lower()
        if c not in pk_candidates and ("id" in cl or "_id" in cl or cl.endswith("id")):
            fk_candidates.append(c)
    print(f"Likely FK(s): {fk_candidates if fk_candidates else '(none detected)'}")

    # Sample row for semantic understanding
    sample = df.limit(1).collect()
    if sample:
        print("\nSample row (first 4000 chars of JSON):")
        print(json.dumps(sample[0].asDict(), default=str, indent=2)[:4000])

    dataset_summaries.append({
        "name": name,
        "rows": n_rows,
        "cols": n_cols,
        "columns": [(f.name, str(f.dataType)) for f in schema.fields],
        "pk": pk_candidates,
        "fk": fk_candidates,
    })

# ---------------------------------------------------------------------------
# 4. Relationships summary
# ---------------------------------------------------------------------------
print("\n\n" + "=" * 90)
print("RELATIONSHIPS SUMMARY")
print("=" * 90)

# Cross-reference column names across datasets
for i, s in enumerate(dataset_summaries):
    for j, s2 in enumerate(dataset_summaries):
        if i >= j:
            continue
        common = set(c for c, _ in s["columns"]) & set(c for c, _ in s2["columns"])
        if common:
            print(f"  {s['name']} <-> {s2['name']}: shared columns = {sorted(common)}")

# ---------------------------------------------------------------------------
# 5. Columns relevant to fraud-detection themes
# ---------------------------------------------------------------------------
print("\n\n" + "=" * 90)
print("COLUMNS RELEVANT TO FRAUD-DETECTION THEMES")
print("=" * 90)

all_cols = {name: set(c for c, _ in df.columns_tuples) for name, df in []}
# rebuild from summaries
all_cols = {}
for s in dataset_summaries:
    all_cols[s["name"]] = set(c for c, _ in s["columns"])

themes = {
    "Discounts":    ["discount", "disc", "promo", "coupon", "comp"],
    "Tips":         ["tip", "gratuity"],
    "Voids":        ["void", "cancel"],
    "Deleted items": ["delete", "removed", "reason"],
    "Employees":    ["employee", "server", "cashier", "user", "staff", "emp"],
    "Locations":    ["location", "store", "restaurant", "site"],
    "Txn amounts":  ["amount", "total", "sales", "subtotal", "tax", "price", "paid", "balance", "cost"],
    "Txn date/time": ["date", "time", "datetime", "timestamp", "opened", "closed", "business"],
}

for theme, keywords in themes.items():
    hits = []
    for name, cols in all_cols.items():
        for c in sorted(cols):
            cl = c.lower()
            if any(kw in cl for kw in keywords):
                hits.append((name, c))
    print(f"\n{theme}:")
    if hits:
        for ds, c in hits:
            print(f"  {ds:<20} -> {c}")
    else:
        print("  (no obvious column name match -- needs deeper inspection)")

print("\n" + "=" * 90)
print("DONE -- review the output above and document findings.")
print("=" * 90)

# COMMAND ----------

#Insights de este primer análisis

#1. El dataset oiginal contiene 2,142,846 comprobantes de 115 fechas de negocio distintas. Sin valores nulos en BusinessDate

#2. La información transaccional tiene una estructura anidada. Dentro de cada registro de checks existen arreglos como ItemsSold, Discounts, Paymnets, Vpids y Tax. Esto premite analizar no solo el check completo, sno también productos, descuentos, pagos y anulaciones asociados a cada transacción.

#3.El identificador Id de checks no es único a nivel global. Hay 2,142,846 registros, pero solo 1,081,811 valores distintos de Id. Por lo tanto, antes de eliminar supuestos duplicados o realizar joins, es necesario determinar una llave compuesta que identifique correctamente cada registro.    Markdown pegado

#4. La base contiene variables especialmente relevantes para detectar patrones sospechosos. Los descuentos incluyen monto, empleado y manager; los voids contienen monto, producto, empleado, manager y categoría; y los pagos incluyen Gratuity, que puede utilizarse para estudiar las propinas.    Markdown pegado

#5. Existe potencial para relacionar múltiples señales de riesgo. La estructura permitiría estudiar posteriormente relaciones como descuentos–propinas, voids–empleados, productos–empleados y diferencias entre locations, en lugar de analizar cada indicador de forma aislada. Esto es una posibilidad analítica derivada de la estructura; todavía no es un hallazgo de fraude.

#6. Las dimensiones presentan identificadores aparentemente únicos. employees.Id, items.Id, locations.Id y revenue_centers.Id tienen tantos valores distintos como filas en sus respectivas tablas, por lo que son candidatos razonables a llaves primarias.    Markdown pegado

#7. LoyaltyId presenta un problema claro de calidad. Los 2,142,846 registros de checks tienen esta variable en NULL, por lo que actualmente no aporta información al análisis y deberá tomarse una decisión sobre su tratamiento durante la limpieza.    Markdown pegado

#8. Las relaciones entre las tablas todavía deben validarse. El análisis automático de Genie relacionó algunas tablas únicamente porque comparten nombres como Id, pero eso no demuestra una relación real. Antes de hacer los joins, debemos validar las foreign keys utilizando los valores contenidos en campos como ItemsSold.ItemId, EmployeeId, LocationId y RevenueCenterId.

#Conclusión: La base cuenta con información transaccional detallada y suficiente para analizar comportamientos potencialmente sospechosos; sin embargo, su estructura anidada y la falta de unicidad del Id de los checks hacen necesario validar primero el nivel de observación, las llaves y las relaciones entre tablas antes de realizar la limpieza y el análisis de fraude.

# COMMAND ----------

# MAGIC %md 4. Identify keys
# MAGIC

# COMMAND ----------

# ============================================================================
# DATA UNDERSTANDING: Transaction grain & key relationships
# Read-only investigation — no raw data is modified.
# ============================================================================

from pyspark.sql.functions import (
    col, count, countDistinct, when, sum as _sum, min as _min,
    max as _max, explode, size, struct, lit, expr, coalesce
)

# ------------------------------------------------------------------
# Re-load datasets (same logic as the profiling cell above)
# ------------------------------------------------------------------
base_path = "/Volumes/workspace/default/applebees_datachefs_raw/OneDrive_1_5-10-2026"
dimensions_path = f"{base_path}/dimensions"
facts_checks_path = f"{base_path}/facts/checks"

checks_df = (
    spark.read
    .option("recursiveFileLookup", "true")
    .option("multiLine", True)
    .json(facts_checks_path)
)

def load_dim(file_entry):
    p = file_entry.path
    if p.endswith(".json"):
        return spark.read.option("multiLine", True).json(p)
    elif p.endswith(".csv"):
        return spark.read.option("header", "true").option("inferSchema", "true").csv(p)
    elif p.endswith(".parquet") or p.endswith(".snappy.parquet"):
        return spark.read.parquet(p)
    else:
        return spark.read.option("multiLine", True).json(p)

dim_dfs = {}
for f in dbutils.fs.ls(dimensions_path):
    key = f.name.split(".")[0].lower().replace(" ", "_")
    dim_dfs[key] = load_dim(f)

total_checks = checks_df.count()

print("=" * 80)
print(f"CHECKS_DF LOADED — {total_checks:,} rows")
print("=" * 80)

# ------------------------------------------------------------------
# 1. What does one row in checks_df represent?
# ------------------------------------------------------------------
print("\n" + "=" * 80)
print("1. TRANSACTION GRAIN")
print("=" * 80)

sample_row = checks_df.limit(1).collect()[0].asDict()
top_level_keys = [k for k in sample_row.keys()]
print(f"Top-level columns ({len(top_level_keys)}): {top_level_keys}")

nested_arrays = ["ItemsSold", "Discounts", "Payments", "Voids", "Tax"]
print(f"\nNested arrays present in schema: {nested_arrays}")
print("\nInterpretation: Each row represents a single restaurant check (comprobante / transacción).")
print("It contains the check header (Id, BusinessDate, LocationId, EmployeeId, etc.)")
print("plus nested arrays of line-item detail (items sold, discounts, payments, voids, tax).")

# ------------------------------------------------------------------
# 2. Test composite keys for uniqueness
# ------------------------------------------------------------------
print("\n" + "=" * 80)
print("2. COMPOSITE KEY CANDIDATES — DUPLICATE TEST")
print("=" * 80)

candidate_keys = {
    "Id":                           ["Id"],
    "LocationId + Id":              ["LocationId", "Id"],
    "BusinessDate + Id":            ["BusinessDate", "Id"],
    "LocationId + BusinessDate + Id":["LocationId", "BusinessDate", "Id"],
}

dup_results = []
for label, cols_list in candidate_keys.items():
    grouped = checks_df.groupBy(*cols_list).agg(count("*").alias("n"))
    dups = grouped.filter(col("n") > 1)
    n_dup_groups = dups.count()
    n_dup_rows = dups.agg(_sum("n")).collect()[0][0] or 0
    dup_results.append((label, n_dup_groups, n_dup_rows))
    print(f"\n  Key: {label}")
    print(f"    Distinct groups with duplicates : {n_dup_groups:,}")
    print(f"    Total rows in dup groups         : {n_dup_rows:,}")

print("\n  ┌─ Summary ───────────────────────────────────────────────────┐")
print(f"  │ {'Key':<38} {'Dup Groups':>12} {'Dup Rows':>12} │")
print(f"  │ {'─'*38} {'─'*12} {'─'*12} │")
for label, dg, dr in dup_results:
    print(f"  │ {label:<38} {dg:>12,} {dr:>12,} │")
print("  └────────────────────────────────────────────────────────────┘")

# ------------------------------------------------------------------
# 3. BusinessDate coverage
# ------------------------------------------------------------------
print("\n" + "=" * 80)
print("3. BUSINESS DATE COVERAGE")
print("=" * 80)

date_stats = checks_df.agg(
    countDistinct("BusinessDate").alias("n_distinct_dates"),
    _min("BusinessDate").alias("min_date"),
    _max("BusinessDate").alias("max_date"),
    count(when(col("BusinessDate").isNull(), 1)).alias("null_dates")
).collect()[0]

print(f"  Distinct BusinessDate values : {date_stats['n_distinct_dates']}")
print(f"  Minimum BusinessDate         : {date_stats['min_date']}")
print(f"  Maximum BusinessDate         : {date_stats['max_date']}")
print(f"  Null BusinessDate count      : {date_stats['null_dates']}")
print(f"  Expected (per inventory)     : 115")
print(f"  Confirmed all 115 present?   : {date_stats['n_distinct_dates'] == 115}")

# ------------------------------------------------------------------
# 4. Nested array inspection
# ------------------------------------------------------------------
print("\n" + "=" * 80)
print("4. NESTED ARRAY INSPECTION")
print("=" * 80)

nested_results = []
for arr_name in nested_arrays:
    if arr_name not in checks_df.columns:
        print(f"  {arr_name}: column not found in checks_df")
        nested_results.append((arr_name, 0, 0.0, 0))
        continue

    # Checks with at least one element in the array (size > 0 and not null)
    has_arr = checks_df.filter(col(arr_name).isNotNull() & (size(col(arr_name)) > 0))
    n_checks_with = has_arr.count()
    pct = round(100.0 * n_checks_with / total_checks, 2) if total_checks else 0.0

    # Total nested records = sum of array sizes across all checks
    total_nested = checks_df.select(
        coalesce(size(col(arr_name)), lit(0)).alias("sz")
    ).agg(_sum("sz").alias("total")).collect()[0]["total"] or 0

    nested_results.append((arr_name, n_checks_with, pct, total_nested))
    print(f"\n  {arr_name}:")
    print(f"    Checks containing ≥1 record : {n_checks_with:,}  ({pct}%)")
    print(f"    Total nested records        : {total_nested:,}")

print("\n  ┌─ Summary ─────────────────────────────────────────────────────────┐")
print(f"  │ {'Array':<18} {'Checks w/ ≥1':>14} {'% of total':>12} {'Total records':>16} │")
print(f"  │ {'─'*18} {'─'*14} {'─'*12} {'─'*16} │")
for name, n_checks, pct, total_recs in nested_results:
    print(f"  │ {name:<18} {n_checks:>14,} {pct:>11.2f}% {total_recs:>16,} │")
print("  └────────────────────────────────────────────────────────────────────┘")

# ------------------------------------------------------------------
# 5. Payments.Gratuity — descriptive statistics
# ------------------------------------------------------------------
print("\n" + "=" * 80)
print("5. PAYMENTS.GRATUITY — TIPS / GRATUITY ANALYSIS")
print("=" * 80)

# Explode Payments and inspect the Gratuity field
payments_exploded = checks_df.select(
    col("Id"), col("BusinessDate"), col("LocationId"),
    explode(col("Payments")).alias("p")
)

gratuity_df = payments_exploded.select(col("p.*"))

# Confirm column exists
pay_cols = payments_exploded.select("p.*").columns
print(f"  Payments sub-schema columns: {pay_cols}")
print(f"  'Gratuity' present in Payments: {'Gratuity' in pay_cols}")

if "Gratuity" in pay_cols:
    grat_stats = gratuity_df.agg(
        count("Gratuity").alias("non_null"),
        count(when(col("Gratuity").isNull(), 1)).alias("nulls"),
        _min("Gratuity").alias("min"),
        _max("Gratuity").alias("max"),
        expr("mean(Gratuity)").alias("mean"),
        expr("percentile_approx(Gratuity, 0.5)").alias("median"),
        expr("stddev(Gratuity)").alias("stddev"),
        _sum("Gratuity").alias("total"),
    ).collect()[0]

    n_payments = payments_exploded.count()
    print(f"\n  Total payment records          : {n_payments:,}")
    print(f"  Gratuity non-null count        : {grat_stats['non_null']:,}")
    print(f"  Gratuity null count            : {grat_stats['nulls']:,}")
    print(f"  Min Gratuity                   : {grat_stats['min']}")
    print(f"  Max Gratuity                   : {grat_stats['max']}")
    print(f"  Mean Gratuity                  : {round(grat_stats['mean'], 4) if grat_stats['mean'] is not None else None}")
    print(f"  Median Gratuity (approx)       : {grat_stats['median']}")
    print(f"  Stddev Gratuity                : {round(grat_stats['stddev'], 4) if grat_stats['stddev'] is not None else None}")
    print(f"  Total Gratuity sum             : {round(grat_stats['total'], 2) if grat_stats['total'] is not None else None}")

    # Distribution of positive gratuities
    pos_grat = gratuity_df.filter(col("Gratuity") > 0)
    n_pos = pos_grat.count()
    print(f"  Payments with Gratuity > 0     : {n_pos:,}  ({round(100.0*n_pos/n_payments, 2) if n_payments else 0}% of payments)")
    print("\n  Interpretation: Payments.Gratuity represents tip/gratuity amounts per payment.")
    print("  It is numeric, often zero or null, and positive values represent actual tips.")
    print("  No outliers removed — all values reported as-is.")
else:
    print("  ⚠ Gratuity not found in Payments sub-schema.")

# ------------------------------------------------------------------
# 6 & 7. Validate FK relationships using actual values
# ------------------------------------------------------------------
print("\n" + "=" * 80)
print("6 & 7. FOREIGN KEY VALIDATION (actual value matching)")
print("=" * 80)

# --- Prepare dimension ID sets ---------------------------------------------------
dim_ids = {}
for key, ddf in dim_dfs.items():
    print(f"  Dimension '{key}': {ddf.count():,} rows, columns = {ddf.columns}")

# employees
emp_ids = set()
if "employees" in dim_dfs:
    emp_df = dim_dfs["employees"]
    id_col = "Id" if "Id" in emp_df.columns else emp_df.columns[0]
    emp_ids = set(r[id_col] for r in emp_df.select(id_col).distinct().collect())

# items
item_ids = set()
if "items" in dim_dfs:
    items_df_local = dim_dfs["items"]
    id_col = "Id" if "Id" in items_df_local.columns else items_df_local.columns[0]
    item_ids = set(r[id_col] for r in items_df_local.select(id_col).distinct().collect())

# locations
loc_ids = set()
if "locations" in dim_dfs:
    loc_df = dim_dfs["locations"]
    id_col = "Id" if "Id" in loc_df.columns else loc_df.columns[0]
    loc_ids = set(r[id_col] for r in loc_df.select(id_col).distinct().collect())

# revenue_centers
rc_ids = set()
if "revenue_centers" in dim_dfs:
    rc_df = dim_dfs["revenue_centers"]
    id_col = "Id" if "Id" in rc_df.columns else rc_df.columns[0]
    rc_ids = set(r[id_col] for r in rc_df.select(id_col).distinct().collect())

print(f"\n  Dimension ID set sizes: employees={len(emp_ids):,}, items={len(item_ids):,}, "
      f"locations={len(loc_ids):,}, revenue_centers={len(rc_ids):,}")

def fk_report(label, fk_df, fk_col, dim_id_set, dim_name):
    """Report orphan FK values vs a dimension ID set."""
    if fk_col not in fk_df.columns:
        print(f"  {label}: column '{fk_col}' not found — SKIPPED")
        return (label, fk_col, dim_name, "column not found", None, None)

    total_fk = fk_df.filter(col(fk_col).isNotNull()).count()
    distinct_fk = fk_df.filter(col(fk_col).isNotNull()).select(fk_col).distinct().count()

    # Collect distinct FK values and compare against dimension set
    fk_values = set(r[fk_col] for r in fk_df.filter(col(fk_col).isNotNull()).select(fk_col).distinct().collect())
    orphans = fk_values - dim_id_set
    n_orphan_values = len(orphans)
    pct_orphan_values = round(100.0 * n_orphan_values / distinct_fk, 4) if distinct_fk else 0.0

    # Count orphan rows (rows whose FK is not in dimension)
    orphan_df = fk_df.filter(col(fk_col).isNotNull() & ~col(fk_col).isin(list(dim_id_set))) if dim_id_set else fk_df.filter(col(fk_col).isNotNull())
    n_orphan_rows = orphan_df.count()
    pct_orphan_rows = round(100.0 * n_orphan_rows / total_fk, 4) if total_fk else 0.0

    status = "✅ CONFIRMED" if n_orphan_values == 0 else ("⚠ UNCERTAIN" if n_orphan_values <= 5 else "❌ BROKEN")

    print(f"\n  {label}")
    print(f"    FK column            : {fk_col}")
    print(f"    Target dimension     : {dim_name} ({len(dim_id_set):,} IDs)")
    print(f"    Total non-null FK    : {total_fk:,}")
    print(f"    Distinct FK values   : {distinct_fk:,}")
    print(f"    Orphan values (no match): {n_orphan_values:,}  ({pct_orphan_values}% of distinct)")
    print(f"    Orphan rows           : {n_orphan_rows:,}  ({pct_orphan_rows}% of non-null FK)")
    print(f"    Status               : {status}")
    return (label, fk_col, dim_name, total_fk, distinct_fk, n_orphan_values, pct_orphan_values, status)

fk_results = []

# --- 6a. ItemsSold.ItemId → items.Id ----------------------------------------------
print("\n  ── 6a. ItemsSold.ItemId → items.Id ──")
items_exploded = checks_df.select(explode(col("ItemsSold")).alias("it"))
items_sub_cols = items_exploded.select("it.*").columns
print(f"  ItemsSold sub-schema columns: {items_sub_cols}")
# Find the item ID column
item_id_col = None
for candidate in ["ItemId", "itemId", "ID", "Id"]:
    if candidate in items_sub_cols:
        item_id_col = candidate
        break
if item_id_col:
    items_fk_df = items_exploded.select(col(f"it.{item_id_col}").alias("fk_val"))
    fk_results.append(fk_report("ItemsSold.ItemId → items.Id", items_fk_df.withColumnRenamed("fk_val", "ItemId"), "ItemId", item_ids, "items"))
else:
    print("  ⚠ Could not find an ItemId-like column in ItemsSold sub-schema.")
    fk_results.append(("ItemsSold.ItemId → items.Id", "(not found)", "items", None, None, None, None, "❌ UNCERTAIN"))

# --- 6b. EmployeeId in ItemsSold, Discounts, Payments, Voids → employees.Id -------
print("\n  ── 6b. Employee IDs in nested arrays → employees.Id ──")

emp_sources = {
    "ItemsSold.EmployeeId": (checks_df.select(explode(col("ItemsSold")).alias("x")), "x"),
    "Discounts.EmployeeId": (checks_df.select(explode(col("Discounts")).alias("x")), "x"),
    "Payments.EmployeeId":  (checks_df.select(explode(col("Payments")).alias("x")), "x"),
    "Voids.EmployeeId":     (checks_df.select(explode(col("Voids")).alias("x")), "x"),
}

for label, (edf, alias) in emp_sources.items():
    sub_cols = edf.select(f"{alias}.*").columns
    # Find employee ID column
    emp_col = None
    for candidate in ["EmployeeId", "employeeId", "EmployeeID", "EmpId"]:
        if candidate in sub_cols:
            emp_col = candidate
            break
    if emp_col:
        emp_fk_df = edf.select(col(f"{alias}.{emp_col}").alias("EmpId"))
        fk_results.append(fk_report(label, emp_fk_df.withColumnRenamed("EmpId", "fk_val").withColumnRenamed("fk_val", "EmpId"), "EmpId", emp_ids, "employees"))
    else:
        # Also check if EmployeeId appears at top level of checks
        print(f"  {label}: no employee-id column found in sub-schema ({sub_cols}). Checking top-level checks.EmployeeId instead.")
        if "EmployeeId" in checks_df.columns:
            emp_fk_df = checks_df.select(col("EmployeeId")).filter(col("EmployeeId").isNotNull())
            fk_results.append(fk_report("checks.EmployeeId → employees.Id", emp_fk_df.withColumnRenamed("EmployeeId", "fk_val").withColumnRenamed("fk_val", "EmpId"), "EmpId", emp_ids, "employees"))
        else:
            fk_results.append((label, "(not found)", "employees", None, None, None, None, "❌ UNCERTAIN"))

# --- 6c. checks.LocationId → locations.Id -----------------------------------------
print("\n  ── 6c. checks.LocationId → locations.Id ──")
if "LocationId" in checks_df.columns:
    loc_fk_df = checks_df.select(col("LocationId")).filter(col("LocationId").isNotNull())
    fk_results.append(fk_report("checks.LocationId → locations.Id", loc_fk_df.withColumnRenamed("LocationId", "fk_val").withColumnRenamed("fk_val", "LocationId"), "LocationId", loc_ids, "locations"))
else:
    fk_results.append(("checks.LocationId → locations.Id", "(not found)", "locations", None, None, None, None, "❌ UNCERTAIN"))

# --- 6d. RevenueCenterId → revenue_centers.Id --------------------------------------
print("\n  ── 6d. RevenueCenterId → revenue_centers.Id ──")

# RevenueCenterId can appear at check level and/or inside nested arrays
rc_sources = {}
if "RevenueCenterId" in checks_df.columns:
    rc_sources["checks.RevenueCenterId"] = checks_df.select(col("RevenueCenterId")).filter(col("RevenueId").isNotNull() if False else col("RevenueCenterId").isNotNull())

# Check ItemsSold for RevenueCenterId
if "ItemsSold" in checks_df.columns:
    items_sub = checks_df.select(explode(col("ItemsSold")).alias("x"))
    items_sub_cols = items_sub.select("x.*").columns
    if "RevenueCenterId" in items_sub_cols:
        rc_sources["ItemsSold.RevenueCenterId"] = items_sub.select(col("x.RevenueCenterId")).filter(col("x.RevenueCenterId").isNotNull())

# Check Discounts
if "Discounts" in checks_df.columns:
    disc_sub = checks_df.select(explode(col("Discounts")).alias("x"))
    disc_sub_cols = disc_sub.select("x.*").columns
    if "RevenueCenterId" in disc_sub_cols:
        rc_sources["Discounts.RevenueCenterId"] = disc_sub.select(col("x.RevenueCenterId")).filter(col("x.RevenueCenterId").isNotNull())

for label, rc_df_local in rc_sources.items():
    fk_results.append(fk_report(label, rc_df_local.withColumnRenamed("RevenueCenterId", "fk_val").withColumnRenamed("fk_val", "RevenueCenterId"), "RevenueCenterId", rc_ids, "revenue_centers"))

if not rc_sources:
    print("  ⚠ RevenueCenterId not found in checks or nested arrays.")
    fk_results.append(("RevenueCenterId → revenue_centers.Id", "(not found)", "revenue_centers", None, None, None, None, "❌ UNCERTAIN"))

# --- Final FK summary table --------------------------------------------------------
print("\n" + "=" * 80)
print("FOREIGN KEY VALIDATION SUMMARY")
print("=" * 80)

print("\n  ┌─ FK Validation ───────────────────────────────────────────────────────────────────────────────┐")
header = f"  │ {'Relationship':<42} {'Target dim':<18} {'Orphan vals':>12} {'% orphan':>10} {'Status':<16} │"
print(header)
print(f"  │ {'─'*42} {'─'*18} {'─'*12} {'─'*10} {'─'*16} │")
for r in fk_results:
    label = r[0]
    dim_name = r[2] if len(r) > 2 else "?"
    n_orphan = r[5] if len(r) > 5 and r[5] is not None else "N/A"
    pct_orphan = r[6] if len(r) > 6 and r[6] is not None else "N/A"
    status = r[-1] if r[-1] else "?"
    n_orphan_s = f"{n_orphan:,}" if isinstance(n_orphan, int) else str(n_orphan)
    pct_s = f"{pct_orphan}%" if isinstance(pct_orphan, (int, float)) else str(pct_orphan)
    print(f"  │ {label:<42} {dim_name:<18} {n_orphan_s:>12} {pct_s:>10} {status:<16} │")
print("  └────────────────────────────────────────────────────────────────────────────────────────────────┘")

print("\n" + "=" * 80)
print("DONE — Data Understanding: grain & key relationships investigation complete.")
print("No raw data was modified.")
print("=" * 80)

# COMMAND ----------

# ============================================================================
# CORRECTED FK VALIDATION & GRATUITY SIGN-CONVENTION INVESTIGATION
# Read-only — no raw data is modified.
# ============================================================================

from pyspark.sql.functions import (
    col, count, countDistinct, when, sum as _sum, min as _min,
    max as _max, explode, size, struct, lit, expr, coalesce, abs as _abs
)

# ------------------------------------------------------------------
# Re-load datasets (same logic as earlier cells)
# ------------------------------------------------------------------
base_path = "/Volumes/workspace/default/applebees_datachefs_raw/OneDrive_1_5-10-2026"
dimensions_path = f"{base_path}/dimensions"
facts_checks_path = f"{base_path}/facts/checks"

checks_df = (
    spark.read
    .option("recursiveFileLookup", "true")
    .option("multiLine", True)
    .json(facts_checks_path)
)

def load_dim(file_entry):
    p = file_entry.path
    if p.endswith(".json"):
        return spark.read.option("multiLine", True).json(p)
    elif p.endswith(".csv"):
        return spark.read.option("header", "true").option("inferSchema", "true").csv(p)
    elif p.endswith(".parquet") or p.endswith(".snappy.parquet"):
        return spark.read.parquet(p)
    else:
        return spark.read.option("multiLine", True).json(p)

dim_dfs = {}
for f in dbutils.fs.ls(dimensions_path):
    key = f.name.split(".")[0].lower().replace(" ", "_")
    dim_dfs[key] = load_dim(f)

total_checks = checks_df.count()

print("=" * 80)
print(f"CHECKS_DF LOADED — {total_checks:,} rows")
print("=" * 80)

# ============================================================================
# PART 1 — CORRECTED FOREIGN-KEY VALIDATION
# ============================================================================
# The previous run reported dimension ID set sizes of 0. Root cause: the keys
# in dim_dfs are derived from filenames (e.g. "employees", "employee",
# "locations", etc.) and may not match the hardcoded strings used previously.
# Additionally, the Id column may have a different name or casing. This cell
# performs a robust diagnosis before any FK validation.

print("\n" + "=" * 80)
print("PART 1: CORRECTED FOREIGN-KEY VALIDATION")
print("=" * 80)

# ------------------------------------------------------------------
# 1a. Diagnose dimension DataFrames — keys, schemas, Id columns
# ------------------------------------------------------------------
print("\n── 1a. DIMENSION DIAGNOSTICS ──")
print(f"\n  dim_dfs keys: {list(dim_dfs.keys())}")

for key, ddf in dim_dfs.items():
    n = ddf.count()
    cols = ddf.columns
    # Find the Id column: try common casings
    id_col = None
    for candidate in ["Id", "id", "ID", "Id_", "_id"]:
        if candidate in cols:
            id_col = candidate
            break
    print(f"\n  Dimension '{key}': {n:,} rows")
    print(f"    Columns: {cols}")
    print(f"    Detected Id column: {id_col}")
    if id_col is not None:
        id_stats = ddf.agg(
            count(col(id_col)).alias("non_null"),
            count(when(col(id_col).isNull(), 1)).alias("nulls"),
            countDistinct(col(id_col)).alias("distinct"),
            _min(col(id_col)).alias("min_id"),
            _max(col(id_col)).alias("max_id"),
        ).collect()[0]
        print(f"    Non-null Id count : {id_stats['non_null']:,}")
        print(f"    Null Id count     : {id_stats['nulls']:,}")
        print(f"    Distinct Id count : {id_stats['distinct']:,}")
        print(f"    Min Id            : {id_stats['min_id']}")
        print(f"    Max Id            : {id_stats['max_id']}")
    else:
        print(f"    ⚠ No Id-like column found among {cols}")
        # Show top-level schema for debugging
        print(f"    Schema:")
        ddf.printSchema()

# ------------------------------------------------------------------
# 1b. Build dimension ID sets with flexible key matching
# ------------------------------------------------------------------
print("\n── 1b. BUILDING DIMENSION ID SETS ──")

def find_dim_df(*possible_keys):
    """Return (key, df) for the first matching key in dim_dfs, else None."""
    for k in possible_keys:
        if k in dim_dfs:
            return k, dim_dfs[k]
    # Fuzzy: check if any key contains one of the possible keys
    for dk in dim_dfs:
        for k in possible_keys:
            if k in dk or dk in k:
                return dk, dim_dfs[dk]
    return None, None

def get_id_set(dim_df, dim_name):
    """Collect distinct non-null Id values from a dimension DataFrame."""
    cols = dim_df.columns
    id_col = None
    for candidate in ["Id", "id", "ID", "Id_", "_id"]:
        if candidate in cols:
            id_col = candidate
            break
    if id_col is None:
        print(f"  ⚠ {dim_name}: no Id column found in {cols}")
        return set(), None
    id_vals = set()
    for r in dim_df.select(col(id_col).alias("id_val")).filter(col(id_col).isNotNull()).distinct().collect():
        v = r["id_val"]
        id_vals.add(v)
    return id_vals, id_col

# Match each dimension
dim_matches = {}
for logical_name, candidates in [
    ("employees",      ["employees", "employee", "emp", "employee_dim"]),
    ("items",           ["items", "item", "menu_items", "item_dim"]),
    ("locations",       ["locations", "location", "restaurant", "location_dim", "restaurants"]),
    ("revenue_centers", ["revenue_centers", "revenue_center", "revenuecenter", "revenue_centers_dim", "rc"]),
]:
    matched_key, matched_df = find_dim_df(*candidates)
    if matched_df is not None:
        id_set, id_col_used = get_id_set(matched_df, logical_name)
        dim_matches[logical_name] = {
            "df": matched_df,
            "key": matched_key,
            "id_col": id_col_used,
            "id_set": id_set,
        }
        print(f"  {logical_name:<18} → matched dim_dfs key '{matched_key}' (Id col = '{id_col_used}'), {len(id_set):,} distinct IDs")
    else:
        dim_matches[logical_name] = None
        print(f"  {logical_name:<18} → ⚠ NOT FOUND among dim_dfs keys {list(dim_dfs.keys())}")

# ------------------------------------------------------------------
# 1c. Pre-FK dimension summary table
# ------------------------------------------------------------------
print("\n── 1c. DIMENSION ID SUMMARY (pre-FK validation) ──")
print()
print(f"  ┌─ Dimension Summary {'─'*50}┐")
print(f"  │ {'Dimension':<20} {'Rows':>10} {'NonNull Id':>12} {'Distinct Id':>12} {'Min Id':>10} {'Max Id':>10} │")
print(f"  │ {'─'*20} {'─'*10} {'─'*12} {'─'*12} {'─'*10} {'─'*10} │")

for logical_name in ["employees", "items", "locations", "revenue_centers"]:
    m = dim_matches.get(logical_name)
    if m is None:
        print(f"  │ {logical_name:<20} {'NOT FOUND':>10} {'':>12} {'':>12} {'':>10} {'':>10} │")
        continue
    ddf = m["df"]
    id_col = m["id_col"]
    row_count = ddf.count()
    if id_col is None:
        print(f"  │ {logical_name:<20} {row_count:>10,} {'N/A':>12} {'N/A':>12} {'N/A':>10} {'N/A':>10} │")
        continue
    stats = ddf.agg(
        count(col(id_col)).alias("non_null"),
        countDistinct(col(id_col)).alias("distinct"),
        _min(col(id_col)).alias("min_id"),
        _max(col(id_col)).alias("max_id"),
    ).collect()[0]
    print(f"  │ {logical_name:<20} {row_count:>10,} {stats['non_null']:>12,} {stats['distinct']:>12,} {str(stats['min_id']):>10} {str(stats['max_id']):>10} │")

print(f"  └{'─'*80}┘")

# ------------------------------------------------------------------
# 1d. FK validation helper (corrected)
# ------------------------------------------------------------------
def fk_report(label, fk_df, fk_col, dim_id_set, dim_name):
    """Report orphan FK values vs a dimension ID set."""
    if fk_col not in fk_df.columns:
        print(f"  {label}: column '{fk_col}' not found — SKIPPED")
        return (label, fk_col, dim_name, "column not found", None, None, None, None, "⚠ SKIPPED")
    if not dim_id_set:
        print(f"  {label}: dimension '{dim_name}' ID set is empty — cannot validate")
        return (label, fk_col, dim_name, 0, 0, 0, 0.0, 0, "⚠ NO DIM IDS")

    total_fk = fk_df.filter(col(fk_col).isNotNull()).count()
    distinct_fk = fk_df.filter(col(fk_col).isNotNull()).select(fk_col).distinct().count()

    # Collect distinct FK values
    fk_values = set()
    for r in fk_df.filter(col(fk_col).isNotNull()).select(fk_col).distinct().collect():
        fk_values.add(r[fk_col])

    orphans = fk_values - dim_id_set
    n_orphan_values = len(orphans)
    pct_orphan_values = round(100.0 * n_orphan_values / distinct_fk, 4) if distinct_fk else 0.0

    # Count orphan rows
    dim_id_list = list(dim_id_set)
    orphan_df = fk_df.filter(col(fk_col).isNotNull() & ~col(fk_col).isin(dim_id_list))
    n_orphan_rows = orphan_df.count()
    pct_orphan_rows = round(100.0 * n_orphan_rows / total_fk, 4) if total_fk else 0.0

    status = "✅ CONFIRMED" if n_orphan_values == 0 else ("⚠ UNCERTAIN" if n_orphan_values <= 5 else "❌ BROKEN")

    print(f"\n  {label}")
    print(f"    FK column             : {fk_col}")
    print(f"    Target dimension      : {dim_name} ({len(dim_id_set):,} IDs)")
    print(f"    Total non-null FK     : {total_fk:,}")
    print(f"    Distinct FK values    : {distinct_fk:,}")
    print(f"    Unmatched distinct IDs: {n_orphan_values:,}  ({pct_orphan_values}% of distinct)")
    print(f"    Unmatched rows        : {n_orphan_rows:,}  ({pct_orphan_rows}% of non-null FK)")
    if 0 < n_orphan_values <= 20:
        print(f"    Sample orphan values  : {sorted(orphans)[:20]}")
    print(f"    Status                : {status}")
    return (label, fk_col, dim_name, total_fk, distinct_fk, n_orphan_values, pct_orphan_values, n_orphan_rows, status)

fk_results = []

# ------------------------------------------------------------------
# 1e. Run FK validations
# ------------------------------------------------------------------
print("\n── 1e. FK VALIDATION RESULTS ──")

# --- ItemsSold.ItemId → items.Id ---
print("\n  ── ItemsSold.ItemId → items.Id ──")
if "ItemsSold" in checks_df.columns:
    items_exploded = checks_df.select(explode(col("ItemsSold")).alias("it"))
    items_sub_cols = items_exploded.select("it.*").columns
    print(f"  ItemsSold sub-schema columns: {items_sub_cols}")
    item_id_col = None
    for candidate in ["ItemId", "itemId", "item_id", "ID", "Id"]:
        if candidate in items_sub_cols:
            item_id_col = candidate
            break
    if item_id_col:
        items_fk_df = items_exploded.select(col(f"it.{item_id_col}").alias("ItemId"))
        m = dim_matches.get("items")
        if m:
            fk_results.append(fk_report("ItemsSold.ItemId → items.Id", items_fk_df, "ItemId", m["id_set"], "items"))
        else:
            print("  ⚠ items dimension not loaded")
    else:
        print("  ⚠ Could not find ItemId column in ItemsSold sub-schema")
        fk_results.append(("ItemsSold.ItemId → items.Id", "(not found)", "items", None, None, None, None, None, "❌ UNCERTAIN"))
else:
    print("  ⚠ ItemsSold column not found in checks_df")
    fk_results.append(("ItemsSold.ItemId → items.Id", "(not found)", "items", None, None, None, None, None, "❌ UNCERTAIN"))

# --- EmployeeId from ItemsSold, Discounts, Payments, Voids → employees.Id ---
print("\n  ── EmployeeId (nested arrays) → employees.Id ──")
m_emp = dim_matches.get("employees")
emp_id_set = m_emp["id_set"] if m_emp else set()

emp_sources = {}
if "ItemsSold" in checks_df.columns:
    emp_sources["ItemsSold.EmployeeId"] = (checks_df.select(explode(col("ItemsSold")).alias("x")), "x")
if "Discounts" in checks_df.columns:
    emp_sources["Discounts.EmployeeId"] = (checks_df.select(explode(col("Discounts")).alias("x")), "x")
if "Payments" in checks_df.columns:
    emp_sources["Payments.EmployeeId"] = (checks_df.select(explode(col("Payments")).alias("x")), "x")
if "Voids" in checks_df.columns:
    emp_sources["Voids.EmployeeId"] = (checks_df.select(explode(col("Voids")).alias("x")), "x")

for label, (edf, alias) in emp_sources.items():
    sub_cols = edf.select(f"{alias}.*").columns
    emp_col = None
    for candidate in ["EmployeeId", "employeeId", "employee_id", "EmployeeID", "EmpId", "emp_id"]:
        if candidate in sub_cols:
            emp_col = candidate
            break
    if emp_col:
        emp_fk_df = edf.select(col(f"{alias}.{emp_col}").alias("EmpId"))
        fk_results.append(fk_report(label, emp_fk_df, "EmpId", emp_id_set, "employees"))
    else:
        print(f"  {label}: no employee-id column in sub-schema ({sub_cols})")
        # Fall back to top-level checks.EmployeeId
        if "EmployeeId" in checks_df.columns:
            emp_fk_df = checks_df.select(col("EmployeeId").alias("EmpId")).filter(col("EmployeeId").isNotNull())
            fk_results.append(fk_report("checks.EmployeeId → employees.Id", emp_fk_df, "EmpId", emp_id_set, "employees"))
        else:
            fk_results.append((label, "(not found)", "employees", None, None, None, None, None, "❌ UNCERTAIN"))

# --- checks.LocationId → locations.Id ---
print("\n  ── checks.LocationId → locations.Id ──")
m_loc = dim_matches.get("locations")
loc_id_set = m_loc["id_set"] if m_loc else set()
if "LocationId" in checks_df.columns:
    loc_fk_df = checks_df.select(col("LocationId")).filter(col("LocationId").isNotNull())
    fk_results.append(fk_report("checks.LocationId → locations.Id", loc_fk_df, "LocationId", loc_id_set, "locations"))
else:
    print("  ⚠ LocationId not found in checks_df")
    fk_results.append(("checks.LocationId → locations.Id", "(not found)", "locations", None, None, None, None, None, "❌ UNCERTAIN"))

# --- checks.RevenueCenterId → revenue_centers.Id ---
print("\n  ── checks.RevenueCenterId → revenue_centers.Id ──")
m_rc = dim_matches.get("revenue_centers")
rc_id_set = m_rc["id_set"] if m_rc else set()
if "RevenueCenterId" in checks_df.columns:
    rc_fk_df = checks_df.select(col("RevenueCenterId")).filter(col("RevenueCenterId").isNotNull())
    fk_results.append(fk_report("checks.RevenueCenterId → revenue_centers.Id", rc_fk_df, "RevenueCenterId", rc_id_set, "revenue_centers"))
else:
    print("  ⚠ RevenueCenterId not found at check level")
    fk_results.append(("checks.RevenueCenterId → revenue_centers.Id", "(not found)", "revenue_centers", None, None, None, None, None, "❌ UNCERTAIN"))

# --- ItemsSold.RevenueCenterId → revenue_centers.Id ---
print("\n  ── ItemsSold.RevenueCenterId → revenue_centers.Id ──")
if "ItemsSold" in checks_df.columns:
    items_sub = checks_df.select(explode(col("ItemsSold")).alias("x"))
    items_sub_cols = items_sub.select("x.*").columns
    if "RevenueCenterId" in items_sub_cols:
        rc_items_fk_df = items_sub.select(col("x.RevenueCenterId")).filter(col("x.RevenueCenterId").isNotNull())
        fk_results.append(fk_report("ItemsSold.RevenueCenterId → revenue_centers.Id", rc_items_fk_df, "RevenueCenterId", rc_id_set, "revenue_centers"))
    else:
        print(f"  ⚠ RevenueCenterId not in ItemsSold sub-schema ({items_sub_cols})")
        fk_results.append(("ItemsSold.RevenueCenterId → revenue_centers.Id", "(not found)", "revenue_centers", None, None, None, None, None, "❌ UNCERTAIN"))
else:
    print("  ⚠ ItemsSold column not found")
    fk_results.append(("ItemsSold.RevenueCenterId → revenue_centers.Id", "(not found)", "revenue_centers", None, None, None, None, None, "❌ UNCERTAIN"))

# --- FK Summary Table ---
print("\n" + "=" * 80)
print("FOREIGN KEY VALIDATION SUMMARY")
print("=" * 80)
print()
print(f"  ┌─ FK Validation {'─'*70}┐")
header = f"  │ {'Relationship':<46} {'Target dim':<18} {'Unmatched':>10} {'% orphan':>9} {'Status':<14} │"
print(header)
print(f"  │ {'─'*46} {'─'*18} {'─'*10} {'─'*9} {'─'*14} │")
for r in fk_results:
    label = r[0]
    dim_name = r[2] if len(r) > 2 else "?"
    n_orphan_vals = r[5] if len(r) > 5 and r[5] is not None else "N/A"
    pct_orphan = r[6] if len(r) > 6 and r[6] is not None else "N/A"
    status = r[-1] if r[-1] else "?"
    n_orphan_s = f"{n_orphan_vals:,}" if isinstance(n_orphan_vals, int) else str(n_orphan_vals)
    pct_s = f"{pct_orphan}%" if isinstance(pct_orphan, (int, float)) else str(pct_orphan)
    print(f"  │ {label:<46} {dim_name:<18} {n_orphan_s:>10} {pct_s:>9} {status:<14} │")
print(f"  └{'─'*98}┘")

# ============================================================================
# PART 2 — PAYMENTS.GRATUITY SIGN-CONVENTION INVESTIGATION
# ============================================================================
print("\n" + "=" * 80)
print("PART 2: PAYMENTS.GRATUITY SIGN-CONVENTION INVESTIGATION")
print("=" * 80)

# Explode Payments
payments_exploded = checks_df.select(
    col("Id").alias("CheckId"),
    col("BusinessDate"),
    col("LocationId"),
    explode(col("Payments")).alias("p")
)

pay_df = payments_exploded.select(
    col("CheckId"),
    col("BusinessDate"),
    col("LocationId"),
    col("p.*")
)

pay_cols = pay_df.columns
print(f"\n  Payments sub-schema columns: {pay_cols}")

# Check if Gratuity exists
if "Gratuity" not in pay_cols:
    # Try case-insensitive
    grat_col = None
    for c in pay_cols:
        if c.lower() == "gratuity":
            grat_col = c
            break
    if grat_col is None:
        print("  ⚠ Gratuity column not found in Payments sub-schema")
        grat_col = "Gratuity"  # will fail gracefully
else:
    grat_col = "Gratuity"

print(f"  Using Gratuity column: '{grat_col}'")

# ------------------------------------------------------------------
# 2a. Sign distribution: < 0, = 0, > 0
# ------------------------------------------------------------------
print("\n── 2a. GRATUITY SIGN DISTRIBUTION ──")

n_total = pay_df.filter(col(grat_col).isNotNull()).count()
n_null = pay_df.filter(col(grat_col).isNull()).count()
n_all = pay_df.count()

n_neg = pay_df.filter(col(grat_col) < 0).count()
n_zero = pay_df.filter(col(grat_col) == 0).count()
n_pos = pay_df.filter(col(grat_col) > 0).count()

print(f"\n  Total payment records       : {n_all:,}")
print(f"  Non-null Gratuity count     : {n_total:,}  ({round(100.0*n_total/n_all, 2) if n_all else 0}% of payments)")
print(f"  Null Gratuity count         : {n_null:,}  ({round(100.0*n_null/n_all, 2) if n_all else 0}% of payments)")
print()
print(f"  ┌─ Gratuity Sign Distribution {'─'*40}┐")
print(f"  │ {'Sign':<18} {'Count':>14} {'% of non-null':>16} │")
print(f"  │ {'─'*18} {'─'*14} {'─'*16} │")
print(f"  │ {'Gratuity < 0':<18} {n_neg:>14,} {round(100.0*n_neg/n_total,4) if n_total else 0:>15.4f}% │")
print(f"  │ {'Gratuity = 0':<18} {n_zero:>14,} {round(100.0*n_zero/n_total,4) if n_total else 0:>15.4f}% │")
print(f"  │ {'Gratuity > 0':<18} {n_pos:>14,} {round(100.0*n_pos/n_total,4) if n_total else 0:>15.4f}% │")
print(f"  └{'─'*74}┘")

# Also show descriptive stats for reference
grat_stats = pay_df.agg(
    _min(col(grat_col)).alias("min"),
    _max(col(grat_col)).alias("max"),
    expr(f"mean({grat_col})").alias("mean"),
    expr(f"percentile_approx({grat_col}, 0.5)").alias("median"),
    expr(f"stddev({grat_col})").alias("stddev"),
    _sum(col(grat_col)).alias("total"),
).collect()[0]
print(f"\n  Descriptive stats (non-null):")
print(f"    Min    : {grat_stats['min']}")
print(f"    Max    : {grat_stats['max']}")
print(f"    Mean   : {round(grat_stats['mean'], 6) if grat_stats['mean'] is not None else None}")
print(f"    Median : {grat_stats['median']}")
print(f"    Stddev : {round(grat_stats['stddev'], 6) if grat_stats['stddev'] is not None else None}")
print(f"    Sum    : {round(grat_stats['total'], 4) if grat_stats['total'] is not None else None}")

# ------------------------------------------------------------------
# 2b. Identify relevant columns for context
# ------------------------------------------------------------------
print("\n── 2b. PAYMENTS SUB-SCHEMA COLUMN INVENTORY ──")

# Look for Amount-like, Category-like, payment-type columns
amount_candidates = [c for c in pay_cols if c.lower() in ["amount", "paymentamount", "payamount", "total", "subtotal"]]
type_candidates = [c for c in pay_cols if any(k in c.lower() for k in ["category", "type", "method", "name", "paymenttype", "tender"])]
print(f"  Amount-like columns  : {amount_candidates}")
print(f"  Type/Category columns: {type_candidates}")
print(f"  All columns           : {pay_cols}")

# Pick the most likely Amount column
amount_col = None
for c in ["Amount", "amount", "PaymentAmount", "Total", "SubTotal"]:
    if c in pay_cols:
        amount_col = c
        break
if amount_col is None and amount_candidates:
    amount_col = amount_candidates[0]

# Pick the most likely category/type column
cat_col = None
for c in ["Category", "category", "PaymentCategory", "Type", "PaymentType", "PaymentMethod", "TenderType", "Name"]:
    if c in pay_cols:
        cat_col = c
        break
if cat_col is None and type_candidates:
    cat_col = type_candidates[0]

print(f"\n  Selected Amount column   : {amount_col}")
print(f"  Selected Category column : {cat_col}")

# Check for credit/cash indicators
cc_candidates = [c for c in pay_cols if any(k in c.lower() for k in ["credit", "cash", "card", "iscredit", "iscash", "tendertype"])]
print(f"  Credit/Cash indicators   : {cc_candidates}")

# ------------------------------------------------------------------
# 2c. Examples: negative, zero, positive Gratuity
# ------------------------------------------------------------------
print("\n── 2c. EXAMPLE PAYMENTS BY GRATUITY SIGN ──")

# Build the select expression for examples
example_cols = ["CheckId", "BusinessDate", "LocationId", "EmployeeId", grat_col]
if amount_col:
    example_cols.append(amount_col)
if cat_col:
    example_cols.append(cat_col)
for c in cc_candidates:
    if c not in example_cols:
        example_cols.append(c)
# Deduplicate while preserving order
seen = set()
example_cols_unique = []
for c in example_cols:
    if c not in seen and c in pay_df.columns:
        example_cols_unique.append(c)
        seen.add(c)

print(f"\n  Example columns: {example_cols_unique}")

# Negative examples
print(f"\n  ── Negative Gratuity examples (Gratuity < 0) — first 10 ──")
neg_examples = pay_df.filter(col(grat_col) < 0).select(*example_cols_unique).orderBy(col(grat_col).asc()).limit(10).collect()
for i, row in enumerate(neg_examples):
    print(f"    [{i+1}] " + " | ".join(f"{c}={row[c]}" for c in example_cols_unique))

# Zero examples
print(f"\n  ── Zero Gratuity examples (Gratuity = 0) — first 10 ──")
zero_examples = pay_df.filter(col(grat_col) == 0).select(*example_cols_unique).limit(10).collect()
for i, row in enumerate(zero_examples):
    print(f"    [{i+1}] " + " | ".join(f"{c}={row[c]}" for c in example_cols_unique))

# Positive examples
print(f"\n  ── Positive Gratuity examples (Gratuity > 0) — first 10 ──")
pos_examples = pay_df.filter(col(grat_col) > 0).select(*example_cols_unique).orderBy(col(grat_col).desc()).limit(10).collect()
for i, row in enumerate(pos_examples):
    print(f"    [{i+1}] " + " | ".join(f"{c}={row[c]}" for c in example_cols_unique))

# ------------------------------------------------------------------
# 2d. Cross-tabulation: Gratuity sign × Payment Category/Type
# ------------------------------------------------------------------
print("\n── 2d. GRATUITY SIGN × PAYMENT CATEGORY/TYPE ──")

if cat_col:
    print(f"  (Using '{cat_col}' as category column)\n")
    sign_cat_df = pay_df.filter(col(grat_col).isNotNull()).withColumn(
        "GratuitySign",
        when(col(grat_col) < 0, lit("negative"))
        .when(col(grat_col) == 0, lit("zero"))
        .otherwise(lit("positive"))
    )
    cross_tab = (sign_cat_df
        .groupBy(cat_col, "GratuitySign")
        .agg(count("*").alias("n"))
        .orderBy(cat_col, "GratuitySign")
        .collect()
    )
    print(f"  {'Category':<30} {'Sign':<12} {'Count':>12}")
    print(f"  {'─'*30} {'─'*12} {'─'*12}")
    for row in cross_tab:
        print(f"  {str(row[cat_col]):<30} {row['GratuitySign']:<12} {row['n']:>12,}")
else:
    print("  ⚠ No category/type column found for cross-tabulation.")

# ------------------------------------------------------------------
# 2e. Compare Gratuity to Amount — sign relationship
# ------------------------------------------------------------------
print("\n── 2e. GRATUITY vs AMOUNT SIGN RELATIONSHIP ──")

if amount_col and grat_col:
    # How often is Gratuity negative when Amount is positive (or vice versa)?
    sign_compare = pay_df.filter(col(grat_col).isNotNull() & col(amount_col).isNotNull()).agg(
        count(when((col(grat_col) < 0) & (col(amount_col) > 0), 1)).alias("grat_neg_amt_pos"),
        count(when((col(grat_col) < 0) & (col(amount_col) < 0), 1)).alias("grat_neg_amt_neg"),
        count(when((col(grat_col) < 0) & (col(amount_col) == 0), 1)).alias("grat_neg_amt_zero"),
        count(when((col(grat_col) > 0) & (col(amount_col) > 0), 1)).alias("grat_pos_amt_pos"),
        count(when((col(grat_col) > 0) & (col(amount_col) < 0), 1)).alias("grat_pos_amt_neg"),
        count(when((col(grat_col) == 0) & (col(amount_col) > 0), 1)).alias("grat_zero_amt_pos"),
        count(when((col(grat_col) == 0) & (col(amount_col) < 0), 1)).alias("grat_zero_amt_neg"),
        count("*").alias("total_both_nonnull"),
    ).collect()[0]

    t = sign_compare["total_both_nonnull"]
    print(f"  (Among {t:,} payments where both Gratuity and {amount_col} are non-null)\n")
    print(f"  {'Pattern':<30} {'Count':>12} {'%':>10}")
    print(f"  {'─'*30} {'─'*12} {'─'*10}")
    for label_key in ["grat_neg_amt_pos", "grat_neg_amt_neg", "grat_neg_amt_zero",
                       "grat_pos_amt_pos", "grat_pos_amt_neg",
                       "grat_zero_amt_pos", "grat_zero_amt_neg"]:
        n = sign_compare[label_key]
        pct = round(100.0 * n / t, 4) if t else 0.0
        print(f"  {label_key:<30} {n:>12,} {pct:>9.4f}%")

    # Correlation between Gratuity and Amount
    corr_val = pay_df.filter(col(grat_col).isNotNull() & col(amount_col).isNotNull()).stat.corr(grat_col, amount_col)
    print(f"\n  Pearson correlation (Gratuity, {amount_col}): {round(corr_val, 6) if corr_val is not None else 'N/A'}")

    # Ratio analysis: |Gratuity| / Amount for negative gratuity
    print(f"\n  Ratio |Gratuity| / {amount_col} for negative Gratuity (first 20):")
    ratio_examples = (pay_df
        .filter((col(grat_col) < 0) & (col(amount_col) > 0) & (col(amount_col) != 0))
        .select(
            col("CheckId"), col(grat_col).alias("Gratuity"), col(amount_col).alias("Amount"),
            (_abs(col(grat_col)) / col(amount_col)).alias("ratio_abs_grat_to_amt"),
        )
        .orderBy(col("ratio_abs_grat_to_amt").desc())
        .limit(20)
        .collect()
    )
    print(f"    {'CheckId':<20} {'Gratuity':>12} {'Amount':>12} {'|Grat|/Amt':>12}")
    print(f"    {'─'*20} {'─'*12} {'─'*12} {'─'*12}")
    for r in ratio_examples:
        print(f"    {str(r['CheckId']):<20} {r['Gratuity']:>12.2f} {r['Amount']:>12.2f} {r['ratio_abs_grat_to_amt']:>12.4f}")
else:
    print("  ⚠ Amount or Gratuity column not available for comparison.")

# ------------------------------------------------------------------
# 2f. Check-level analysis: sum of Gratuity per check
# ------------------------------------------------------------------
print("\n── 2f. CHECK-LEVEL GRATUITY SUMMARIES ──")

check_grat = pay_df.filter(col(grat_col).isNotNull()).groupBy("CheckId").agg(
    _sum(col(grat_col)).alias("total_gratuity"),
    count("*").alias("n_payments"),
    _sum(col(amount_col)).alias("total_amount") if amount_col else lit(0).alias("total_amount"),
)

# Distribution of check-level total gratuity signs
check_signs = check_grat.agg(
    count(when(col("total_gratuity") < 0, 1)).alias("checks_neg_grat"),
    count(when(col("total_gratuity") == 0, 1)).alias("checks_zero_grat"),
    count(when(col("total_gratuity") > 0, 1)).alias("checks_pos_grat"),
    count("*").alias("total_checks_with_grat"),
).collect()[0]

tc = check_signs["total_checks_with_grat"]
print(f"  Checks with ≥1 non-null Gratuity payment: {tc:,}")
print(f"  Checks where summed Gratuity < 0: {check_signs['checks_neg_grat']:,}  ({round(100.0*check_signs['checks_neg_grat']/tc,2) if tc else 0}%)")
print(f"  Checks where summed Gratuity = 0: {check_signs['checks_zero_grat']:,}  ({round(100.0*check_signs['checks_zero_grat']/tc,2) if tc else 0}%)")
print(f"  Checks where summed Gratuity > 0: {check_signs['checks_pos_grat']:,}  ({round(100.0*check_signs['checks_pos_grat']/tc,2) if tc else 0}%)")

# ------------------------------------------------------------------
# 2g. Sign-convention assessment
# ------------------------------------------------------------------
print("\n── 2g. SIGN-CONVENTION ASSESSMENT ──")
print()
print("  Observations:")
print(f"    1. {n_neg:,} of {n_total:,} non-null Gratuity values ({round(100.0*n_neg/n_total,2) if n_total else 0}%) are NEGATIVE.")
print(f"    2. Only {n_pos:,} ({round(100.0*n_pos/n_total,4) if n_total else 0}%) are POSITIVE.")
print(f"    3. {n_zero:,} ({round(100.0*n_zero/n_total,2) if n_total else 0}%) are EXACTLY ZERO.")
if amount_col:
    grat_neg_amt_pos_n = sign_compare["grat_neg_amt_pos"] if 'sign_compare' in dir() else 0
    print(f"    4. Among payments with both fields non-null, {grat_neg_amt_pos_n:,} have negative Gratuity alongside positive {amount_col}.")
    if corr_val is not None:
        print(f"    5. Pearson correlation between Gratuity and {amount_col}: {round(corr_val, 6)}.")
print()
print("  Interpretation:")
print("    The overwhelmingly negative distribution (Gratuity < 0 in the vast majority of non-null cases)")
print("    is consistent with a POS system sign convention where tips/gratuities are stored as negative")
print("    values (credits to the customer/employee) while payment amounts are positive (debits). Under")
print("    this convention, |Gratuity| represents the actual tip amount.")
print()
print("    However, the extremely small number of positive values and the very high proportion of zeros")
print("    mean that the convention cannot be fully confirmed from sign patterns alone without domain")
print("    documentation or a data dictionary.")
print()
print("  ┌──────────────────────────────────────────────────────────────────────────┐")
print("  │ SIGN-CONVENTION STATUS: UNCERTAIN — consistent with a POS convention     │")
print("  │ where negative Gratuity = actual tip amount (|Gratuity|), but not         │")
print("  │ confirmable from the data alone. No values have been transformed.        │")
print("  └──────────────────────────────────────────────────────────────────────────┘")

print("\n" + "=" * 80)
print("DONE — Corrected FK validation & Gratuity sign-convention investigation complete.")
print("No raw data was modified.")
print("=" * 80)

# COMMAND ----------

#Insights
#1. Se identificó EmployeeId = 0 como valor sin correspondencia en algunas estructuras transaccionales. Su frecuencia es mínima y su significado debe evaluarse antes de definir su tratamiento durante la limpieza.
#2. Los valores de Gratuity presentan una convención de signo atípica: 46.28% son negativos y prácticamente no existen valores positivos. La evidencia es consistente con una convención del sistema POS donde el valor absoluto podría representar la propina, pero esto requiere validación adicional antes de transformar la variable.

# COMMAND ----------

# MAGIC %md
# MAGIC 5. Missing Values

# COMMAND ----------

# ============================================================================
# 5. MISSING VALUES — DIAGNOSTIC ONLY (read-only, no raw data modified)
# ============================================================================
# This cell performs a comprehensive missing-value audit across the five
# logical datasets and every nested array inside checks_df.
# Nothing is cleaned, filled, dropped, or transformed.
# ============================================================================

from pyspark.sql.functions import (
    col, count, countDistinct, when, sum as _sum, min as _min, max as _max,
    explode, size, lit, isnan, length, trim
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
SEVERITY_THRESHOLDS = [
    (0.0,  "None"),
    (5.0,  "Low"),
    (20.0, "Moderate"),
    (100.0, "High"),
]

def severity_label(pct):
    if pct == 0:
        return "None"
    elif pct <= 5:
        return "Low"
    elif pct <= 20:
        return "Moderate"
    elif pct < 100:
        return "High"
    else:
        return "Complete"

def pct_fmt(pct):
    return f"{pct:.4f}%" if pct is not None else "N/A"

def analyze_top_level(df, dataset_name, total_rows=None):
    """Return list of dicts with missing-value stats for every top-level column."""
    if total_rows is None:
        total_rows = df.count()
    rows_out = []
    for c in df.columns:
        col_ref = col(c)
        stats = df.agg(
            count(col_ref).alias("non_null"),
            count(when(col_ref.isNull(), True)).alias("null_cnt"),
            countDistinct(col_ref).alias("distinct"),
        ).collect()[0]
        null_cnt = stats["null_cnt"]
        null_pct = round(100.0 * null_cnt / total_rows, 4) if total_rows else 0.0
        dtype = dict(df.dtypes)[c]
        rows_out.append({
            "dataset": dataset_name,
            "column": c,
            "dtype": dtype,
            "total_rows": total_rows,
            "null_count": null_cnt,
            "null_pct": null_pct,
            "distinct": stats["distinct"],
            "nested": False,
        })
    # sort by null_pct descending
    rows_out.sort(key=lambda r: (-r["null_pct"], r["column"]))
    return rows_out

def analyze_nested_array(checks_df, array_col, label_name, total_checks=None):
    """Explode an array column from checks_df and report per-field missingness."""
    if array_col not in checks_df.columns:
        print(f"  ⚠ '{array_col}' not found in checks_df — SKIPPED")
        return [], 0, 0

    if total_checks is None:
        total_checks = checks_df.count()

    # Count empty arrays vs non-empty
    arr_info = checks_df.agg(
        count(when(col(array_col).isNull(), True)).alias("null_arr"),
        count(when(col(array_col).isNotNull() & (size(col(array_col)) == 0), True)).alias("empty_arr"),
        count(when(col(array_col).isNotNull() & (size(col(array_col)) > 0), True)).alias("nonempty_arr"),
    ).collect()[0]
    print(f"\n  ── {label_name} (checks.{array_col}) ──")
    print(f"    Checks with null array   : {arr_info['null_arr']:,}")
    print(f"    Checks with empty array  : {arr_info['empty_arr']:,}")
    print(f"    Checks with non-empty arr: {arr_info['nonempty_arr']:,}")

    if arr_info["nonempty_arr"] == 0:
        print(f"    ⚠ No non-empty arrays — cannot inspect nested fields")
        return [], arr_info['null_arr'], arr_info['empty_arr']

    # Explode and select sub-fields
    exploded = checks_df.filter(col(array_col).isNotNull() & (size(col(array_col)) > 0)).select(explode(col(array_col)).alias("x"))
    total_nested = exploded.count()
    sub_cols = exploded.select("x.*").columns
    print(f"    Total nested records      : {total_nested:,}")
    print(f"    Sub-fields               : {sub_cols}")

    rows_out = []
    for sc in sub_cols:
        stats = exploded.agg(
            count(col(f"x.{sc}")).alias("non_null"),
            count(when(col(f"x.{sc}").isNull(), True)).alias("null_cnt"),
            countDistinct(col(f"x.{sc}")).alias("distinct"),
        ).collect()[0]
        null_cnt = stats["null_cnt"]
        null_pct = round(100.0 * null_cnt / total_nested, 4) if total_nested else 0.0
        dtype = dict(exploded.select("x.*").dtypes)[sc]
        rows_out.append({
            "dataset": f"checks.{array_col}",
            "column": sc,
            "dtype": dtype,
            "total_rows": total_nested,
            "null_count": null_cnt,
            "null_pct": null_pct,
            "distinct": stats["distinct"],
            "nested": True,
        })
    rows_out.sort(key=lambda r: (-r["null_pct"], r["column"]))
    return rows_out, arr_info['null_arr'], arr_info['empty_arr']

# ------------------------------------------------------------------
# Gather dimension DataFrames
# ------------------------------------------------------------------
datasets = {"checks_df": checks_df}
# reuse dim_matches if available, otherwise rebuild quickly
for logical_name, candidates in [
    ("employees",      ["employees", "employee", "emp", "employee_dim"]),
    ("items",          ["items", "item", "menu_items", "item_dim"]),
    ("locations",      ["locations", "location", "restaurant", "location_dim", "restaurants"]),
    ("revenue_centers",["revenue_centers", "revenue_center", "revenuecenter", "revenue_centers_dim", "rc"]),
]:
    matched_key, matched_df = None, None
    if 'dim_matches' in dir() and isinstance(dim_matches, dict) and dim_matches.get(logical_name):
        matched_df = dim_matches[logical_name]["df"]
        matched_key = dim_matches[logical_name]["key"]
    else:
        for k in candidates:
            if k in dim_dfs:
                matched_key, matched_df = k, dim_dfs[k]
                break
        if matched_df is None:
            for dk in dim_dfs:
                for k in candidates:
                    if k in dk or dk in k:
                        matched_key, matched_df = dk, dim_dfs[dk]
                        break
                if matched_df is not None:
                    break
    if matched_df is not None:
        datasets[logical_name] = matched_df
        print(f"  Dimension '{logical_name}' loaded from dim_dfs['{matched_key}']")
    else:
        print(f"  ⚠ Dimension '{logical_name}' not found in dim_dfs")

print("\n" + "=" * 80)
print("SECTION 5: MISSING VALUES — DIAGNOSTIC AUDIT")
print("=" * 80)

# ============================================================================
# 5.1 — TOP-LEVEL MISSING VALUES (every dataset)
# ============================================================================
print("\n" + "─" * 80)
print("5.1  TOP-LEVEL MISSING VALUES")
print("─" * 80)

all_top_rows = []
total_checks = checks_df.count()

for ds_name, ds_df in datasets.items():
    print(f"\n  ── {ds_name} ──")
    tr = total_checks if ds_name == "checks_df" else ds_df.count()
    rows = analyze_top_level(ds_df, ds_name, tr)
    all_top_rows.extend(rows)
    # print compact table
    print(f"    {'Column':<35} {'Dtype':<18} {'Rows':>12} {'Nulls':>12} {'Null %':>10} {'Distinct':>10}")
    print(f"    {'─'*35} {'─'*18} {'─'*12} {'─'*12} {'─'*10} {'─'*10}")
    for r in rows:
        print(f"    {r['column']:<35} {str(r['dtype']):<18} {r['total_rows']:>12,} {r['null_count']:>12,} {pct_fmt(r['null_pct']):>10} {r['distinct']:>10,}")

# ============================================================================
# 5.2 — NESTED ARRAYS IN CHECKS
# ============================================================================
print("\n" + "─" * 80)
print("5.2  NESTED ARRAYS IN CHECKS")
print("─" * 80)

nested_arrays = ["ItemsSold", "Discounts", "Payments", "Voids", "Tax"]
all_nested_rows = []
array_empty_summary = []

for arr in nested_arrays:
    rows, n_null_arr, n_empty_arr = analyze_nested_array(checks_df, arr, arr, total_checks)
    all_nested_rows.extend(rows)
    array_empty_summary.append({
        "array": arr,
        "null_array": n_null_arr,
        "empty_array": n_empty_arr,
    })

# Print nested detail tables
for arr in nested_arrays:
    arr_rows = [r for r in all_nested_rows if r["dataset"] == f"checks.{arr}"]
    if not arr_rows:
        continue
    print(f"\n  ── checks.{arr} — per-field missingness ──")
    print(f"    {'Field':<30} {'Dtype':<18} {'Records':>12} {'Nulls':>12} {'Null %':>10} {'Distinct':>10}")
    print(f"    {'─'*30} {'─'*18} {'─'*12} {'─'*12} {'─'*10} {'─'*10}")
    for r in arr_rows:
        print(f"    {r['column']:<30} {str(r['dtype']):<18} {r['total_rows']:>12,} {r['null_count']:>12,} {pct_fmt(r['null_pct']):>10} {r['distinct']:>10,}")

# Empty-array summary
print(f"\n  ── Empty / Null Array Summary ──")
print(f"    {'Array':<18} {'Null Arrays':>12} {'Empty Arrays':>14} {'Non-empty':>12}")
print(f"    {'─'*18} {'─'*12} {'─'*14} {'─'*12}")
for a in array_empty_summary:
    ne = total_checks - a["null_array"] - a["empty_array"]
    print(f"    {a['array']:<18} {a['null_array']:>12,} {a['empty_array']:>14,} {ne:>12,}")

# ============================================================================
# 5.3 — FULLY MISSING / LOW-INFORMATION COLUMNS
# ============================================================================
print("\n" + "─" * 80)
print("5.3  FULLY MISSING OR LOW-INFORMATION COLUMNS")
print("─" * 80)

all_rows = all_top_rows + all_nested_rows

fully_null = [r for r in all_rows if r["null_pct"] == 100.0]
over_50 = [r for r in all_rows if 50.0 < r["null_pct"] < 100.0]
single_val = [r for r in all_rows if r["distinct"] == 1 and r["null_pct"] < 100.0]

print(f"\n  ── 100% NULL columns ({len(fully_null)}) ──")
if fully_null:
    print(f"    {'Dataset':<28} {'Column':<35} {'Nulls':>12} {'Rows':>12}")
    print(f"    {'─'*28} {'─'*35} {'─'*12} {'─'*12}")
    for r in fully_null:
        print(f"    {r['dataset']:<28} {r['column']:<35} {r['null_count']:>12,} {r['total_rows']:>12,}")
else:
    print("    (none)")

print(f"\n  ── >50% NULL columns ({len(over_50)}) ──")
if over_50:
    print(f"    {'Dataset':<28} {'Column':<35} {'Null %':>10} {'Distinct':>10}")
    print(f"    {'─'*28} {'─'*35} {'─'*10} {'─'*10}")
    for r in over_50:
        print(f"    {r['dataset']:<28} {r['column']:<35} {pct_fmt(r['null_pct']):>10} {r['distinct']:>10,}")
else:
    print("    (none)")

print(f"\n  ── Single distinct non-null value ({len(single_val)}) ──")
if single_val:
    print(f"    {'Dataset':<28} {'Column':<35} {'Distinct':>10} {'Null %':>10}")
    print(f"    {'─'*28} {'─'*35} {'─'*10} {'─'*10}")
    for r in single_val:
        print(f"    {r['dataset']:<28} {r['column']:<35} {r['distinct']:>10,} {pct_fmt(r['null_pct']):>10}")
else:
    print("    (none)")

# Special attention fields
print(f"\n  ── Previously Observed Fields (special attention) ──")
special_fields = ["LoyaltyId", "ExternalId", "IsEmancipated", "ServiceCharge"]
print(f"    {'Field':<20} {'Found In':<30} {'Nulls':>12} {'Null %':>10} {'Distinct':>10}")
print(f"    {'─'*20} {'─'*30} {'─'*12} {'─'*10} {'─'*10}")
for sf in special_fields:
    found = [r for r in all_rows if r["column"].lower() == sf.lower()]
    if found:
        for r in found:
            print(f"    {r['column']:<20} {r['dataset']:<30} {r['null_count']:>12,} {pct_fmt(r['null_pct']):>10} {r['distinct']:>10,}")
    else:
        print(f"    {sf:<20} {'NOT FOUND':<30} {'':>12} {'':>10} {'':>10}")

# ============================================================================
# 5.4 — KEY FIELDS NULL CHECK
# ============================================================================
print("\n" + "─" * 80)
print("5.4  KEY FIELDS — NULL VALIDATION")
print("─" * 80)

key_field_results = []

def check_key_nulls(df, col_name, label, total_rows=None):
    """Check for nulls in a key column. EmployeeId=0 is NOT counted as null."""
    if col_name not in df.columns:
        key_field_results.append((label, col_name, "NOT FOUND", None, None, "⚠ NOT FOUND"))
        return
    if total_rows is None:
        total_rows = df.count()
    n_null = df.filter(col(col_name).isNull()).count()
    n_zero = 0
    if col_name.lower().endswith("employeeid") or col_name.lower() == "employeeid":
        n_zero = df.filter(col(col_name) == 0).count()
    pct = round(100.0 * n_null / total_rows, 4) if total_rows else 0.0
    status = "✅ NO NULLS" if n_null == 0 else f"❌ {n_null:,} NULLS"
    key_field_results.append((label, col_name, total_rows, n_null, pct, n_zero, status))
    extra = f"  (also {n_zero:,} zero-values, NOT counted as null)" if n_zero > 0 else ""
    print(f"  {label:<52} nulls={n_null:>6,} / {total_rows:>8,}  ({pct_fmt(pct):>10}){extra}")

print("\n  ── checks_df key fields ──")
check_key_nulls(checks_df, "LocationId",      "checks.LocationId")
check_key_nulls(checks_df, "BusinessDate",    "checks.BusinessDate")
check_key_nulls(checks_df, "Id",             "checks.Id")
check_key_nulls(checks_df, "RevenueCenterId", "checks.RevenueCenterId")

print("\n  ── Dimension Id columns ──")
for ds_name in ["employees", "items", "locations", "revenue_centers"]:
    if ds_name not in datasets:
        key_field_results.append((f"{ds_name}.Id", "Id", "NOT FOUND", None, None, "⚠ NOT FOUND"))
        print(f"  {ds_name}.Id  — dimension not loaded")
        continue
    ddf = datasets[ds_name]
    id_col = None
    for cand in ["Id", "id", "ID"]:
        if cand in ddf.columns:
            id_col = cand
            break
    if id_col:
        check_key_nulls(ddf, id_col, f"{ds_name}.{id_col}")
    else:
        key_field_results.append((f"{ds_name}.Id", "(none)", ddf.count(), None, None, "⚠ NO ID COLUMN"))
        print(f"  {ds_name}: no Id-like column found in {ddf.columns}")

print("\n  ── Nested key fields (EmployeeId, ItemId, RevenueCenterId) ──")
for arr in ["ItemsSold", "Discounts", "Payments", "Voids"]:
    if arr not in checks_df.columns:
        print(f"  checks.{arr}: column not found")
        continue
    exploded = checks_df.filter(col(arr).isNotNull() & (size(col(arr)) > 0)).select(explode(col(arr)).alias("x"))
    total_n = exploded.count()
    if total_n == 0:
        print(f"  checks.{arr}: no records")
        continue
    sub_cols = exploded.select("x.*").columns
    # EmployeeId
    for cand in ["EmployeeId", "employeeId", "employee_id", "EmpId"]:
        if cand in sub_cols:
            check_key_nulls(exploded.select(col(f"x.{cand}").alias("EmployeeId")), "EmployeeId", f"{arr}.EmployeeId", total_n)
            break
    # ItemId (only for ItemsSold)
    if arr == "ItemsSold":
        for cand in ["ItemId", "itemId", "item_id"]:
            if cand in sub_cols:
                check_key_nulls(exploded.select(col(f"x.{cand}").alias("ItemId")), "ItemId", f"{arr}.ItemId", total_n)
                break
    # RevenueCenterId
    for cand in ["RevenueCenterId", "revenueCenterId", "revenue_center_id"]:
        if cand in sub_cols:
            check_key_nulls(exploded.select(col(f"x.{cand}").alias("RevenueCenterId")), "RevenueCenterId", f"{arr}.RevenueCenterId", total_n)
            break

# ============================================================================
# 5.5 — MISSINGNESS PATTERNS (by LocationId, by BusinessDate)
# ============================================================================
print("\n" + "─" * 80)
print("5.5  MISSINGNESS PATTERNS")
print("─" * 80)

# Identify top-level check columns with > 0 nulls
check_null_cols = [r["column"] for r in all_top_rows if r["dataset"] == "checks_df" and r["null_count"] > 0]

if not check_null_cols:
    print("\n  No top-level null columns in checks_df — skipping pattern analysis.")
else:
    # --- By LocationId ---
    if "LocationId" in checks_df.columns:
        print("\n  ── Missingness by LocationId (top 15 locations with most null rows) ──")
        loc_pattern = checks_df.groupBy("LocationId").agg(
            count(lit(1)).alias("total_rows"),
            *[_sum(when(col(c).isNull(), 1).otherwise(0)).alias(f"null_{c}") for c in check_null_cols]
        ).orderBy(col("total_rows").desc()).limit(15)
        print(f"    {'LocationId':<15} {'TotalRows':>10} ", end="")
        for c in check_null_cols[:8]:
            print(f"{f'null_{c}':<16}", end="")
        print()
        print(f"    {'─'*15} {'─'*10} " + " ".join(["─"*16 for _ in check_null_cols[:8]]))
        for r in loc_pattern.collect():
            print(f"    {str(r['LocationId']):<15} {r['total_rows']:>10,} ", end="")
            for c in check_null_cols[:8]:
                print(f"{r[f'null_{c}']:>16,}", end=" ")
            print()
        if len(check_null_cols) > 8:
            print(f"    ... and {len(check_null_cols)-8} more null columns not shown")

    # --- By BusinessDate ---
    if "BusinessDate" in checks_df.columns:
        print("\n  ── Missingness by BusinessDate (top 15 dates with most null rows) ──")
        date_pattern = checks_df.groupBy("BusinessDate").agg(
            count(lit(1)).alias("total_rows"),
            *[_sum(when(col(c).isNull(), 1).otherwise(0)).alias(f"null_{c}") for c in check_null_cols]
        ).orderBy(col("total_rows").desc()).limit(15)
        print(f"    {'BusinessDate':<18} {'TotalRows':>10} ", end="")
        for c in check_null_cols[:8]:
            print(f"{f'null_{c}':<16}", end="")
        print()
        print(f"    {'─'*18} {'─'*10} " + " ".join(["─"*16 for _ in check_null_cols[:8]]))
        for r in date_pattern.collect():
            print(f"    {str(r['BusinessDate']):<18} {r['total_rows']:>10,} ", end="")
            for c in check_null_cols[:8]:
                print(f"{r[f'null_{c}']:>16,}", end=" ")
            print()
        if len(check_null_cols) > 8:
            print(f"    ... and {len(check_null_cols)-8} more null columns not shown")

# ============================================================================
# 5.6 — FINAL SUMMARY TABLE
# ============================================================================
print("\n" + "=" * 80)
print("5.6  FINAL MISSING-VALUES SUMMARY TABLE")
print("=" * 80)

# Build interpretations
def interpret(r):
    col_name = r["column"]
    pct = r["null_pct"]
    distinct = r["distinct"]
    ds = r["dataset"]
    if pct == 100.0:
        return "Column fully null — no information; candidate for removal"
    if pct > 50.0:
        return "Majority null — low completeness; investigate before use"
    if distinct == 1 and pct < 100.0:
        return "Single value — low variance; verify if informative"
    if col_name.lower() == "loyaltyid":
        return "Expected nulls — not all customers use loyalty"
    if col_name.lower() == "externalid":
        return "Possible optional identifier — assess coverage"
    if col_name.lower() == "isemancipated":
        return "Check if applicable to all records"
    if col_name.lower() == "servicecharge":
        return "May be null when no service charge applied"
    if col_name.lower() in ["gratuity"]:
        return "Null when no gratuity/tip on payment"
    if col_name.lower().startswith("employee") and pct > 0:
        return "Missing employee reference — verify against dimension"
    if pct == 0:
        return "Complete — no missing values"
    if pct <= 5:
        return "Low nullity — minor impact expected"
    if pct <= 20:
        return "Moderate nullity — evaluate per use case"
    return "High nullity — evaluate necessity before analysis"

summary_rows = []
for r in all_rows:
    sev = severity_label(r["null_pct"])
    interp = interpret(r)
    summary_rows.append({
        "Dataset": r["dataset"],
        "Column": r["column"],
        "Null Count": r["null_count"],
        "Null %": r["null_pct"],
        "Severity": sev,
        "Initial Interpretation": interp,
    })

# Sort by severity order then null % desc
sev_order = {"Complete": 0, "High": 1, "Moderate": 2, "Low": 3, "None": 4}
summary_rows.sort(key=lambda x: (sev_order[x["Severity"]], -x["Null %"], x["Dataset"], x["Column"]))

# Print
print(f"\n  {'Dataset':<24} {'Column':<28} {'Nulls':>10} {'Null %':>10} {'Severity':<12} {'Initial Interpretation'}")
print(f"  {'─'*24} {'─'*28} {'─'*10} {'─'*10} {'─'*12} {'─'*50}")
for s in summary_rows:
    print(f"  {s['Dataset']:<24} {s['Column']:<28} {s['Null Count']:>10,} {pct_fmt(s['Null %']):>10} {s['Severity']:<12} {s['Initial Interpretation']}")

# Severity counts
print(f"\n  ── Severity distribution ──")
sev_counts = {}
for s in summary_rows:
    sev_counts[s["Severity"]] = sev_counts.get(s["Severity"], 0) + 1
print(f"    {'Severity':<14} {'Columns':>10}")
print(f"    {'─'*14} {'─'*10}")
for sev in ["None", "Low", "Moderate", "High", "Complete"]:
    print(f"    {sev:<14} {sev_counts.get(sev, 0):>10}")

print("\n" + "=" * 80)
print("MISSING-VALUES DIAGNOSTIC COMPLETE — no raw data was modified.")
print("=" * 80)

# COMMAND ----------

#La base presenta una alta completitud en las variables transaccionales relevantes para el análisis. Los campos asociados con productos, descuentos, pagos y voids no presentan valores nulos. Los principales problemas de completitud se concentran en tres variables completamente vacías (LoyaltyId, IsEmancipated y ExternalId) y en algunos atributos descriptivos de empleados y ubicaciones.

# COMMAND ----------

# MAGIC %md
# MAGIC Duplicates

# COMMAND ----------

# ============================================================================
# 6. DUPLICATES — DIAGNOSTIC ONLY (read-only, no raw data modified)
# ============================================================================
# Confirmed composite key for checks: LocationId + BusinessDate + Id
# A repeated Id alone is NOT a duplicate.
# Nothing is cleaned, deleted, deduplicated, or modified.
# ============================================================================

from pyspark.sql.functions import (
    col, count, countDistinct, when, sum as _sum, collect_set,
    explode, size, lit, struct, concat_ws, row_number, broadcast
)
from pyspark.sql.window import Window

print("=" * 80)
print("SECTION 6: DUPLICATES — DIAGNOSTIC AUDIT")
print("=" * 80)

# ------------------------------------------------------------------
# Helper: exact duplicate detector for any DataFrame
# ------------------------------------------------------------------
def analyze_exact_duplicates(df, label, total_rows=None):
    """Count exact duplicate rows (all columns identical) and report groups."""
    if total_rows is None:
        total_rows = df.count()
    if total_rows == 0:
        print(f"\n  ── {label}: 0 rows — SKIPPED ──")
        return {"label": label, "total_rows": 0, "exact_dup_groups": 0,
                "rows_in_exact_dups": 0, "pct": 0.0}

    # Group by all columns, count occurrences > 1
    all_cols = df.columns
    grouped = df.groupBy(all_cols).agg(count(lit(1)).alias("_cnt"))
    dup_groups = grouped.filter(col("_cnt") > 1)
    n_dup_groups = dup_groups.count()
    rows_in_dups = dup_groups.agg(_sum(col("_cnt")).alias("r")).collect()[0]["r"] or 0
    pct = round(100.0 * rows_in_dups / total_rows, 4) if total_rows else 0.0

    print(f"\n  ── {label} — Exact row-level duplicates ──")
    print(f"    Total rows            : {total_rows:,}")
    print(f"    Exact dup groups      : {n_dup_groups:,}")
    print(f"    Rows in exact dups    : {rows_in_dups:,}")
    print(f"    % rows in exact dups  : {pct:.4f}%")
    if n_dup_groups > 0:
        print(f"    Sample dup groups (top 5 by count):")
        for r in dup_groups.orderBy(col("_cnt").desc()).limit(5).collect():
            preview = {k: v for k, v in r.asDict().items() if k != "_cnt"}
            print(f"      count={r['_cnt']}  →  {preview}")
    return {"label": label, "total_rows": total_rows, "exact_dup_groups": n_dup_groups,
            "rows_in_exact_dups": rows_in_dups, "pct": pct}


def analyze_composite_key_duplicates(df, key_cols, label, total_rows=None):
    """Find duplicate groups based on a composite key (not exact duplicates)."""
    if total_rows is None:
        total_rows = df.count()
    if total_rows == 0:
        return {"label": label, "key_cols": key_cols, "total_rows": 0,
                "dup_key_groups": 0, "rows_in_dup_keys": 0, "pct": 0.0}

    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        print(f"  ⚠ {label}: key columns {missing} not found — SKIPPED")
        return {"label": label, "key_cols": key_cols, "total_rows": total_rows,
                "dup_key_groups": None, "rows_in_dup_keys": None, "pct": None,
                "note": f"Missing columns: {missing}"}

    grouped = df.groupBy(key_cols).agg(count(lit(1)).alias("_cnt"))
    dup_groups = grouped.filter(col("_cnt") > 1)
    n_dup_groups = dup_groups.count()
    rows_in_dup = dup_groups.agg(_sum(col("_cnt")).alias("r")).collect()[0]["r"] or 0
    pct = round(100.0 * rows_in_dup / total_rows, 4) if total_rows else 0.0

    print(f"\n  ── {label} — Composite-key duplicates ({', '.join(key_cols)}) ──")
    print(f"    Total rows            : {total_rows:,}")
    print(f"    Dup key groups        : {n_dup_groups:,}")
    print(f"    Rows in dup keys      : {rows_in_dup:,}")
    print(f"    % rows in dup keys    : {pct:.4f}%")
    if n_dup_groups > 0:
        print(f"    Sample dup key groups (top 10 by count):")
        for r in dup_groups.orderBy(col("_cnt").desc()).limit(10).collect():
            preview = {k: r[k] for k in key_cols}
            print(f"      count={r['_cnt']}  →  {preview}")
    return {"label": label, "key_cols": key_cols, "total_rows": total_rows,
            "dup_key_groups": n_dup_groups, "rows_in_dup_keys": rows_in_dup, "pct": pct}


def analyze_business_field_duplicates(df, fields, label, total_rows=None):
    """Check repeated business fields — distinguish from exact duplicates."""
    if total_rows is None:
        total_rows = df.count()
    print(f"\n  ── {label} — Business-field repetition (not necessarily duplicates) ──")
    print(f"    Total rows: {total_rows:,}")
    for f in fields:
        if f not in df.columns:
            print(f"    {f:<22}: NOT FOUND")
            continue
        non_null = df.filter(col(f).isNotNull()).count()
        distinct = df.select(f).distinct().count()
        repeated_values = non_null - distinct
        # how many groups have count > 1
        grouped = df.groupBy(f).agg(count(lit(1)).alias("_cnt"))
        dup_groups = grouped.filter((col("_cnt") > 1) & col(f).isNotNull()).count()
        rows_in_dup = grouped.filter((col("_cnt") > 1) & col(f).isNotNull()).agg(_sum(col("_cnt")).alias("r")).collect()[0]["r"] or 0
        pct = round(100.0 * rows_in_dup / non_null, 4) if non_null else 0.0
        print(f"    {f:<22}: distinct={distinct:>8,}  repeated_values={repeated_values:>8,}  "
              f"dup_groups={dup_groups:>6,}  rows_in_reps={rows_in_dup:>8,}  ({pct:.4f}%)")
        # show top 5 most repeated values
        if dup_groups > 0:
            top_reps = grouped.filter((col("_cnt") > 1) & col(f).isNotNull()).orderBy(col("_cnt").desc()).limit(5).collect()
            for r in top_reps:
                print(f"        ↳ value={str(r[f])[:40]:<40} count={r['_cnt']:,}")


# ------------------------------------------------------------------
# Gather dimension DataFrames (reuse from Section 5 if available)
# ------------------------------------------------------------------
datasets = {"checks_df": checks_df}
for logical_name, candidates in [
    ("employees",       ["employees", "employee", "emp", "employee_dim"]),
    ("items",           ["items", "item", "menu_items", "item_dim"]),
    ("locations",       ["locations", "location", "restaurant", "location_dim", "restaurants"]),
    ("revenue_centers", ["revenue_centers", "revenue_center", "revenuecenter", "revenue_centers_dim", "rc"]),
]:
    matched_key, matched_df = None, None
    if 'dim_matches' in dir() and isinstance(dim_matches, dict) and dim_matches.get(logical_name):
        matched_df = dim_matches[logical_name]["df"]
        matched_key = dim_matches[logical_name]["key"]
    else:
        for k in candidates:
            if k in dim_dfs:
                matched_key, matched_df = k, dim_dfs[k]
                break
        if matched_df is None:
            for dk in dim_dfs:
                for k in candidates:
                    if k in dk or dk in k:
                        matched_key, matched_df = dk, dim_dfs[dk]
                        break
                if matched_df is not None:
                    break
    if matched_df is not None:
        datasets[logical_name] = matched_df
        print(f"  Dimension '{logical_name}' loaded from dim_dfs['{matched_key}']")
    else:
        print(f"  ⚠ Dimension '{logical_name}' not found in dim_dfs")

# Store all results for final summary
all_results = []

# ============================================================================
# 6.1 — CHECKS: ROW-LEVEL DUPLICATES
# ============================================================================
print("\n" + "─" * 80)
print("6.1  CHECKS — ROW-LEVEL DUPLICATES")
print("─" * 80)

total_checks = checks_df.count()
print(f"\n  Total checks rows: {total_checks:,}")

# 6.1a — Exact duplicate rows (all top-level columns identical)
print(f"\n  ── 6.1a Exact duplicate rows (all columns) ──")
exact_res = analyze_exact_duplicates(checks_df, "checks_df (exact)", total_checks)
all_results.append({
    "dataset": "checks_df", "test": "Exact row-level duplicates",
    "dup_groups": exact_res["exact_dup_groups"], "rows_involved": exact_res["rows_in_exact_dups"],
    "pct": exact_res["pct"],
    "interpretation": ("No duplicate issue" if exact_res["exact_dup_groups"] == 0
                        else "Potential technical duplicate — exact identical rows")
})

# 6.1b — Composite key: LocationId + BusinessDate + Id
COMPOSITE_KEY = ["LocationId", "BusinessDate", "Id"]
print(f"\n  ── 6.1b Composite-key duplicates ({' + '.join(COMPOSITE_KEY)}) ──")
key_res = analyze_composite_key_duplicates(checks_df, COMPOSITE_KEY, "checks_df (composite key)", total_checks)
all_results.append({
    "dataset": "checks_df", "test": f"Duplicate composite key ({' + '.join(COMPOSITE_KEY)})",
    "dup_groups": key_res["dup_key_groups"], "rows_involved": key_res["rows_in_dup_keys"],
    "pct": key_res["pct"],
    "interpretation": ("No duplicate issue" if key_res["dup_key_groups"] == 0
                        else "Potential technical duplicate — same composite key in multiple rows")
})

# 6.1c — Id-only repetition (for comparison — NOT a duplicate indicator)
print(f"\n  ── 6.1c Id-only repetition (NOT classified as duplicates) ──")
id_grouped = checks_df.groupBy("Id").agg(count(lit(1)).alias("_cnt"))
id_dup_groups = id_grouped.filter(col("_cnt") > 1).count()
id_rows_in_dup = id_grouped.filter(col("_cnt") > 1).agg(_sum(col("_cnt")).alias("r")).collect()[0]["r"] or 0
id_pct = round(100.0 * id_rows_in_dup / total_checks, 4) if total_checks else 0.0
print(f"    Repeated Id groups     : {id_dup_groups:,}")
print(f"    Rows with repeated Id  : {id_rows_in_dup:,}  ({id_pct:.4f}%)")
print(f"    ⚠ These are NOT duplicates — Id repeats across locations/dates by design.")
all_results.append({
    "dataset": "checks_df", "test": "Repeated Id (NOT a duplicate — different LocationId/Date)",
    "dup_groups": id_dup_groups, "rows_involved": id_rows_in_dup, "pct": id_pct,
    "interpretation": "Legitimate repetition likely — Id is not unique without LocationId+BusinessDate"
})

# ============================================================================
# 6.2 — DIMENSION DATASETS
# ============================================================================
print("\n" + "─" * 80)
print("6.2  DIMENSION DATASETS — DUPLICATES")
print("─" * 80)

dim_business_fields = {
    "employees":       ["PayrollEmployeeId", "LocationEmployeeId", "Name"],
    "items":           ["Name", "Plu"],
    "locations":       ["Name"],
    "revenue_centers": ["Name"],
}

for ds_name in ["employees", "items", "locations", "revenue_centers"]:
    if ds_name not in datasets:
        print(f"\n  ⚠ {ds_name}: dimension not loaded — SKIPPED")
        all_results.append({
            "dataset": ds_name, "test": "(not loaded)", "dup_groups": None,
            "rows_involved": None, "pct": None, "interpretation": "N/A — dimension not loaded"
        })
        continue

    ddf = datasets[ds_name]
    total_dim = ddf.count()
    print(f"\n  {'═' * 70}")
    print(f"  {ds_name.upper()}  ({total_dim:,} rows)")
    print(f"  {'═' * 70}")

    # Exact duplicates
    exact_res = analyze_exact_duplicates(ddf, ds_name, total_dim)
    all_results.append({
        "dataset": ds_name, "test": "Exact row-level duplicates",
        "dup_groups": exact_res["exact_dup_groups"], "rows_involved": exact_res["rows_in_exact_dups"],
        "pct": exact_res["pct"],
        "interpretation": ("No duplicate issue" if exact_res["exact_dup_groups"] == 0
                            else "Potential technical duplicate — exact identical rows")
    })

    # Duplicate Id values
    id_col = None
    for cand in ["Id", "id", "ID"]:
        if cand in ddf.columns:
            id_col = cand
            break
    if id_col:
        print(f"\n  ── {ds_name}.{id_col} — Duplicate ID values ──")
        id_grouped = ddf.groupBy(id_col).agg(count(lit(1)).alias("_cnt"))
        dup_id_groups = id_grouped.filter((col("_cnt") > 1) & col(id_col).isNotNull()).count()
        rows_in_dup_ids = id_grouped.filter((col("_cnt") > 1) & col(id_col).isNotNull()).agg(_sum(col("_cnt")).alias("r")).collect()[0]["r"] or 0
        pct_ids = round(100.0 * rows_in_dup_ids / total_dim, 4) if total_dim else 0.0
        print(f"    Dup ID groups   : {dup_id_groups:,}")
        print(f"    Rows in dup IDs : {rows_in_dup_ids:,}  ({pct_ids:.4f}%)")
        if dup_id_groups > 0:
            print(f"    Sample (top 5):")
            for r in id_grouped.filter((col("_cnt") > 1) & col(id_col).isNotNull()).orderBy(col("_cnt").desc()).limit(5).collect():
                print(f"      {id_col}={r[id_col]}  count={r['_cnt']:,}")
        all_results.append({
            "dataset": ds_name, "test": f"Duplicate {id_col} values",
            "dup_groups": dup_id_groups, "rows_involved": rows_in_dup_ids, "pct": pct_ids,
            "interpretation": ("No duplicate issue" if dup_id_groups == 0
                                else "Potential technical duplicate — same ID in multiple rows")
        })
    else:
        print(f"\n  ── {ds_name}: no Id-like column found ──")
        all_results.append({
            "dataset": ds_name, "test": "Duplicate Id (no Id column)",
            "dup_groups": None, "rows_involved": None, "pct": None,
            "interpretation": "N/A — no Id column found"
        })

    # Business field repetition
    biz_fields = dim_business_fields.get(ds_name, [])
    available_biz = [f for f in biz_fields if f in ddf.columns]
    if available_biz:
        analyze_business_field_duplicates(ddf, available_biz, ds_name, total_dim)
        for f in available_biz:
            grouped = ddf.groupBy(f).agg(count(lit(1)).alias("_cnt"))
            dup_g = grouped.filter((col("_cnt") > 1) & col(f).isNotNull()).count()
            all_results.append({
                "dataset": ds_name, "test": f"Repeated {f} (business field)",
                "dup_groups": dup_g, "rows_involved": None, "pct": None,
                "interpretation": ("No repetition" if dup_g == 0
                                    else "Legitimate repetition likely — verify if exact duplicates exist")
            })
    else:
        print(f"\n  ── {ds_name}: no business fields found among {biz_fields} ──")

# ============================================================================
# 6.3 — NESTED TRANSACTION ARRAYS
# ============================================================================
print("\n" + "─" * 80)
print("6.3  NESTED TRANSACTION ARRAYS — DUPLICATES WITHIN PARENT CHECK")
print("─" * 80)

nested_arrays = ["ItemsSold", "Discounts", "Payments", "Voids", "Tax"]
COMPOSITE_KEY = ["LocationId", "BusinessDate", "Id"]

for arr in nested_arrays:
    print(f"\n  {'═' * 70}")
    print(f"  checks.{arr}")
    print(f"  {'═' * 70}")

    if arr not in checks_df.columns:
        print(f"    ⚠ Column '{arr}' not found in checks_df — SKIPPED")
        all_results.append({
            "dataset": f"checks.{arr}", "test": "Nested duplicates (column not found)",
            "dup_groups": None, "rows_involved": None, "pct": None,
            "interpretation": "N/A — column not found"
        })
        continue

    # Explode with parent key preserved
    exploded = (checks_df
                .filter(col(arr).isNotNull() & (size(col(arr)) > 0))
                .select(*[col(c).alias(f"_parent_{c}") for c in COMPOSITE_KEY],
                        explode(col(arr)).alias("_rec")))

    total_nested = exploded.count()
    if total_nested == 0:
        print(f"    No nested records — SKIPPED")
        all_results.append({
            "dataset": f"checks.{arr}", "test": "Nested exact duplicates",
            "dup_groups": 0, "rows_involved": 0, "pct": 0.0,
            "interpretation": "No data — empty array"
        })
        continue

    print(f"    Total nested records: {total_nested:,}")

    # Get sub-field names
    sub_cols = exploded.select("_rec.*").columns
    print(f"    Sub-fields: {sub_cols}")

    # Build a flattened DataFrame with parent key + all sub-fields
    flat = exploded.select(
        *[col(f"_parent_{c}").alias(c) for c in COMPOSITE_KEY],
        *[col(f"_rec.{sc}").alias(sc) for sc in sub_cols]
    )

    # 6.3a — Exact duplicate nested records within the same parent check
    all_flat_cols = COMPOSITE_KEY + sub_cols
    grouped = flat.groupBy(all_flat_cols).agg(count(lit(1)).alias("_cnt"))
    dup_within_check = grouped.filter(col("_cnt") > 1)
    n_dup_groups = dup_within_check.count()
    rows_in_dup = dup_within_check.agg(_sum(col("_cnt")).alias("r")).collect()[0]["r"] or 0
    pct_dup = round(100.0 * rows_in_dup / total_nested, 4) if total_nested else 0.0

    # Count distinct parent checks that contain at least one duplicate group
    checks_with_dups = dup_within_check.select("LocationId", "BusinessDate", "Id").distinct().count() if n_dup_groups > 0 else 0

    print(f"\n    ── Exact duplicate records within same parent check ──")
    print(f"      Dup groups (within check)     : {n_dup_groups:,}")
    print(f"      Nested records in dup groups   : {rows_in_dup:,}")
    print(f"      % of nested records involved   : {pct_dup:.4f}%")
    print(f"      Parent checks containing dups   : {checks_with_dups:,}")

    if n_dup_groups > 0:
        print(f"      Sample dup groups (top 5):")
        for r in dup_within_check.orderBy(col("_cnt").desc()).limit(5).collect():
            preview = {k: r[k] for k in COMPOSITE_KEY}
            rec_preview = {k: r[k] for k in sub_cols if k in r.asDict()}
            print(f"        count={r['_cnt']}  parent={preview}  record={rec_preview}")

    all_results.append({
        "dataset": f"checks.{arr}", "test": "Exact duplicate nested records (within same check)",
        "dup_groups": n_dup_groups, "rows_involved": rows_in_dup, "pct": pct_dup,
        "interpretation": ("No duplicate issue" if n_dup_groups == 0
                            else "Potential technical duplicate OR legitimate repetition — investigate per business rules")
    })

    # 6.3b — For ItemsSold: test if Sequence or line identifier distinguishes repeated items
    if arr == "ItemsSold":
        print(f"\n    ── ItemsSold: Sequence / line-identifier analysis ──")
        line_id_cands = [c for c in sub_cols if c.lower() in ["sequence", "seq", "lineno", "linenumber", "order", "orderindex"]]
        if line_id_cands:
            line_id = line_id_cands[0]
            print(f"      Found line identifier: '{line_id}'")
            # Check if adding the line identifier to the grouping eliminates duplicates
            extended_key = COMPOSITE_KEY + [line_id]
            ext_grouped = flat.groupBy(extended_key + [c for c in sub_cols if c != line_id]).agg(count(lit(1)).alias("_cnt"))
            ext_dups = ext_grouped.filter(col("_cnt") > 1).count()
            print(f"      Dup groups with {line_id} included: {ext_dups:,}")
            if ext_dups < n_dup_groups:
                print(f"      ✅ Adding '{line_id}' reduces duplicates: {n_dup_groups:,} → {ext_dups:,}")
                print(f"         Many repeated items may be distinguished by {line_id}.")
                all_results.append({
                    "dataset": f"checks.{arr}", "test": f"Dup groups reduced by including {line_id}",
                    "dup_groups": ext_dups, "rows_involved": None, "pct": None,
                    "interpretation": f"Sequence/line-id '{line_id}' distinguishes many repeated items — legitimate repetition likely"
                })
            else:
                print(f"      ⚠ '{line_id}' does NOT reduce duplicates — repeated items may be true duplicates.")
                all_results.append({
                    "dataset": f"checks.{arr}", "test": f"Dup groups with {line_id} included",
                    "dup_groups": ext_dups, "rows_involved": None, "pct": None,
                    "interpretation": "Line identifier does not resolve duplicates — uncertain / requires investigation"
                })
        else:
            print(f"      No line identifier found among sub-fields: {sub_cols}")
            print(f"      Cannot distinguish repeated items by sequence — treat as potential duplicates.")
            all_results.append({
                "dataset": f"checks.{arr}", "test": "No line identifier available",
                "dup_groups": None, "rows_involved": None, "pct": None,
                "interpretation": "Uncertain / requires investigation — no Sequence or line-id field to distinguish items"
            })

    # 6.3c — Cross-check duplicates: same record across DIFFERENT checks (NOT duplicates)
    cross_grouped = flat.groupBy(sub_cols).agg(
        count(lit(1)).alias("_cnt"),
        countDistinct("LocationId", "BusinessDate", "Id").alias("_distinct_checks")
    )
    cross_dups = cross_grouped.filter((col("_cnt") > 1) & (col("_distinct_checks") > 1))
    n_cross = cross_dups.count()
    if n_cross > 0:
        print(f"\n    ── Identical records across DIFFERENT checks (NOT duplicates) ──")
        print(f"      Records appearing in multiple checks: {n_cross:,} groups")
        print(f"      ⚠ These are NOT duplicates — same item/payment can legitimately appear in different checks.")

# ============================================================================
# 6.4 — FINAL SUMMARY TABLE
# ============================================================================
print("\n" + "=" * 80)
print("6.4  FINAL DUPLICATES SUMMARY TABLE")
print("=" * 80)

print(f"\n  {'Dataset':<26} {'Duplicate Test':<50} {'Groups':>8} {'Rows':>10} {'% Rows':>10}  Interpretation")
print(f"  {'─'*26} {'─'*50} {'─'*8} {'─'*10} {'─'*10}  {'─'*50}")

for r in all_results:
    ds = str(r["dataset"])[:25]
    test = str(r["test"])[:49]
    groups = f"{r['dup_groups']:,}" if r["dup_groups"] is not None else "N/A"
    rows = f"{r['rows_involved']:,}" if r["rows_involved"] is not None else "N/A"
    pct = f"{r['pct']:.4f}%" if r["pct"] is not None else "N/A"
    interp = str(r["interpretation"])[:80]
    print(f"  {ds:<26} {test:<50} {groups:>8} {rows:>10} {pct:>10}  {interp}")

print(f"\n  Legend:")
print(f"    • 'No duplicate issue'           → no duplicates detected")
print(f"    • 'Potential technical duplicate'  → exact identical rows found; investigate")
print(f"    • 'Legitimate repetition likely'   → repeated values explained by business logic")
print(f"    • 'Uncertain / requires investigation' → needs manual review")
print(f"\n  ⚠ No data was cleaned, deleted, deduplicated, or modified. This audit is diagnostic only.")

# COMMAND ----------

#Insights: El 30.67% de los registros de voids pertenece a grupos de eventos idénticos repetidos dentro de una misma transacción. Aunque estas repeticiones pueden responder a operaciones legítimas, su concentración es relevante para el análisis de comportamientos potencialmente anómalos y debe conservarse para etapas posteriores.

# COMMAND ----------

# MAGIC %md
# MAGIC Data Types

# COMMAND ----------

# ============================================================================
# 7. DATA TYPES — DIAGNOSTIC ONLY (read-only, no raw data modified)
# ============================================================================
# This step is diagnostic only. No casting, overwriting, cleaning, or
# transforming of any raw data is performed.
# It checks whether each variable is stored with a type suitable for analysis.
# ============================================================================

from pyspark.sql.functions import (
    col, count, when, lit, isnan, isnull, min as _min, max as _max,
    collect_list, size, explode, struct, to_date, to_timestamp,
    date_format, trim, length, regexp_replace, lower, upper, desc, asc
)
from pyspark.sql.types import (
    StringType, IntegerType, LongType, DoubleType, DecimalType,
    FloatType, BooleanType, DateType, TimestampType, ArrayType, StructType
)
from datetime import datetime
import re

print("=" * 80)
print("SECTION 7: DATA TYPES — DIAGNOSTIC AUDIT")
print("=" * 80)

# ------------------------------------------------------------------
# Helper: gather dimension DataFrames (reuse logic from Section 5/6)
# ------------------------------------------------------------------
datasets = {"checks_df": checks_df}
for logical_name, candidates in [
    ("employees",       ["employees", "employee", "emp", "employee_dim"]),
    ("items",           ["items", "item", "menu_items", "item_dim"]),
    ("locations",       ["locations", "location", "restaurant", "location_dim", "restaurants"]),
    ("revenue_centers", ["revenue_centers", "revenue_center", "revenuecenter", "revenue_centers_dim", "rc"]),
]:
    matched_key, matched_df = None, None
    if 'dim_matches' in dir() and isinstance(dim_matches, dict) and dim_matches.get(logical_name):
        matched_df = dim_matches[logical_name]["df"]
        matched_key = dim_matches[logical_name]["key"]
    else:
        for k in candidates:
            if k in dim_dfs:
                matched_key, matched_df = k, dim_dfs[k]
                break
        if matched_df is None:
            for dk in dim_dfs:
                for k in candidates:
                    if k in dk or dk in k:
                        matched_key, matched_df = dk, dim_dfs[dk]
                        break
                if matched_df is not None:
                    break
    if matched_df is not None:
        datasets[logical_name] = matched_df
        print(f"  Dimension '{logical_name}' loaded from dim_dfs['{matched_key}']")
    else:
        print(f"  ⚠ Dimension '{logical_name}' not found in dim_dfs")

# ------------------------------------------------------------------
# Helper: get Spark type label
# ------------------------------------------------------------------
def type_label(dtype):
    """Return a human-readable label for a Spark data type."""
    if isinstance(dtype, StringType):       return "string"
    if isinstance(dtype, IntegerType):      return "int"
    if isinstance(dtype, LongType):         return "long"
    if isinstance(dtype, DoubleType):      return "double"
    if isinstance(dtype, FloatType):        return "float"
    if isinstance(dtype, DecimalType):     return f"decimal({dtype.precision},{dtype.scale})"
    if isinstance(dtype, BooleanType):     return "boolean"
    if isinstance(dtype, DateType):         return "date"
    if isinstance(dtype, TimestampType):   return "timestamp"
    if isinstance(dtype, ArrayType):
        return f"array<{type_label(dtype.elementType)}>"
    if isinstance(dtype, StructType):
        return f"struct<{', '.join(f'{f.name}:{type_label(f.dataType)}' for f in dtype.fields)}>"
    return str(dtype)

# ------------------------------------------------------------------
# Helper: recommend analytical type
# ------------------------------------------------------------------
def recommend_type(col_name, dtype):
    """Heuristic recommendation for analytical type based on column name and type."""
    cname = col_name.lower()
    # Date/time fields
    date_fields = ["businessdate", "dateofbirth", "hiredate", "opendate", "closeddate"]
    time_fields = ["opentime", "closetime", "ordertime"]
    if any(f in cname for f in date_fields):
        if isinstance(dtype, DateType): return "date (already correct)"
        if isinstance(dtype, TimestampType): return "date (consider extracting date part)"
        return "date"
    if any(f in cname for f in time_fields):
        if isinstance(dtype, TimestampType): return "timestamp"
        if isinstance(dtype, StringType): return "timestamp or string (time-of-day)"
        return "timestamp"
    # Monetary fields
    money_fields = ["amount", "grossprice", "soldprice", "gratuity", "servicecharge"]
    if any(f in cname for f in money_fields):
        if isinstance(dtype, (DoubleType, FloatType)):
            return "decimal(18,2) (monetary — prefer decimal over double)"
        if isinstance(dtype, DecimalType):
            return f"decimal (already correct — {dtype.precision},{dtype.scale})"
        return "decimal(18,2)"
    # Quantity fields
    if "quantity" in cname:
        if isinstance(dtype, (IntegerType, LongType)): return "integer (already correct)"
        if isinstance(dtype, (DoubleType, FloatType, DecimalType)): return "integer (if whole numbers only) or decimal"
        return "integer or decimal"
    # ID fields
    if cname in ["id", "locationid", "employeeid", "managerid", "itemid",
                 "revenuecenterid", "discountid", "paymentcategoryid",
                 "voidcategoryid", "payrollemployeeid", "locationemployeeid"] or "id" == cname[-2:] or cname.endswith("id"):
        if isinstance(dtype, (IntegerType, LongType)):
            return "string (categorical — IDs are labels, not quantities)"
        if isinstance(dtype, StringType):
            return "string (already correct)"
        return "string (categorical)"
    # PLU fields
    if "plu" in cname:
        return "string (PLU codes are categorical)"
    # Boolean fields
    if isinstance(dtype, BooleanType):
        return "boolean (already correct)"
    # Traffic count
    if "trafficcount" in cname:
        if isinstance(dtype, (IntegerType, LongType)): return "integer (already correct)"
        return "integer"
    # Name fields
    if cname in ["name"] or cname.endswith("name"):
        return "string (already correct)"
    # Default
    return type_label(dtype) + " (review needed)"

# ------------------------------------------------------------------
# Helper: check if action is needed
# ------------------------------------------------------------------
def action_needed(col_name, dtype):
    """Return 'no change needed', 'recommended conversion', 'required conversion', or 'uncertain'."""
    rec = recommend_type(col_name, dtype)
    if "already correct" in rec:
        return "no change needed"
    if any(k in rec for k in ["consider", "if whole numbers", "review needed"]):
        return "uncertain"
    if "prefer decimal" in rec or "categorical" in rec:
        return "recommended conversion"
    if rec.startswith("date") and not isinstance(dtype, DateType):
        return "required conversion"
    if rec.startswith("timestamp") and isinstance(dtype, StringType):
        return "required conversion"
    return "recommended conversion"

# ------------------------------------------------------------------
# Helper: get representative non-null examples
# ------------------------------------------------------------------
def get_examples(df, col_name, n=5):
    """Get n representative non-null, non-empty examples from a column."""
    try:
        rows = (df
                .filter(col(col_name).isNotNull())
                .select(col_name)
                .distinct()
                .limit(n)
                .collect())
        return [r[col_name] for r in rows if r[col_name] is not None]
    except Exception:
        return []

# ------------------------------------------------------------------
# Helper: test date parsing with observed format
# ------------------------------------------------------------------
def test_date_parsing(df, col_name, fmt, exclude_nulls=True):
    """Test whether all non-null values can be parsed using the given format.
    Returns (total_non_null, parse_failures, pct_fail)."""
    try:
        total = df.filter(col(col_name).isNotNull()).count()
        if total == 0:
            return 0, 0, 0.0
        # Try parsing: to_date returns null on parse failure
        parsed = df.filter(col(col_name).isNotNull()).withColumn("_parsed", to_date(col(col_name), fmt))
        failures = parsed.filter(col("_parsed").isNull()).count()
        pct = round(100.0 * failures / total, 4) if total else 0.0
        return total, failures, pct
    except Exception as e:
        return -1, -1, -1.0

def test_timestamp_parsing(df, col_name, fmt, exclude_nulls=True):
    """Test whether all non-null values can be parsed as timestamp with the given format."""
    try:
        total = df.filter(col(col_name).isNotNull()).count()
        if total == 0:
            return 0, 0, 0.0
        parsed = df.filter(col(col_name).isNotNull()).withColumn("_parsed", to_timestamp(col(col_name), fmt))
        failures = parsed.filter(col("_parsed").isNull()).count()
        pct = round(100.0 * failures / total, 4) if total else 0.0
        return total, failures, pct
    except Exception as e:
        return -1, -1, -1.0

def infer_date_format(examples):
    """Infer likely date format from example values."""
    if not examples:
        return None
    for ex in examples:
        s = str(ex)
        # Try common date patterns
        patterns = [
            ("yyyy-MM-dd", r"^\d{4}-\d{2}-\d{2}$"),
            ("MM/dd/yyyy", r"^\d{2}/\d{2}/\d{4}$"),
            ("dd/MM/yyyy", r"^\d{2}/\d{2}/\d{4}$"),
            ("yyyy-MM-dd HH:mm:ss", r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"),
            ("MM/dd/yyyy HH:mm:ss", r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}$"),
            ("yyyy-MM", r"^\d{4}-\d{2}$"),
        ]
        for fmt, pat in patterns:
            if re.match(pat, s):
                return fmt
    return None

def infer_timestamp_format(examples):
    """Infer likely timestamp format from example values."""
    if not examples:
        return None
    for ex in examples:
        s = str(ex)
        patterns = [
            ("yyyy-MM-dd HH:mm:ss", r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"),
            ("yyyy-MM-dd HH:mm:ss.SSS", r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{1,3}$"),
            ("HH:mm:ss", r"^\d{2}:\d{2}:\d{2}$"),
            ("HH:mm", r"^\d{2}:\d{2}$"),
            ("MM/dd/yyyy HH:mm:ss", r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}$"),
        ]
        for fmt, pat in patterns:
            if re.match(pat, s):
                return fmt
    return None

# ------------------------------------------------------------------
# Storage for summary
# ------------------------------------------------------------------
type_inventory = []
date_time_results = []
numeric_results = []
identifier_results = []
boolean_categorical_results = []
final_summary = []

# ============================================================================
# 7.1 — CURRENT DATA TYPES: COMPLETE INVENTORY
# ============================================================================
print("\n" + "═" * 80)
print("7.1  CURRENT DATA TYPES — COMPLETE INVENTORY")
print("═" * 80)

# --- Top-level columns of each dataset ---
for ds_name in ["checks_df", "employees", "items", "locations", "revenue_centers"]:
    if ds_name not in datasets:
        print(f"\n  ⚠ {ds_name}: not loaded — SKIPPED")
        continue
    df = datasets[ds_name]
    print(f"\n  ── {ds_name} ({df.count():,} rows) ──")
    print(f"  {'Column':<30} {'Current Type':<35} {'Recommended Analytical Type':<45} {'Action Needed'}")
    print(f"  {'-'*30} {'-'*35} {'-'*45} {'-'*25}")
    for field in df.schema.fields:
        cname = field.name
        dtype = field.dataType
        cur_type = type_label(dtype)
        rec = recommend_type(cname, dtype)
        action = action_needed(cname, dtype)
        print(f"  {cname:<30} {cur_type:<35} {rec:<45} {action}")
        type_inventory.append({
            "dataset": ds_name, "column": cname,
            "current_type": cur_type, "recommended_type": rec,
            "action_needed": action
        })

# --- Nested fields inside checks_df arrays ---
print("\n  ── Nested array fields in checks_df ──")
nested_arrays = ["ItemsSold", "Discounts", "Payments", "Voids", "Tax"]
for arr in nested_arrays:
    if arr not in checks_df.columns:
        print(f"\n    ⚠ Array '{arr}' not found — SKIPPED")
        continue
    arr_type = checks_df.schema[arr].dataType
    if not isinstance(arr_type, ArrayType) or not isinstance(arr_type.elementType, StructType):
        print(f"\n    ⚠ '{arr}' is not an array of structs — SKIPPED")
        continue
    print(f"\n    ── checks.{arr} (array<struct>) ──")
    print(f"    {'Sub-field':<30} {'Current Type':<35} {'Recommended Analytical Type':<45} {'Action Needed'}")
    print(f"    {'-'*30} {'-'*35} {'-'*45} {'-'*25}")
    for field in arr_type.elementType.fields:
        cname = field.name
        dtype = field.dataType
        cur_type = type_label(dtype)
        rec = recommend_type(cname, dtype)
        action = action_needed(cname, dtype)
        print(f"    {cname:<30} {cur_type:<35} {rec:<45} {action}")
        type_inventory.append({
            "dataset": f"checks.{arr}", "column": cname,
            "current_type": cur_type, "recommended_type": rec,
            "action_needed": action
        })

# ============================================================================
# 7.2 — DATE AND TIME FIELDS
# ============================================================================
print("\n" + "═" * 80)
print("7.2  DATE AND TIME FIELDS — DETAILED INSPECTION")
print("═" * 80)

# Define date/time fields to inspect
# (dataset_name, column_name, is_nested, parent_array, is_time_field)
date_time_fields = [
    ("checks_df", "BusinessDate", False, None, False),
    ("checks_df", "OpenTime", False, None, True),
    ("checks_df", "CloseTime", False, None, True),
    ("employees", "DateOfBirth", False, None, False),
    ("employees", "HireDate", False, None, False),
    ("locations", "OpenDate", False, None, False),
    ("locations", "ClosedDate", False, None, False),
    # Nested: ItemsSold.OrderTime
    ("checks_df", "OrderTime", True, "ItemsSold", True),
]

for ds_name, col_name, is_nested, parent_arr, is_time in date_time_fields:
    label = f"{ds_name}.{col_name}" if not is_nested else f"{ds_name}.{parent_arr}.{col_name}"
    print(f"\n  {'─' * 70}")
    print(f"  ── {label} ──")
    print(f"  {'─' * 70}")

    # Get the DataFrame / column
    if is_nested:
        if parent_arr not in checks_df.columns:
            print(f"    ⚠ Parent array '{parent_arr}' not found — SKIPPED")
            date_time_results.append({
                "field": label, "current_type": "N/A",
                "examples": [], "observed_format": None,
                "total_non_null": 0, "parse_failures": 0, "pct_fail": 0.0,
                "recommendation": "N/A — parent array not found",
                "decision": "N/A"
            })
            continue
        arr_type = checks_df.schema[parent_arr].dataType
        if not isinstance(arr_type, ArrayType) or not isinstance(arr_type.elementType, StructType):
            print(f"    ⚠ Parent array '{parent_arr}' is not array<struct> — SKIPPED")
            continue
        sub_fields = arr_type.elementType.fields
        if col_name not in [f.name for f in sub_fields]:
            print(f"    ⚠ Field '{col_name}' not found in {parent_arr} — SKIPPED")
            date_time_results.append({
                "field": label, "current_type": "N/A",
                "examples": [], "observed_format": None,
                "total_non_null": 0, "parse_failures": 0, "pct_fail": 0.0,
                "recommendation": "N/A — field not found",
                "decision": "N/A"
            })
            continue
        dtype = [f.dataType for f in sub_fields if f.name == col_name][0]
        # Explode to get examples
        exploded = (checks_df
                    .filter(col(parent_arr).isNotNull() & (size(col(parent_arr)) > 0))
                    .select(explode(col(parent_arr)).alias("_rec")))
        examples = get_examples(exploded.select(col(f"_rec.{col_name}").alias(col_name)), col_name, 5)
        total_non_null = exploded.filter(col(f"_rec.{col_name}").isNotNull()).count()
    else:
        if ds_name not in datasets:
            print(f"    ⚠ Dataset '{ds_name}' not loaded — SKIPPED")
            date_time_results.append({
                "field": label, "current_type": "N/A",
                "examples": [], "observed_format": None,
                "total_non_null": 0, "parse_failures": 0, "pct_fail": 0.0,
                "recommendation": "N/A — dataset not loaded",
                "decision": "N/A"
            })
            continue
        df = datasets[ds_name]
        if col_name not in df.columns:
            print(f"    ⚠ Column '{col_name}' not found in {ds_name} — SKIPPED")
            date_time_results.append({
                "field": label, "current_type": "N/A",
                "examples": [], "observed_format": None,
                "total_non_null": 0, "parse_failures": 0, "pct_fail": 0.0,
                "recommendation": "N/A — column not found",
                "decision": "N/A"
            })
            continue
        dtype = df.schema[col_name].dataType
        examples = get_examples(df, col_name, 5)
        total_non_null = df.filter(col(col_name).isNotNull()).count()

    cur_type = type_label(dtype)
    print(f"    Current type     : {cur_type}")
    print(f"    Non-null count   : {total_non_null:,}")
    print(f"    Examples (up to 5): {examples}")

    # Infer format
    if is_time:
        observed_fmt = infer_timestamp_format(examples)
    else:
        observed_fmt = infer_date_format(examples)
    print(f"    Observed format  : {observed_fmt if observed_fmt else 'could not infer from examples'}")

    # Test parsing
    parse_failures = 0
    pct_fail = 0.0
    parse_decision = "no change needed"

    if isinstance(dtype, (DateType, TimestampType)):
        print(f"    Parse test       : Already a {type_label(dtype)} — no string parsing needed")
        parse_decision = "no change needed (already typed)"
        recommendation = f"retain as {type_label(dtype)}"
    elif observed_fmt:
        if is_time:
            total_nn, fails, pct = test_timestamp_parsing(
                exploded.select(col(f"_rec.{col_name}").alias(col_name)) if is_nested else df,
                col_name, observed_fmt
            )
        else:
            total_nn, fails, pct = test_date_parsing(
                exploded.select(col(f"_rec.{col_name}").alias(col_name)) if is_nested else df,
                col_name, observed_fmt
            )
        parse_failures = fails
        pct_fail = pct
        print(f"    Parse test       : format='{observed_fmt}'  non-null={total_nn:,}  failures={fails:,}  ({pct:.4f}%)")

        if fails == 0:
            if is_time:
                recommendation = f"convert to timestamp (format: {observed_fmt})"
                parse_decision = "required conversion" if isinstance(dtype, StringType) else "recommended conversion"
            else:
                recommendation = f"convert to date (format: {observed_fmt})"
                parse_decision = "required conversion" if isinstance(dtype, StringType) else "recommended conversion"
        else:
            recommendation = f"parse issues — {fails:,} values ({pct:.4f}%) fail format '{observed_fmt}' — investigate before casting"
            parse_decision = "uncertain — parse failures detected"
    else:
        print(f"    Parse test       : could not infer format — manual inspection needed")
        recommendation = f"uncertain — inspect manually; current type is {cur_type}"
        parse_decision = "uncertain"

    # Special handling for ClosedDate: nulls are valid, don't count as failures
    if col_name == "ClosedDate":
        total_nulls = df.filter(col(col_name).isNull()).count() if not is_nested else 0
        print(f"    Note             : ClosedDate has {total_nulls:,} null values (valid — open locations) — excluded from parse failures")

    print(f"    Recommendation   : {recommendation}")
    print(f"    Decision         : {parse_decision}")

    date_time_results.append({
        "field": label, "current_type": cur_type,
        "examples": examples, "observed_format": observed_fmt,
        "total_non_null": total_non_null, "parse_failures": parse_failures,
        "pct_fail": pct_fail, "recommendation": recommendation,
        "decision": parse_decision
    })

    if parse_decision != "no change needed (already typed)":
        final_summary.append({
            "dataset": ds_name if not is_nested else f"checks.{parent_arr}",
            "column": col_name, "current_type": cur_type,
            "recommended_type": recommendation,
            "parse_failures": f"{parse_failures:,} ({pct_fail:.4f}%)",
            "decision": parse_decision
        })

# ============================================================================
# 7.3 — NUMERIC AND MONETARY FIELDS
# ============================================================================
print("\n" + "═" * 80)
print("7.3  NUMERIC AND MONETARY FIELDS — TYPE REVIEW")
print("═" * 80)

# (array_name, field_name) for nested; (None, field_name) for top-level
numeric_fields = [
    ("Discounts", "Amount"),
    ("ItemsSold", "GrossPrice"),
    ("ItemsSold", "SoldPrice"),
    ("ItemsSold", "Quantity"),
    ("Payments", "Amount"),
    ("Payments", "Gratuity"),
    ("Voids", "Amount"),
    ("Voids", "Quantity"),
    ("Tax", "Amount"),
    (None, "TrafficCount"),
    (None, "ServiceCharge"),
]

for arr_name, field_name in numeric_fields:
    label = f"checks.{arr_name}.{field_name}" if arr_name else f"checks_df.{field_name}"
    print(f"\n  ── {label} ──")

    if arr_name:
        if arr_name not in checks_df.columns:
            print(f"    ⚠ Array '{arr_name}' not found — SKIPPED")
            numeric_results.append({
                "field": label, "current_type": "N/A",
                "recommended_type": "N/A", "decision": "N/A — not found"
            })
            continue
        arr_type = checks_df.schema[arr_name].dataType
        if not isinstance(arr_type, ArrayType) or not isinstance(arr_type.elementType, StructType):
            print(f"    ⚠ '{arr_name}' is not array<struct> — SKIPPED")
            continue
        sub_fields = arr_type.elementType.fields
        if field_name not in [f.name for f in sub_fields]:
            print(f"    ⚠ Field '{field_name}' not found in {arr_name} — SKIPPED")
            numeric_results.append({
                "field": label, "current_type": "N/A",
                "recommended_type": "N/A", "decision": "N/A — not found"
            })
            continue
        dtype = [f.dataType for f in sub_fields if f.name == field_name][0]
    else:
        if field_name not in checks_df.columns:
            print(f"    ⚠ Column '{field_name}' not found in checks_df — SKIPPED")
            numeric_results.append({
                "field": label, "current_type": "N/A",
                "recommended_type": "N/A", "decision": "N/A — not found"
            })
            continue
        dtype = checks_df.schema[field_name].dataType

    cur_type = type_label(dtype)
    rec = recommend_type(field_name, dtype)
    decision = action_needed(field_name, dtype)

    # Additional info: is it integer or has decimals?
    if isinstance(dtype, (IntegerType, LongType)):
        note = "stored as integer"
    elif isinstance(dtype, (DoubleType, FloatType)):
        note = "stored as floating-point — monetary values ideally use decimal for precision"
    elif isinstance(dtype, DecimalType):
        note = f"stored as decimal({dtype.precision},{dtype.scale}) — appropriate for monetary"
    else:
        note = f"stored as {cur_type} — may not be suitable for numeric analysis"

    print(f"    Current type     : {cur_type}")
    print(f"    Note            : {note}")
    print(f"    Recommended    : {rec}")
    print(f"    Decision        : {decision}")
    print(f"    ⚠ Not classifying negative/zero/large/fractional values as errors in this step.")

    numeric_results.append({
        "field": label, "current_type": cur_type,
        "recommended_type": rec, "decision": decision, "note": note
    })

    if decision != "no change needed":
        final_summary.append({
            "dataset": f"checks.{arr_name}" if arr_name else "checks_df",
            "column": field_name, "current_type": cur_type,
            "recommended_type": rec, "parse_failures": "N/A (type review)",
            "decision": decision
        })

# ============================================================================
# 7.4 — IDENTIFIER FIELDS
# ============================================================================
print("\n" + "═" * 80)
print("7.4  IDENTIFIER FIELDS — TYPE REVIEW")
print("═" * 80)

# (dataset, column_name, is_nested, parent_array)
identifier_fields = [
    ("checks_df", "Id", False, None),
    ("checks_df", "LocationId", False, None),
    ("employees", "PayrollEmployeeId", False, None),
    ("employees", "LocationEmployeeId", False, None),
    ("locations", "Id", False, None),
    ("items", "Id", False, None),
    ("revenue_centers", "Id", False, None),
    # Nested identifiers
    ("checks_df", "EmployeeId", True, "ItemsSold"),
    ("checks_df", "ManagerId", True, "ItemsSold"),
    ("checks_df", "ItemId", True, "ItemsSold"),
    ("checks_df", "RevenueCenterId", True, "ItemsSold"),
    ("checks_df", "DiscountId", True, "Discounts"),
    ("checks_df", "PaymentCategoryId", True, "Payments"),
    ("checks_df", "VoidCategoryId", True, "Voids"),
    # PLU fields
    ("items", "Plu", False, None),
]

# Also check top-level checks_df for EmployeeId, ManagerId, etc.
for extra in ["EmployeeId", "ManagerId", "ItemId", "RevenueCenterId", "DiscountId", "PaymentCategoryId", "VoidCategoryId"]:
    if extra in checks_df.columns:
        identifier_fields.append(("checks_df", extra, False, None))

seen = set()
for ds_name, col_name, is_nested, parent_arr in identifier_fields:
    key = (ds_name, col_name, is_nested, parent_arr)
    if key in seen:
        continue
    seen.add(key)

    label = f"{ds_name}.{parent_arr}.{col_name}" if is_nested else f"{ds_name}.{col_name}"
    print(f"\n  ── {label} ──")

    if is_nested:
        if parent_arr not in checks_df.columns:
            print(f"    ⚠ Parent array '{parent_arr}' not found — SKIPPED")
            continue
        arr_type = checks_df.schema[parent_arr].dataType
        if not isinstance(arr_type, ArrayType) or not isinstance(arr_type.elementType, StructType):
            print(f"    ⚠ '{parent_arr}' is not array<struct> — SKIPPED")
            continue
        sub_fields = arr_type.elementType.fields
        if col_name not in [f.name for f in sub_fields]:
            print(f"    ⚠ Field '{col_name}' not found in {parent_arr} — SKIPPED")
            continue
        dtype = [f.dataType for f in sub_fields if f.name == col_name][0]
    else:
        if ds_name not in datasets:
            print(f"    ⚠ Dataset '{ds_name}' not loaded — SKIPPED")
            continue
        df = datasets[ds_name]
        if col_name not in df.columns:
            print(f"    ⚠ Column '{col_name}' not found in {ds_name} — SKIPPED")
            continue
        dtype = df.schema[col_name].dataType

    cur_type = type_label(dtype)
    is_numeric_id = isinstance(dtype, (IntegerType, LongType, DoubleType, FloatType, DecimalType))
    rec = recommend_type(col_name, dtype)
    decision = action_needed(col_name, dtype)

    if is_numeric_id:
        rationale = ("IDs stored as numeric — safe for joins but analytically they are labels, "
                     "not quantities. Converting to string/categorical prevents accidental aggregation (SUM, AVG).")
    elif isinstance(dtype, StringType):
        rationale = "IDs stored as string — already categorical, safe for joins and grouping."
    else:
        rationale = f"IDs stored as {cur_type} — review suitability."

    print(f"    Current type     : {cur_type}")
    print(f"    Rationale        : {rationale}")
    print(f"    Recommended      : {rec}")
    print(f"    Decision         : {decision}")

    identifier_results.append({
        "field": label, "current_type": cur_type,
        "recommended_type": rec, "decision": decision, "rationale": rationale
    })

    if decision != "no change needed":
        final_summary.append({
            "dataset": f"checks.{parent_arr}" if is_nested else ds_name,
            "column": col_name, "current_type": cur_type,
            "recommended_type": rec, "parse_failures": "N/A (ID review)",
            "decision": decision
        })

# ============================================================================
# 7.5 — BOOLEAN AND CATEGORICAL FIELDS
# ============================================================================
print("\n" + "═" * 80)
print("7.5  BOOLEAN AND CATEGORICAL FIELDS — VALIDATION")
print("═" * 80)

# --- 7.5a: Boolean fields ---
print("\n  ── 7.5a Boolean fields: verify only valid boolean values ──")
boolean_fields_found = []
for ds_name in ["checks_df", "employees", "items", "locations", "revenue_centers"]:
    if ds_name not in datasets:
        continue
    df = datasets[ds_name]
    for field in df.schema.fields:
        if isinstance(field.dataType, BooleanType):
            col_name = field.name
            total = df.count()
            non_null = df.filter(col(col_name).isNotNull()).count()
            true_count = df.filter(col(col_name) == True).count()
            false_count = df.filter(col(col_name) == False).count()
            null_count = total - non_null
            print(f"\n    {ds_name}.{col_name}")
            print(f"      type=boolean  total={total:,}  non-null={non_null:,}  true={true_count:,}  false={false_count:,}  null={null_count:,}")
            if true_count + false_count == non_null:
                print(f"      ✅ All non-null values are valid booleans (true/false)")
                decision = "no change needed"
            else:
                print(f"      ⚠ Non-null count != true+false — unexpected values present!")
                decision = "investigate — invalid boolean values"
            boolean_fields_found.append({
                "field": f"{ds_name}.{col_name}", "decision": decision
            })

# Check nested boolean fields
for arr in nested_arrays:
    if arr not in checks_df.columns:
        continue
    arr_type = checks_df.schema[arr].dataType
    if not isinstance(arr_type, ArrayType) or not isinstance(arr_type.elementType, StructType):
        continue
    for field in arr_type.elementType.fields:
        if isinstance(field.dataType, BooleanType):
            col_name = field.name
            exploded = (checks_df
                        .filter(col(arr).isNotNull() & (size(col(arr)) > 0))
                        .select(explode(col(arr)).alias("_rec")))
            total = exploded.count()
            non_null = exploded.filter(col(f"_rec.{col_name}").isNotNull()).count()
            true_count = exploded.filter(col(f"_rec.{col_name}") == True).count()
            false_count = exploded.filter(col(f"_rec.{col_name}") == False).count()
            print(f"\n    checks.{arr}.{col_name}")
            print(f"      type=boolean  total={total:,}  non-null={non_null:,}  true={true_count:,}  false={false_count:,}  null={total - non_null:,}")
            if true_count + false_count == non_null:
                print(f"      ✅ All non-null values are valid booleans (true/false)")
                decision = "no change needed"
            else:
                print(f"      ⚠ Non-null count != true+false — unexpected values present!")
                decision = "investigate — invalid boolean values"
            boolean_fields_found.append({
                "field": f"checks.{arr}.{col_name}", "decision": decision
            })

if not boolean_fields_found:
    print("\n    No boolean fields found in any dataset.")

# --- 7.5b: Categorical fields (numeric/string acting as categories) ---
print("\n  ── 7.5b Categorical fields: identify numeric/string fields functioning as categories ──")

categorical_indicators = ["type", "category", "status", "level", "flag", "code", "reason", "method", "class"]
for ds_name in ["checks_df", "employees", "items", "locations", "revenue_centers"]:
    if ds_name not in datasets:
        continue
    df = datasets[ds_name]
    total = df.count()
    for field in df.schema.fields:
        cname_lower = field.name.lower()
        # Check if the column name suggests a category
        is_category_by_name = any(ind in cname_lower for ind in categorical_indicators)
        # Check if numeric column has very few distinct values (likely categorical)
        if isinstance(field.dataType, (IntegerType, LongType)) and not any(
            k in cname_lower for k in ["id", "quantity", "count", "year", "number"]
        ):
            try:
                distinct = df.select(field.name).distinct().count()
                if distinct <= 20 or is_category_by_name:
                    print(f"\n    {ds_name}.{field.name} (type={type_label(field.dataType)}, distinct={distinct:,})")
                    print(f"      → Low-cardinality numeric — likely categorical. Consider string/categorical.")
                    boolean_categorical_results.append({
                        "field": f"{ds_name}.{field.name}", "type": "categorical",
                        "decision": "recommended conversion to string/categorical"
                    })
            except Exception:
                pass
        elif isinstance(field.dataType, StringType) and is_category_by_name:
            try:
                distinct = df.select(field.name).distinct().count()
                if distinct <= 50:
                    print(f"\n    {ds_name}.{field.name} (type=string, distinct={distinct:,})")
                    print(f"      → String field with category-like name and low cardinality — confirm categorical.")
                    boolean_categorical_results.append({
                        "field": f"{ds_name}.{field.name}", "type": "categorical",
                        "decision": "no change needed (already string) — confirm categorical usage"
                    })
            except Exception:
                pass

# Check nested fields for categorical-like columns
for arr in nested_arrays:
    if arr not in checks_df.columns:
        continue
    arr_type = checks_df.schema[arr].dataType
    if not isinstance(arr_type, ArrayType) or not isinstance(arr_type.elementType, StructType):
        continue
    for field in arr_type.elementType.fields:
        cname_lower = field.name.lower()
        is_category_by_name = any(ind in cname_lower for ind in categorical_indicators)
        if isinstance(field.dataType, (IntegerType, LongType)) and not any(
            k in cname_lower for k in ["id", "quantity", "count", "year", "number"]
        ):
            try:
                exploded = (checks_df
                            .filter(col(arr).isNotNull() & (size(col(arr)) > 0))
                            .select(explode(col(arr)).alias("_rec")))
                distinct = exploded.select(col(f"_rec.{field.name}")).distinct().count()
                if distinct <= 20 or is_category_by_name:
                    print(f"\n    checks.{arr}.{field.name} (type={type_label(field.dataType)}, distinct={distinct:,})")
                    print(f"      → Low-cardinality numeric — likely categorical. Consider string/categorical.")
                    boolean_categorical_results.append({
                        "field": f"checks.{arr}.{field.name}", "type": "categorical",
                        "decision": "recommended conversion to string/categorical"
                    })
            except Exception:
                pass
        elif isinstance(field.dataType, StringType) and is_category_by_name:
            try:
                exploded = (checks_df
                            .filter(col(arr).isNotNull() & (size(col(arr)) > 0))
                            .select(explode(col(arr)).alias("_rec")))
                distinct = exploded.select(col(f"_rec.{field.name}")).distinct().count()
                if distinct <= 50:
                    print(f"\n    checks.{arr}.{field.name} (type=string, distinct={distinct:,})")
                    print(f"      → String field with category-like name and low cardinality — confirm categorical.")
                    boolean_categorical_results.append({
                        "field": f"checks.{arr}.{field.name}", "type": "categorical",
                        "decision": "no change needed (already string) — confirm categorical usage"
                    })
            except Exception:
                pass

# ============================================================================
# 7.6 — FINAL SUMMARY
# ============================================================================
print("\n" + "═" * 80)
print("7.6  FINAL SUMMARY — FIELDS REQUIRING ATTENTION")
print("═" * 80)
print()
print(f"  {'Dataset':<22} {'Column':<22} {'Current Type':<20} {'Recommended Type':<50} {'Parse Failures':<25} {'Decision for Cleaning Stage'}")
print(f"  {'-'*22} {'-'*22} {'-'*20} {'-'*50} {'-'*25} {'-'*35}")

if not final_summary:
    print("  ✅ No fields require attention — all types appear suitable for analysis.")
else:
    # Deduplicate by (dataset, column)
    seen_keys = set()
    for row in final_summary:
        key = (row["dataset"], row["column"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        print(f"  {row['dataset']:<22} {row['column']:<22} {row['current_type']:<20} {row['recommended_type']:<50} {row['parse_failures']:<25} {row['decision']}")

print(f"\n  Total fields flagged: {len(seen_keys) if final_summary else 0}")

# Summary counts by decision
from collections import Counter
decision_counts = Counter(r["decision"] for r in final_summary)
print("\n  Decision breakdown:")
for dec, cnt in decision_counts.most_common():
    print(f"    {dec:<45} : {cnt}")

print("\n" + "=" * 80)
print("SECTION 7 COMPLETE — DATA TYPES DIAGNOSTIC AUDIT FINISHED")
print("⚠ No raw data was modified, cast, or transformed. All results are diagnostic only.")
print("=" * 80)

# COMMAND ----------

#La revisión de tipos de datos mostró una estructura consistente: las variables numéricas y booleanas son adecuadas para el análisis, mientras que las principales transformaciones necesarias corresponden a campos temporales almacenados como texto. Las fechas presentan formatos uniformes y 0% de errores de parsing, lo que permitirá transformarlas de forma segura y habilitar análisis temporales posteriores.

# COMMAND ----------

# ============================================================================
# STEP 7 — FINAL TIMESTAMP VALIDATION (read-only, no data modified)
# ===========================================================================
# Validates complete non-null values of:
#   • checks_df.OpenTime
#   • checks_df.CloseTime
#   • checks_df.ItemsSold.OrderTime
# Tests parsing against observed pattern:  yyyy-MM-dd'T'HH:mm
# Also scans for seconds, milliseconds, timezone info, or other formats.
# No original field is transformed or overwritten.
# ===========================================================================

from pyspark.sql.functions import (
    col, count, when, lit, isnull, size, explode,
    to_timestamp, min as _min, max as _max, length, trim
)
from pyspark.sql.types import ArrayType, StructType
import re

FMT = "yyyy-MM-dd'T'HH:mm"

print("=" * 90)
print("STEP 7 — FINAL TIMESTAMP VALIDATION  (format under test: yyyy-MM-dd'T'HH:mm)")
print("=" * 90)

# ------------------------------------------------------------------
# Helper: pattern-detection for alternative formats
# ------------------------------------------------------------------
SECOND_RE     = re.compile(r"\d{2}:\d{2}:\d{2}")          # contains seconds
MILLI_RE      = re.compile(r"\.\d{1,6}(?:[^0-9]|$)")        # contains fractional seconds
TZ_RE         = re.compile(r"[Zz]|[+-]\d{2}:?\d{2}$|[+-]\d{2}$")  # timezone suffix
FULL_DT_TM_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$")

def detect_alternatives(spark_df, col_name):
    """Return counts of distinct non-null values with seconds, milliseconds, timezone, or other format."""
    total = spark_df.filter(col(col_name).isNotNull()).count()
    if total == 0:
        return {"total": 0, "seconds": 0, "milliseconds": 0, "timezone": 0,
                "other_format": 0, "other_examples": []}
    distinct_vals = (spark_df
                     .filter(col(col_name).isNotNull())
                     .select(col_name)
                     .distinct()
                     .collect())
    vals = [r[col_name] for r in distinct_vals]
    sec_count = sum(1 for v in vals if SECOND_RE.search(str(v)))
    ms_count  = sum(1 for v in vals if MILLI_RE.search(str(v)))
    tz_count  = sum(1 for v in vals if TZ_RE.search(str(v)))
    other_count = sum(1 for v in vals if not FULL_DT_TM_RE.match(str(v)))
    other_examples = [str(v) for v in vals if not FULL_DT_TM_RE.match(str(v))][:10]
    return {"total": total, "seconds": sec_count, "milliseconds": ms_count,
            "timezone": tz_count, "other_format": other_count,
            "other_examples": other_examples}

def _print_alternatives(alt):
    print(f"  ── Alternative-format scan (distinct non-null values) ──")
    print(f"    Distinct values checked         : {alt['total']:,}")
    print(f"    Values with seconds (HH:mm:ss)  : {alt['seconds']:,}")
    print(f"    Values with milliseconds        : {alt['milliseconds']:,}")
    print(f"    Values with timezone info       : {alt['timezone']:,}")
    print(f"    Values NOT matching yyyy-MM-dd'T'HH:mm : {alt['other_format']:,}")
    if alt['other_examples']:
        print(f"    Other-format examples (up to 10): {alt['other_examples']}")
    else:
        print(f"    Other-format examples           : (none — all match expected pattern)")

# ------------------------------------------------------------------
# Helper: full validation of one field against FMT
# ------------------------------------------------------------------
def validate_timestamp_field(df, col_name, label):
    """Validate every non-null value in df.col_name against FMT."""
    print(f"\n{'─' * 90}")
    print(f"  FIELD: {label}")
    print(f"{'─' * 90}")

    non_null_df = df.filter(col(col_name).isNotNull())
    total_non_null = non_null_df.count()

    if total_non_null == 0:
        print(f"    Total non-null records : 0")
        print(f"    Successfully parsed    : 0")
        print(f"    Failed records         : 0")
        print(f"    Failure percentage     : N/A")
        print(f"    Min parsed timestamp   : N/A")
        print(f"    Max parsed timestamp   : N/A")
        print(f"    Failed value examples  : (none)")
        _print_alternatives(detect_alternatives(df, col_name))
        return

    # Parse with to_timestamp; failures become NULL (no overwrite of original col)
    parsed_df = non_null_df.withColumn("_parsed", to_timestamp(col(col_name), FMT))
    parsed_ok_df   = parsed_df.filter(col("_parsed").isNotNull())
    parsed_fail_df = parsed_df.filter(col("_parsed").isNull())

    success_count = parsed_ok_df.count()
    fail_count    = parsed_fail_df.count()
    pct_fail      = round(100.0 * fail_count / total_non_null, 6) if total_non_null else 0.0

    min_ts = None
    max_ts = None
    if success_count > 0:
        row = parsed_ok_df.agg(_min("_parsed").alias("mn"), _max("_parsed").alias("mx")).collect()[0]
        min_ts = row["mn"]
        max_ts = row["mx"]

    failed_examples = []
    if fail_count > 0:
        failed_examples = [r[col_name] for r in
                          parsed_fail_df.select(col_name).distinct().limit(10).collect()]

    print(f"    Total non-null records : {total_non_null:,}")
    print(f"    Successfully parsed    : {success_count:,}")
    print(f"    Failed records         : {fail_count:,}")
    print(f"    Failure percentage     : {pct_fail:.6f}%")
    print(f"    Min parsed timestamp   : {min_ts}")
    print(f"    Max parsed timestamp   : {max_ts}")
    print(f"    Failed value examples  : {failed_examples if failed_examples else '(none)'}")
    _print_alternatives(detect_alternatives(non_null_df, col_name))

# ------------------------------------------------------------------
# 1) checks_df.OpenTime
# ------------------------------------------------------------------
validate_timestamp_field(checks_df, "OpenTime", "checks_df.OpenTime")

# ------------------------------------------------------------------
# 2) checks_df.CloseTime
# ------------------------------------------------------------------
validate_timestamp_field(checks_df, "CloseTime", "checks_df.CloseTime")

# ------------------------------------------------------------------
# 3) checks_df.ItemsSold.OrderTime  (nested inside ItemsSold array)
# ------------------------------------------------------------------
label3 = "checks_df.ItemsSold.OrderTime"
print(f"\n{'─' * 90}")
print(f"  FIELD: {label3}")
print(f"{'─' * 90}")

if "ItemsSold" not in checks_df.columns:
    print(f"    ⚠ 'ItemsSold' array not found in checks_df — SKIPPED")
else:
    arr_type = checks_df.schema["ItemsSold"].dataType
    if not isinstance(arr_type, ArrayType) or not isinstance(arr_type.elementType, StructType):
        print(f"    ⚠ 'ItemsSold' is not an array<struct> — SKIPPED")
    else:
        sub_fields = [f.name for f in arr_type.elementType.fields]
        if "OrderTime" not in sub_fields:
            print(f"    ⚠ 'OrderTime' not found inside ItemsSold struct — SKIPPED")
        else:
            # Explode ItemsSold to get one row per element (read-only; no overwrite)
            exploded = (checks_df
                        .filter(col("ItemsSold").isNotNull() & (size(col("ItemsSold")) > 0))
                        .select(explode(col("ItemsSold")).alias("_rec"))
                        .select(col("_rec.OrderTime").alias("OrderTime")))
            validate_timestamp_field(exploded, "OrderTime", label3)

print(f"\n{'=' * 90}")
print("STEP 7 — FINAL TIMESTAMP VALIDATION COMPLETE")
print("=" * 90)


# COMMAND ----------

#Las variables temporales presentan una alta consistencia. OpenTime, CloseTime y OrderTime siguen uniformemente el formato yyyy-MM-dd'T'HH:mm, con 0% de errores de parsing en más de 28 millones de registros. Esto permite convertirlas de forma segura a timestamp y habilita análisis posteriores por hora del día, duración de transacciones y secuencia temporal de operaciones.

# COMMAND ----------

# MAGIC %md
# MAGIC Invalid/Inconsistent values

# COMMAND ----------

# ============================================================================
# STEP 8 — INVALID / INCONSISTENT VALUES  (diagnostic only, read-only)
# ============================================================================
# NO raw data is modified, filtered, imputed, cast, winsorized, capped,
# deduplicated, or overwritten.  This step is purely diagnostic.
#
# Goal: distinguish
#   1. Likely data-quality error
#   2. Legitimate business/system behavior
#   3. Unusual behavior requiring further investigation
#
# Confirmed check key:  LocationId + BusinessDate + Id
# ============================================================================

from pyspark.sql.functions import (
    col, count, when, lit, isnull, isnan, size, explode, struct,
    to_timestamp, to_date, min as _min, max as _max,
    sum as _sum, avg, expr, round as _round, unix_timestamp,
    lower, upper, trim, concat, length, desc, asc
)
from pyspark.sql.types import (
    StringType, IntegerType, LongType, DoubleType, DecimalType,
    FloatType, BooleanType, DateType, TimestampType, ArrayType, StructType
)
from functools import reduce

FMT_TS = "yyyy-MM-dd'T'HH:mm"

print("=" * 100)
print("STEP 8 — INVALID / INCONSISTENT VALUES  (Diagnostic Only — Read-Only)")
print("=" * 100)

# ------------------------------------------------------------------
# Dimension lookup (reuse from previous steps)
# ------------------------------------------------------------------
dim_dfs = dim_dfs if 'dim_dfs' in dir() else {}
emp_df, loc_df = None, None
emp_key, loc_key = None, None
for k in dim_dfs:
    kl = k.lower()
    if emp_df is None and 'emp' in kl:
        emp_df, emp_key = dim_dfs[k], k
    if loc_df is None and ('loc' in kl or 'restaurant' in kl):
        loc_df, loc_key = dim_dfs[k], k

if emp_df is not None:
    print(f"  Employees dim : dim_dfs['{emp_key}']")
if loc_df is not None:
    print(f"  Locations dim : dim_dfs['{loc_key}']")

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def pct(n, d):
    return round(100.0 * n / d, 4) if d and d > 0 else 0.0

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def struct_field_names(df, array_col):
    """Return list of field names inside an array<struct> column, or None."""
    if array_col not in df.columns:
        return None
    dt = df.schema[array_col].dataType
    if isinstance(dt, ArrayType) and isinstance(dt.elementType, StructType):
        return [f.name for f in dt.elementType.fields]
    return None

def numeric_stats(df, col_name):
    """Return dict with min, median, mean, p95, p99, max for a numeric column."""
    row = df.agg(
        _min(col(col_name)).alias("mn"),
        _max(col(col_name)).alias("mx"),
        avg(col(col_name)).alias("mean")
    ).collect()[0]
    try:
        q = df.na.drop(subset=[col_name]).stat.approxQuantile(
            col_name, [0.5, 0.95, 0.99], 0.001)
        med, p95, p99 = q[0], q[1], q[2]
    except Exception:
        med = p95 = p99 = None
    return {
        "min": row["mn"],
        "median": med,
        "mean": round(row["mean"], 4) if row["mean"] is not None else None,
        "p95": p95,
        "p99": p99,
        "max": row["mx"],
    }

def make_exploded(df, array_col):
    """Return DataFrame with check-key + all fields from array<struct>, or None."""
    fields = struct_field_names(df, array_col)
    if fields is None:
        return None
    base_sel = [col("LocationId"), col("BusinessDate"), col("Id")]
    if "OpenTime" in df.columns:
        base_sel += [col("OpenTime"), col("CloseTime")]
    exploded = (df.filter(col(array_col).isNotNull() & (size(col(array_col)) > 0))
               .select(*base_sel, explode(col(array_col)).alias("_e")))
    sel = list(base_sel)
    for f in fields:
        sel.append(col(f"_e.{f}").alias(f))
    return exploded.select(*sel)

def get_numeric_df(df, col_name, field_name):
    """Return DF with check key + field value from a struct or array<struct> col."""
    if col_name not in df.columns:
        return None
    dt = df.schema[col_name].dataType
    base_sel = [col("LocationId"), col("BusinessDate"), col("Id")]
    if isinstance(dt, ArrayType) and isinstance(dt.elementType, StructType):
        sub = [f.name for f in dt.elementType.fields]
        if field_name not in sub:
            return None
        return (df.filter(col(col_name).isNotNull() & (size(col(col_name)) > 0))
                .select(*base_sel, explode(col(col_name)).alias("_e"))
                .select(*base_sel, col(f"_e.{field_name}").alias(field_name)))
    elif isinstance(dt, StructType):
        sub = [f.name for f in dt.fields]
        if field_name not in sub:
            return None
        return (df.filter(col(col_name).isNotNull())
                .select(*base_sel, col(f"{col_name}.{field_name}").alias(field_name)))
    return None

def analyze_numeric_field(df, col_name, label):
    """Full sign/zero/stats analysis for a numeric column in df."""
    if df is None or col_name not in df.columns:
        print(f"\n  ⚠ {label}: field '{col_name}' not available — SKIPPED")
        return None
    nn_df = df.filter(col(col_name).isNotNull())
    nn = nn_df.count()
    if nn == 0:
        print(f"\n  ── {label} ({col_name}) ──  No non-null values")
        return {"total": 0, "negative": 0, "zero": 0, "positive": 0}
    neg = nn_df.filter(col(col_name) < 0).count()
    zero = nn_df.filter(col(col_name) == 0).count()
    pos = nn_df.filter(col(col_name) > 0).count()
    stats = numeric_stats(nn_df, col_name)
    print(f"\n  ── {label} ({col_name}) ──")
    print(f"    Total non-null  : {nn:,}")
    print(f"    Negative        : {neg:,}  ({pct(neg, nn):.4f}%)")
    print(f"    Zero            : {zero:,}  ({pct(zero, nn):.4f}%)")
    print(f"    Positive        : {pos:,}  ({pct(pos, nn):.4f}%)")
    print(f"    min={stats['min']}  median={stats['median']}  mean={stats['mean']}")
    print(f"    p95={stats['p95']}  p99={stats['p99']}  max={stats['max']}")
    if neg > 0:
        ex = nn_df.filter(col(col_name) < 0).select(
            "LocationId", "BusinessDate", "Id", col_name).distinct().limit(5).collect()
        print(f"    Negative examples:")
        for r in ex:
            print(f"      Loc={r['LocationId']} Date={r['BusinessDate']} Id={r['Id']}  {col_name}={r[col_name]}")
    if zero > 0 and neg == 0:  # only show zero examples if no negative examples shown
        ex = nn_df.filter(col(col_name) == 0).select(
            "LocationId", "BusinessDate", "Id", col_name).distinct().limit(5).collect()
        print(f"    Zero examples:")
        for r in ex:
            print(f"      Loc={r['LocationId']} Date={r['BusinessDate']} Id={r['Id']}  {col_name}={r[col_name]}")
    return {"total": nn, "negative": neg, "zero": zero, "positive": pos, **stats}

# ------------------------------------------------------------------
# Summary accumulator
# ------------------------------------------------------------------
findings = []

def add_finding(issue, structure, records, pct_val, severity, interpretation, treatment):
    findings.append({
        "Issue": issue,
        "Structure/Field": structure,
        "Records Affected": records,
        "%": pct_val,
        "Severity": severity,
        "Initial Interpretation": interpretation,
        "Recommended Treatment": treatment,
    })

# ------------------------------------------------------------------
# Build exploded views (read-only; original data untouched)
# ------------------------------------------------------------------
print("\n  Building exploded views (read-only)...")
exploded_items    = make_exploded(checks_df, "ItemsSold")
exploded_payments = make_exploded(checks_df, "Payments")
exploded_voids    = make_exploded(checks_df, "Voids")
exploded_discounts= make_exploded(checks_df, "Discounts")

total_items    = exploded_items.count()    if exploded_items    else 0
total_payments = exploded_payments.count() if exploded_payments else 0
total_voids    = exploded_voids.count()    if exploded_voids    else 0
total_discounts= exploded_discounts.count() if exploded_discounts else 0

print(f"  Exploded item records     : {total_items:,}")
print(f"  Exploded payment records  : {total_payments:,}")
print(f"  Exploded void records     : {total_voids:,}")
print(f"  Exploded discount records : {total_discounts:,}")

TOTAL_CHECKS = checks_df.count()
print(f"\n  Total checks in checks_df : {TOTAL_CHECKS:,}")

# ===========================================================================
# 8.1 — CHECK STRUCTURE AND TEMPORAL CONSISTENCY
# ===========================================================================
print("\n" + "═" * 100)
print("8.1  CHECK STRUCTURE AND TEMPORAL CONSISTENCY")
print("═" * 100)

base = (checks_df
        .withColumn("_open",  to_timestamp(col("OpenTime"),  FMT_TS))
        .withColumn("_close", to_timestamp(col("CloseTime"), FMT_TS))
        .withColumn("_bdate", to_date(col("BusinessDate"), "yyyy-MM-dd")))

dur_df = base.withColumn(
    "_dur_min",
    (unix_timestamp(col("_close")) - unix_timestamp(col("_open"))) / 60.0)

close_before_open = dur_df.filter(col("_close") < col("_open")).count()
open_eq_close     = dur_df.filter(col("_close") == col("_open")).count()
neg_dur           = dur_df.filter(col("_dur_min") < 0).count()

print(f"  Checks where CloseTime < OpenTime : {close_before_open:,}  ({pct(close_before_open, TOTAL_CHECKS):.4f}%)")
print(f"  Checks where OpenTime = CloseTime  : {open_eq_close:,}  ({pct(open_eq_close, TOTAL_CHECKS):.4f}%)")
print(f"  Checks with negative duration      : {neg_dur:,}  ({pct(neg_dur, TOTAL_CHECKS):.4f}%)")

valid_dur = dur_df.filter(col("_open").isNotNull() & col("_close").isNotNull())
dur_count = valid_dur.count()
print(f"  Checks with valid Open & Close     : {dur_count:,}")

if dur_count > 0:
    ds = numeric_stats(valid_dur, "_dur_min")
    print(f"  Duration (min): min={ds['min']}  median={ds['median']}  mean={ds['mean']}  p95={ds['p95']}  p99={ds['p99']}  max={ds['max']}")
    for hrs in (6, 12, 24):
        c = valid_dur.filter(col("_dur_min") > hrs * 60).count()
        print(f"  Checks lasting >{hrs}h              : {c:,}  ({pct(c, dur_count):.4f}%)")

if close_before_open > 0:
    print(f"  Examples (CloseTime < OpenTime):")
    for r in dur_df.filter(col("_close") < col("_open")).select(
            "LocationId", "BusinessDate", "Id", "OpenTime", "CloseTime").limit(5).collect():
        print(f"    Loc={r['LocationId']} Date={r['BusinessDate']} Id={r['Id']}  Open={r['OpenTime']}  Close={r['CloseTime']}")

# ── ItemsSold.OrderTime vs OpenTime / CloseTime ──
print(f"\n  ── ItemsSold.OrderTime temporal consistency ──")
if exploded_items is not None and "OrderTime" in exploded_items.columns:
    ot_df = (exploded_items
             .withColumn("_open",  to_timestamp(col("OpenTime"),  FMT_TS))
             .withColumn("_close", to_timestamp(col("CloseTime"), FMT_TS))
             .withColumn("_order", to_timestamp(col("OrderTime"), FMT_TS)))
    ot_total = ot_df.count()
    ot_before_open = ot_df.filter(col("_order") < col("_open")).count()
    ot_after_close = ot_df.filter(col("_order") > col("_close")).count()
    print(f"  Total item records (exploded)    : {ot_total:,}")
    print(f"  OrderTime before OpenTime         : {ot_before_open:,}  ({pct(ot_before_open, ot_total):.4f}%)")
    print(f"  OrderTime after CloseTime         : {ot_after_close:,}  ({pct(ot_after_close, ot_total):.4f}%)")
    if ot_before_open > 0:
        print(f"  Examples (OrderTime < OpenTime):")
        for r in ot_df.filter(col("_order") < col("_open")).select(
                "LocationId", "BusinessDate", "Id", "OrderTime", "OpenTime").limit(5).collect():
            print(f"    Loc={r['LocationId']} Date={r['BusinessDate']} Id={r['Id']}  Order={r['OrderTime']}  Open={r['OpenTime']}")
    if ot_after_close > 0:
        print(f"  Examples (OrderTime > CloseTime):")
        for r in ot_df.filter(col("_order") > col("_close")).select(
                "LocationId", "BusinessDate", "Id", "OrderTime", "CloseTime").limit(5).collect():
            print(f"    Loc={r['LocationId']} Date={r['BusinessDate']} Id={r['Id']}  Order={r['OrderTime']}  Close={r['CloseTime']}")
else:
    print("  ⚠ ItemsSold.OrderTime not found — SKIPPED")

# ===========================================================================
# 8.2 — CHECKS WITH EMPTY TRANSACTION ARRAYS
# ===========================================================================
print("\n" + "═" * 100)
print("8.2  CHECKS WITH EMPTY TRANSACTION ARRAYS")
print("═" * 100)

def is_empty_array(col_name):
    if col_name not in checks_df.columns:
        return lit(False)
    return col(col_name).isNull() | (size(col(col_name)) == 0)

items_empty   = is_empty_array("ItemsSold")
payments_empty = is_empty_array("Payments")

c_no_items    = checks_df.filter(items_empty).count()
c_no_payments = checks_df.filter(payments_empty).count()
c_neither     = checks_df.filter(items_empty & payments_empty).count()
print(f"  Checks with no ItemsSold          : {c_no_items:,}  ({pct(c_no_items, TOTAL_CHECKS):.4f}%)")
print(f"  Checks with no Payments            : {c_no_payments:,}  ({pct(c_no_payments, TOTAL_CHECKS):.4f}%)")
print(f"  Checks with neither Items nor Payments : {c_neither:,}  ({pct(c_neither, TOTAL_CHECKS):.4f}%)")

# Do these empty checks contain Discounts, Voids, Tax, or non-zero TrafficCount?
empty_checks = checks_df.filter(items_empty | payments_empty)

for arr_name in ("Discounts", "Voids"):
    if arr_name in checks_df.columns:
        has_arr = col(arr_name).isNotNull() & (size(col(arr_name)) > 0)
        n = empty_checks.filter(has_arr).count()
        print(f"  Empty-array checks with {arr_name} present : {n:,}  ({pct(n, c_no_items + c_no_payments - c_neither):.4f}%)")

tax_df = get_numeric_df(checks_df, "Tax", "Amount")
if tax_df is not None:
    # checks that have Tax.Amount != 0 among empty checks
    ec_tax = empty_checks.join(
        tax_df.filter(col("Amount") != 0).select("LocationId", "BusinessDate", "Id").distinct(),
        ["LocationId", "BusinessDate", "Id"], "inner"
    ).count()
    print(f"  Empty-array checks with non-zero Tax : {ec_tax:,}")

if "TrafficCount" in checks_df.columns:
    tc_nonzero = empty_checks.filter(col("TrafficCount") > 0).count()
    print(f"  Empty-array checks with TrafficCount > 0 : {tc_nonzero:,}  ({pct(tc_nonzero, c_no_items + c_no_payments - c_neither):.4f}%)")

# Top locations for empty-ItemsSold checks
if c_no_items > 0:
    print(f"\n  Top 10 locations for no-ItemsSold checks:")
    checks_df.filter(items_empty).groupBy("LocationId").agg(
        count("*").alias("n")).orderBy(desc("n")).limit(10).show(truncate=False)

# ===========================================================================
# 8.3 — QUANTITY VALIDATION
# ===========================================================================
print("\n" + "═" * 100)
print("8.3  QUANTITY VALIDATION")
print("═" * 100)

def quantity_analysis(df, col_name, label):
    if df is None or col_name not in df.columns:
        print(f"\n  ⚠ {label}: '{col_name}' not found — SKIPPED")
        return
    nn = df.filter(col(col_name).isNotNull())
    total = nn.count()
    if total == 0:
        print(f"\n  {label}: no non-null values")
        return
    neg = nn.filter(col(col_name) < 0).count()
    zero = nn.filter(col(col_name) == 0).count()
    pos = nn.filter(col(col_name) > 0).count()
    frac = nn.filter((col(col_name) % lit(1)) != 0).count()
    stats = numeric_stats(nn, col_name)
    print(f"\n  ── {label} ({col_name}) ──")
    print(f"    Total non-null     : {total:,}")
    print(f"    Negative           : {neg:,}  ({pct(neg, total):.4f}%)")
    print(f"    Zero               : {zero:,}  ({pct(zero, total):.4f}%)")
    print(f"    Positive           : {pos:,}  ({pct(pos, total):.4f}%)")
    print(f"    Fractional/whole   : {frac:,}  ({pct(frac, total):.4f}%)  (non-whole quantities)")
    print(f"    min={stats['min']}  median={stats['median']}  p95={stats['p95']}  p99={stats['p99']}  max={stats['max']}")
    if neg > 0:
        print(f"    Negative examples:")
        for r in nn.filter(col(col_name) < 0).select(
                "LocationId", "BusinessDate", "Id", col_name).distinct().limit(5).collect():
            print(f"      Loc={r['LocationId']} Date={r['BusinessDate']} Id={r['Id']}  {col_name}={r[col_name]}")
    if frac > 0:
        print(f"    Fractional examples:")
        for r in nn.filter((col(col_name) % lit(1)) != 0).select(
                "LocationId", "BusinessDate", "Id", col_name).distinct().limit(5).collect():
            print(f"      Loc={r['LocationId']} Date={r['BusinessDate']} Id={r['Id']}  {col_name}={r[col_name]}")

quantity_analysis(exploded_items, "Quantity", "ItemsSold.Quantity")
quantity_analysis(exploded_voids, "Quantity", "Voids.Quantity")

# ===========================================================================
# 8.4 — MONETARY SIGN AND ZERO ANALYSIS
# ===========================================================================
print("\n" + "═" * 100)
print("8.4  MONETARY SIGN AND ZERO ANALYSIS")
print("═" * 100)

monetary_targets = [
    (exploded_items,    "GrossPrice", "ItemsSold.GrossPrice"),
    (exploded_items,    "SoldPrice",  "ItemsSold.SoldPrice"),
    (exploded_discounts,"Amount",     "Discounts.Amount"),
    (exploded_payments, "Amount",     "Payments.Amount"),
    (exploded_payments, "Gratuity",   "Payments.Gratuity"),
    (exploded_voids,    "Amount",     "Voids.Amount"),
]
# Tax.Amount (might be struct or array<struct>)
tax_amt_df = get_numeric_df(checks_df, "Tax", "Amount")
if tax_amt_df is not None:
    monetary_targets.append((tax_amt_df, "Amount", "Tax.Amount"))

monetary_results = {}
for df, cname, label in monetary_targets:
    monetary_results[label] = analyze_numeric_field(df, cname, label)

# ===========================================================================
# 8.5 — SPECIAL IDENTIFIER VALUES
# ===========================================================================
print("\n" + "═" * 100)
print("8.5  SPECIAL IDENTIFIER VALUES")
print("═" * 100)

# EmployeeId = 0 in each array
for arr_name, expl_df in [("ItemsSold", exploded_items), ("Discounts", exploded_discounts),
                          ("Payments", exploded_payments), ("Voids", exploded_voids)]:
    if expl_df is not None and "EmployeeId" in expl_df.columns:
        total_e = expl_df.count()
        emp0 = expl_df.filter(col("EmployeeId") == 0).count()
        emp_neg = expl_df.filter(col("EmployeeId") < 0).count()
        print(f"  {arr_name}.EmployeeId == 0 : {emp0:,} / {total_e:,}  ({pct(emp0, total_e):.4f}%)")
        if emp_neg > 0:
            print(f"  {arr_name}.EmployeeId < 0   : {emp_neg:,} / {total_e:,}  ({pct(emp_neg, total_e):.4f}%)")

# Check Id <= 0
id_le0 = checks_df.filter(col("Id") <= 0).count()
print(f"\n  Check Id <= 0 : {id_le0:,}  ({pct(id_le0, TOTAL_CHECKS):.4f}%)")

# Other top-level IDs <= 0
for id_col in ["LocationId", "RevenueCenterId", "ManagerId"]:
    if id_col in checks_df.columns:
        n = checks_df.filter(col(id_col) <= 0).count()
        print(f"  {id_col} <= 0   : {n:,}  ({pct(n, TOTAL_CHECKS):.4f}%)")

# ItemId <= 0 in ItemsSold
if exploded_items is not None and "ItemId" in exploded_items.columns:
    n = exploded_items.filter(col("ItemId") <= 0).count()
    print(f"  ItemsSold.ItemId <= 0 : {n:,}  ({pct(n, total_items):.4f}%)")

# RevenueCenterId in ItemsSold (if present)
if exploded_items is not None and "RevenueCenterId" in exploded_items.columns:
    n = exploded_items.filter(col("RevenueCenterId") <= 0).count()
    print(f"  ItemsSold.RevenueCenterId <= 0 : {n:,}  ({pct(n, total_items):.4f}%)")

# ===========================================================================
# 8.6 — EMPLOYEE STATUS CONSISTENCY
# ===========================================================================
print("\n" + "═" * 100)
print("8.6  EMPLOYEE STATUS CONSISTENCY")
print("═" * 100)

if emp_df is not None:
    emp_id_col    = find_col(emp_df, ["EmployeeId", "Id", "EmployeeID", "employee_id"])
    deleted_col   = find_col(emp_df, ["Deleted", "deleted", "IsDeleted", "is_deleted"])
    hiredate_col  = find_col(emp_df, ["HireDate", "hiredate", "hire_date", "HiringDate"])
    dob_col       = find_col(emp_df, ["DateOfBirth", "dateofbirth", "DOB", "dob", "BirthDate"])
    print(f"  Employee dim ID col  : {emp_id_col}")
    print(f"  Deleted col          : {deleted_col}")
    print(f"  HireDate col         : {hiredate_col}")
    print(f"  DateOfBirth col      : {dob_col}")

    if emp_id_col and deleted_col:
        _emp_select_exprs = [
            col(emp_id_col).alias("e_id"),
            col(deleted_col).alias("e_deleted"),
        ]
        if hiredate_col:
            _emp_select_exprs.append(to_date(col(hiredate_col), "yyyy-MM-dd").alias("e_hire"))
        if dob_col:
            _emp_select_exprs.append(to_date(col(dob_col), "yyyy-MM-dd").alias("e_dob"))
        emp_dim_sel = emp_df.select(*_emp_select_exprs)

        # Distinct EmployeeIds from all transaction arrays
        tx_emp_parts = []
        for expl_df in [exploded_items, exploded_payments, exploded_voids, exploded_discounts]:
            if expl_df is not None and "EmployeeId" in expl_df.columns:
                tx_emp_parts.append(expl_df.select(col("EmployeeId").alias("emp_id")))
        if tx_emp_parts:
            all_tx_emps = tx_emp_parts[0]
            for p in tx_emp_parts[1:]:
                all_tx_emps = all_tx_emps.unionByName(p)
            distinct_tx = all_tx_emps.filter(col("emp_id").isNotNull() & (col("emp_id") != 0)).distinct()
            total_distinct_tx = distinct_tx.count()
            emp_zero_count = all_tx_emps.filter(col("emp_id") == 0).count()

            joined = distinct_tx.join(emp_dim_sel, col("emp_id") == col("e_id"), "left")
            matched_deleted  = joined.filter(col("e_deleted") == True).count()
            matched_active   = joined.filter(col("e_deleted") == False).count()
            unmatched        = joined.filter(col("e_id").isNull()).count()

            print(f"\n  Distinct transaction EmployeeIds (excl. 0) : {total_distinct_tx:,}")
            print(f"  Matched Deleted=true                     : {matched_deleted:,}")
            print(f"  Matched Deleted=false                    : {matched_active:,}")
            print(f"  Unmatched (not in employees dim)         : {unmatched:,}")
            print(f"  Transaction records with EmployeeId=0    : {emp_zero_count:,}")

            # Transaction record counts by Deleted status (via ItemsSold)
            if exploded_items is not None and "EmployeeId" in exploded_items.columns:
                emp_recs = (exploded_items
                    .filter(col("EmployeeId").isNotNull() & (col("EmployeeId") != 0))
                    .groupBy("EmployeeId").agg(count("*").alias("n_recs")))
                emp_status = emp_recs.join(
                    emp_dim_sel.select("e_id", "e_deleted"),
                    col("EmployeeId") == col("e_id"), "left")
                print(f"\n  Item-level transaction records by Deleted status:")
                emp_status.groupBy("e_deleted").agg(
                    _sum("n_recs").alias("total_records")).orderBy(desc("total_records")).show(truncate=False)

            # Transactions before HireDate
            if hiredate_col and exploded_items is not None and "OrderTime" in exploded_items.columns:
                print(f"\n  ── Transactions before HireDate ──")
                tx_hire = (exploded_items
                    .filter(col("EmployeeId").isNotNull() & (col("EmployeeId") != 0) & col("OrderTime").isNotNull())
                    .join(emp_dim_sel.select("e_id", "e_hire"),
                          col("EmployeeId") == col("e_id"), "inner")
                    .withColumn("_order", to_timestamp(col("OrderTime"), FMT_TS)))
                if "e_hire" in tx_hire.columns:
                    before_hire = tx_hire.filter(col("_order").isNotNull() & col("e_hire").isNotNull() & (col("_order") < col("e_hire"))).count()
                    print(f"  Item records with OrderTime before HireDate : {before_hire:,}")
                    if before_hire > 0:
                        for r in tx_hire.filter(
                                col("_order").isNotNull() & col("e_hire").isNotNull() & (col("_order") < col("e_hire"))
                            ).select("EmployeeId", "LocationId", "BusinessDate", "Id", "OrderTime", "e_hire"
                            ).limit(5).collect():
                            print(f"    EmpId={r['EmployeeId']} Loc={r['LocationId']} Date={r['BusinessDate']} Id={r['Id']}  Order={r['OrderTime']}  Hire={r['e_hire']}")
else:
    print("  ⚠ Employees dimension not found — SKIPPED")

# ===========================================================================
# 8.7 — DATE CONSISTENCY
# ===========================================================================
print("\n" + "═" * 100)
print("8.7  DATE CONSISTENCY")
print("═" * 100)

# DateOfBirth after analysis period
if emp_df is not None and dob_col:
    dob_parsed = emp_df.withColumn("_dob", to_date(col(dob_col), "yyyy-MM-dd"))
    # Analysis period: min and max BusinessDate from checks_df
    date_range = base.agg(_min("_bdate").alias("d_min"), _max("_bdate").alias("d_max")).collect()[0]
    analysis_end = date_range["d_max"]
    analysis_start = date_range["d_min"]
    print(f"  Analysis period (BusinessDate) : {analysis_start}  to  {analysis_end}")
    dob_after = dob_parsed.filter(col("_dob").isNotNull() & (col("_dob") > analysis_end)).count()
    total_emp = emp_df.count()
    print(f"  Employees with DateOfBirth after analysis end : {dob_after:,} / {total_emp:,}  ({pct(dob_after, total_emp):.4f}%)")

# Location OpenDate after transactions / ClosedDate before transactions
if loc_df is not None:
    loc_id_col      = find_col(loc_df, ["LocationId", "Id", "LocationID", "location_id"])
    opendate_col    = find_col(loc_df, ["OpenDate", "opendate", "OpeningDate"])
    closeddate_col  = find_col(loc_df, ["ClosedDate", "closeddate", "ClosingDate"])
    print(f"  Location dim ID col   : {loc_id_col}")
    print(f"  OpenDate col          : {opendate_col}")
    print(f"  ClosedDate col        : {closeddate_col}")

    if loc_id_col:
        check_dates = base.select("LocationId", "_bdate").distinct()
        _loc_select_exprs = [col(loc_id_col).alias("l_id")]
        if opendate_col:
            _loc_select_exprs.append(to_date(col(opendate_col), "yyyy-MM-dd").alias("l_open"))
        if closeddate_col:
            _loc_select_exprs.append(to_date(col(closeddate_col), "yyyy-MM-dd").alias("l_closed"))
        loc_sel = loc_df.select(*_loc_select_exprs)

        joined_loc = check_dates.join(loc_sel, col("LocationId") == col("l_id"), "left")

        if opendate_col:
            open_after = joined_loc.filter(
                col("l_open").isNotNull() & (col("_bdate") < col("l_open"))).count()
            print(f"  Checks at location before its OpenDate : {open_after:,}")
        if closeddate_col:
            closed_before = joined_loc.filter(
                col("l_closed").isNotNull() & (col("_bdate") > col("l_closed"))).count()
            print(f"  Checks at location after its ClosedDate : {closed_before:,}")
else:
    print("  ⚠ Locations dimension not found — SKIPPED")

# ===========================================================================
# 8.8 — CROSS-FIELD BUSINESS CONSISTENCY
# ===========================================================================
print("\n" + "═" * 100)
print("8.8  CROSS-FIELD BUSINESS CONSISTENCY")
print("═" * 100)

# GrossPrice vs SoldPrice
if exploded_items is not None and "GrossPrice" in exploded_items.columns and "SoldPrice" in exploded_items.columns:
    gp_sp = exploded_items.filter(col("GrossPrice").isNotNull() & col("SoldPrice").isNotNull())
    total_gp_sp = gp_sp.count()
    sold_gt_gross = gp_sp.filter(col("SoldPrice") > col("GrossPrice")).count()
    sold_neg_gp_pos = gp_sp.filter((col("SoldPrice") < 0) & (col("GrossPrice") > 0)).count()
    print(f"  Items where SoldPrice > GrossPrice      : {sold_gt_gross:,}  ({pct(sold_gt_gross, total_gp_sp):.4f}%)")
    print(f"  Items where SoldPrice<0 & GrossPrice>0   : {sold_neg_gp_pos:,}  ({pct(sold_neg_gp_pos, total_gp_sp):.4f}%)")
    if sold_gt_gross > 0:
        print(f"  Examples (SoldPrice > GrossPrice):")
        for r in gp_sp.filter(col("SoldPrice") > col("GrossPrice")).select(
                "LocationId", "BusinessDate", "Id", "GrossPrice", "SoldPrice").limit(5).collect():
            print(f"    Loc={r['LocationId']} Date={r['BusinessDate']} Id={r['Id']}  Gross={r['GrossPrice']}  Sold={r['SoldPrice']}")

# Payment boolean flags vs PayCatName
if exploded_payments is not None and "PayCatName" in exploded_payments.columns:
    pay_struct = checks_df.schema["Payments"].dataType.elementType
    bool_fields = [f.name for f in pay_struct.fields if isinstance(f.dataType, BooleanType)]
    print(f"\n  ── Payment flag consistency ──")
    print(f"  Boolean fields in Payments struct : {bool_fields}")
    if bool_fields:
        # Multiple flags true simultaneously
        sum_expr = reduce(lambda a, b: a + b, [col(f).cast("int") for f in bool_fields])
        multi_flag = exploded_payments.withColumn("_n_true", sum_expr).filter(col("_n_true") > 1)
        multi_count = multi_flag.count()
        print(f"  Payment records with >1 boolean flag true : {multi_count:,}")
        if multi_count > 0:
            print(f"  Top PayCatName for multi-flag records:")
            multi_flag.groupBy("PayCatName").agg(count("*").alias("n")).orderBy(desc("n")).show(10, truncate=False)

        # Zero flags true (no category flag set)
        no_flag = exploded_payments.withColumn("_n_true", sum_expr).filter(col("_n_true") == 0)
        no_flag_count = no_flag.count()
        print(f"  Payment records with 0 boolean flags true  : {no_flag_count:,}")
        if no_flag_count > 0:
            print(f"  Top PayCatName for zero-flag records:")
            no_flag.groupBy("PayCatName").agg(count("*").alias("n")).orderBy(desc("n")).show(10, truncate=False)

    # PayCatName distribution
    print(f"\n  PayCatName distribution (top 20):")
    exploded_payments.groupBy("PayCatName").agg(
        count("*").alias("n"),
        _round(count("*") * 100.0 / total_payments, 4).alias("pct")
    ).orderBy(desc("n")).show(20, truncate=False)

# Void Amount / Quantity combinations
if exploded_voids is not None and "Amount" in exploded_voids.columns and "Quantity" in exploded_voids.columns:
    print(f"\n  ── Void Amount / Quantity combinations ──")
    vq = exploded_voids.filter(col("Amount").isNotNull() & col("Quantity").isNotNull())
    total_vq = vq.count()
    amt_pos_qty_neg = vq.filter((col("Amount") > 0) & (col("Quantity") < 0)).count()
    amt_neg_qty_pos = vq.filter((col("Amount") < 0) & (col("Quantity") > 0)).count()
    amt_zero_qty_nonzero = vq.filter((col("Amount") == 0) & (col("Quantity") != 0)).count()
    amt_nonzero_qty_zero = vq.filter((col("Amount") != 0) & (col("Quantity") == 0)).count()
    print(f"  Void: Amount>0 & Quantity<0   : {amt_pos_qty_neg:,}  ({pct(amt_pos_qty_neg, total_vq):.4f}%)")
    print(f"  Void: Amount<0 & Quantity>0   : {amt_neg_qty_pos:,}  ({pct(amt_neg_qty_pos, total_vq):.4f}%)")
    print(f"  Void: Amount=0 & Quantity!=0  : {amt_zero_qty_nonzero:,}  ({pct(amt_zero_qty_nonzero, total_vq):.4f}%)")
    print(f"  Void: Amount!=0 & Quantity=0  : {amt_nonzero_qty_zero:,}  ({pct(amt_nonzero_qty_zero, total_vq):.4f}%)")

# ===========================================================================
# FINAL OUTPUT — PRIORITIZED SUMMARY TABLE
# ===========================================================================
print("\n" + "═" * 100)
print("FINAL OUTPUT — PRIORITIZED FINDINGS TABLE")
print("═" * 100)

# Populate findings based on results (conditionally)
add_finding(
    "CloseTime before OpenTime", "checks_df.OpenTime/CloseTime",
    close_before_open, pct(close_before_open, TOTAL_CHECKS),
    "Medium" if close_before_open > 0 else "Low",
    "Potential fraud-relevant anomaly" if close_before_open > 0 else "Likely legitimate/system behavior",
    "Investigate before cleaning" if close_before_open > 0 else "Keep")

add_finding(
    "Zero-duration checks (OpenTime = CloseTime)", "checks_df.OpenTime/CloseTime",
    open_eq_close, pct(open_eq_close, TOTAL_CHECKS),
    "Medium" if pct(open_eq_close, TOTAL_CHECKS) > 0.1 else "Low",
    "Uncertain — business validation required",
    "Keep and flag")

if 'ot_before_open' in dir() and total_items > 0:
    add_finding(
        "OrderTime before OpenTime", "ItemsSold.OrderTime",
        ot_before_open, pct(ot_before_open, total_items),
        "Medium" if ot_before_open > 0 else "Low",
        "Uncertain — business validation required" if ot_before_open > 0 else "Likely legitimate/system behavior",
        "Investigate before cleaning" if ot_before_open > 0 else "Keep")

add_finding(
    "Checks with no ItemsSold", "checks_df.ItemsSold (empty)",
    c_no_items, pct(c_no_items, TOTAL_CHECKS),
    "Medium",
    "Uncertain — business validation required",
    "Keep and flag")

add_finding(
    "Checks with no Payments", "checks_df.Payments (empty)",
    c_no_payments, pct(c_no_payments, TOTAL_CHECKS),
    "Medium",
    "Uncertain — business validation required",
    "Keep and flag")

add_finding(
    "Checks with neither Items nor Payments", "checks_df.ItemsSold/Payments (empty)",
    c_neither, pct(c_neither, TOTAL_CHECKS),
    "High" if c_neither > 100 else "Medium",
    "Likely data-quality issue",
    "Candidate for correction/exclusion")

# Quantity findings
if exploded_items is not None and "Quantity" in exploded_items.columns:
    nn = exploded_items.filter(col("Quantity").isNotNull())
    q_total = nn.count()
    q_neg = nn.filter(col("Quantity") < 0).count()
    q_zero = nn.filter(col("Quantity") == 0).count()
    add_finding("Negative quantities", "ItemsSold.Quantity", q_neg, pct(q_neg, q_total),
        "Medium" if q_neg > 0 else "Low",
        "Uncertain — business validation required" if q_neg > 0 else "Likely legitimate/system behavior",
        "Keep and flag" if q_neg > 0 else "Keep")
    add_finding("Zero quantities", "ItemsSold.Quantity", q_zero, pct(q_zero, q_total),
        "Low",
        "Uncertain — business validation required" if q_zero > 0 else "Likely legitimate/system behavior",
        "Keep and flag" if q_zero > 0 else "Keep")

# Monetary findings
for label, res in monetary_results.items():
    if res and res.get("total", 0) > 0:
        neg = res["negative"]
        zero = res["zero"]
        if neg > 0:
            add_finding(f"Negative values in {label}", label, neg, pct(neg, res["total"]),
                "Medium" if pct(neg, res["total"]) < 5 else "High",
                "Uncertain — business validation required" if "Gratuity" in label else "Potential fraud-relevant anomaly",
                "Keep and flag" if "Gratuity" in label else "Investigate before cleaning")
        if zero > 0 and pct(zero, res["total"]) > 0.1:
            add_finding(f"Zero values in {label}", label, zero, pct(zero, res["total"]),
                "Low",
                "Uncertain — business validation required",
                "Keep and flag")

# Employee findings
if 'matched_deleted' in dir() and 'total_distinct_tx' in dir():
    if total_distinct_tx > 0:
        add_finding("Deleted employees with transactions", "employees.Deleted vs ItemsSold.EmployeeId",
            matched_deleted, pct(matched_deleted, total_distinct_tx),
            "Medium",
            "Likely legitimate/system behavior",
            "Keep and flag")
        if unmatched > 0:
            add_finding("Transaction EmployeeIds not in employees dim", "ItemsSold.EmployeeId",
                unmatched, pct(unmatched, total_distinct_tx),
                "High" if pct(unmatched, total_distinct_tx) > 10 else "Medium",
                "Potential fraud-relevant anomaly",
                "Investigate before cleaning")

# Print table
header = f"{'Issue':<45} {'Structure/Field':<35} {'Records':>12} {'%':>10} {'Severity':<10} {'Interpretation':<40} {'Treatment':<30}"
print(header)
print("-" * len(header))
for f in findings:
    print(f"{f['Issue']:<45} {f['Structure/Field']:<35} {f['Records Affected']:>12,} {f['%']:>10.4f} {f['Severity']:<10} {f['Initial Interpretation']:<40} {f['Recommended Treatment']:<30}")

# ===========================================================================
# 5 MOST IMPORTANT FINDINGS FOR FRAUD-FOCUSED EDA
# ===========================================================================
print("\n" + "═" * 100)
print("TOP 5 FINDINGS FOR FRAUD-FOCUSED EDA")
print("═" * 100)
print("""
1. CHECKS WITH NEITHER ITEMS NOR PAYMENTS — These checks carry no transactional
   payload.  Whether they represent system artifacts, training data, or manual
   overrides, they warrant investigation as candidates for exclusion and as
   potential indicators of manipulated or voided-out transactions.

2. NEGATIVE MONETARY VALUES (esp. Payments.Gratuity, Discounts.Amount, Voids.Amount)
   — The sign conventions are uncertain.  Negative gratuity with positive payment
   suggests an accounting/POS convention rather than error, but the pattern and
   concentration of negative values across locations/employees should be profiled
   for fraud-relevant anomalies.

3. CLOSETIME BEFORE OPENTIME / ORDERTIME OUTSIDE CHECK WINDOW — Temporal
   inconsistencies may indicate system clock issues, manual edits, or unusual
   operational patterns.  Cross-midnight transactions are legitimate, but
   checks where the close precedes the open are structurally impossible and
   should be investigated before any cleaning.

4. UNMATCHED EMPLOYEE IDS / DELETED EMPLOYEES WITH TRANSACTIONS — If a
   significant number of transaction EmployeeIds do not match the employees
   dimension, or if Deleted=true employees are actively transacting, this
   could signal orphaned system IDs, role/permission abuse, or data-integrity
   gaps.  These do NOT prove fraud but define high-priority sub-populations.

5. EMPTY ARRAY CHECKS WITH DISCOUNTS / VOIDS / TAX / TRAFFICCOUNT — Checks
   with no items or payments but with voids, discounts, or tax amounts may
   represent voided transactions, comp entries, or system-generated records.
   Their spatial and temporal concentration is a fraud-relevant signal.
""")
print("=" * 100)
print("STEP 8 — DIAGNOSTIC COMPLETE  (no data was modified)")
print("=" * 100)

# COMMAND ----------

#La base presenta alta consistencia general y pocas anomalías estructurales. No existen checks con duración negativa ni cierres anteriores a la apertura, y más del 99% de las cantidades son positivas. Los principales casos que requieren seguimiento son checks con duración cero, pequeñas inconsistencias entre OrderTime y la ventana del check, y 36,658 registros realizados antes de la HireDate registrada del empleado.    Markdown pegado

#Los checks con arrays vacíos no deben eliminarse automáticamente. Se encontraron 11,508 checks sin ItemsSold y 3,856 sin Payments; sin embargo, una parte importante de los casos analizados con arrays vacíos contiene Voids (8,990; 63.3%), lo que sugiere operaciones anuladas o especiales que pueden ser relevantes para detectar patrones sospechosos.    Markdown pegado

#Los valores monetarios extremos deben conservarse. Los precios, pagos, impuestos y voids negativos son poco frecuentes y pueden representar reversos u operaciones especiales. Gratuity es la excepción: 46.28% de sus valores son negativos, reforzando la posibilidad de una convención del POS que todavía no debemos modificar.    Markdown pegado

#La integridad de empleados y ubicaciones es alta. Todos los EmployeeIds transaccionales distintos de 0 encuentran correspondencia en employees; tampoco existen transacciones de ubicaciones antes de su apertura o después de su cierre. Los empleados Deleted=true sí presentan transacciones, por lo que no deben eliminarse automáticamente.

# COMMAND ----------

# MAGIC %md
# MAGIC 9. Outliers

# COMMAND ----------

# ============================================================================
# STEP 9 — OUTLIERS: MONETARY VALUES  (diagnostic only, read-only)
# ============================================================================
# NO raw data is modified, filtered, imputed, capped, winsorized, or removed.
# This step is purely diagnostic.
#
# Goal: identify extreme monetary values using the IQR method and high
# percentiles, and display the top-5 records for each variable.
# Outliers are NOT classified as errors or fraud.
# Gratuity is NOT transformed.
#
# Confirmed check key:  LocationId + BusinessDate + Id
# ============================================================================

from pyspark.sql.functions import (
    col, count, when, lit, isnull, desc, asc
)

print("=" * 100)
print("STEP 9 — OUTLIERS: MONETARY VALUES  (Diagnostic Only — Read-Only)")
print("=" * 100)

# ------------------------------------------------------------------
# Reuse exploded views from Step 8 (read-only; original data untouched)
# ------------------------------------------------------------------
exploded_items     = exploded_items     if 'exploded_items'     in dir() else make_exploded(checks_df, "ItemsSold")
exploded_payments  = exploded_payments  if 'exploded_payments'  in dir() else make_exploded(checks_df, "Payments")
exploded_voids     = exploded_voids     if 'exploded_voids'     in dir() else make_exploded(checks_df, "Voids")
exploded_discounts = exploded_discounts if 'exploded_discounts' in dir() else make_exploded(checks_df, "Discounts")

# ------------------------------------------------------------------
# Helper: full outlier report for one numeric field
# ------------------------------------------------------------------
def outlier_report(df, col_name, label):
    """
    Diagnostic report for a single numeric column in an exploded DF.
    Prints Q1/median/Q3, IQR, upper IQR threshold, count & % above threshold,
    P95/P99/max, and the 5 highest-value records.
    """
    if df is None or col_name not in df.columns:
        print(f"\n  ⚠ {label}: field '{col_name}' not available — SKIPPED")
        return None

    nn_df = df.filter(col(col_name).isNotNull())
    nn = nn_df.count()
    if nn == 0:
        print(f"\n  ── {label} ({col_name}) ──  No non-null values")
        return None

    # Quartiles and high percentiles via approxQuantile
    try:
        q = nn_df.stat.approxQuantile(col_name, [0.25, 0.50, 0.75, 0.95, 0.99], 0.001)
        q1, median, q3, p95, p99 = q[0], q[1], q[2], q[3], q[4]
    except Exception:
        q1 = median = q3 = p95 = p99 = None

    max_val = nn_df.agg({col_name: "max"}).collect()[0][0]

    iqr = (q3 - q1) if (q1 is not None and q3 is not None) else None
    upper_thr = (q3 + 1.5 * iqr) if iqr is not None else None

    # Count above upper IQR threshold
    if upper_thr is not None:
        n_above = nn_df.filter(col(col_name) > upper_thr).count()
        pct_above = round(100.0 * n_above / nn, 4) if nn > 0 else 0.0
    else:
        n_above = None
        pct_above = None

    # ── Print report ──
    print(f"\n  ── {label} ({col_name}) ──")
    print(f"    Total non-null      : {nn:,}")
    print(f"    Q1                   : {q1}")
    print(f"    Median               : {median}")
    print(f"    Q3                   : {q3}")
    print(f"    IQR (Q3 − Q1)        : {iqr}")
    print(f"    Upper IQR threshold  : {upper_thr}")
    print(f"    Above threshold      : {n_above:,}  ({pct_above:.4f}% of non-null)" if n_above is not None else "    Above threshold      : N/A")
    print(f"    P95                  : {p95}")
    print(f"    P99                  : {p99}")
    print(f"    Max                  : {max_val}")

    # ── Top-5 highest values ──
    select_cols = ["LocationId", "BusinessDate", "Id"]
    if "EmployeeId" in df.columns:
        select_cols += ["EmployeeId"]
    select_cols += [col_name]

    print(f"    Top 5 highest values:")
    top5 = (nn_df
            .select(*select_cols)
            .orderBy(desc(col_name))
            .limit(5)
            .collect())
    for i, r in enumerate(top5, 1):
        emp_str = f"  Emp={r['EmployeeId']}" if "EmployeeId" in r and r['EmployeeId'] is not None else ""
        print(f"      {i}. Loc={r['LocationId']}  Date={r['BusinessDate']}  Id={r['Id']}{emp_str}  {col_name}={r[col_name]}")

    return {
        "total_non_null": nn,
        "Q1": q1,
        "median": median,
        "Q3": q3,
        "IQR": iqr,
        "upper_iqr_threshold": upper_thr,
        "n_above": n_above,
        "pct_above": pct_above,
        "P95": p95,
        "P99": p99,
        "max": max_val,
    }

# ------------------------------------------------------------------
# Run outlier reports for each monetary variable
# ------------------------------------------------------------------
outlier_results = {}

targets = [
    (exploded_items,     "SoldPrice", "ItemsSold.SoldPrice"),
    (exploded_discounts, "Amount",    "Discounts.Amount"),
    (exploded_payments,  "Amount",    "Payments.Amount"),
    (exploded_payments,  "Gratuity",  "Payments.Gratuity"),
    (exploded_voids,     "Amount",    "Voids.Amount"),
]

for df, cname, label in targets:
    outlier_results[label] = outlier_report(df, cname, label)

# ------------------------------------------------------------------
# Summary table
# ------------------------------------------------------------------
print("\n" + "═" * 100)
print("9.1  OUTLIER SUMMARY TABLE")
print("═" * 100)
hdr = (f"  {'Variable':<25s} {'N':>10s} {'Q1':>12s} {'Med':>12s} {'Q3':>12s} "
       f"{'IQR':>12s} {'UpperThr':>14s} {'AboveN':>10s} {'Above%':>8s} {'P95':>12s} {'P99':>12s} {'Max':>14s}")
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for label, r in outlier_results.items():
    if r is None:
        print(f"  {label:<25s}  — not available")
        continue
    def fmt(v, w):
        return f"{v:>{w},}" if isinstance(v, (int, float)) and v is not None else f"{'N/A':>{w}s}"
    print(f"  {label:<25s} {r['total_non_null']:>10,} "
          f"{fmt(r['Q1'], 12)} {fmt(r['median'], 12)} {fmt(r['Q3'], 12)} "
          f"{fmt(r['IQR'], 12)} {fmt(r['upper_iqr_threshold'], 14)} "
          f"{fmt(r['n_above'], 10)} {fmt(r['pct_above'], 8)} "
          f"{fmt(r['P95'], 12)} {fmt(r['P99'], 12)} {fmt(r['max'], 14)}")

print("\n  ⚠ Outliers are NOT classified as errors or fraud.")
print("  ⚠ Gratuity was NOT transformed — values reported as-is.")
print("\n" + "=" * 100)

# COMMAND ----------

#Las variables monetarias presentan distribuciones asimétricas y valores extremos relevantes, especialmente en descuentos, pagos y voids. Debido a la estructura particular de los datos, el método IQR puede sobreidentificar outliers en variables con alta concentración de ceros, por lo que los valores extremos se conservarán y se analizarán mediante percentiles y patrones por empleado y ubicación, en lugar de eliminarse automáticamente.

# COMMAND ----------

# ============================================================================
# STEP 9 (cont.) — EXTREME VALUE CONCENTRATION  (diagnostic only, read-only)
# ============================================================================
# NO raw data is modified, filtered, imputed, capped, winsorized, or removed.
# This step is purely diagnostic.
#
# For Discounts.Amount, Payments.Amount, and Voids.Amount we use P99 as the
# extreme-value threshold.  For values strictly above P99 we report:
#   - number of records
#   - number of distinct employees
#   - top 10 employees by count  (count + total amount)
#   - top 10 locations by count  (count + total amount)
#
# No employee or location is classified as fraud.
#
# Confirmed check key:  LocationId + BusinessDate + Id
# ============================================================================

from pyspark.sql.functions import col, count, sum as _sum, desc, collect_set, size as _size

print("\n" + "=" * 100)
print("STEP 9 (cont.) — EXTREME VALUE CONCENTRATION  (P99 threshold, Diagnostic Only)")
print("=" * 100)

# ------------------------------------------------------------------
# Exploded views (reuse from Step 8/9 if available, otherwise rebuild)
# ------------------------------------------------------------------
def _ensure(name, builder):
    return globals().get(name) if name in globals() else builder()

exploded_discounts = _ensure("exploded_discounts", lambda: make_exploded(checks_df, "Discounts"))
exploded_payments  = _ensure("exploded_payments",  lambda: make_exploded(checks_df, "Payments"))
exploded_voids     = _ensure("exploded_voids",     lambda: make_exploded(checks_df, "Voids"))

# ------------------------------------------------------------------
# Helper: determine the EmployeeId column in an exploded DF
# ------------------------------------------------------------------
def _emp_col(df):
    for c in ("EmployeeId", "employeeId", "EmployeeID"):
        if c in df.columns:
            return c
    return None

# ------------------------------------------------------------------
# Core function: concentration report for one exploded monetary field
# ------------------------------------------------------------------
def concentration_report(df, col_name, label):
    """
    For values strictly above P99, print:
      - number of records
      - number of distinct employees
      - top 10 employees by count (count + total amount)
      - top 10 locations by count  (count + total amount)
    No fraud classification.
    """
    if df is None or col_name not in df.columns:
        print(f"\n  ⚠ {label}: field '{col_name}' not available — SKIPPED")
        return None

    nn_df = df.filter(col(col_name).isNotNull())
    nn = nn_df.count()
    if nn == 0:
        print(f"\n  ── {label} ({col_name}) ──  No non-null values")
        return None

    # P99 threshold
    try:
        p99 = nn_df.stat.approxQuantile(col_name, [0.99], 0.001)[0]
    except Exception:
        p99 = None

    if p99 is None:
        print(f"\n  ⚠ {label}: could not compute P99 — SKIPPED")
        return None

    print(f"\n  {'═' * 96}")
    print(f"  {label}  ({col_name})")
    print(f"  {'═' * 96}")
    print(f"    Total non-null records : {nn:,}")
    print(f"    P99 threshold          : {p99:,.4f}")

    # Records strictly above P99
    extreme = nn_df.filter(col(col_name) > p99)
    n_extreme = extreme.count()
    print(f"    Records above P99      : {n_extreme:,}  ({pct(n_extreme, nn):.4f}% of non-null)")

    if n_extreme == 0:
        print(f"    No records above P99 — nothing to report.")
        return {"label": label, "p99": p99, "n_extreme": 0}

    # ── Number of distinct employees ──
    emp_c = _emp_col(df)
    if emp_c is not None:
        n_emp = extreme.select(col(emp_c)).distinct().count()
        print(f"    Distinct employees (above P99): {n_emp:,}")
    else:
        n_emp = None
        print(f"    Distinct employees: EmployeeId column not available in exploded view")

    # ── Top 10 employees by count ──
    if emp_c is not None:
        print(f"\n    Top 10 employees by count of extreme records:")
        print(f"      {'EmployeeId':<15s} {'Count':>8s} {'TotalAmount':>16s}")
        print(f"      {'-'*15} {'-'*8} {'-'*16}")
        emp_top = (extreme
                   .groupBy(col(emp_c).alias("EmployeeId"))
                   .agg(count("*").alias("cnt"),
                        _sum(col(col_name)).alias("total_amt"))
                   .orderBy(desc("cnt"))
                   .limit(10)
                   .collect())
        for r in emp_top:
            print(f"      {str(r['EmployeeId']):<15s} {r['cnt']:>8,} {r['total_amt']:>16,.2f}")
    else:
        print(f"\n    Top 10 employees: EmployeeId column not available — SKIPPED")

    # ── Top 10 locations by count ──
    loc_c = "LocationId" if "LocationId" in df.columns else None
    if loc_c is not None:
        print(f"\n    Top 10 locations by count of extreme records:")
        print(f"      {'LocationId':<15s} {'Count':>8s} {'TotalAmount':>16s}")
        print(f"      {'-'*15} {'-'*8} {'-'*16}")
        loc_top = (extreme
                   .groupBy(col(loc_c).alias("LocationId"))
                   .agg(count("*").alias("cnt"),
                        _sum(col(col_name)).alias("total_amt"))
                   .orderBy(desc("cnt"))
                   .limit(10)
                   .collect())
        for r in loc_top:
            print(f"      {str(r['LocationId']):<15s} {r['cnt']:>8,} {r['total_amt']:>16,.2f}")
    else:
        print(f"\n    Top 10 locations: LocationId column not available — SKIPPED")

    print(f"\n    ⚠ No employee or location is classified as fraud. Values are reported as-is.")
    return {"label": label, "p99": p99, "n_extreme": n_extreme, "n_employees": n_emp}

# ------------------------------------------------------------------
# Run concentration reports
# ------------------------------------------------------------------
concentration_targets = [
    (exploded_discounts, "Amount", "Discounts.Amount"),
    (exploded_payments,  "Amount", "Payments.Amount"),
    (exploded_voids,     "Amount", "Voids.Amount"),
]

concentration_results = {}
for df, cname, label in concentration_targets:
    concentration_results[label] = concentration_report(df, cname, label)

# ------------------------------------------------------------------
# Compact summary
# ------------------------------------------------------------------
print(f"\n{'═' * 100}")
print("EXTREME VALUE CONCENTRATION — SUMMARY")
print(f"{'═' * 100}")
print(f"  {'Variable':<25s} {'P99':>14s} {'Records>':>10s} {'DistEmp':>10s}")
print(f"  {'-'*25} {'-'*14} {'-'*10} {'-'*10}")
for label, r in concentration_results.items():
    if r is None:
        print(f"  {label:<25s}  — not available")
        continue
    p99_str = f"{r['p99']:>14,.4f}" if r['p99'] is not None else f"{'N/A':>14s}"
    ne_str  = f"{r['n_extreme']:>10,}"
    emp_str = f"{r['n_employees']:>10,}" if r.get('n_employees') is not None else f"{'N/A':>10s}"
    print(f"  {label:<25s} {p99_str} {ne_str} {emp_str}")

print(f"\n  ⚠ Diagnostic only — no data was modified.")
print(f"  ⚠ No employee or location is classified as fraud.")
print(f"\n{'=' * 100}")

# COMMAND ----------

#Las variables monetarias presentan distribuciones asimétricas con valores extremos relevantes, especialmente en descuentos, pagos y voids. Los eventos superiores al percentil 99 representan aproximadamente 1% de los registros y se distribuyen entre múltiples empleados y ubicaciones, por lo que no existe evidencia suficiente para considerar estos valores como errores o fraude. Los outliers se conservarán para análisis posteriores, donde deberán evaluarse en relación con el volumen total de actividad de cada empleado y ubicación.

# COMMAND ----------

# MAGIC %md
# MAGIC Cleaning decision
# MAGIC
# MAGIC La limpieza se realizó de forma conservadora debido al objetivo de detección de fraude. Se eliminaron únicamente variables sin información analítica y se corrigieron tipos de datos claramente identificados. Los valores extremos, transacciones inusuales y eventos repetidos se conservaron, ya que podrían representar señales relevantes para identificar comportamientos potencialmente sospechosos.

# COMMAND ----------

# MAGIC %md
# MAGIC Cleaning

# COMMAND ----------

from pyspark.sql import functions as F

# =========================
# CHECKS CLEAN
# =========================

checks_clean = (
    checks_df
    .drop("LoyaltyId")
    .withColumn("BusinessDate", F.to_date("BusinessDate", "yyyy-MM-dd"))
    .withColumn(
        "OpenTime",
        F.to_timestamp("OpenTime", "yyyy-MM-dd'T'HH:mm")
    )
    .withColumn(
        "CloseTime",
        F.to_timestamp("CloseTime", "yyyy-MM-dd'T'HH:mm")
    )
)

# =========================
# EMPLOYEES CLEAN
# =========================

employees_clean = (
    dim_dfs["employees/"]
    .drop("IsEmancipated")
    .withColumn("DateOfBirth", F.to_date("DateOfBirth", "yyyy-MM-dd"))
    .withColumn("HireDate", F.to_date("HireDate", "yyyy-MM-dd"))
)

# =========================
# ITEMS CLEAN
# =========================

items_clean = (
    dim_dfs["items/"]
    .drop("ExternalId")
)

# =========================
# LOCATIONS CLEAN
# =========================

locations_clean = (
    dim_dfs["locations/"]
    .withColumn("OpenDate", F.to_date("OpenDate", "yyyy-MM-dd"))
    .withColumn("ClosedDate", F.to_date("ClosedDate", "yyyy-MM-dd"))
)

# =========================
# REVENUE CENTERS CLEAN
# =========================

revenue_centers_clean = dim_dfs["revenue_centers/"]

print("Clean DataFrames created successfully.")

# COMMAND ----------

# ==========================================
# 12.2 - Convert nested ItemsSold.OrderTime
# ==========================================

checks_clean = checks_clean.withColumn(
    "ItemsSold",
    F.transform(
        "ItemsSold",
        lambda item: item.withField(
            "OrderTime",
            F.to_timestamp(
                item["OrderTime"],
                "yyyy-MM-dd'T'HH:mm"
            )
        )
    )
)

print("ItemsSold.OrderTime converted successfully.")

# COMMAND ----------

checks_clean.select(
    F.explode("ItemsSold").alias("item")
).select(
    "item.OrderTime"
).printSchema()

# COMMAND ----------

# ==========================================
# 12.3 - Create quality flags
# ==========================================

checks_clean = (
    checks_clean
    # Check con duración cero
    .withColumn(
        "flag_zero_duration",
        F.col("OpenTime") == F.col("CloseTime")
    )

    # Check sin productos
    .withColumn(
        "flag_no_items",
        F.size("ItemsSold") == 0
    )

    # Check sin pagos
    .withColumn(
        "flag_no_payments",
        F.size("Payments") == 0
    )

    # Check sin productos ni pagos
    .withColumn(
        "flag_empty_check",
        (F.size("ItemsSold") == 0) &
        (F.size("Payments") == 0)
    )
)

print("Quality flags created successfully.")

# COMMAND ----------

checks_clean.select(
    "flag_zero_duration",
    "flag_no_items",
    "flag_no_payments",
    "flag_empty_check"
).groupBy(
    "flag_zero_duration",
    "flag_no_items",
    "flag_no_payments",
    "flag_empty_check"
).count().show()

# COMMAND ----------

print("RAW rows:", checks_df.count())
print("CLEAN rows:", checks_clean.count())

print("\nCLEAN SCHEMA:")
checks_clean.printSchema()

print("\nComposite key duplicates:")
(
    checks_clean
    .groupBy("LocationId", "BusinessDate", "Id")
    .count()
    .filter(F.col("count") > 1)
    .count()
)

# COMMAND ----------

#El proceso de limpieza conservó el 100% de los 2,142,846 checks originales y mantuvo la unicidad de la llave compuesta LocationId + BusinessDate + Id. Se eliminaron únicamente variables completamente vacías, se estandarizaron las variables temporales y se incorporaron flags de calidad para conservar transacciones atípicas potencialmente relevantes para el análisis de fraude.

# COMMAND ----------

# MAGIC %md
# MAGIC Joins

# COMMAND ----------

# ==========================================
# STEP 13.1 - Join check-level dimensions
# ==========================================

checks_enriched = (
    checks_clean

    # Location
    .join(
        locations_clean.select(
            F.col("Id").alias("LocationId"),
            F.col("Name").alias("LocationName"),
            "OpenDate",
            "ClosedDate",
            "TimeZoneName"
        ),
        on="LocationId",
        how="left"
    )

    # Revenue Center
    .join(
        revenue_centers_clean.select(
            F.col("Id").alias("RevenueCenterId"),
            F.col("Name").alias("RevenueCenterName")
        ),
        on="RevenueCenterId",
        how="left"
    )
)

print("Check-level joins completed.")
print("Rows:", checks_enriched.count())

# COMMAND ----------

checks_enriched.select(
    "LocationId",
    "LocationName",
    "RevenueCenterId",
    "RevenueCenterName"
).show(10, truncate=False)

# COMMAND ----------

# ==========================================
# STEP 13.2 - Create ItemsSold event table
# ==========================================

items_events = (
    checks_enriched
    .select(
        "LocationId",
        "LocationName",
        "BusinessDate",
        F.col("Id").alias("CheckId"),
        F.explode("ItemsSold").alias("item")
    )
    .select(
        "LocationId",
        "LocationName",
        "BusinessDate",
        "CheckId",
        "item.*"
    )
)

print("Items event rows:", items_events.count())

# COMMAND ----------

items_events.select(
    "LocationId",
    "BusinessDate",
    "CheckId",
    "EmployeeId",
    "ItemId",
    "ItemName",
    "Quantity",
    "SoldPrice"
).show(10, truncate=False)

# COMMAND ----------

# ==========================================
# STEP 13.3 - Enrich ItemsSold events
# ==========================================

items_enriched = (
    items_events

    # Employee information
    .join(
        employees_clean.select(
            F.col("Id").alias("EmployeeId"),
            F.col("Name").alias("EmployeeName"),
            "HireDate",
            F.col("Deleted").alias("EmployeeDeleted")
        ),
        on="EmployeeId",
        how="left"
    )

    # Item dimension information
    .join(
        items_clean.select(
            F.col("Id").alias("ItemId"),
            F.col("Name").alias("ItemDimName"),
            "SalesCategoryId"
        ),
        on="ItemId",
        how="left"
    )
)

print("Items enriched rows:", items_enriched.count())

# COMMAND ----------

items_enriched.select(
    "BusinessDate",
    "CheckId",
    "EmployeeId",
    "EmployeeName",
    "EmployeeDeleted",
    "ItemId",
    "ItemName",
    "SalesCategoryId",
    "Quantity",
    "SoldPrice"
).show(10, truncate=False)

# COMMAND ----------

# ==========================================
# STEP 13.4 - Transaction event tables
# ==========================================

# DISCOUNTS
discount_events = (
    checks_enriched
    .select(
        "LocationId", "LocationName", "BusinessDate",
        F.col("Id").alias("CheckId"),
        F.explode("Discounts").alias("d")
    )
    .select(
        "LocationId", "LocationName", "BusinessDate",
        "CheckId", "d.*"
    )
)

# PAYMENTS
payment_events = (
    checks_enriched
    .select(
        "LocationId", "LocationName", "BusinessDate",
        F.col("Id").alias("CheckId"),
        F.explode("Payments").alias("p")
    )
    .select(
        "LocationId", "LocationName", "BusinessDate",
        "CheckId", "p.*"
    )
)

# VOIDS
void_events = (
    checks_enriched
    .select(
        "LocationId", "LocationName", "BusinessDate",
        F.col("Id").alias("CheckId"),
        F.explode("Voids").alias("v")
    )
    .select(
        "LocationId", "LocationName", "BusinessDate",
        "CheckId", "v.*"
    )
)

print("Discounts:", discount_events.count())
print("Payments:", payment_events.count())
print("Voids:", void_events.count())

# COMMAND ----------

# MAGIC %md
# MAGIC Analysis master

# COMMAND ----------

# ==========================================
# STEP 14.1 - Aggregate events to check level
# ==========================================

check_key = ["LocationId", "BusinessDate", "CheckId"]

# ITEMS
items_agg = (
    items_enriched
    .groupBy(*check_key)
    .agg(
        F.sum("Quantity").alias("total_item_quantity"),
        F.sum("SoldPrice").alias("total_sold_price"),
        F.count("*").alias("item_event_count"),
        F.countDistinct("EmployeeId").alias("item_employee_count")
    )
)

# DISCOUNTS
discounts_agg = (
    discount_events
    .groupBy(*check_key)
    .agg(
        F.count("*").alias("discount_count"),
        F.sum("Amount").alias("discount_amount"),
        F.max("Amount").alias("max_discount"),
        F.countDistinct("EmployeeId").alias("discount_employee_count")
    )
)

# PAYMENTS
payments_agg = (
    payment_events
    .groupBy(*check_key)
    .agg(
        F.count("*").alias("payment_count"),
        F.sum("Amount").alias("payment_amount"),
        F.sum("Gratuity").alias("gratuity_raw"),
        F.countDistinct("EmployeeId").alias("payment_employee_count")
    )
)

# VOIDS
voids_agg = (
    void_events
    .groupBy(*check_key)
    .agg(
        F.count("*").alias("void_count"),
        F.sum("Amount").alias("void_amount"),
        F.max("Amount").alias("max_void"),
        F.countDistinct("EmployeeId").alias("void_employee_count")
    )
)

print("Aggregations created successfully.")

# COMMAND ----------

# ==========================================
# STEP 14.2 - Build analysis_master
# ==========================================

analysis_master = (
    checks_enriched
    .withColumnRenamed("Id", "CheckId")

    .join(items_agg, check_key, "left")
    .join(discounts_agg, check_key, "left")
    .join(payments_agg, check_key, "left")
    .join(voids_agg, check_key, "left")
)

# Fill with 0 only when the absence of an event
# logically means zero activity
zero_fill_cols = [
    "total_item_quantity",
    "total_sold_price",
    "item_event_count",
    "item_employee_count",
    "discount_count",
    "discount_amount",
    "discount_employee_count",
    "payment_count",
    "payment_amount",
    "payment_employee_count",
    "void_count",
    "void_amount",
    "void_employee_count"
]

analysis_master = analysis_master.fillna(
    0,
    subset=zero_fill_cols
)

print("analysis_master created successfully.")
print("Rows:", analysis_master.count())

# COMMAND ----------

# MAGIC %md
# MAGIC Validacion final

# COMMAND ----------

# ==========================================
# STEP 15 - Final Validation
# ==========================================

print("1. ROW COUNT")
print("RAW:   ", checks_df.count())
print("MASTER:", analysis_master.count())

print("\n2. COMPOSITE KEY DUPLICATES")
duplicate_keys = (
    analysis_master
    .groupBy("LocationId", "BusinessDate", "CheckId")
    .count()
    .filter(F.col("count") > 1)
    .count()
)
print("Duplicate keys:", duplicate_keys)

print("\n3. DATE RANGE")
analysis_master.select(
    F.min("BusinessDate").alias("min_date"),
    F.max("BusinessDate").alias("max_date")
).show()

print("\n4. EVENT TOTALS")
analysis_master.select(
    F.sum("item_event_count").alias("items"),
    F.sum("discount_count").alias("discounts"),
    F.sum("payment_count").alias("payments"),
    F.sum("void_count").alias("voids")
).show()

# COMMAND ----------

# MAGIC %md
# MAGIC Documentación de todo lo que se hizo

# COMMAND ----------

# MAGIC %md
# MAGIC ## Entendimiento y Limpieza de Datos — Resumen Final
# MAGIC
# MAGIC ### Dataset
# MAGIC El dataset de Applebee's contiene **2,142,846 checks** correspondientes a **127 ubicaciones**, con información de **115 fechas de operación entre el 28 de abril y el 20 de agosto de 2026**. 
# MAGIC
# MAGIC La información transaccional se encuentra almacenada en estructuras anidadas que incluyen productos vendidos, descuentos, pagos, impuestos y voids.
# MAGIC
# MAGIC ### Estructura de los datos
# MAGIC El `Id` de los checks no es único por sí solo. La llave compuesta validada es:
# MAGIC
# MAGIC **LocationId + BusinessDate + CheckId**
# MAGIC
# MAGIC Esta combinación identifica de manera única cada check dentro del dataset.
# MAGIC
# MAGIC ### Principales hallazgos de calidad
# MAGIC - No se identificaron checks exactamente duplicados.
# MAGIC - Las variables transaccionales presentan un alto nivel de completitud.
# MAGIC - `LoyaltyId`, `IsEmancipated` y `ExternalId` se encontraban completamente vacías.
# MAGIC - Las fechas y timestamps presentaron formatos consistentes y pudieron convertirse correctamente.
# MAGIC - Se encontró una alta integridad referencial entre ubicaciones, productos, empleados y revenue centers.
# MAGIC - Los valores monetarios extremos, voids repetidos y otras transacciones atípicas se conservaron para el análisis de posibles patrones de riesgo.
# MAGIC
# MAGIC ### Estrategia de limpieza
# MAGIC Se utilizó una estrategia de limpieza conservadora debido a que las transacciones inusuales pueden representar señales relevantes para el análisis de fraude.
# MAGIC
# MAGIC Las principales acciones realizadas fueron:
# MAGIC - Eliminar únicamente variables completamente vacías.
# MAGIC - Convertir las variables de fecha y hora a los tipos de datos correspondientes.
# MAGIC - Conservar outliers y valores transaccionales inusuales.
# MAGIC - Conservar descuentos, pagos y voids repetidos.
# MAGIC - Crear flags de calidad en lugar de eliminar checks atípicos.
# MAGIC - Mantener los valores originales de `Gratuity`, debido a que la convención de signos del sistema POS requiere validación adicional.
# MAGIC
# MAGIC ### Dataset analítico final
# MAGIC Se construyó `analysis_master` con:
# MAGIC
# MAGIC **2,142,846 filas — una fila por check**
# MAGIC
# MAGIC El dataset integra variables agregadas relacionadas con:
# MAGIC - Productos vendidos.
# MAGIC - Descuentos.
# MAGIC - Pagos.
# MAGIC - Gratuity en su formato original.
# MAGIC - Voids.
# MAGIC - Ubicación y revenue center.
# MAGIC - Flags de calidad de los datos.
# MAGIC
# MAGIC ### Validación final
# MAGIC El dataset final conserva:
# MAGIC - **2,142,846 checks**
# MAGIC - **28,558,925 registros de ItemsSold**
# MAGIC - **240,699 registros de Discounts**
# MAGIC - **2,232,092 registros de Payments**
# MAGIC - **178,303 registros de Voids**
# MAGIC - **0 duplicados en la llave compuesta**
# MAGIC
# MAGIC No se eliminaron registros transaccionales durante el proceso de limpieza.
# MAGIC
# MAGIC ### Conclusión
# MAGIC La base final mantiene la información transaccional original y conserva las anomalías que podrían ser relevantes para el análisis de fraude. La limpieza se enfocó en mejorar la estructura, consistencia y facilidad de análisis de los datos sin eliminar comportamientos inusuales que deberán investigarse en las siguientes etapas del EDA.

# COMMAND ----------

# MAGIC %md
# MAGIC Guardar la base de datos

# COMMAND ----------

# ==========================================
# STEP 17 - Save analysis_master as Delta
# ==========================================

table_name = "workspace.default.applebees_analysis_master"

(
    analysis_master
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(table_name)
)

print(f"Delta table saved successfully: {table_name}")

# COMMAND ----------

# Read saved Delta table

analysis_master_saved = spark.table(
    "workspace.default.applebees_analysis_master"
)

print("Saved rows:", analysis_master_saved.count())

display(analysis_master_saved.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC Extra

# COMMAND ----------

# DBTITLE 1,Save fraud-relevant event tables
# ==========================================
# Save fraud-relevant event tables
# (Recreates prerequisites after kernel restart)
# ==========================================

from pyspark.sql import functions as F

# --- Load raw data ---
base_path = "/Volumes/workspace/default/applebees_datachefs_raw/OneDrive_1_5-10-2026"
dimensions_path = f"{base_path}/dimensions"
facts_checks_path = f"{base_path}/facts/checks"

checks_df = (
    spark.read
    .option("recursiveFileLookup", "true")
    .option("multiLine", True)
    .json(facts_checks_path)
)

def load_dim(file_entry):
    p = file_entry.path
    if p.endswith(".json"):
        return spark.read.option("multiLine", True).json(p)
    elif p.endswith(".csv"):
        return spark.read.option("header", "true").option("inferSchema", "true").csv(p)
    elif p.endswith(".parquet") or p.endswith(".snappy.parquet"):
        return spark.read.parquet(p)
    else:
        return spark.read.option("multiLine", True).json(p)

dim_dfs = {}
for f in dbutils.fs.ls(dimensions_path):
    key = f.name.split(".")[0].lower().replace(" ", "_")
    dim_dfs[key] = load_dim(f)

# --- Clean checks and dimensions ---
checks_clean = (
    checks_df
    .drop("LoyaltyId")
    .withColumn("BusinessDate", F.to_date("BusinessDate", "yyyy-MM-dd"))
    .withColumn("OpenTime", F.to_timestamp("OpenTime", "yyyy-MM-dd'T'HH:mm"))
    .withColumn("CloseTime", F.to_timestamp("CloseTime", "yyyy-MM-dd'T'HH:mm"))
)

locations_clean = (
    dim_dfs["locations/"]
    .withColumn("OpenDate", F.to_date("OpenDate", "yyyy-MM-dd"))
    .withColumn("ClosedDate", F.to_date("ClosedDate", "yyyy-MM-dd"))
)

revenue_centers_clean = dim_dfs["revenue_centers/"]

# --- Enrich checks (joins for LocationName, RevenueCenterName) ---
checks_enriched = (
    checks_clean
    .join(
        locations_clean.select(
            F.col("Id").alias("LocationId"),
            F.col("Name").alias("LocationName"),
            "OpenDate",
            "ClosedDate",
            "TimeZoneName"
        ),
        on="LocationId",
        how="left"
    )
    .join(
        revenue_centers_clean.select(
            F.col("Id").alias("RevenueCenterId"),
            F.col("Name").alias("RevenueCenterName")
        ),
        on="RevenueCenterId",
        how="left"
    )
)

# --- Create event tables ---
discount_events = (
    checks_enriched
    .select(
        "LocationId", "LocationName", "BusinessDate",
        F.col("Id").alias("CheckId"),
        F.explode("Discounts").alias("d")
    )
    .select(
        "LocationId", "LocationName", "BusinessDate",
        "CheckId", "d.*"
    )
)

payment_events = (
    checks_enriched
    .select(
        "LocationId", "LocationName", "BusinessDate",
        F.col("Id").alias("CheckId"),
        F.explode("Payments").alias("p")
    )
    .select(
        "LocationId", "LocationName", "BusinessDate",
        "CheckId", "p.*"
    )
)

void_events = (
    checks_enriched
    .select(
        "LocationId", "LocationName", "BusinessDate",
        F.col("Id").alias("CheckId"),
        F.explode("Voids").alias("v")
    )
    .select(
        "LocationId", "LocationName", "BusinessDate",
        "CheckId", "v.*"
    )
)

# --- Save event tables ---
discount_events.write.format("delta").mode("overwrite").saveAsTable(
    "workspace.default.applebees_discount_events"
)

payment_events.write.format("delta").mode("overwrite").saveAsTable(
    "workspace.default.applebees_payment_events"
)

void_events.write.format("delta").mode("overwrite").saveAsTable(
    "workspace.default.applebees_void_events"
)

print("Event tables saved successfully.")