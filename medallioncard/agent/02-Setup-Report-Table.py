# Databricks notebook source
# MAGIC %pip install mlflow==2.10.1 lxml==4.9.3 transformers==4.30.2 langchain==0.1.5 databricks-vectorsearch==0.22 databricks-sdk==0.28.0 databricks-feature-store==0.17.0
# MAGIC %pip install dspy-ai -U
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

# MAGIC %run ../../_resources/00-init $reset_all_data=false

# COMMAND ----------

sql(f"CREATE CATALOG IF NOT EXISTS {catalog};")
sql(f"USE CATALOG {catalog};")
sql(f"CREATE SCHEMA IF NOT EXISTS {dbName};")
sql(f"USE SCHEMA {dbName};")
sql(f"CREATE VOLUME IF NOT EXISTS {volume};")

# COMMAND ----------

# すでに同名のテーブルが存在する場合は削除
sql(f"drop table if exists {report_bronze_table_name}")

# 生データを読み込み、デルタ・テーブルを作成して保存
raw_data_url = "https://raw.githubusercontent.com/hiouchiy/Pratical_RAG_Project/refs/heads/main/medallioncard/agent/daily_report.json"
!wget $raw_data_url -O /tmp/report.json

unity_catalog_volume_path = f'/Volumes/{catalog}/{dbName}/{volume}/report.json'
!cp /tmp/report.json $unity_catalog_volume_path

spark.read.option("multiline","true").json(unity_catalog_volume_path).write.mode('overwrite').saveAsTable(report_bronze_table_name)

display(spark.table(report_bronze_table_name))

# COMMAND ----------

sql(f"DROP TABLE IF EXISTS {report_silver_table_name};")

sql(f"""
--インデックスを作成するには、テーブルのChange Data Feedを有効にします
CREATE TABLE IF NOT EXISTS {report_silver_table_name} (
  id BIGINT GENERATED BY DEFAULT AS IDENTITY,
  emp_id STRING NOT NULL,
  emp_name STRING NOT NULL,
  team_id STRING NOT NULL,
  report_date DATE NOT NULL,
  report STRING NOT NULL
) TBLPROPERTIES (delta.enableChangeDataFeed = true); 
""")

from pyspark.sql.functions import lit, to_date, col
spark.table(report_bronze_table_name).withColumn("report_date", to_date(col("date"), "yyyy-MM-dd")).drop("date").write.mode('overwrite').saveAsTable(report_silver_table_name)

# comment = """各チーム名は以下の通り。
# ENT: Enterprise Team, 大企業向け営業を担当
# MMC: Mid-Market Companies Team, 中堅企業向け営業を担当
# SMB: Small and Medium Business Team, 中小企業向け営業を担当
# SUT: Startup Team, スタートアップ企業向け営業を担当"""
comment = "A table including daily reports from sales team"

spark.sql(f'COMMENT ON TABLE {report_silver_table_name} IS "{comment}"')

display(spark.table(report_silver_table_name))

# COMMAND ----------


