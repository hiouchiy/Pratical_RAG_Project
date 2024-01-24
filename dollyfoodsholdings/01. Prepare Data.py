# Databricks notebook source
# MAGIC %md
# MAGIC ## まずはお使いの環境に応じて以下のパラメータをセットしてください。

# COMMAND ----------

catalog = "japan_practical_demo"
dbName = "genie_demo"
volume = "raw_data"
raw_data_table_name = "sales_result"

print('catalog =',catalog)
print('dbName =',dbName)
print('volume =',volume)
print('raw_data_table_name =',raw_data_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## あとは以下のセルを一つ一つ実行するだけです。

# COMMAND ----------

spark.conf.set("my.catalogName", catalog)
spark.conf.set("my.schemaName", dbName)
spark.conf.set("my.volumeName", volume)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE CATALOG IF NOT EXISTS ${my.catalogName};
# MAGIC USE CATALOG ${my.catalogName};
# MAGIC CREATE SCHEMA IF NOT EXISTS ${my.catalogName}.${my.schemaName};
# MAGIC USE SCHEMA ${my.schemaName};
# MAGIC CREATE VOLUME IF NOT EXISTS ${my.catalogName}.${my.schemaName}.${my.volumeName};

# COMMAND ----------

# Drop if table existing
sql(f"drop table if exists {raw_data_table_name}")

# Read raw data and create delta table to store it
raw_data_url = "https://raw.githubusercontent.com/hiouchiy/Pratical_RAG_Project/main/dollyfoodsholdings/financial_data_all_group_companies_2019_2023.csv"
!wget $raw_data_url -O /tmp/financial_data_all_group_companies_2019_2023.csv

unity_catalog_volume_path = f'/Volumes/{catalog}/{dbName}/{volume}/financial_data_all_group_companies_2019_2023.csv'
!cp /tmp/financial_data_all_group_companies_2019_2023.csv $unity_catalog_volume_path

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql import functions as F

schema = StructType([ \
    StructField("Code",StringType(),True), \
    StructField("Year",IntegerType(),True), \
    StructField("Month",IntegerType(),True), \
    StructField("Revenue", DoubleType(), True), \
    StructField("Expenses", DoubleType(), True), \
    StructField("NetProfit", DoubleType(), True) \
  ])

csv_df = spark.read.format("csv").option("header", "true").schema(schema).load(unity_catalog_volume_path)
csv_df = csv_df.withColumn( "date", F.make_date( "year", "month", "month" ) ).withColumn( "yearmonth", F.trunc( "date", "month" ) )

display(csv_df)

csv_df.write.mode('overwrite').saveAsTable(raw_data_table_name)

# COMMAND ----------

comment = """グループ企業コードと会社名の対応関係は以下の通り。
・G001=BCDマート株式会社（略称:Bマート）
・G002=YXZモール株式会社（略称:モール）
・G003=株式会社Fastショップ（略称:ショップ）
・G004=Goodレストラン株式会社（略称:グッレス）
・G005=UA食品株式会社（略称:U食）

【留意事項】
会計年度は原則4月から翌年3月まで。ただし、G003のみ2月から翌年1月までとする。"""

spark.sql(f'COMMENT ON TABLE {raw_data_table_name} IS "{comment}"')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 以上でデータ準備は完了です。Genie Data Roomへ移動しましょう。 

# COMMAND ----------


