# Databricks notebook source
# MAGIC %md
# MAGIC ## まずはお使いの環境に応じて以下のパラメータをセットしてください。

# COMMAND ----------

catalog = "japan_practical_demo"
dbName = "dollyfoodsholdings"
volume = "raw_data"
raw_data_table_name = "sales_result"
company_table_name = "company_info"

print('catalog =', catalog)
print('dbName =', dbName)
print('volume =', volume)
print('raw_data_table_name =', raw_data_table_name)
print('company_table_name =', company_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## あとは以下のセルを一つ一つ実行するだけです。

# COMMAND ----------

spark.conf.set("my.catalogName", catalog)
spark.conf.set("my.schemaName", dbName)
spark.conf.set("my.volumeName", volume)

# COMMAND ----------

sql(f"CREATE CATALOG IF NOT EXISTS {catalog};")
sql(f"USE CATALOG {catalog};")
sql(f"CREATE SCHEMA IF NOT EXISTS {dbName};")
sql(f"USE SCHEMA {dbName};")
sql(f"CREATE VOLUME IF NOT EXISTS {volume};")

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
・G001=ブルーフィッシュ・シーフード株式会社（略称: 「ブルーフィッシュ」または「BFS」）
・G002=グリーンガーデンカフェ株式会社（略称: 「グリーンカフェ」または「GGC」）
・G003=サンセット・イタリアーナ株式会社（略称: 「サンセット」または「SIT」）
・G004=ナチュラルハーベストデリ株式会社（略称: 「ナチュデリ」または「NHD」）
・G005=桜庵株式会社（略称: 「さくら」または「SA」）

【留意事項】
会計年度は原則4月から翌年3月まで。ただし、G003のみ2月から翌年1月までとする。"""

spark.sql(f'COMMENT ON TABLE {raw_data_table_name} IS "{comment}"')

# COMMAND ----------

# Drop if table existing
sql(f"drop table if exists {company_table_name}")

# Read raw data and create delta table to store it
raw_data_url = "https://raw.githubusercontent.com/hiouchiy/Pratical_RAG_Project/main/dollyfoodsholdings/company_master.csv"
!wget $raw_data_url -O /tmp/company_master.csv

unity_catalog_volume_path = f'/Volumes/{catalog}/{dbName}/{volume}/company_master.csv'
!cp /tmp/company_master.csv $unity_catalog_volume_path

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql import functions as F

schema = StructType([ \
    StructField("Code", StringType(),True), \
    StructField("Year", IntegerType(),True), \
    StructField("NumOfEmployees", IntegerType(),True), \
    StructField("YearOfEstablishment", IntegerType(), True) \
  ])

csv_df = spark.read.format("csv").option("header", "true").schema(schema).load(unity_catalog_volume_path)

display(csv_df)

csv_df.write.mode('overwrite').saveAsTable(company_table_name)

# COMMAND ----------

sql(f"ALTER TABLE {company_table_name} ALTER COLUMN Code COMMENT 'グループ会社コード'")
sql(f"ALTER TABLE {company_table_name} ALTER COLUMN Year COMMENT '会計年度'")
sql(f"ALTER TABLE {company_table_name} ALTER COLUMN NumOfEmployees COMMENT '従業員数'")
sql(f"ALTER TABLE {company_table_name} ALTER COLUMN YearOfEstablishment COMMENT '設立年'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 以上でデータ準備は完了です。Genie Data Roomへ移動しましょう。 
