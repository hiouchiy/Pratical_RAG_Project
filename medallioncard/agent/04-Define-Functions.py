# Databricks notebook source
# MAGIC %run ./config

# COMMAND ----------

# MAGIC %md
# MAGIC ## チームマスタ検索

# COMMAND ----------

sql(f"""
CREATE OR REPLACE FUNCTION
{catalog}.{dbName}.get_team_info (
  target_team_code STRING COMMENT 'Team code to be searched'
)
returns table(team_id STRING, team_code STRING, team_name STRING, description STRING)
COMMENT 'Get team information from team master table'
return
(SELECT team_id, team_code, team_name, description
FROM {catalog}.{dbName}.{team_master_silver_table_name} AS t
WHERE team_code = target_team_code)
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 日報データ検索

# COMMAND ----------

#処理スピードのために、日報データを一部フィルター
sql(f"""
CREATE OR REPLACE FUNCTION
{catalog}.{dbName}.search_report (
  target_team_id STRING COMMENT 'Team ID to be searched',
  target_date DATE COMMENT 'Date to searching report'
)
returns table(emp_name STRING, report STRING)
COMMENT 'Search report by team ID and date'
return
(SELECT emp_name, report
FROM {catalog}.{dbName}.{report_silver_table_name} 
WHERE team_id = target_team_id AND report_date = target_date AND emp_id in ( "SA345682", "SA678905", "SA890136", "SA012348", "SA234571", "SA901347", "SA456782", "SA890137", "SA678904", "SA789025", "SA123459"))
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 日付取得

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION
# MAGIC hiroshi.medallioncard_agent.get_relative_date (
# MAGIC   days_offset INT COMMENT 'The offset for the date. 0 means today, -1 means yesterday, +1 means tomorrow.')
# MAGIC RETURNS STRING
# MAGIC LANGUAGE PYTHON
# MAGIC DETERMINISTIC
# MAGIC COMMENT 'Get the date in yyyy-mm-dd format based on the given offset'
# MAGIC AS $$
# MAGIC   from datetime import datetime, timedelta
# MAGIC
# MAGIC   target_date = datetime(2024, 11, 14) + timedelta(days=days_offset)
# MAGIC   return target_date.strftime('%Y-%m-%d')
# MAGIC $$

# COMMAND ----------

# MAGIC %md
# MAGIC ## ベクトル検索(APIコール)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION
# MAGIC hiroshi.medallioncard_agent.search_similar_case_studies (
# MAGIC   query_text STRING COMMENT 'a query to search for similar case studies'
# MAGIC )
# MAGIC RETURNS STRING
# MAGIC LANGUAGE PYTHON
# MAGIC DETERMINISTIC
# MAGIC COMMENT 'Performs a semantic search to find similar case studies based on the provided input sentence.'
# MAGIC AS $$
# MAGIC   import requests
# MAGIC   import os
# MAGIC
# MAGIC   databricks_url = 'YOUR_URL'
# MAGIC   token = 'YOUR_TOKEN' 
# MAGIC   headers={"Authorization": f"Bearer {token}"}
# MAGIC
# MAGIC   try:
# MAGIC       response = requests.get(databricks_url+"/api/2.0/vector-search/indexes/hiroshi.medallioncard_agent.case_silver_table_vs_index/query", params ={"columns":["case", "url"],"query_text":query_text,"num_results":3}, headers=headers).json()
# MAGIC       chunked_texts = [entry[0].replace('スカイコーポレーション', '株式会社レノバリ').replace('アーバンテック', 'アーバンソルテック') for entry in response['result']['data_array']]
# MAGIC
# MAGIC       count = 0
# MAGIC       for chunk in chunked_texts:
# MAGIC         chunked_texts[count] = "事例" + str(count+1) + ": " + chunk
# MAGIC         count = count + 1
# MAGIC       
# MAGIC       return "\n\n".join(chunked_texts)
# MAGIC       
# MAGIC   except requests.exceptions.RequestException as e:
# MAGIC       return f"Error: {e}"
# MAGIC $$

# COMMAND ----------

# MAGIC %md
# MAGIC ## 以下はDummy関数

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION
# MAGIC hiroshi.medallioncard_agent.get_contract_info (
# MAGIC   contract_id STRING COMMENT 'contract ID to be searched'
# MAGIC )
# MAGIC RETURNS STRING
# MAGIC LANGUAGE PYTHON
# MAGIC DETERMINISTIC
# MAGIC COMMENT 'Get a contract information from the contract management service'
# MAGIC AS $$
# MAGIC   return "this is dummy"
# MAGIC $$

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION
# MAGIC hiroshi.medallioncard_agent.send_notification (
# MAGIC   target_emp_id STRING COMMENT 'target employee ID to send a notification'
# MAGIC )
# MAGIC RETURNS STRING
# MAGIC LANGUAGE PYTHON
# MAGIC DETERMINISTIC
# MAGIC COMMENT 'Send a notificaiton'
# MAGIC AS $$
# MAGIC   return "this is dummy"
# MAGIC $$

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION
# MAGIC hiroshi.medallioncard_agent.get_approval_status (
# MAGIC   app_id STRING COMMENT 'app ID to be searched'
# MAGIC )
# MAGIC RETURNS STRING
# MAGIC LANGUAGE PYTHON
# MAGIC DETERMINISTIC
# MAGIC COMMENT 'Get approval status from the Approval Workflow Service'
# MAGIC AS $$
# MAGIC   return "this is dummy"
# MAGIC $$

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION
# MAGIC hiroshi.medallioncard_agent.send_email (
# MAGIC   comp_id STRING COMMENT 'comp ID to be searched'
# MAGIC )
# MAGIC RETURNS STRING
# MAGIC LANGUAGE PYTHON
# MAGIC DETERMINISTIC
# MAGIC COMMENT 'Get approval status from the Approval Workflow Service'
# MAGIC AS $$
# MAGIC   return "this is dummy"
# MAGIC $$

# COMMAND ----------

# MAGIC %md
# MAGIC ## 関数削除処理

# COMMAND ----------

sql(f"""
DROP FUNCTION hiroshi.medallioncard_agent.get_opportunity_info
""")

# COMMAND ----------


