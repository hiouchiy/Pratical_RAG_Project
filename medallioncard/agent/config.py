# Databricks notebook source
# MAGIC %md 
# MAGIC ## Configuration file
# MAGIC
# MAGIC 別のカタログでデモを実行するには、ここでカタログとスキーマを変更してください。
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=2556758628403379&notebook=%2Fconfig&demo_name=llm-rag-chatbot&event=VIEW&path=%2F_dbdemos%2Fdata-science%2Fllm-rag-chatbot%2Fconfig&version=1">

# COMMAND ----------

catalog = "hiroshi"
dbName = db = "medallioncard_agent"
volume = "raw_data"
report_bronze_table_name = "report_bronze_table"
report_silver_table_name = "report_silver_table"
team_master_bronze_table_name = "team_master_bronze_table"
team_master_silver_table_name = "team_master_silver_table"
case_bronze_table_name = "case_bronze_table"
case_silver_table_name = "case_silver_table"
registered_model_name = "medallioncard_agent_model"

VECTOR_SEARCH_ENDPOINT_NAME="one-env-shared-endpoint-10"
embedding_endpoint_name = "multilingual-e5-large-embedding"
instruct_endpoint_name = "databricks-dbrx-instruct"
case_endpoint_name = "medallioncard-case-fs-endpoint"

databricks_token_secrets_scope = "dbdemos"
databricks_token_secrets_key = "databricks_token"
databricks_host_secrets_scope = "dbdemos"
databricks_host_secrets_key = "databricks_host"

print('VECTOR_SEARCH_ENDPOINT_NAME =',VECTOR_SEARCH_ENDPOINT_NAME)
print('catalog =',catalog)
print('dbName =',dbName)
print('volume =',volume)
print('report_bronze_table_name =',report_bronze_table_name)
print('report_silver_table_name =',report_silver_table_name)
print('team_master_bronze_table_name =',team_master_bronze_table_name)
print('team_master_silver_table_name =',team_master_silver_table_name)
print('registered_model_name =',registered_model_name)
print('case_bronze_table_name =',case_bronze_table_name)
print('case_silver_table_name =',case_silver_table_name)
print('embedding_endpoint_name =',embedding_endpoint_name)
print('instruct_endpoint_name =',instruct_endpoint_name)
print('case_endpoint_name =',case_endpoint_name)
