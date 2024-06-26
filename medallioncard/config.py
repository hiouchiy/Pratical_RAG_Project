# Databricks notebook source
# MAGIC %md 
# MAGIC ## Configuration file
# MAGIC
# MAGIC 別のカタログでデモを実行するには、ここでカタログとスキーマを変更してください。
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=2556758628403379&notebook=%2Fconfig&demo_name=llm-rag-chatbot&event=VIEW&path=%2F_dbdemos%2Fdata-science%2Fllm-rag-chatbot%2Fconfig&version=1">

# COMMAND ----------

catalog = "japan_practical_demo"
dbName = db = "medallioncard"
volume = "raw_data"
faq_bronze_table_name = "faq_bronze_table"
faq_silver_table_name = "faq_silver_table"
user_bronze_table_name = "user_bronze_table"
user_silver_table_name = "user_silver_table"
registered_model_name = "medallioncard_chatbot_model"

VECTOR_SEARCH_ENDPOINT_NAME="dbdemos_vs_endpoint"
embedding_endpoint_name = "multilingual-e5-large-embedding"
instruct_endpoint_name = "databricks-dbrx-instruct"
user_endpoint_name = "medallioncard-user-info-fs-endpoint"

databricks_token_secrets_scope = "medallioncard"
databricks_token_secrets_key = "databricks_token"

print('VECTOR_SEARCH_ENDPOINT_NAME =',VECTOR_SEARCH_ENDPOINT_NAME)
print('catalog =',catalog)
print('dbName =',dbName)
print('volume =',volume)
print('faq_bronze_table_name =',faq_bronze_table_name)
print('faq_silver_table_name =',faq_silver_table_name)
print('registered_model_name =',registered_model_name)
print('user_bronze_table_name =',user_bronze_table_name)
print('user_silver_table_name =',user_silver_table_name)
print('embedding_endpoint_name =',embedding_endpoint_name)
print('instruct_endpoint_name =',instruct_endpoint_name)
print('user_endpoint_name =',user_endpoint_name)
