# Databricks notebook source
# MAGIC %md 
# MAGIC ## Configuration file
# MAGIC
# MAGIC 別のカタログでデモを実行するには、ここでカタログとスキーマを変更してください。
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=2556758628403379&notebook=%2Fconfig&demo_name=llm-rag-chatbot&event=VIEW&path=%2F_dbdemos%2Fdata-science%2Fllm-rag-chatbot%2Fconfig&version=1">

# COMMAND ----------

VECTOR_SEARCH_ENDPOINT_NAME="my_vs_endpoint"

DATABRICKS_SITEMAP_URL = "https://docs.databricks.com/ja/doc-sitemap.xml"

catalog = "my_catalog"
dbName = db = "my_schema"
volume = "raw_data"
raw_data_table_name = "raw_query"
embed_table_name = "airbricks_documentation"
registered_model_name = "airbricks_chatbot_model"

embedding_endpoint_name = "my_embedding_llm_endpoint"
instruct_endpoint_name = "my_instruct_llm_endpoint"

databricks_token_secrets_scope = "my_token_scope"
databricks_token_secrets_key = "my_token_key"
databricks_host_secrets_scope = "my_host_scope"
databricks_host_secrets_key = "my_host_key"

print('VECTOR_SEARCH_ENDPOINT_NAME =',VECTOR_SEARCH_ENDPOINT_NAME)
print('DATABRICKS_SITEMAP_URL =',DATABRICKS_SITEMAP_URL)
print('catalog =',catalog)
print('dbName =',dbName)
print('volume =',volume)
print('raw_data_table_name =',raw_data_table_name)
print('embed_table_name =',embed_table_name)
print('registered_model_name =',registered_model_name)
print('embed_table_name =',embed_table_name)
print('embedding_endpoint_name =',embedding_endpoint_name)
print('instruct_endpoint_name =',instruct_endpoint_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### License
# MAGIC This demo installs the following external libraries on top of DBR(ML):
# MAGIC
# MAGIC
# MAGIC | Library | License |
# MAGIC |---------|---------|
# MAGIC | langchain     | [MIT](https://github.com/langchain-ai/langchain/blob/master/LICENSE)     |
# MAGIC | lxml      | [BSD-3](https://pypi.org/project/lxml/)     |
# MAGIC | transformers      | [Apache 2.0](https://github.com/huggingface/transformers/blob/main/LICENSE)     |
# MAGIC | unstructured      | [Apache 2.0](https://github.com/Unstructured-IO/unstructured/blob/main/LICENSE.md)     |
# MAGIC | llama-index      | [MIT](https://github.com/run-llama/llama_index/blob/main/LICENSE)     |
# MAGIC | tesseract      | [Apache 2.0](https://github.com/tesseract-ocr/tesseract/blob/main/LICENSE)     |
# MAGIC | poppler-utils      | [MIT](https://github.com/skmetaly/poppler-utils/blob/master/LICENSE)     |
# MAGIC | textstat      | [MIT](https://pypi.org/project/textstat/)     |
# MAGIC | tiktoken      | [MIT](https://github.com/openai/tiktoken/blob/main/LICENSE)     |
# MAGIC | evaluate      | [Apache2](https://pypi.org/project/evaluate/)     |
# MAGIC | torch      | [BDS-3](https://github.com/intel/torch/blob/master/LICENSE.md)     |
# MAGIC | tiktoken      | [MIT](https://github.com/openai/tiktoken/blob/main/LICENSE)     |
# MAGIC
# MAGIC
# MAGIC
# MAGIC
