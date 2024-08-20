# Databricks notebook source
# MAGIC %md
# MAGIC # Databricks上のMLFlowで`multilingual-e5-large`モデルを管理する
# MAGIC
# MAGIC この例では、[multilingual-e5-large model](https://huggingface.co/intfloat/multilingual-e5-large)を `sentence_transformers` フレーバーで MLFLow にロギングし、Unity Catalog でモデルを管理し、モデル提供エンドポイントを作成する方法を示します。
# MAGIC
# MAGIC このノートブックの環境
# MAGIC - ランタイム: 15.2 ML Runtime
# MAGIC - インスタンス: AWS の `i3.xlarge` または Azure の `Standard_D4DS_v5` 
# MAGIC

# COMMAND ----------

# サービングエンドポイントの名前を指定
endpoint_name = 'multilingual-e5-large-embedding'

# COMMAND ----------

databricks_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# COMMAND ----------

# サービングエンドポイントの作成または更新
from datetime import timedelta
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput, ServedModelInputWorkloadSize, ServedModelInputWorkloadType

serving_endpoint_name = endpoint_name

w = WorkspaceClient()

existing_endpoint = next(
    (e for e in w.serving_endpoints.list() if e.name == serving_endpoint_name), None
)
serving_endpoint_url = f"{databricks_url}/ml/endpoints/{serving_endpoint_name}"
if not existing_endpoint == None:
    print(f"Deleting the endpoint {serving_endpoint_url}, this will take a few minutes...")
    w.serving_endpoints.delete(name=serving_endpoint_name)
    
displayHTML('Your Model Endpoint Serving is now deleted.')

# COMMAND ----------


