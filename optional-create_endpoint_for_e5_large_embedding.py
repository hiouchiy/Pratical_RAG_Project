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

# MAGIC %md
# MAGIC ## モデルをMLFlowに記録する

# COMMAND ----------

from sentence_transformers import SentenceTransformer
model_name = "intfloat/multilingual-e5-large"

model = SentenceTransformer(model_name)

# COMMAND ----------

import mlflow
import pandas as pd

# 入出力スキーマの定義
sentences = ["これは例文です", "各文章は変換されます"]
signature = mlflow.models.infer_signature(
    sentences,
    model.encode(sentences),
)

# MLFlowのSentence Transformerフレーバーを使って登録
with mlflow.start_run() as run:  
    mlflow.sentence_transformers.log_model(
      model, 
      "multilingual-e5-large-embedding", 
      signature=signature,
      input_example=sentences)

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルを Unity Catalog に登録する
# MAGIC  デフォルトでは、MLflowはDatabricksワークスペースのモデルレジストリにモデルを登録します。代わりにUnity Catalogにモデルを登録するには、[ドキュメント](https://docs.databricks.com/machine-learning/manage-model-lifecycle/index.html)に従い、レジストリサーバーをDatabricks Unity Catalogに設定します。
# MAGIC
# MAGIC  Unity Catalogにモデルを登録するには、ワークスペースでUnity Catalogが有効になっている必要があるなど、[いくつかの要件](https://docs.databricks.com/machine-learning/manage-model-lifecycle/index.html#requirements)があります。
# MAGIC

# COMMAND ----------

# Unityカタログにモデルを登録するためにMLflow Pythonクライアントを設定する
import mlflow
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %sql
# MAGIC USE CATALOG dev;
# MAGIC CREATE SCHEMA IF NOT EXISTS models;

# COMMAND ----------

# Unityカタログへのモデル登録

registered_name = "dev.models.multilingual-e5-large" # UCモデル名は<カタログ名>.<スキーマ名>.<モデル名>のパターンに従っており、カタログ名、スキーマ名、登録モデル名に対応していることに注意してください。
result = mlflow.register_model(
    "runs:/"+run.info.run_id+"/multilingual-e5-large-embedding",
    registered_name,
)

# COMMAND ----------

from mlflow import MlflowClient
client = MlflowClient()

# 上記のセルに登録されている正しいモデルバージョンを選択
client.set_registered_model_alias(name=registered_name, alias="Champion", version=result.version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unityカタログからモデルを読み込む

# COMMAND ----------

import mlflow
import pandas as pd

loaded_model = mlflow.pyfunc.load_model(f"models:/{registered_name}@Champion")

# ロードされたモデルを使って予測を立てる
loaded_model.predict(
  ["MLとは何か？", "大規模言語モデルとは何か？"],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデル提供エンドポイントの作成
# MAGIC モデルが登録されたら、APIを使用してDatabricks GPU Model Serving Endpointを作成し、`bge-large-en`モデルをサービングしていきます。
# MAGIC
# MAGIC 以下のデプロイにはGPUモデルサービングが必要です。GPU モデルサービングの詳細については、Databricks チームにお問い合わせいただくか、サインアップ [こちら](https://docs.google.com/forms/d/1-GWIlfjlIaclqDz6BPODI2j1Xg4f4WbFvBXyebBpN-Y/edit) してください。

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

model_version = result  # mlflow.register_modelの返された結果

serving_endpoint_name = endpoint_name
latest_model_version = model_version.version
model_name = model_version.name

w = WorkspaceClient()
endpoint_config = EndpointCoreConfigInput(
    name=serving_endpoint_name,
    served_models=[
        ServedModelInput(
            model_name=model_name,
            model_version=latest_model_version,
            workload_type=ServedModelInputWorkloadType.CPU,
            workload_size=ServedModelInputWorkloadSize.SMALL,
            scale_to_zero_enabled=True
        )
    ]
)

existing_endpoint = next(
    (e for e in w.serving_endpoints.list() if e.name == serving_endpoint_name), None
)
serving_endpoint_url = f"{databricks_url}/ml/endpoints/{serving_endpoint_name}"
if existing_endpoint == None:
    print(f"Creating the endpoint {serving_endpoint_url}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.create_and_wait(name=serving_endpoint_name, config=endpoint_config, timeout=timedelta(minutes=60))
else:
    print(f"Updating the endpoint {serving_endpoint_url} to version {latest_model_version}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.update_config_and_wait(served_models=endpoint_config.served_models, name=serving_endpoint_name, timeout=timedelta(minutes=60))
    
displayHTML(f'Your Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{serving_endpoint_name}">Model Serving Endpoint page</a> for more details.')

# COMMAND ----------

# MAGIC %md
# MAGIC モデルサービングエンドポイントの準備ができたら、同じワークスペースで実行されているMLflow Deployments SDKで簡単にクエリできます。

# COMMAND ----------

import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")

embeddings_response = client.predict(
    endpoint=endpoint_name,
    inputs={
        "inputs": ["おはようございます"]
    }
)
embeddings_response['predictions']

# COMMAND ----------

import time

start = time.time()

# If you get timeout error (from the endpoint not yet being ready), then rerun this.
endpoint_response = w.serving_endpoints.query(name=endpoint_name, dataframe_records=['こんにちは', 'おはようございます'])

end = time.time()

print(endpoint_response)
print(f'Time taken for querying endpoint in seconds: {end-start}')

# COMMAND ----------


