# Databricks notebook source
# MAGIC %md 
# MAGIC ### 実行環境:
# MAGIC
# MAGIC - Databricks Runtime 15.2 ML 以上
# MAGIC - g5.8xlarge (single cluster)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## 　オプション/ELYZA-japanese-Llama-2-13b-instructモデルサービング環境の構築
# MAGIC https://docs.databricks.com/en/machine-learning/model-serving/create-manage-serving-endpoints.html
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-inference-1.png?raw=true" style="float: right; margin-left: 10px"  width="600px;">
# MAGIC
# MAGIC [ELYZA-japanese-Llama-2-13b-instruct](https://huggingface.co/elyza/ELYZA-japanese-Llama-2-13b-instruct)は、Metaがオープンソースとして公開している「Llama 2 13B」をベースに、約180億トークンの日本語テキストで追加事前学習を行った日本語向けLLMモデルです。
# MAGIC 事後学習には、ELYZA独自の高品質な指示データセットを用いています。また、複数ターンからなる対話にも対応しており、過去の対話を引き継いでユーザーからの指示を遂行することができます。
# MAGIC
# MAGIC このNotebookではダウンロードした「ELYZA-japanese-Llama-2-13b」をUnityCatalogに登録し、カスタムモデルサービング・エンドポイントを作成します。
# MAGIC
# MAGIC モデルサービングは、MLflow 機械学習モデルをスケーラブルな REST API エンドポイントとして公開し、モデルをデプロイするための高可用性かつ低遅延のサービスを提供します。このサービスは、需要の変化に合わせて自動的にスケールアップまたはスケールダウンし、レイテンシーのパフォーマンスを最適化しながらインフラストラクチャのコストを節約します。この機能はサーバーレス コンピューティングを使用します。
# MAGIC
# MAGIC Model Serving を使用すると、Databricks でホストされているモデルや外部プロバイダーからのモデルを含む、すべてのモデルを1か所で集中管理および制御できます。
# MAGIC

# COMMAND ----------

# MAGIC %pip install tf-keras
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,コンフィグ(環境に合わせて修正してください）
CATALOG = "japan_practical_demo"
SCHEMA = "models"

# COMMAND ----------

sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG};")
sql(f"USE CATALOG {CATALOG};")
sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};")
sql(f"USE SCHEMA {SCHEMA};")

# COMMAND ----------

# DBTITLE 1,ダウンロードロケーションからモデルをダウンロード
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# model_name = 'elyza/ELYZA-japanese-Llama-2-13b-instruct'
model_name = 'elyza/ELYZA-japanese-Llama-2-13b-fast-instruct'

# Load model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
  model_name, 
  torch_dtype="auto", 
  cache_dir="/local_disk0/.cache/huggingface/")

# COMMAND ----------

# DBTITLE 1,プロンプトの指定
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。"

text = "クマが海辺に行ってアザラシと出会い、Databricsの予測IOについて語り合い仲良くなるいうプロットの短編小説を400文字以内で書いてください。"

prompt = "{b_inst} {system}{prompt} {e_inst} ".format(
#    bos_token=tokenizer.bos_token,
    b_inst=B_INST,
    system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",
    prompt=text,
    e_inst=E_INST,
)

# COMMAND ----------

# DBTITLE 1,ダウンロードしたモデルをMLflowに登録
import mlflow
from mlflow.models import infer_signature
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

import pandas as pd


# Define model signature including params
input_example = {"prompt": prompt}
inference_config = {
  "temperature": 1.0,
  "max_new_tokens": 100,
}
signature = infer_signature(
  model_input=input_example,
  model_output="Machien Learning is...",
  params=inference_config
)

# Log the model with its details such as artifacts, pip requirements and input example
# This may take about 1.7 minutes to complete
with mlflow.start_run() as run:  
  mlflow.transformers.log_model(
    transformers_model={
      "tokenizer": tokenizer,
      "model": model,
    },
    task = "text-generation",
    artifact_path="model",
    pip_requirements=["torch", "transformers", "accelerate", "sentencepiece"],
    input_example=input_example,
    signature=signature,
    # メタデータ・タスクを追加することで、後で作成されるモデル・サービング・エンドポイントが最適化されます。
    # llm/v1/chatを指定しないと推論時にフォーマットエラーとなった。
    #metadata={"task": "llm/v1/completions"}
    metadata={"task": "llm/v1/chat"}
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルをUnityカタログに登録

# COMMAND ----------

# Unity カタログにモデルを登録するための MLflow Python クライアントの設定
import mlflow
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# Unityカタログへのモデル登録
# This may take 2.2 minutes to complete

#registered_name = "models.default.llama2_7b_completions" # Note that the UC model name follows the pattern <catalog_name>.<schema_name>.<model_name>, corresponding to the catalog, schema, and registered model name

registered_name = f"{CATALOG}.{SCHEMA}.ELYZA-japanese-Llama-2-13b-fast-instruct"
print(registered_name)

result = mlflow.register_model(
    "runs:/"+run.info.run_id+"/model",
    registered_name,
)

# COMMAND ----------

print('version=',result.version)
print(result)

# COMMAND ----------

from mlflow import MlflowClient
client = MlflowClient()

# 上記のセルに登録されている正しいモデルバージョンを選択してください。
print('version=',result.version)
client.set_registered_model_alias(name=registered_name, alias="Champion", version=result.version)

# COMMAND ----------

# DBTITLE 1,モデルの動作確認
import mlflow
import pandas as pd
import torch

torch.cuda.empty_cache()

# モデル名称
print(registered_name)

# モデルをUDFとしてロード
loaded_model = mlflow.pyfunc.load_model(f"models:/{registered_name}@Champion")

# ロードされたモデルを使った予測
story = loaded_model.predict(
  {"prompt": prompt}, 
  params={
    #"temperature": 0.5,
    "max_new_tokens": 2000,
  }
)

# COMMAND ----------

print(story[0])

# COMMAND ----------

# MAGIC %md ## Serving endpointの作成
# MAGIC プロビジョンド スループット モードでモデルをデプロイします。<br>
# MAGIC https://docs.databricks.com/en/machine-learning/foundation-models/deploy-prov-throughput-foundation-model-apis.html

# COMMAND ----------

import requests
import json

# Name of the registered MLflow model
model_name = registered_name

# Get the latest version of the MLflow model
model_version = result.version

# Get the API endpoint and token for the current notebook context
API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get() 
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

headers = {"Context-Type": "text/json", "Authorization": f"Bearer {API_TOKEN}"}

response = requests.get(url=f"{API_ROOT}/api/2.0/serving-endpoints/get-model-optimization-info/{model_name}/{model_version}", headers=headers)

print(json.dumps(response.json(), indent=4))

# COMMAND ----------

from mlflow.deployments import get_deploy_client

# !export DATABRICKS_HOST = f"{API_ROOT}/api/2.0/serving-endpoints"
# !export DATABRICKS_TOKEN = API_TOKEN

client = get_deploy_client("databricks")

endpoint_name = "elyza-japanese-llama2-13b-endpoint"
endpoint = client.create_endpoint(
    name=endpoint_name,
    config={
        "served_entities": [
            {
                "entity_name": model_name,
                "entity_version": model_version,
                "min_provisioned_throughput": response.json()['throughput_chunk_size'],
                "max_provisioned_throughput": response.json()['throughput_chunk_size'],
            }
        ]
    },
)

print(json.dumps(endpoint, indent=4))

# COMMAND ----------

# MAGIC %md ## Serving endpointの作成が完了するまで待ちます。
# MAGIC 左のメニューの[Serving](https://e2-demo-west.cloud.databricks.com/ml/endpoints?o=2556758628403379#)からステータスを確認できます。<br>
# MAGIC
# MAGIC モデルのサイズによっては、モデルのデプロイに 1 時間以上かかる場合があります。<br>
# MAGIC Databricks は、この時間を短縮するために積極的に取り組んでいます。

# COMMAND ----------

chat_response = client.predict(
    endpoint=endpoint_name,
    inputs={
        "messages": [
            {
              "role": "user",
              "content": prompt
            }
        ],
        "temperature": 0.5,
        "max_tokens": 128
    }
)

print(chat_response)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 以上です。お疲れ様でした！
