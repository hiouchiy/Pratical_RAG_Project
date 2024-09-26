# Databricks notebook source
# MAGIC %md
# MAGIC Runtime: 15.4 ML GPU
# MAGIC
# MAGIC Node: AWS or Azure VM instance including more than a A10 GPUs

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルを推論する
# MAGIC
# MAGIC まずはモデルをダウンロードして、推論してみる。

# COMMAND ----------

# MAGIC %pip install --no-build-isolation flash_attn
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

model = AutoModelForCausalLM.from_pretrained("weblab-GENIAC/Tanuki-8B-dpo-v1.0", device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("weblab-GENIAC/Tanuki-8B-dpo-v1.0")
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

messages = [
    {"role": "system", "content": "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。"},
    {"role": "user", "content": "たぬきに純粋理性批判は理解できますか？"}
]

input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
output_ids = model.generate(input_ids,
                            max_new_tokens=1024,
                            temperature=0.5,
                            streamer=streamer)


# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルをモデルレジストリに登録する
# MAGIC Unity Catalogのモデルレジストリへモデルを登録する。

# COMMAND ----------

catalog = "YOUR_CATALOG_NAME"
schema = "models"
registered_model_name = f"{catalog}.{schema}.tanuki-8b-dpo-v1_0"

# COMMAND ----------

# Make sure that the catalog and the schema exist
_ = spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}") 
_ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}") 

# COMMAND ----------

# MAGIC %md
# MAGIC モデルをMLFlowのモデルレジストリへ登録する。登録時に`task="llm/v1/chat"`を指定することで、本モデルのSignatureをOpenAI互換のものに設定できる。
# MAGIC
# MAGIC https://mlflow.org/docs/latest/llms/transformers/tutorials/conversational/pyfunc-chat-model.html

# COMMAND ----------

from transformers import pipeline
import mlflow

mlflow.set_registry_uri("databricks-uc")

generator = pipeline(
    "text-generation",
    tokenizer="weblab-GENIAC/Tanuki-8B-dpo-v1.0",
    model="weblab-GENIAC/Tanuki-8B-dpo-v1.0",
)

with mlflow.start_run() as run:
    model_info = mlflow.transformers.log_model(
        transformers_model=generator,
        artifact_path="tanuki-8b-dpo-v1.0",
        task="llm/v1/chat",
        registered_model_name=registered_model_name,
    )

# COMMAND ----------

from mlflow import MlflowClient
client = MlflowClient()

# from mlflow.tracking.client import MlflowClient
def get_latest_model_version(model_name):
  client = MlflowClient()
  model_version_infos = client.search_model_versions("name = '%s'" % model_name)
  return max([int(model_version_info.version) for model_version_info in model_version_infos])

latest_version = get_latest_model_version(model_name=registered_model_name)

# 上記のセルに登録されている正しいモデルバージョンを選択
client.set_registered_model_alias(name=registered_model_name, alias="Champion", version=latest_version)

# COMMAND ----------

model_champion_uri = "models:/{model_name}@Champion".format(model_name=registered_model_name)
 
print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_champion_uri))
champion_model = mlflow.pyfunc.load_model(model_champion_uri)

champion_model.predict({"messages": messages, "max_tokens": 1024})

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルをデプロイする
# MAGIC サービングエンドポイントとしてモデルをデプロイする

# COMMAND ----------

# サービングエンドポイントの名前を指定
endpoint_name = 'endpoint-tanuki-8B-dpo-v1-0'

databricks_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# COMMAND ----------

# サービングエンドポイントの作成または更新
from mlflow.deployments import get_deploy_client

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput, ServedModelInputWorkloadSize, ServedModelInputWorkloadType
from datetime import timedelta

import requests
import json

# Name of the registered MLflow model
model_name = registered_model_name

# Get the latest version of the MLflow model
model_version = latest_version

headers = {"Context-Type": "text/json", "Authorization": f"Bearer {token}"}

response = requests.get(url=f"{databricks_url}/api/2.0/serving-endpoints/get-model-optimization-info/{model_name}/{model_version}", headers=headers)

print(json.dumps(response.json(), indent=4))

client = get_deploy_client("databricks")

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

# MAGIC %md
# MAGIC ## エンドポイントをテストする

# COMMAND ----------

import os
import openai
from openai import OpenAI

API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get() 
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

client = OpenAI(
    api_key=API_TOKEN,
    base_url=f"{API_ROOT}/serving-endpoints"
)

response = client.chat.completions.create(
    model=endpoint_name,
    messages=[
      {
        "role": "system",
        "content": "You're a helpful assistant. You have to answer all questions in Japanese."
      },
      {
        "role": "user",
        "content": "Mixture-of-experts（MoE）モデルとは何ですか?",
      }
    ],
    max_tokens=2000,
    temperature=0.1
)

print("【回答】")
print(response.choices[0].message.content)

# COMMAND ----------

from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")
response = client.predict(
    endpoint=endpoint_name,
    inputs={
        "messages": [{"role": "user", "content": "Mixture-of-experts（MoE）モデルとは何ですか?"}],
        "temperature": 0.0,
        "n": 1,
        "max_tokens": 500,
    },
)

print("【回答】")
print(response['choices'][0]['message']['content'])

# COMMAND ----------


