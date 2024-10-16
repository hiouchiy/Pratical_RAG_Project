# Databricks notebook source
# MAGIC %md 
# MAGIC ### 環境
# MAGIC - ノードタイプ: サーバーレス、または、15.4 ML LTS

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC # RAGエージェントの作成／デプロイ／評価
# MAGIC
# MAGIC 前回の[01-Data-Preparation-and-Index]($./01-Data-Preparation-and-Index [DO NOT EDIT])ノートブックで、RAGアプリケーションに必要な以下のコンポーネントを準備しました。
# MAGIC - FAQデータのベクトルインデックス (Embeddingモデル含む)
# MAGIC - 文章生成LLM（Llama3.1-70B）のエンドポイント
# MAGIC
# MAGIC このノートブックでは、Mosaic AI Agent Frameworkを使用して、これらコンポーネントを繋ぎ合わせてユーザーからの質問に適切に回答する RAGエージェント・アプリ（チェーンやオーケストレーターとも呼ばれる）を作成し、それをエンドポイントとしてデプロイします。
# MAGIC
# MAGIC さらに、Mosaic AI Agent Evaluationを使用して、RAGエージェント・アプリの評価を行います。

# COMMAND ----------

# MAGIC %pip install -U -qqqq databricks-agents mlflow mlflow-skinny databricks-vectorsearch langchain==0.2.11 langchain_core==0.2.23 langchain_community==0.2.10 openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,バックアップ（使用しません）
# MAGIC %pip install --quiet -U databricks-agents==0.1.0 mlflow-skinny==2.14.0 mlflow==2.14.0 mlflow[gateway] langchain==0.2.1 langchain_core==0.2.5 langchain_community==0.2.4 databricks-vectorsearch==0.38 databricks-sdk==0.23.0 openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,コンフィグ(環境に合わせて修正してください）
# MAGIC %run ./config

# COMMAND ----------

# MAGIC %md
# MAGIC ### RAGエージェント・アプリで使用されるパラメータをYAMLファイルとして保存

# COMMAND ----------

import yaml
import mlflow

rag_chain_config = {
      "vector_search_endpoint_name": VECTOR_SEARCH_ENDPOINT_NAME,
      "vector_search_index_name": f"{catalog}.{dbName}.{embed_table_name}_vs_index",
      "llm_endpoint_name": instruct_endpoint_name,
}
config_file_name = 'rag_chain_config.yaml'
try:
    with open(config_file_name, 'w') as f:
        yaml.dump(rag_chain_config, f)
except:
    print('pass to work on build job')

# COMMAND ----------

# MAGIC %md
# MAGIC ### RAGエージェントの実装コード
# MAGIC
# MAGIC Langchain、および、pyfunc.PythonModelベースで実装できます。
# MAGIC
# MAGIC 実装コードは"chain"および"chain_pyfunc"にそれぞれあります。

# COMMAND ----------

# MAGIC %md
# MAGIC ### RAGエージェント・アプリをUnity Catalogに登録

# COMMAND ----------

import os

# Specify the full path to the chain notebook
chain_notebook_path = os.path.join(os.getcwd(), "chain_langchain")

# Specify the full path to the config file (.yaml)
config_file_path = os.path.join(os.getcwd(), "rag_chain_config.yaml")

print(f"Chain notebook path: {chain_notebook_path}")
print(f"Chain notebook path: {config_file_path}")

# COMMAND ----------

user_account_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

# COMMAND ----------

import mlflow

# Set the experiment name
mlflow.set_experiment(f"/Users/{user_account_name}/airbricks_rag_experiment")

# Log the model to MLflow
# TODO: remove example_no_conversion once this papercut is fixed
with mlflow.start_run(run_name="airbricks_rag_chatbot"):
    # Tag to differentiate from the data pipeline runs
    mlflow.set_tag("type", "chain")

    input_example = {
        "messages": [{"role": "user", "content": "Zenith ZR-450のタッチスクリーン操作パネルの反応が鈍いです。どうしたら良いですか？"}]
    }

    logged_chain_info = mlflow.langchain.log_model(
        lc_model=chain_notebook_path,  # Chain code file e.g., /path/to/the/chain.py
        model_config=config_file_path,  # Chain configuration set in 00_config
        artifact_path="chain",  # Required by MLflow
        input_example=input_example,  # Save the chain's input schema.  MLflow will execute the chain before logging & capture it's output schema.
        example_no_conversion=True,  # Required by MLflow to use the input_example as the chain's schema
    )

    # # Attach the data pipeline's configuration as parameters
    # mlflow.log_params(_flatten_nested_params({"data_pipeline": data_pipeline_config}))

    # # Attach the data pipeline configuration 
    # mlflow.log_dict(data_pipeline_config, "data_pipeline_config.json")

# COMMAND ----------

# DBTITLE 1,バックアップ（使用しません）
import numpy as np
import pandas as pd

import mlflow
from mlflow.models import infer_signature

with mlflow.start_run(run_name="airbricks_rag_chatbot"):
  # 入出力スキーマの定義
  input_example = {
    "messages": [
        {
            "role": "user",
            "content": "新しいエアコンを選ぶ際に最も重要なことは何ですか？",
        }
    ],
  }

  output_response = {
    'id': 'chatcmpl_e048d1af-4b9c-4cc9-941f-0311ac5aa7ab',
    'choices': [
      {
        'finish_reason': 'stop', 
        'index': 0,
        'logprobs': "",
        'message': {
          'content': '新しいエアコンを選ぶ際に最も重要なことは、冷却能力、エネルギー効率、サイズ、特定の機能（例えば空気浄化やWi-Fi接続）など、ご自宅やオフィスのニーズに最適な特性を考慮することです。',
          'role': 'assistant'
          }
        }
      ],
    'created': 1719722525,
    'model': 'dbrx-instruct-032724',
    'object': 'chat.completion',
    'usage': {'completion_tokens': 279,'prompt_tokens': 1386,'total_tokens': 1665}
  }

  signature = infer_signature(
    model_input=input_example, 
    model_output=output_response)
  
  logged_chain_info = mlflow.pyfunc.log_model(
    python_model=chain_notebook_path,
    model_config=config_file_path,
    artifact_path="chain",
    signature=signature, 
    input_example=input_example,
    example_no_conversion=True,
    pip_requirements=[
      "databricks-agents==0.1.0",
      "langchain==0.2.1",
      "langchain_core==0.2.5",
      "langchain_community==0.2.4",
      "databricks-vectorsearch==0.38",
      "databricks-sdk==0.23.0",
      "openai"]
  )

# COMMAND ----------

chain = mlflow.langchain.load_model(logged_chain_info.model_uri)
chain.invoke(input_example)

# COMMAND ----------

# DBTITLE 1,バックアップ（使用しません）
# Test the chain locally
chain = mlflow.pyfunc.load_model(logged_chain_info.model_uri)
chain.predict(input_example)

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")

model_name = f"{catalog}.{dbName}.{registered_model_name}"
uc_model_info = mlflow.register_model(model_uri=logged_chain_info.model_uri, name=model_name)

# COMMAND ----------

### Test the registered model
registered_agent = mlflow.langchain.load_model(f"models:/{model_name}/{uc_model_info.version}")

registered_agent.invoke(input_example)

# COMMAND ----------

# DBTITLE 1,バックアップ（使用しません）
### Test the registered model
registered_agent = mlflow.pyfunc.load_model(f"models:/{model_name}/{uc_model_info.version}")

registered_agent.predict(input_example)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### RAGエージェント・アプリをサービング・エンドポイントとしてデプロイ
# MAGIC
# MAGIC 最後のステップは、エージェント・アプリをエンドポイントとしてデプロイすることです。
# MAGIC Databricks Agentライブラリを使用します。
# MAGIC
# MAGIC
# MAGIC これで、チェーンの評価、および、フロントエンドアプリからリクエストを送信できるようになります。

# COMMAND ----------

import os
import mlflow
from databricks import agents

deployment_info = agents.deploy(
    model_name, 
    uc_model_info.version 
)

browser_url = mlflow.utils.databricks_utils.get_browser_hostname()
print(f"\n\nView deployment status: https://{browser_url}/ml/endpoints/{deployment_info.endpoint_name}")

review_instructions = """### 株式会社エアブリックス FAQチャットボットのテスト手順

チャットボットの品質向上のためにぜひフィードバックを提供ください。

1. **多様な質問をお試しください**：
   - 実際のお客様が尋ねると予想される多様な質問を入力ください。これは、予想される質問を効果的に処理できるか否かを確認するのに役立ちます。

2. **回答に対するフィードバック**：
   - 質問の後、フィードバックウィジェットを使って、チャットボットの回答を評価してください。
   - 回答が間違っていたり、改善すべき点がある場合は、「回答の編集（Edit Response）」で修正してください。皆様の修正により、アプリケーションの精度を向上できます。

3. **回答に付随している参考文献の確認**：
   - 質問に対してシステムから回答される各参考文献をご確認ください。
   - Good👍／Bad👎機能を使って、その文書が質問内容に関連しているかどうかを評価ください。

チャットボットの評価にお時間を割いていただき、ありがとうございます。エンドユーザーに高品質の製品をお届けするためには、皆様のご協力が不可欠です。"""

agents.set_review_instructions(model_name, review_instructions)

# COMMAND ----------

# DBTITLE 1,バックアップ（使用しません）
import os
import mlflow
from databricks import agents

deployment_info = agents.deploy(
    model_name, 
    uc_model_info.version, 
    environment_vars={
        "DATABRICKS_TOKEN": "{{secrets/"+databricks_token_secrets_scope+"/"+databricks_token_secrets_key+"}}"
    })

browser_url = mlflow.utils.databricks_utils.get_browser_hostname()
print(f"\n\nView deployment status: https://{browser_url}/ml/endpoints/{deployment_info.endpoint_name}")

review_instructions = """### 株式会社エアブリックス FAQチャットボットのテスト手順

チャットボットの品質向上のためにぜひフィードバックを提供ください。

1. **多様な質問をお試しください**：
   - 実際のお客様が尋ねると予想される多様な質問を入力ください。これは、予想される質問を効果的に処理できるか否かを確認するのに役立ちます。

2. **回答に対するフィードバック**：
   - 質問の後、フィードバックウィジェットを使って、チャットボットの回答を評価してください。
   - 回答が間違っていたり、改善すべき点がある場合は、「回答の編集（Edit Response）」で修正してください。皆様の修正により、アプリケーションの精度を向上できます。

3. **回答に付随している参考文献の確認**：
   - 質問に対してシステムから回答される各参考文献をご確認ください。
   - Good👍／Bad👎機能を使って、その文書が質問内容に関連しているかどうかを評価ください。

チャットボットの評価にお時間を割いていただき、ありがとうございます。エンドユーザーに高品質の製品をお届けするためには、皆様のご協力が不可欠です。"""

agents.set_review_instructions(model_name, review_instructions)

# COMMAND ----------

import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate
from databricks.sdk.errors import NotFound, ResourceDoesNotExist

# Wait for the Review App to be ready
print("\nWaiting for endpoint to deploy.  This can take 15 - 20 minutes.", end="")
w = WorkspaceClient()
while w.serving_endpoints.get(deployment_info.endpoint_name).state.ready == EndpointStateReady.NOT_READY or w.serving_endpoints.get(deployment_info.endpoint_name).state.config_update == EndpointStateConfigUpdate.IN_PROGRESS:
    print(".", end="")
    time.sleep(30)

print(f"\n\nReview App: {deployment_info.review_app_url}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Mosaic AI Agent Evaluationのレビューアプリを使用して人手でフィードバックを行う
# MAGIC 関係者にMosaic AI Agent Evaluation のレビューアプリ へのアクセス権を与え、フィードバックを行ってもらいましょう。
# MAGIC アクセスを簡単にするため、関係者はDatabricksアカウントを持っている必要はありません。

# COMMAND ----------

from databricks import agents

user_list = ["someone@databricks.com"]
agents.set_permissions(model_name=model_name, users=user_list, permission_level=agents.PermissionLevel.CAN_QUERY)

print(f"Share this URL with your stakeholders: {deployment_info.review_app_url}")

# COMMAND ----------

# MAGIC %md
# MAGIC エンドポイントのデプロイ状況は[Serving Endpoint UI](#/mlflow/endpoints)で確認できます。デプロイ完了まで数分程度要します。
# MAGIC なお、Feedbackというのが、レビューアプリ用のエンドポイントです。
# MAGIC
# MAGIC デプロイ完了後、レビューアプリのURLにアクセスして人手によるフィードバックを行いましょう。

# COMMAND ----------

# MAGIC %md
# MAGIC ### Mosaic AI Agent Evaluation "LLM-as-a-judge" を使用してRAGエージェントの自動評価を行う
# MAGIC
# MAGIC 参考：https://docs.databricks.com/en/generative-ai/agent-evaluation/evaluation-set.html

# COMMAND ----------

import mlflow
import pandas as pd
eval_set  = [
    {
      "request_id": "1",
      "request": "Zenith ZR-450のタッチスクリーン操作パネルの反応が鈍いです。どうしたら良いですか？",
      "expected_retrieved_context": [
        {
            "doc_uri": "https://example.com/1855",
        }
      ],
      "expected_response": "Zenith ZR-450のタッチスクリーンが鈍い場合の具体的な対処法は以下の通りです：\n	1.	画面を清掃してください。\n	2.	改善しない場合は、ファームウェアのアップデートを確認してください。\n	3.	それでも解決しない場合は、サポートセンターに連絡して技術的なサポートを受けてください。\n\nこちらが正しい対応手順となります。"
    },
    {
      "request_id": "2",
      "request": "エアコンを買い換えるタイミングの判断基準は何ですか？",
      "expected_retrieved_context": [
        {
            "doc_uri": "https://example.com/8321",
        },
        {
            "doc_uri": "https://example.com/3029",
        },
        {
            "doc_uri": "https://example.com/3724",
        },
        {
            "doc_uri": "https://example.com/2849",
        }
      ],
      "expected_response": "エアコンを買い換えるタイミングの判断基準は以下の通りです：\n\n	1.	使用年数が10年以上経過した。\n	2.	修理が頻繁に必要になった。\n	3.	電気代が増加し、エネルギー効率が低下している。\n	4.	最新技術や機能を活用したいと考えている場合。"
    },
    {
      "request_id": "3",
      "request": "リビングが３０平米なのですが、どの製品がベスト？",
      "expected_retrieved_context": [
        {
            "doc_uri": "https://example.com/9662",
        },
        {
            "doc_uri": "https://example.com/4609",
        },
        {
            "doc_uri": "https://example.com/1885",
        }
      ],
      "expected_response": "30平米のリビングに適したエアコンとしては、以下の製品が候補となります：\n\n	1.	EcoSmart TY-700:\n	•	冷却能力：7.0 kW\n	•	広い空間に対応可能で、効率的な冷暖房が可能です。\n	2.	Zenith ZR-450:\n	•	冷却能力：4.5 kW\n	•	少し小さめの冷却能力ですが、30平米程度の部屋には十分なパフォーマンスを発揮します。\n\nいずれも、広さに応じた冷却能力を持つため、好みに応じて選択できます。"
    }
]
#### Convert dictionary to a pandas DataFrame
eval_set_df = pd.DataFrame(eval_set)


model_name = f"{catalog}.{dbName}.{registered_model_name}"

###
# mlflow.evaluate() call
###
# with mlflow.start_run(run_id=logged_chain_info.run_id):
with mlflow.start_run(run_name="new_eval_run"):
  evaluation_results = mlflow.evaluate(
      data=eval_set_df,
      model=f"models:/{model_name}/{uc_model_info.version}",
      model_type="databricks-agent",
  )

# COMMAND ----------

evaluation_results.tables["eval_results"]

# COMMAND ----------

# MAGIC %md
# MAGIC ### 最後にデプロイされたRAGエージェントにRESTクライアントからアクセスしてみましょう

# COMMAND ----------

import requests
import json

data = {
  "messages": [{"role": "user", "content": "エアコンの買い換えを決める際の判断基準はありますか？"}]
}

databricks_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Context-Type": "text/json", "Authorization": f"Bearer {databricks_token}"}

response = requests.post(
    url=f"{databricks_host}/serving-endpoints/{deployment_info.endpoint_name}/invocations", json=data, headers=headers
)

print(response.json()["choices"][0]["message"]["content"])

# COMMAND ----------

# MAGIC %md ## おまけ：レビューアプリ名を検索
# MAGIC
# MAGIC このノートブックの状態を失い、レビューアプリのURLを見つける必要がある場合は、このセルを実行します。
# MAGIC
# MAGIC または、レビューアプリのURLを次のように作成することもできます。
# MAGIC
# MAGIC `https://<your-workspace-url>/ml/reviews/{UC_CATALOG}.{UC_SCHEMA}.{UC_MODEL_NAME}/{UC_MODEL_VERSION_NUMBER}/instructions`

# COMMAND ----------

active_deployments = agents.list_deployments()

active_deployment = next((item for item in active_deployments if item.model_name == model_name), None)

print(f"Review App URL: {active_deployment.review_app_url}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 以上です。
# MAGIC
# MAGIC リソースを解放するために、本サンプルで作成したすべてのリソースを削除ください。
# MAGIC
