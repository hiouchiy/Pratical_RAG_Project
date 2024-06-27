# Databricks notebook source
# MAGIC %md 
# MAGIC ### 環境
# MAGIC - ノードタイプ: サーバーレス

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC # RAGによるチャットボットの作成
# MAGIC
# MAGIC 前回の[01-Data-Preparation-and-Index]($./01-Data-Preparation-and-Index [DO NOT EDIT])ノートブックで、RAGアプリケーションに必要なコンポーネントを準備しました。具体的には以下の通りです。
# MAGIC - ユーザーマスタのオンラインテーブル
# MAGIC - FAQデータのベクトルインデックス (Embeddingモデル含む)
# MAGIC - 文章生成LLM（DBRX）のエンドポイント
# MAGIC
# MAGIC このノートブックでは、これらコンポーネントを繋ぎ合わせて、ユーザーからの質問に適切に回答するRAGチェーン・アプリ（オーケストレーターとも呼ばれる）を作成し、それをエンドポイントとしてデプロイします。
# MAGIC
# MAGIC RAGチェーン・アプリの処理フローは以下のようになります：
# MAGIC
# MAGIC 1. ユーザーから質問とユーザーIDが RAGチェーン・アプリ のエンドポイントへ送信される
# MAGIC 1. RAGチェーン・アプリにて、ユーザーIDをキーとして、ユーザーマスタから当該ユーザーの属性情報を取得
# MAGIC 1. RAGチェーン・アプリにて、質問に関連した情報を抜き出すべく、ベクトルインデックスに対して類似検索を実施
# MAGIC 1. RAGチェーン・アプリにて、ユーザー属性情報、質問関連情報、質問を組み合わせてプロンプトを作成
# MAGIC 1. RAGチェーン・アプリにて、プロンプトを文章生成LLM（DBRX）のエンドポイントへ送信
# MAGIC 1. ユーザーにLLMの出力を返す

# COMMAND ----------

# MAGIC %pip install --quiet -U databricks-agents==0.1.0 mlflow-skinny==2.14.0 mlflow==2.14.0 mlflow[gateway] langchain==0.2.1 langchain_core==0.2.5 langchain_community==0.2.4 databricks-vectorsearch==0.38 databricks-sdk==0.23.0 openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,コンフィグ(環境に合わせて修正してください）
# MAGIC %run ./config

# COMMAND ----------

# MAGIC %md
# MAGIC ### RAGチェーン・アプリで使用されるパラメータをYAMLファイルとして保存

# COMMAND ----------

import yaml
import mlflow

rag_chain_config = {
      "vector_search_endpoint_name": VECTOR_SEARCH_ENDPOINT_NAME,
      "vector_search_index_name": f"{catalog}.{dbName}.{faq_silver_table_name}_vs_index",
      "llm_endpoint_name": instruct_endpoint_name,
      "user_info_endpoint_name": user_endpoint_name,
}
config_file_name = 'rag_chain_config.yaml'
try:
    with open(config_file_name, 'w') as f:
        yaml.dump(rag_chain_config, f)
except:
    print('pass to work on build job')
model_config = mlflow.models.ModelConfig(development_config=config_file_name)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### Chainオーケストレーターの構築
# MAGIC
# MAGIC それでは、フロントエンドアプリから質問を取得し、FAQデータ（ベクトル検索）とカード会員マスタ（特徴量サービング）から関連情報を検索し、プロンプトを拡張するレトリーバーと、回答を作成するチャットモデルを1つのチェーンに統合しましょう。
# MAGIC
# MAGIC 必要に応じて、様々なテンプレートを試し、AIアシスタントの口調や性格を皆様の要求に合うように調整してください。

# COMMAND ----------

# MAGIC %md
# MAGIC ### モデルをUnityカタログ内のモデルレジストリに保存
# MAGIC
# MAGIC モデルの準備ができたので、Unity Catalogに登録します

# COMMAND ----------

import os

# Specify the full path to the chain notebook
chain_notebook_path = os.path.join(os.getcwd(), "02-RAG-Chain-App")

# Specify the full path to the config file (.yaml)
config_file_path = os.path.join(os.getcwd(), config_file_name)

print(f"Chain notebook path: {chain_notebook_path}")
print(f"Chain notebook path: {config_file_path}")

# COMMAND ----------

import numpy as np
import pandas as pd

import mlflow
from mlflow.models import infer_signature

with mlflow.start_run(run_name="medallioncard_rag_chatbot"):
  # 入出力スキーマの定義
  input_example = {
    "messages": [
        {
            "role": "user",
            "content": "現在のランクから一つ上のランクに行くためにはどういった条件が必要ですか？",
        }
    ],
  }

  output_response = {
    "id": "chatcmpl-64b2187c10c942819a5a554da07bea34",
    "object": "chat.completion",
    "created": 1711339597,
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "ゴールドランクの場合は・・・",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 13, "completion_tokens": 1, "total_tokens": 14},
  }

  params={
    "id": ["111"],
  }

  signature = infer_signature(
    model_input=input_example, 
    model_output=output_response, 
    params=params)
  
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

# Test the chain locally
chain = mlflow.pyfunc.load_model(logged_chain_info.model_uri)
chain.predict(input_example, params={"id": ["222"]})

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")

model_name = f"{catalog}.{dbName}.{registered_model_name}"
uc_model_info = mlflow.register_model(model_uri=logged_chain_info.model_uri, name=model_name)

# COMMAND ----------

### Test the registered model
model = mlflow.pyfunc.load_model(f"models:/{model_name}/{uc_model_info.version}")

model.predict({"messages": [{"role": "user", "content": "現在のランクから一つ上のランクに行くためにはどういった条件が必要ですか？"}]}, params={"id":["333"]})

# COMMAND ----------

# MAGIC %md 
# MAGIC ### チャットモデルをサービング・エンドポイントとしてデプロイ
# MAGIC
# MAGIC 最後のステップは、チェーンをエンドポイントとしてデプロイすることです。
# MAGIC Databricks の Agentライブラリを使用します。
# MAGIC
# MAGIC
# MAGIC これで、チェーンの評価、および、フロントエンドアプリからリクエストを送信できるようになります。

# COMMAND ----------

from databricks import agents
deployment_info = agents.deploy(
    model_name, 
    uc_model_info.version, 
    environment_vars={
        "DATABRICKS_TOKEN": "{{secrets/"+databricks_token_secrets_scope+"/"+databricks_token_secrets_key+"}}"
    })

review_instructions = """### メダリオン・カード株式会社 会員向けチャットボットのテスト手順

チャットボットの品質向上のためにぜひフィードバックを提供ください。

1. **多様な質問をお試しください**：
   - 実際の会員様が尋ねると予想される多様な質問を入力ください。これは、予想される質問を効果的に処理できるか否かを確認するのに役立ちます。

2. **回答に対するフィードバック**：
   - 質問の後、フィードバックウィジェットを使って、チャットボットの回答を評価してください。
   - 回答が間違っていたり、改善すべき点がある場合は、「回答の編集（Edit Response）」で修正してください。皆様の修正により、アプリケーションの精度を向上できます。

3. **回答に付随している参考文献の確認**：
   - 質問に対してシステムから回答される各参考文献をご確認ください。
   - Good👍／Bad👎機能を使って、その文書が質問内容に関連しているかどうかを評価ください。

チャットボットの評価にお時間を割いていただき、ありがとうございます。エンドユーザーに高品質の製品をお届けするためには、皆様のご協力が不可欠です。"""

agents.set_review_instructions(model_name, review_instructions)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Mosaic AIエージェント評価アプリを使用して人手でフィードバックを行う
# MAGIC 関係者にMosaic AIエージェント評価アプリへのアクセス権を与え、フィードバックを行ってもらいましょう。
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

# COMMAND ----------

import mlflow
import pandas as pd
eval_set  = [
    {
      "request_id": "1",
      "request": "私の現在のランクには空港ラウンジ特典はついていますか？",
    },
    {
      "request_id": "2",
      "request": "現在のランクから一つ上のランクに行くためにはどういう条件が必要ですか？",
    },
    {
      "request_id": "3",
      "request": "私のランクの特典を全て教えてください。",
    }
]
#### Convert dictionary to a pandas DataFrame
eval_set_df = pd.DataFrame(eval_set)


model_name = f"{catalog}.{dbName}.{registered_model_name}"

###
# mlflow.evaluate() call
###
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
  "messages": [{"role": "user", "content": "私の現在のランクには空港ラウンジ特典はついていますか？"}],
  "id": ["222"]
}

databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Context-Type": "text/json", "Authorization": f"Bearer {databricks_token}"}

response = requests.post(
    url=f"{host}/serving-endpoints/{deployment_info.endpoint_name}/invocations", json=data, headers=headers
)

print(json.dumps(response.json(), ensure_ascii=False))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### おまけ：GradioをUIとして使ってみましょう！
# MAGIC
# MAGIC Gradio で作成したUIからアクセスすることもできます。([License](https://github.com/gradio-app/gradio/blob/main/LICENSE))。
# MAGIC
# MAGIC *Note: このUIはHuggingFace Spaceによってホストされているものを、本ノートブック上で表示をしております。*

# COMMAND ----------

def display_gradio_app(space_name = "databricks-demos-chatbot"):
    displayHTML(f'''<div style="margin: auto; width: 1000px"><iframe src="https://{space_name}.hf.space" frameborder="0" width="1000" height="950" style="margin: auto"></iframe></div>''')
    
display_gradio_app("hiouchiy-medallioncardcorporation-dbrx")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 以上です。
# MAGIC
# MAGIC リソースを解放するために、本サンプルで作成したすべてのリソースを削除ください。
# MAGIC
