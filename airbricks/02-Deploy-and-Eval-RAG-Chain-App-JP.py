# Databricks notebook source
# MAGIC %md 
# MAGIC ### 環境
# MAGIC - ノードタイプ: サーバーレス

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC # RAGエージェントの作成／デプロイ／評価
# MAGIC
# MAGIC 前回の[01-Data-Preparation-and-Index]($./01-Data-Preparation-and-Index [DO NOT EDIT])ノートブックで、RAGアプリケーションに必要な以下のコンポーネントを準備しました。
# MAGIC - FAQデータのベクトルインデックス (Embeddingモデル含む)
# MAGIC - 文章生成LLM（DBRX）のエンドポイント
# MAGIC
# MAGIC このノートブックでは、Mosaic AI Agent Frameworkを使用して、これらコンポーネントを繋ぎ合わせてユーザーからの質問に適切に回答する RAGエージェント・アプリ（チェーンやオーケストレーターとも呼ばれる）を作成し、それをエンドポイントとしてデプロイします。
# MAGIC
# MAGIC さらに、Mosaic AI Agent Evaluationを使用して、RAGエージェント・アプリの評価を行います。

# COMMAND ----------

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

import os

API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
os.environ["DATABRICKS_HOST"] = API_ROOT
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DATABRICKS_TOKEN"] = API_TOKEN

# COMMAND ----------

# MAGIC %md
# MAGIC ### RAGエージェントを実装する
# MAGIC
# MAGIC このデモではpyfunc.PythonModelベースの実装をします。
# MAGIC
# MAGIC 実装、および動作確認後、このセルのコードを以下のマジックコマンドを使用して.pyファイルとして書き出します。
# MAGIC
# MAGIC %%writefile chain.py

# COMMAND ----------

# %%writefile chain.py
import os

import pandas as pd

import mlflow
import mlflow.deployments

from databricks.vector_search.client import VectorSearchClient
from langchain_core.prompts.chat import HumanMessagePromptTemplate
from openai import OpenAI

class AirbricksRAGAgentApp(mlflow.pyfunc.PythonModel):

    def __init__(self):
        """
        コンストラクタ
        """

        self.model_config = mlflow.models.ModelConfig(development_config="rag_chain_config.yaml")

        try:
            # サービングエンドポイントのホストに"DB_MODEL_SERVING_HOST_URL"が自動設定されるので、その内容をDATABRICKS_HOSTにも設定
            os.environ["DATABRICKS_HOST"] = os.environ["DB_MODEL_SERVING_HOST_URL"]
        except:
            pass

        vsc = VectorSearchClient(disable_notice=True)
        self.vs_index = vsc.get_index(
            endpoint_name=self.model_config.get("vector_search_endpoint_name"),
            index_name=self.model_config.get("vector_search_index_name")
        )

        # 特徴量サービングアクセス用クライアントの取得
        self.deploy_client = mlflow.deployments.get_deploy_client("databricks")

        # LLM基盤モデルのエンドポイントのクライアントを取得
        self.chat_model = OpenAI(
            api_key=os.environ.get("DATABRICKS_TOKEN"),
            base_url=os.environ.get("DATABRICKS_HOST") + "/serving-endpoints",
        )

        # システムプロンプトを準備
        self.SYSTEM_MESSAGE = "【参考情報】のみを参考にしながら【質問】にできるだけ正確に答えてください。わからない場合や、質問が適切でない場合は、分からない旨を答えてください。【参考情報】に記載されていない事実を答えるのはやめてください。"

        # ヒューマンプロンプトテンプレートを準備
        human_template = """【参考情報】
{context}

【質問】
{question}"""
        self.HUMAN_MESSAGE = HumanMessagePromptTemplate.from_template(human_template)

        
    def _find_relevant_doc(self, question, num_results = 10, relevant_threshold = 0.7):
        """
        ベクター検索インデックスにリクエストを送信し、類似コンテンツを検索
        """

        results = self.vs_index.similarity_search(
            query_text=question,
            columns=["query", "response"],
            num_results=num_results)
        
        docs = results.get('result', {}).get('data_array', [])

        #関連性スコアでフィルタリングします。0.7以下は、関連性の高いコンテンツがないことを意味する
        returned_docs = []
        for doc in docs:
          if doc[-1] > relevant_threshold:
            returned_docs.append({"query": doc[0], "response": doc[1]})

        return returned_docs
    

    def _build_prompt(self, docs, question):
        """
        プロンプトの構築
        """

        context = ""
        for doc in docs:
          context = context + doc['response'] + "\n\n"

        human_message = self.HUMAN_MESSAGE.format_messages(
          context=context, 
          question=question
        )
    
        prompt=[
            {
                "role": "system",
                "content": self.SYSTEM_MESSAGE
            },
            {
                "role": "user",
                "content": human_message[0].content,
            }
        ]

        return prompt


    @mlflow.trace(name="predict_rag")
    def predict(self, context, model_input, params=None):
        """
        推論メイン関数
        """

        if isinstance(model_input, pd.DataFrame):
            model_input = model_input.to_dict(orient="records")[0]

        # FAQデータからベクター検索を用いて質問と類似している情報を検索
        with mlflow.start_span(name="_find_relevant_doc") as span:
            question = model_input["messages"][-1]["content"]
            docs = self._find_relevant_doc(question)
            span.set_inputs({"question": question})
            span.set_outputs({"docs": docs})

        # プロンプトの構築
        with mlflow.start_span(name="_build_prompt") as span:
            prompt = self._build_prompt(docs, question)
            span.set_inputs({"question": question, "docs": docs})
            span.set_outputs({"prompt": prompt})

        # LLMに回答を生成させる
        with mlflow.start_span(name="generate_answer") as span:
            response = self.chat_model.chat.completions.create(
                model=self.model_config.get("llm_endpoint_name"),
                messages=prompt,
                max_tokens=2000,
                temperature=0.1
            )
            span.set_inputs({"question": question, "prompt": prompt})
            span.set_outputs({"answer": response})
        
        
        # 回答データを整形して返す.
        # ChatCompletionResponseの形式で返さないと後々エラーとなる。
        return response.to_dict()


mlflow.models.set_model(model=AirbricksRAGAgentApp())

# COMMAND ----------

# MAGIC %md
# MAGIC ### RAGエージェントをテストする

# COMMAND ----------

input_example = {
  "messages": [{"role": "user", "content": "新しいエアコンを選ぶ際に最も重要なことは何ですか？"}]
}

rag_model = AirbricksRAGAgentApp()
rag_model.predict(None, model_input=input_example)

# COMMAND ----------

# MAGIC %md
# MAGIC ### RAGエージェント・アプリをUnity Catalogに登録

# COMMAND ----------

import os

# Specify the full path to the chain notebook
chain_notebook_path = os.path.join(os.getcwd(), "chain.py")

# Specify the full path to the config file (.yaml)
config_file_path = os.path.join(os.getcwd(), config_file_name)

print(f"Chain notebook path: {chain_notebook_path}")
print(f"Chain notebook path: {config_file_path}")

# COMMAND ----------

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
registered_agent = mlflow.pyfunc.load_model(f"models:/{model_name}/{uc_model_info.version}")

registered_agent.predict(
  {"messages": [{"role": "user", "content": "新しいエアコンを選ぶ際に最も重要なことは何ですか？"}]}
)

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

from databricks import agents
deployment_info = agents.deploy(
    model_name, 
    uc_model_info.version, 
    environment_vars={
        "DATABRICKS_TOKEN": "{{secrets/"+databricks_token_secrets_scope+"/"+databricks_token_secrets_key+"}}"
    })

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

# COMMAND ----------

import mlflow
import pandas as pd
eval_set  = [
    {
      "request_id": "1",
      "request": "Zenith ZR-450のタッチスクリーン操作パネルの反応が鈍いです。どうしたら良いですか？",
    },
    {
      "request_id": "2",
      "request": "エアコンを買い換えるタイミングの判断基準は何ですか？",
    },
    {
      "request_id": "3",
      "request": "リビングが３０平米なのですが、どの製品がベスト？",
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
  "messages": [{"role": "user", "content": "エアコンを買い換えるタイミングの判断基準は何ですか？"}]
}

databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Context-Type": "text/json", "Authorization": f"Bearer {databricks_token}"}

response = requests.post(
    url=f"{API_ROOT}/serving-endpoints/{deployment_info.endpoint_name}/invocations", json=data, headers=headers
)

print(response.json()["choices"][0]["message"]["content"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 以上です。
# MAGIC
# MAGIC リソースを解放するために、本サンプルで作成したすべてのリソースを削除ください。
# MAGIC
