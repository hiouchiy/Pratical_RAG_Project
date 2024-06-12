# Databricks notebook source
# MAGIC %md 
# MAGIC ### 環境
# MAGIC - ノードタイプ: サーバーレス

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC # 2/ RAGによるチャットボットの作成
# MAGIC
# MAGIC 前のノートブックでベクターサーチインデックスと特徴量サービングの準備を行いました。
# MAGIC このノートブックでは、RAGを実行するためのモデルサービングエンドポイントを作成し、デプロイしてみましょう。
# MAGIC
# MAGIC フローは次のようになります：
# MAGIC
# MAGIC - ユーザーが質問します
# MAGIC - 質問はサーバレスチャットボットのRAGエンドポイントに送られます。
# MAGIC - エンドポイントはエンベッディングを計算し、Vector Search Indexを活用して質問に似たドキュメントを検索します。
# MAGIC - エンドポイントは、docでエンリッチされたプロンプトを作成します。
# MAGIC - プロンプトはModel Serving Endpointに送られます。
# MAGIC - ユーザーに出力を表示します！
# MAGIC
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=2556758628403379&notebook=%2F01-quickstart%2F02-Deploy-RAG-Chatbot-Model&demo_name=llm-rag-chatbot&event=VIEW&path=%2F_dbdemos%2Fdata-science%2Fllm-rag-chatbot%2F01-quickstart%2F02-Deploy-RAG-Chatbot-Model&version=1">

# COMMAND ----------

# MAGIC %md 
# MAGIC *注：RAGはDatabricks Vector Searchを使ってドキュメント検索を行います。このノートブックでは、検索インデックスが使用できる状態になっていることを前提としています。*
# MAGIC
# MAGIC 前回の[01-Data-Preparation-and-Index]($./01-Data-Preparation-and-Index [DO NOT EDIT])ノートブックを必ず実行してください。
# MAGIC

# COMMAND ----------

# MAGIC %pip install mlflow==2.10.1 langchain==0.1.5 databricks-vectorsearch==0.22 databricks-sdk==0.28.0 mlflow[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,コンフィグ(環境に合わせて修正してください）
# MAGIC %run ./config

# COMMAND ----------

# DBTITLE 1,カタログ初期化及びデモ用のインポートとヘルパーのインストール
# MAGIC %run ../_resources/00-init $reset_all_data=false

# COMMAND ----------

# MAGIC %md
# MAGIC   
# MAGIC ###  このデモを動かすにはシークレットが必要です：
# MAGIC Model Serving Endpoint は Vector Search Index を認証するためにシークレットを必要とします。 (see [Documentation](https://docs.databricks.com/en/security/secrets/secrets.html)).  <br/>
# MAGIC **Note: 共有のデモ・ワークスペースを使用していて、シークレットが設定されていることを確認した場合は、以下の手順を実行せず、その値を上書きしないでください。**<br/>
# MAGIC
# MAGIC - ラップトップまたはこのクラスタターミナルで[Databricks CLI](https://docs.databricks.com/en/dev-tools/cli/install.html) をセットアップする必要があります。 <br/>
# MAGIC `pip install databricks-cli` <br/>
# MAGIC - CLIを設定します。ワークスペースのURLとプロフィールページのPATトークンが必要です。e<br>
# MAGIC `databricks configure`
# MAGIC - dbdemosスコープを作成します。<br/>
# MAGIC `databricks secrets create-scope dbdemos`
# MAGIC - サービスプリンシパルのシークレットを保存します。これはモデルエンドポイントが認証するために使われます。これがデモ/テストである場合、あなたの[PAT token](https://docs.databricks.com/en/dev-tools/auth/pat.html)を利用できます。<br>
# MAGIC `databricks secrets put-secret dbdemos rag_sp_token`
# MAGIC
# MAGIC *Note: サービスプリンシパルがVector Searchインデックスにアクセスできることを確認してください。:*
# MAGIC
# MAGIC ```
# MAGIC spark.sql('GRANT USAGE ON CATALOG <catalog> TO `<YOUR_SP>`');
# MAGIC spark.sql('GRANT USAGE ON DATABASE <catalog>.<db> TO `<YOUR_SP>`');
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC import databricks.sdk.service.catalog as c
# MAGIC WorkspaceClient().grants.update(c.SecurableType.TABLE, <index_name>, 
# MAGIC                                 changes=[c.PermissionsChange(add=[c.Privilege["SELECT"]], principal="<YOUR_SP>")])
# MAGIC   ```

# COMMAND ----------

# DBTITLE 1,モデルの認証設定
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
os.environ['DATABRICKS_HOST'] = host

os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get(databricks_token_secrets_scope, databricks_token_secrets_key)

index_name=f"{catalog}.{db}.{embed_table_name}_vs_index"

print("HOST: " + os.environ['DATABRICKS_HOST'])
print("TOKEN: " + os.environ['DATABRICKS_TOKEN'])
print("Vector Search Index Name: " + index_name)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### チャットモデルの構築 （DBRX-instruct 基盤モデルへのクエリ）
# MAGIC
# MAGIC 今回は DBRX 基盤モデルを使って回答を生成します。
# MAGIC
# MAGIC Note: 複数のタイプのエンドポイントやラングチェーンモデルを使用することができます：
# MAGIC
# MAGIC - Databricks Foundationモデル（今回使用するものです）
# MAGIC - ファインチューニングしたモデル
# MAGIC - 外部のモデルプロバイダ（Azure OpenAIなど）

# COMMAND ----------

# DBTITLE 1,chat_modelモデルの定義（まだRAGは使用していない状態です）
# Databricks Foundation LLMモデルのテスト
from langchain.chat_models import ChatDatabricks
from langchain_core.messages import HumanMessage, SystemMessage

##############################################
# chat_modelモデルの定義(カスタムモデルを使用)
##############################################
chat_model = ChatDatabricks(
  endpoint=instruct_endpoint_name, 
  max_tokens = 2000, 
  temprature = 0.1)

messages = [
    SystemMessage(content="You're a helpful assistant"),
    HumanMessage(content="What is a mixture of experts model?"),
]

response = chat_model.invoke(messages)

print(f"Test chat model: {response}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### Chainオーケストレーターの構築
# MAGIC
# MAGIC それでは、フロントエンドアプリから質問を取得し、FAQデータ（ベクトル検索）とカード会員マスタ（特徴量サービング）から関連情報を検索し、プロンプトを拡張するレトリーバーと、回答を作成するチャットモデルを1つのチェーンに統合しましょう。
# MAGIC
# MAGIC 必要に応じて、様々なテンプレートを試し、AIアシスタントの口調や性格を皆様の要求に合うように調整してください。

# COMMAND ----------

import mlflow
from mlflow.pyfunc import PythonModel
from typing import Dict

class ChatbotRAGOrchestratorApp(mlflow.pyfunc.PythonModel):

    def __init__(self):
        """
        コンストラクタ
        """

        from databricks.vector_search.client import VectorSearchClient
        import mlflow.deployments
        from langchain.chat_models import ChatDatabricks
        from langchain_core.messages import SystemMessage
        from databricks import sql
        import os

        # ベクトル検索インデックスを取得
        vsc = VectorSearchClient(
          workspace_url=os.environ["DATABRICKS_HOST"], 
          personal_access_token=os.environ["DATABRICKS_TOKEN"])
        self.vs_index = vsc.get_index(
            endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
            index_name=index_name
        )

        # 特徴量サービングアクセス用クライアントの取得
        self.deploy_client = mlflow.deployments.get_deploy_client("databricks")

        # LLM基盤モデルのエンドポイントのクライアントを取得
        self.chat_model = ChatDatabricks(endpoint=instruct_endpoint_name, max_tokens = 2000)

        # システムプロンプトを準備
        self.SYSTEM_MESSAGE = SystemMessage(content="【ユーザー情報】と【参考情報】のみを参考にしながら【質問】にできるだけ正確に答えてください。わからない場合や、質問が適切でない場合は、分からない旨を答えてください。【参考情報】に記載されていない事実を答えるのはやめてください。")
        
    def _find_relevant_doc(self, question, rank, num_results = 10, relevant_threshold = 0.7):
        """
        ベクター検索インデックスにリクエストを送信し、類似コンテンツを検索
        """

        results = self.vs_index.similarity_search(
            query_text=question,
            columns=["usertype", "query", "response"],
            num_results=num_results,
            filters={"usertype": ("general", rank)})
        
        docs = results.get('result', {}).get('data_array', [])

        #関連性スコアでフィルタリングします。0.7以下は、関連性の高いコンテンツがないことを意味する
        returned_docs = []
        for doc in docs:
          if doc[-1] > relevant_threshold:
            returned_docs.append({"query": doc[1], "response": doc[2]})

        return returned_docs
    
    def _get_user_info(self, user_id):
        """
        カード会員マスタから当該ユーザーの属性情報を取得
        """

        result = self.deploy_client.predict(
          endpoint=user_endpoint_name,
          inputs={"dataframe_records": [{"id": user_id}]},
        )
        name = result['outputs'][0]['name']
        rank = result['outputs'][0]['type']
        birthday = result['outputs'][0]['birthday']
        since = result['outputs'][0]['since']

        return name, rank, birthday, since
    
    def _build_prompt(self, name, rank, birthday, since, docs):
        """
        プロンプトの構築
        """

        prompt = f"""【ユーザー情報】
名前：{name}
ランク：{rank}
生年月日：{birthday}
入会日：{since}


【参考情報】
"""

        for doc in docs:
          prompt = prompt + doc['response'] + "\n\n"

        #Final instructions
        prompt += f"\n\n 【質問】\n{question}"

        return prompt

    def predict(self, context, model_input, params=None):
        """
        推論メイン関数
        """

        # カード会員マスタから当該ユーザーの属性情報を取得
        userId = model_input['id'][0]
        name, rank, birthday, since = self._get_user_info(userId)

        #ベクター検索で質問と類似している情報を検索
        question = model_input['query'][0]
        docs = self._find_relevant_doc(question, rank)

        # プロンプトの構築
        prompt = self._build_prompt(name, rank, birthday, since, docs)

        # LLMに回答を生成させる
        from langchain_core.messages import HumanMessage
        query = [self.SYSTEM_MESSAGE, HumanMessage(content=prompt)]
        response = self.chat_model.invoke(query)
        
        # 回答データをパッケージング
        answers = [{"answer": response.content, "prompt": prompt}]
        return answers

# COMMAND ----------

question = "現在のランクから一つ上のランクに行くためにはどういった条件が必要ですか？"

proxy_model = ChatbotRAGOrchestratorApp()
results = proxy_model.predict(None, pd.DataFrame({"id": ["111"], "query": [question]}))
print(results[0]["answer"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### モデルをUnityカタログレジストリに保存
# MAGIC
# MAGIC モデルの準備ができたので、Unity Catalogスキーマに登録します

# COMMAND ----------

import numpy as np
import pandas as pd

import mlflow
from mlflow.models import infer_signature
from mlflow.models.signature import ModelSignature
from mlflow.types import ColSpec, DataType, ParamSchema, ParamSpec, Schema

mlflow.set_registry_uri("databricks-uc")
model_name = f"{catalog}.{db}.{registered_model_name}"

with mlflow.start_run(run_name="chatbot_rag_ja") as run:

    # 入出力スキーマの定義
    input_schema = Schema(
      [
        ColSpec(DataType.string, "id"),
        ColSpec(DataType.string, "query"),
      ]
    )

    output_schema = Schema(
      [
        ColSpec(DataType.string, "answer"),
        ColSpec(DataType.string, "prompt")
      ]
    )

    parameters = ParamSchema(
      [
        ParamSpec("temperature", DataType.float, np.float32(0.1), None),
        ParamSpec("max_tokens", DataType.integer, np.int32(1000), None),
      ]
    )

    signature = ModelSignature(inputs=input_schema, outputs=output_schema, params=parameters)

    input_data_example = pd.DataFrame({"id": ["111"], "query": [question]})

    mlflow.pyfunc.log_model(
      artifact_path="medallion_rag_model", 
      python_model=ChatbotRAGOrchestratorApp(), 
      signature=signature, 
      registered_model_name=model_name,
      input_example=input_data_example,
      pip_requirements=[
            "langchain==0.1.5",
            "databricks-vectorsearch==0.22",
            "databricks-sdk==0.28.0"]
    ) 
print(run.info.run_id)

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()
client.set_registered_model_alias(name=model_name, alias="Champion", version=get_latest_model_version(model_name))

# COMMAND ----------

import mlflow

loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name}@Champion")

answer = loaded_model.predict(
    pd.DataFrame({"id": ["111"], "query": [question]})
)

print(answer)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### チャットモデルをサービング・エンドポイントとしてデプロイする
# MAGIC
# MAGIC モデルはUnity Catalogに保存されます。最後のステップは、Model Servingとしてデプロイすることです。
# MAGIC
# MAGIC これでアシスタントのフロントエンドからリクエストを送信できるようになります。

# COMMAND ----------

# サービングエンドポイントの作成または更新
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput, ServedModelInputWorkloadSize

serving_endpoint_name = f"medallioncard_rag_endpoint_{catalog}_{db}"[:63]
latest_model_version = get_latest_model_version(model_name)

w = WorkspaceClient()
endpoint_config = EndpointCoreConfigInput(
    name=serving_endpoint_name,
    served_models=[
        ServedModelInput(
            model_name=model_name,
            model_version=latest_model_version,
            workload_size=ServedModelInputWorkloadSize.SMALL,
            scale_to_zero_enabled=False,
            environment_vars={
                "DATABRICKS_HOST": "{{secrets/"+databricks_host_secrets_scope+"/"+databricks_host_secrets_key+"}}",
                "DATABRICKS_TOKEN": "{{secrets/"+databricks_token_secrets_scope+"/"+databricks_token_secrets_key+"}}"
            }
        )
    ]
)

existing_endpoint = next(
    (e for e in w.serving_endpoints.list() if e.name == serving_endpoint_name), None
)
serving_endpoint_url = f"{host}/ml/endpoints/{serving_endpoint_name}"
if existing_endpoint == None:
    print(f"Creating the endpoint {serving_endpoint_url}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.create_and_wait(name=serving_endpoint_name, config=endpoint_config)
else:
    print(f"Updating the endpoint {serving_endpoint_url} to version {latest_model_version}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.update_config_and_wait(served_models=endpoint_config.served_models, name=serving_endpoint_name)
    
displayHTML(f'Your Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{serving_endpoint_name}">Model Serving Endpoint page</a> for more details.')

# COMMAND ----------

# MAGIC %md
# MAGIC これでエンドポイントが配置されました！エンドポイントは[Serving Endpoint UI](#/mlflow/endpoints)で確認できます。
# MAGIC
# MAGIC PythonでRESTクエリを実行してみましょう。ご覧のように、`test sentence` docを送信すると、回答が返されます。

# COMMAND ----------

answer = requests.post(
  f"{host}/serving-endpoints/{serving_endpoint_name}/invocations", 
  json={
    "inputs": { 
        "id": ["222"], 
        "query": ["私の現在のランクには空港ラウンジ特典はついていますか？"] 
    }
  },
  headers={
    'Authorization': f'Bearer {dbutils.secrets.get(scope=databricks_token_secrets_scope, key=databricks_token_secrets_key)}', 
  }
).json()

print(answer)
print(answer['predictions'][0]['answer'])

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### GradioをUIとして使ってみましょう！
# MAGIC
# MAGIC あとはチャットボットのUIをデプロイするだけです。以下は Gradio を使った簡単な例です。([License](https://github.com/gradio-app/gradio/blob/main/LICENSE)). Explore the chatbot gradio [implementation](https://huggingface.co/spaces/databricks-demos/chatbot/blob/main/app.py).
# MAGIC
# MAGIC *Note: このUIはDatabricksによってホストされ、デモ用にメンテナンスされています。Lakehouse Appsでその方法をすぐにお見せします！*

# COMMAND ----------

display_gradio_app("hiouchiy-medallioncardcorporation-dbrx")

# COMMAND ----------

# MAGIC %md
# MAGIC ## おめでとうございます！あなたは最初のGenAI RAGモデルをデプロイしました！
# MAGIC
# MAGIC レイクハウスのAIを活用し、社内のナレッジベースに同じロジックを導入する準備が整いました。
# MAGIC
# MAGIC LakehouseのAIがお客様のGenAIの課題を解決するためにどのようなユニークなポジションにあるかを見てきました：
# MAGIC
# MAGIC - Databricksのエンジニアリング機能によるデータ取り込みと準備の簡素化
# MAGIC - フルマネージドインデックスによるベクトル検索の展開の高速化
# MAGIC - Databricks LLama 2基盤モデルエンドポイントの活用
# MAGIC - リアルタイムモデルエンドポイントの導入によるRAGの実行とQ&A機能の提供
# MAGIC
# MAGIC Lakehouse AIは、お客様の生成AIの展開を加速させる唯一の機能を提供しています。
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next: ready to take it to a next level?
# MAGIC
# MAGIC Open the [02-advanced/01-PDF-Advanced-Data-Preparation]($../02-advanced/01-PDF-Advanced-Data-Preparation) notebook series to learn more about unstructured data, advanced chain, model evaluation and monitoring.

# COMMAND ----------

# MAGIC %md # Cleanup
# MAGIC
# MAGIC リソースを解放するために、コメントを削除して以下のセルを実行してください。

# COMMAND ----------

# /!\ THIS WILL DROP YOUR DEMO SCHEMA ENTIRELY /!\ 
# cleanup_demo(catalog, db, serving_endpoint_name, f"{catalog}.{db}.databricks_documentation_vs_index")
