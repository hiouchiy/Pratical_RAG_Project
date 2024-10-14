# Databricks notebook source
# MAGIC %md 
# MAGIC ### 環境
# MAGIC - ノードタイプ: サーバレス

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC # RAGのためのデータとモデルの準備
# MAGIC
# MAGIC このノートブックでは、チャットボットの回答精度を向上させるため、独自のドメイン固有データを用いて、ベクトル検索のインデックス、および、オンラインテーブルを作成します。
# MAGIC
# MAGIC 具体的には、架空のエアコンメーカー「株式会社エアブリックス」を例に取り、以下のドキュメントを使用します：
# MAGIC
# MAGIC - FAQデータ（非構造化データ）　→ ベクトル検索用インデックス化
# MAGIC
# MAGIC さらに、回答生成に用いるLLMとして以下のものを使用します。
# MAGIC
# MAGIC - DBRX Instruct (データブリックス上で基盤モデルAPIとして提供中)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0.ライブラリのインストール & 外部モジュールのロード

# COMMAND ----------

# MAGIC %pip install mlflow==2.10.1 lxml==4.9.3 transformers==4.30.2 langchain==0.1.5 databricks-vectorsearch==0.22 databricks-sdk==0.28.0 databricks-feature-store==0.17.0 openai
# MAGIC %pip install dspy-ai -U
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,コンフィグ(環境に合わせて修正してください）
# MAGIC %run ./config

# COMMAND ----------

# DBTITLE 1,カタログ初期化及びデモ用のインポートとヘルパーのインストール
# MAGIC %run ../_resources/00-init $reset_all_data=false

# COMMAND ----------

# MAGIC %md
# MAGIC ---

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## ステップ1. FAQデータをベクトルインデックス化
# MAGIC
# MAGIC まずは、[FAQデータ](https://raw.githubusercontent.com/hiouchiy/Pratical_RAG_Project/main/medallioncard/qa.json)（JSON）をベクトル検索のためにインデックス化していきましょう。
# MAGIC
# MAGIC 主な手順は以下の通りです：
# MAGIC
# MAGIC - FAQデータの元ファイル（JSON）をダウンロード
# MAGIC - JSONファイルをBronzeテーブル（Delta Table）として保存
# MAGIC - Bronzeテーブルを加工し、Silverテーブル（Delta Table）として保存
# MAGIC - ベクトル検索用のエンドポイントを作成
# MAGIC - Silverテーブルからベクトル検索インデックスを作成

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1-1.データを保存しておくためのカタログ/スキーマ/ボリュームを作成
# MAGIC _（※この操作はGUIでも実施可能）_

# COMMAND ----------

sql(f"CREATE CATALOG IF NOT EXISTS {catalog};")
sql(f"USE CATALOG {catalog};")
sql(f"CREATE SCHEMA IF NOT EXISTS {dbName};")
sql(f"USE SCHEMA {dbName};")
sql(f"CREATE VOLUME IF NOT EXISTS {volume};")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1-2.FAQデータ（JSON）をBronzeテーブル（Delta Table）として保存
# MAGIC 元データはJSON形式です。それをダウンロードして、Delta Tableの形式で保存しておきます。この際、特にスキーマ定義などは厳密に行わず、ありのままのデータを保存します。このようなテーブルをDatabricksのメダリオンアーキテクチャーではBronzeテーブルと呼びます。

# COMMAND ----------

# すでに同名のテーブルが存在する場合は削除
sql(f"drop table if exists {raw_data_table_name}")

# 生データを読み込み、デルタ・テーブルを作成して保存
raw_data_url = "https://raw.githubusercontent.com/hiouchiy/Pratical_RAG_Project/refs/heads/simple_ver/airbricks/query.json"
!wget $raw_data_url -O /tmp/query.json

unity_catalog_volume_path = f'/Volumes/{catalog}/{dbName}/{volume}/query.json'
!cp /tmp/query.json $unity_catalog_volume_path

spark.read.option("multiline","true").json(unity_catalog_volume_path).write.mode('overwrite').saveAsTable(raw_data_table_name)

display(spark.table(raw_data_table_name))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1-3.FAQデータのSilverテーブルを作成
# MAGIC 先ほど作成したFAQデータのBronzeテーブルを元に、スキーマ定義を厳密に行い、（本サンプルでは実施しませんが）データクレンジングなど加工を施したデータをSilverテーブルとして保存します。

# COMMAND ----------

sql(f"DROP TABLE IF EXISTS {embed_table_name};")

sql(f"""
--インデックスを作成するには、テーブルのChange Data Feedを有効にします
CREATE TABLE IF NOT EXISTS {embed_table_name} (
  id BIGINT GENERATED BY DEFAULT AS IDENTITY,
  query STRING,
  response STRING,
  url STRING
) TBLPROPERTIES (delta.enableChangeDataFeed = true); 
""")

spark.table(raw_data_table_name).write.mode('overwrite').saveAsTable(embed_table_name)

display(spark.table(embed_table_name))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### 1-4. Silverテーブルからベクトル検索用インデックスを作成
# MAGIC
# MAGIC Databricksは複数のタイプのベクトル検索インデックスを提供します：
# MAGIC
# MAGIC - **マネージドエンベッディング**：テキストカラムとエンドポイント名を指定すると、DatabricksがDeltaテーブルとインデックスを同期します。
# MAGIC - **自己管理型エンベッディング**：エンベッディングを計算し、デルタテーブルのフィールドとして保存すると、Databricksがインデックスを同期します。
# MAGIC - **ダイレクトインデックス**: デルタテーブルを持たずにインデックスを使用・更新したい場合
# MAGIC
# MAGIC このデモでは、**マネージドエンベッディング** インデックスを設定する方法を紹介します。

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC #### 1-4-1. Embedding モデルのエンドポイントを確認
# MAGIC _（※この操作はGUIでも実施可能）_
# MAGIC
# MAGIC DatabricksはEmbeddingの計算やモデルの評価のためにいくつかのエンドポイントをサポートしています：
# MAGIC - **基盤モデルエンドポイント**：Databricks が提供するマネージド・エンドポイント (例: llama2-70B, MPT...)
# MAGIC - **外部エンドポイント**：外部モデルへのゲートウェイとして動作するエンドポイント  (例: Azure OpenAI)
# MAGIC - **カスタムモデル用のエンドポイント**：Databricksモデルサービス上でホストされるファインチューニングされたモデルのエンドポイント
# MAGIC
# MAGIC このデモでは、3つ目のオプションである**カスタムモデル(e5-large)のエンドポイント**を使用しますが、必要に応じて基盤モデルのBGE（埋め込み）へ変更することも可能です。 
# MAGIC
# MAGIC なお、カスタムのEmbeddingモデルエンドポイントをDatabricks上にデプロイする手順は[こちら](https://github.com/hiouchiy/databricks-ml-examples/tree/master/llm-models/embedding/e5/multilingual-e5-large)を参照ください。

# COMMAND ----------

# DBTITLE 1,埋め込みエンドポイントとしてmultilingual-e5-largeモデルを使用
import mlflow.deployments
deploy_client = mlflow.deployments.get_deploy_client("databricks")

# カスタムEmbeddingモデル
response = deploy_client.predict(
  endpoint = embedding_endpoint_name, 
  inputs = {"inputs": ["新しいエアコンを選ぶ際に最も重要なことは何ですか？", "Vortex VX-600の製品スペックは？"]}
)
embeddings = [e for e in response.predictions]

print(embeddings)

# Databricksの基盤モデル「databricks-bge-large-en」への切り替えも簡単
# embedding_endpoint_name = "databricks-bge-large-en"
# response = deploy_client.predict(
#   endpoint = embedding_endpoint_name, 
#   inputs = {"input": ["新しいエアコンを選ぶ際に最も重要なことは何ですか？", "Vortex VX-600の製品スペックは？"]}
# )
# embeddings = [e for e in response.data]

# print(embeddings)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC #### 1-4-2. ベクトル検索エンドポイントを作成
# MAGIC _（※この操作はGUIでも実施可能）_
# MAGIC
# MAGIC 次に、Databricks Vector Searchを設定します。
# MAGIC
# MAGIC Vector search インデックスは、埋め込みデータを提供するためにVector searchエンドポイントを使用します（Vector Search APIエンドポイントと考えることができます）。<br/>
# MAGIC 複数のインデックスが同じエンドポイントを使用できます。まずは一つ作ってみましょう。

# COMMAND ----------

# DBTITLE 1,Vector searchエンドポイントの作成
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

if VECTOR_SEARCH_ENDPOINT_NAME not in [e['name'] for e in vsc.list_endpoints().get('endpoints', [])]:
    vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")

wait_for_vs_endpoint_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME)
print(f"Endpoint named {VECTOR_SEARCH_ENDPOINT_NAME} is ready.")

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC エンドポイントは [Vector Search Endpoints UI](#/setting/clusters/vector-search) で確認できます。エンドポイント名をクリックすると、そのエンドポイントによって提供されているすべてのインデックスが表示されます。

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC #### 1-4-3. ベクトル検索インデックスを作成
# MAGIC _（※この操作はGUIでも実施可能）_
# MAGIC
# MAGIC 次に、インデックスを作成します。

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c
import time

#インデックスの元となるテーブル
source_table_fullname = f"{catalog}.{db}.{embed_table_name}"

#インデックスを格納する場所
vs_index_fullname = f"{catalog}.{db}.{embed_table_name}_vs_index"

#すでに同名のインデックスが存在すれば削除
if index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname):
  print(f"Deleting index {vs_index_fullname} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}...")
  vsc.delete_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
  while True:
    if index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname):
      time.sleep(1)
      print(".")
    else:      
      break

#インデックスを新規作成
print(f"Creating index {vs_index_fullname} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}...")
vsc.create_delta_sync_index(
  endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
  index_name=vs_index_fullname,
  pipeline_type="TRIGGERED",
  source_table_name=source_table_fullname,
  primary_key="id",
  embedding_source_column="response",
  embedding_model_endpoint_name=embedding_endpoint_name
)

#インデックスの準備ができ、すべてエンベッディングが作成され、インデックスが作成されるのを待ちましょう。
wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
print(f"index {vs_index_fullname} on table {source_table_fullname} is ready")

# COMMAND ----------

# MAGIC %md
# MAGIC 作成したインデックスを更新する場合は以下のコードを実行します。

# COMMAND ----------

#同期をトリガーして、テーブルに保存された新しいデータでベクターサーチのコンテンツを更新
vs_index = vsc.get_index(
  VECTOR_SEARCH_ENDPOINT_NAME, 
  vs_index_fullname)
vs_index.sync()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 1-5. Similarity検索を試す
# MAGIC
# MAGIC 試しに類似コンテンツを検索してみましょう。
# MAGIC
# MAGIC *Note:`similarity_search` は filters パラメータもサポートしています。これは、RAGシステムにセキュリティレイヤーを追加するのに便利です。誰がエンドポイントへのアクセスを行うかに基づいて、機密性の高いコンテンツをフィルタリングすることができます（例えば、ユーザー情報に基づいて特定の部署をフィルタリングするなど）。*

# COMMAND ----------

# インデックスへの参照を取得
vs_index = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)

results = vs_index.similarity_search(
  query_text="新しいエアコンを選ぶ際に最も重要なことは何ですか？",
  columns=["query", "response", "url"],
  num_results=3)
docs = results.get('result', {}).get('data_array', [])
docs

# COMMAND ----------

# MAGIC %md
# MAGIC ---

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## ステップ2. 回答生成LLMの準備
# MAGIC _（※この操作はGUIでも実施可能）_
# MAGIC
# MAGIC 今回はDatabricksが提供する基盤モデルAPIから DBRX を使って回答を生成します。
# MAGIC その他にも、以下のモデルエンドポイントを利用可能です。
# MAGIC - Databricks Foundationモデル（今回使用するものです）
# MAGIC - ファインチューニングしたモデル
# MAGIC - 外部のモデルプロバイダ（Azure OpenAIなど）
# MAGIC
# MAGIC 参考として、日本語LLMのELYZA-7bをDatabricks上にエンドポイントとしてデプロイする手順は[こちら](https://github.com/hiouchiy/Pratical_RAG_Project/blob/main/Optional-Register_ELYZA_AI.py)をご参照ください。

# COMMAND ----------

# MAGIC %md
# MAGIC ### LangchainのChatDatabricksクラスを使用する場合

# COMMAND ----------

# Databricks Foundation LLMモデルのテスト
from langchain.chat_models import ChatDatabricks
from langchain_core.messages import HumanMessage, SystemMessage

##############################################
# chat_modelモデルの定義(カスタムモデルを使用)
##############################################
chat_model = ChatDatabricks(
  endpoint=instruct_endpoint_name, 
  max_tokens = 2000,
  temperature=0.1)

messages = [
    SystemMessage(content="You're a helpful assistant. You have to answer all questions in Japanese."),
    HumanMessage(content="Mixture-of-experts（MoE）モデルとは何ですか?"),
]

response = chat_model.invoke(messages)

print(f"Test chat model: {response}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### OpenAIのライブラリーを使用する場合

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
    model=instruct_endpoint_name,
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

print(f"Test chat model: {response}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## （おまけ）DSPyのDatabricksRMでベクトル検索

# COMMAND ----------

import dspy
from dspy.retrieve.databricks_rm import DatabricksRM

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

retrieve = DatabricksRM( # Set up retrieval from our vector search
            databricks_index_name=f"{catalog}.{db}.{embed_table_name}_vs_index",
            databricks_endpoint=url, 
            databricks_token=token,
            columns=["id", "query", "response"],
            text_column_name="response",
            docs_id_column_name="id",
            k=5
        )

# COMMAND ----------

retrieve("現在の私のランクの特典を教えてください。", query_type="text")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 次のステップ RAGを使ったチャットボットモデルのデプロイ
# MAGIC
# MAGIC Databricks Mosaic AIを使用すると、数行のコードと設定だけで、ドキュメントの取り込みと準備、その上でのVector Searchインデックスのデプロイを簡単に行うことができます。
# MAGIC
# MAGIC これにより、データプロジェクトが簡素化、高速化され、次のステップである、プロンプトのオーグメンテーションによるリアルタイムチャットボットのエンドポイントの作成に集中できるようになります。
# MAGIC
# MAGIC [02-Deploy-RAG-Chatbot-Model]($./02-Deploy-RAG-Chatbot-Model) チャットボットエンドポイントを作成し、デプロイするためのノートブックを開いてください。
