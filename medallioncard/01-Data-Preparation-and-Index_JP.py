# Databricks notebook source
# MAGIC %md 
# MAGIC ### 環境
# MAGIC - ノードタイプ: サーバレス

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC # 1/ RAGのためのデータ準備
# MAGIC
# MAGIC ### 独自データをベクトルインデックス化、または、オンラインテーブルとしてサービングする
# MAGIC
# MAGIC このノートブックでは、チャットボットがより良い回答を提供できるように、独自のドメイン固有なデータを用いて、Vector Searchのインデックスを作成します。
# MAGIC
# MAGIC この例では架空のクレジット会社「メダリオンカード株式会社」を例に取り、以下のドキュメントを使用します：
# MAGIC
# MAGIC - カード会員向けFAQデータ（非構造化データ）　→ ベクトル検索用インデックス化
# MAGIC - カード会員マスタ（構造化データ）　→ オンラインテーブル化

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0.ライブラリのインストール & 外部モジュールのロード

# COMMAND ----------

# MAGIC %pip install mlflow==2.10.1 lxml==4.9.3 transformers==4.30.2 langchain==0.1.5 databricks-vectorsearch==0.22 databricks-sdk==0.28.0 databricks-feature-store==0.17.0
# MAGIC %pip install dspy-ai -U
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,コンフィグ(環境に合わせて修正してください）
# MAGIC %run ./config

# COMMAND ----------

# DBTITLE 1,カタログ初期化及びデモ用のインポートとヘルパーのインストール
# MAGIC %run ../_resources/00-init $reset_all_data=false

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## 1.データ準備
# MAGIC
# MAGIC まず、生のデータセットを Delta Tableとして作成してみましょう。
# MAGIC
# MAGIC 主な手順は以下の通りです：
# MAGIC
# MAGIC - [FAQデータ](https://raw.githubusercontent.com/hiouchiy/Pratical_RAG_Project/main/medallioncard/qa.json)（非構造化データ）をダウンロードしてDelta Tableとして保存
# MAGIC - [カード会員マスタ](https://raw.githubusercontent.com/hiouchiy/Pratical_RAG_Project/main/medallioncard/user.json)（構造化データ）をダウンロードしてDelta Tableとして保存

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1-1.データを保存しておくためのカタログ/スキーマ/ボリュームを作成

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
sql(f"drop table if exists {faq_bronze_table_name}")

# 生データを読み込み、デルタ・テーブルを作成して保存
raw_data_url = "https://raw.githubusercontent.com/hiouchiy/Pratical_RAG_Project/main/medallioncard/qa.json"
!wget $raw_data_url -O /tmp/qa.json

unity_catalog_volume_path = f'/Volumes/{catalog}/{dbName}/{volume}/qa.json'
!cp /tmp/qa.json $unity_catalog_volume_path

spark.read.option("multiline","true").json(unity_catalog_volume_path).write.mode('overwrite').saveAsTable(faq_bronze_table_name)

display(spark.table(faq_bronze_table_name))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1-3.カード会員マスタをDelta Tableとして保存
# MAGIC 元データはJSON形式です。それをダウンロードして、Delta Tableの形式で保存しておきます。この際、特にスキーマ定義などは厳密に行わず、ありのままのデータを保存します。このようなテーブルをDatabricksのメダリオンアーキテクチャーではBronzeテーブルと呼びます。

# COMMAND ----------

# すでに同名のテーブルが存在する場合は削除
sql(f"drop table if exists {user_bronze_table_name}")

# 生データを読み込み、デルタ・テーブルを作成して保存
raw_data_url = "https://raw.githubusercontent.com/hiouchiy/Pratical_RAG_Project/main/medallioncard/user.json"
!wget $raw_data_url -O /tmp/user.json

unity_catalog_volume_path = f'/Volumes/{catalog}/{dbName}/{volume}/user.json'
!cp /tmp/user.json $unity_catalog_volume_path

spark.read.option("multiline","true").json(unity_catalog_volume_path).write.mode('overwrite').saveAsTable(user_bronze_table_name)

display(spark.table(user_bronze_table_name))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## 2.FAQデータを使ってベクターサーチインデックスを作成
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
# MAGIC ### 2-1.Embedding モデルのエンドポイントを確認
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
  inputs = {"inputs": ["Apache Sparkとはなんですか?", "ビッグデータとはなんですか？"]}
)
embeddings = [e for e in response.predictions]

print(embeddings)

# Databricksの基盤モデル「databricks-bge-large-en」への切り替えも簡単
# response = deploy_client.predict(
#   endpoint = "databricks-bge-large-en", 
#   inputs = {"input": ["Apache Sparkとはなんですか?", "ビッグデータとはなんですか？"]}
# )
# embeddings = [e for e in response.data]

# print(embeddings)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2-2.FAQデータのSilverテーブルを作成
# MAGIC 先ほど作成したFAQデータのBronzeテーブルを元に、スキーマ定義を厳密に行い、（本サンプルでは実施しませんが）データクレンジングなど加工を施したデータをSilverテーブルとして保存します。

# COMMAND ----------

sql(f"DROP TABLE IF EXISTS {faq_silver_table_name};")

sql(f"""
--インデックスを作成するには、テーブルのChange Data Feedを有効にします
CREATE TABLE IF NOT EXISTS {faq_silver_table_name} (
  id BIGINT GENERATED BY DEFAULT AS IDENTITY,
  usertype STRING,
  query STRING,
  response STRING
) TBLPROPERTIES (delta.enableChangeDataFeed = true); 
""")

spark.table(faq_bronze_table_name).write.mode('overwrite').saveAsTable(faq_silver_table_name)

display(spark.table(faq_silver_table_name))

# COMMAND ----------

display(
  sql(f"""select * from {faq_silver_table_name} where response like '%%' order by id;""")
)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### 2-3.Vector Search インデックスを作成
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

from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c
import time

#インデックスの元となるテーブル
source_table_fullname = f"{catalog}.{db}.{faq_silver_table_name}"

#インデックスを格納する場所
vs_index_fullname = f"{catalog}.{db}.{faq_silver_table_name}_vs_index"

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
# MAGIC ### 2-4.類似検索を試す
# MAGIC
# MAGIC 試しに類似コンテンツを検索してみましょう。
# MAGIC
# MAGIC *Note:`similarity_search` は filters パラメータもサポートしています。これは、RAGシステムにセキュリティレイヤーを追加するのに便利です。誰がエンドポイントへのアクセスを行うかに基づいて、機密性の高いコンテンツをフィルタリングすることができます（例えば、ユーザー情報に基づいて特定の部署をフィルタリングするなど）。*

# COMMAND ----------

# インデックスへの参照を取得
vs_index = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)

results = vs_index.similarity_search(
  query_text="現在の私のランクの特典を教えてください。",
  columns=["usertype", "query", "response"],
  num_results=5,
  filters={"usertype": ("general", "gold")})
docs = results.get('result', {}).get('data_array', [])
docs

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.カード会員マスタを使って特徴量サービングを作成

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3-1.カード会員マスタのBronzeテーブルからSilverテーブルを作成

# COMMAND ----------

from databricks import feature_engineering

fe = feature_engineering.FeatureEngineeringClient()

# Where we want to store our index
sql(f"DROP TABLE IF EXISTS {user_silver_table_name};")

sql(f"""
--インデックスを作成するには、テーブルのChange Data Feedを有効にします
CREATE TABLE IF NOT EXISTS {user_silver_table_name} (
    id VARCHAR(255),
    type STRING,
    name STRING,
    birthday STRING,
    since STRING,
    CONSTRAINT users_pk PRIMARY KEY(id)
) TBLPROPERTIES (delta.enableChangeDataFeed = true);
""")

# spark.table(user_bronze_table_name).write.mode('overwrite').saveAsTable(user_silver_table_name)
# Create the feature table
user_bronze_table_df = spark.table(user_bronze_table_name)
fe.write_table(name=user_silver_table_name, df=user_bronze_table_df)
display(spark.table(user_silver_table_name))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3-2.カード会員マスタのSilverテーブルからオンラインテーブルを作成

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy

w = WorkspaceClient()

# オンライン・テーブルの作成
source_table_full_name = f"{catalog}.{db}.{user_silver_table_name}"
spec = OnlineTableSpec(
  primary_key_columns=["id"],
  source_table_full_name=source_table_full_name,
  run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({'triggered': 'true'})
)

# オンラインテーブルを格納する場所
user_info_fs_online_fullname = source_table_full_name + "_online"
w.online_tables.create(name=user_info_fs_online_fullname, spec=spec)


# COMMAND ----------

# MAGIC %md
# MAGIC ### 3-3.オンラインテーブルへアクセスするためのエンドポイントを作成

# COMMAND ----------

from databricks.feature_engineering.entities.feature_lookup import FeatureLookup
from databricks.feature_engineering import FeatureEngineeringClient, FeatureFunction
from databricks.feature_engineering.entities.feature_serving_endpoint import (
    EndpointCoreConfig,
    ServedEntity
)

# Create a lookup to fetch features by key
features=[
  FeatureLookup(
    table_name=user_silver_table_name,
    lookup_key="id"
  )
]

fe = FeatureEngineeringClient()

# Create feature spec with the lookup for features
user_spec_name = f"{catalog}.{db}.user_info_spec"
try:
  fe.create_feature_spec(name=user_spec_name, features=features)
except Exception as e:
  if "already exists" in str(e):
    fe.delete_feature_spec(name=user_spec_name)
    fe.create_feature_spec(name=user_spec_name, features=features)
    # pass
  else:
    raise e
  
# Create endpoint for serving user budget preferences
try:
  fe.create_feature_serving_endpoint(
    name=user_endpoint_name, 
    config=EndpointCoreConfig(
      served_entities=ServedEntity(
        feature_spec_name=user_spec_name, 
        workload_size="Small", 
        scale_to_zero_enabled=False)
      )
    )

  # Print endpoint creation status
  print("Started creating endpoint " + user_endpoint_name)
except Exception as e:
  if "already exists" in str(e):
    pass
  else:
    raise e

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3-4.エンドポイントをテストする

# COMMAND ----------

import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")
response = client.predict(
    endpoint=user_endpoint_name,
    inputs={
        "dataframe_records": [
            {"id": "111"},
        ]
    },
)
print(response['outputs'][0])

# COMMAND ----------

# MAGIC %md
# MAGIC ## （おまけ）DSPyのDatabricksRMでベクトル検索

# COMMAND ----------

import dspy
from dspy.retrieve.databricks_rm import DatabricksRM

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

retrieve = DatabricksRM( # Set up retrieval from our vector search
            databricks_index_name=f"{catalog}.{db}.{faq_silver_table_name}_vs_index",
            databricks_endpoint=url, 
            databricks_token=token,
            columns=["id", "usertype", "query", "response"],
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
