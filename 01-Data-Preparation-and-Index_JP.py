# Databricks notebook source
# MAGIC %md 
# MAGIC ### 環境
# MAGIC - Runtime: 14.2 ML GPU
# MAGIC - Node type: g5.8xlarge (Single Node)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC # 1/ LLMチャットボットRAGのためのデータ準備
# MAGIC
# MAGIC ### 企業の独自データをDatabricksベクターサーチでインデックス化する
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-self-managed-flow-1.png?raw=true" style="float: right; width: 800px; margin-left: 10px">
# MAGIC
# MAGIC このノートブックでは、チャットボットがより良い回答を提供できるように、ドキュメントページをインジェストし、Vector Searchインデックスでインデックスを作成します。
# MAGIC
# MAGIC 高品質のデータを準備することは、チャットボットのパフォーマンスにとって重要です。次のステップは、時間をかけてご自身のデータセットで実施することをお勧めします。
# MAGIC
# MAGIC Lakehouse AIはお客様のAIやLLMプロジェクトを加速させる最先端のソリューションを提供し、スケーラブルにデータの取り込みと準備を簡素化します。
# MAGIC
# MAGIC この例では[docs.databricks.com](docs.databricks.com)のDatabricksドキュメントを使用します：
# MAGIC
# MAGIC - ウェブページのダウンロード
# MAGIC - ページを小さなテキストチャンクに分割します。
# MAGIC - Databricks Foundationモデルを使用してエンべディングを計算します。
# MAGIC - Delta TableにとしてVector Searchインデックスを作成します。
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=2556758628403379&notebook=%2F01-quickstart%2F01-Data-Preparation-and-Index&demo_name=llm-rag-chatbot&event=VIEW&path=%2F_dbdemos%2Fdata-science%2Fllm-rag-chatbot%2F01-quickstart%2F01-Data-Preparation-and-Index&version=1">

# COMMAND ----------

# DBTITLE 1,必要な外部ライブラリのインストール 
# MAGIC %pip install mlflow==2.9.0 lxml==4.9.3 transformers==4.30.2 langchain==0.0.344 databricks-vectorsearch==0.22 databricks-sdk==0.12.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,コンフィグ(環境に合わせて修正してください）
# MAGIC %run ./config

# COMMAND ----------

# DBTITLE 1,カタログ初期化及びデモ用のインポートとヘルパーのインストール
# MAGIC %run ./_resources/00-init $reset_all_data=false

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Databricks ドキュメントのサイトマップとページの抽出
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-self-managed-prep-1.png?raw=true" style="float: right; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC まず、生のデータセットを Delta Lake テーブルとして作成してみましょう。
# MAGIC
# MAGIC このデモでは、`docs.databricks.com`からいくつかのドキュメントページを直接ダウンロードし、HTMLコンテンツを保存します。
# MAGIC
# MAGIC 主な手順は以下の通りです：
# MAGIC
# MAGIC - スクリプトを実行して `sitemap.xml` ファイルからページの URL を抽出します。
# MAGIC - ウェブページをダウンロード
# MAGIC - BeautifulSoupを使ってArticleBodyを抽出します。
# MAGIC - HTMLの結果をデルタレイクのテーブルに保存

# COMMAND ----------

spark.conf.set("my.catalogName", catalog)
spark.conf.set("my.schemaName", dbName)
spark.conf.set("my.volumeName", volume)
spark.conf.set("my.embbedTableName", embed_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE CATALOG IF NOT EXISTS ${my.catalogName};
# MAGIC USE CATALOG ${my.catalogName};
# MAGIC CREATE SCHEMA IF NOT EXISTS ${my.catalogName}.${my.schemaName};
# MAGIC USE SCHEMA ${my.schemaName};
# MAGIC CREATE VOLUME IF NOT EXISTS ${my.catalogName}.${my.schemaName}.${my.volumeName};

# COMMAND ----------

# Drop if table existing
sql(f"drop table if exists {raw_data_table_name}")

# Read raw data and create delta table to store it
raw_data_url = "https://raw.githubusercontent.com/hiouchiy/Pratical_RAG_Project/main/query.json"
!wget $raw_data_url -O /tmp/query.json

unity_catalog_volume_path = f'/Volumes/{catalog}/{dbName}/{volume}/query.json'
!cp /tmp/query.json $unity_catalog_volume_path

spark.read.option("multiline","true").json(unity_catalog_volume_path).write.mode('overwrite').saveAsTable(raw_data_table_name)

display(spark.table(raw_data_table_name))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## ベクターサーチインデックスの作成
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/databricks-vector-search-type.png?raw=true" style="float: right" width="800px">
# MAGIC
# MAGIC Databricksは複数のタイプのベクトル検索インデックスを提供します：
# MAGIC
# MAGIC - **Managedエンベッディング**：テキストカラムとエンドポイント名を指定すると、DatabricksがDeltaテーブルとインデックスを同期します。
# MAGIC - **自己管理型エンベッディング**：エンベッディングを計算し、デルタテーブルのフィールドとして保存すると、Databricksがインデックスを同期します。
# MAGIC - **ダイレクトインデックス**: デルタテーブルを持たずにインデックスを使用・更新したい場合
# MAGIC
# MAGIC このデモでは、**自己管理型エンベッディング** インデックスを設定する方法を紹介します。
# MAGIC
# MAGIC そのためには、まずチャンクのエンベッディングを計算し、デルタレイクのテーブルフィールドとして `array&ltfloat&gt` 型のデータとして保存する必要があります。

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Databricks BGE Embeddings Foundation モデルのエンドポイント
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-self-managed-prep-4.png?raw=true" style="float: right; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC Foundation ModelsはDatabricksによって提供されており、すぐに使用することができます。
# MAGIC
# MAGIC Databricksはエンベッディングの計算やモデルの評価のためにいくつかのエンドポイントをサポートしています：
# MAGIC - Databricks が提供する **ファウンデーションモデルエンドポイント** (例: llama2-70B, MPT...)
# MAGIC - 外部モデルへのゲートウェイとして動作する **外部エンドポイント** (例: Azure OpenAI)
# MAGIC - Databricksモデルサービス上でホストされるファインチューニングされた**カスタムモデル用のエンドポイント**
# MAGIC
# MAGIC [Model Serving Endpoints page](/ml/endpoints)を開いて、ファウンデーションモデルを試してください。
# MAGIC
# MAGIC このデモでは、ファウンデーションモデルのBGE（埋め込み）とllama2-70B（チャット）を使用します。 <br/><br/>
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/databricks-foundation-models.png?raw=true" width="600px" >

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

# COMMAND ----------

# DBTITLE 1,ドキュメントをdatabricks_documentationテーブルの作成
# MAGIC %sql
# MAGIC --Note that we need to enable Change Data Feed on the table to create the index
# MAGIC CREATE TABLE IF NOT EXISTS ${my.embbedTableName} (
# MAGIC   id BIGINT GENERATED BY DEFAULT AS IDENTITY,
# MAGIC   query STRING,
# MAGIC   response STRING,
# MAGIC   embedding ARRAY <FLOAT>
# MAGIC ) TBLPROPERTIES (delta.enableChangeDataFeed = true); 

# COMMAND ----------

# MAGIC %md
# MAGIC ### チャンクのエンベッディングを計算し、デルタテーブルに保存します。
# MAGIC
# MAGIC 最後のステップは、すべてのドキュメントチャンクの埋め込みを計算することです。foundationモデルのエンドポイントを使ってエンベッディングを計算するudfを作りましょう。
# MAGIC
# MAGIC *この部分は、通常、新しいドキュメントページが更新されるとすぐに実行されるプロダクショングレードのジョブとしてセットアップされることに注意してください。<br/> これは、Delta Live Tableパイプラインとしてセットアップすることで、インクリメンタルにデータ更新を処理することができます。*

# COMMAND ----------

import mlflow.deployments
deploy_client = mlflow.deployments.get_deploy_client("databricks")

@pandas_udf("array<float>")
def get_embedding(contents: pd.Series) -> pd.Series:
    def get_embeddings(batch):
        response = deploy_client.predict(
            endpoint=embedding_endpoint_name, 
            inputs={"inputs": batch})
        return [e for e in response.predictions]

    # エンベッディングモデルはリクエストごとに最大150の入力を取るので、コンテンツを150のアイテムごとに分割します。
    max_batch_size = 150
    batches = [contents.iloc[i:i + max_batch_size] for i in range(0, len(contents), max_batch_size)]

    # Process each batch and collect the results
    all_embeddings = []
    for batch in batches:
        all_embeddings += get_embeddings(batch.tolist())

    return pd.Series(all_embeddings)

# COMMAND ----------

(spark.table(raw_data_table_name)
      .withColumn('embedding', get_embedding('response'))
      .write.mode('overwrite').saveAsTable(embed_table_name))

display(spark.table(embed_table_name))

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from ${my.embbedTableName}
# MAGIC where response like '%%' order by id;

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### データセットの準備ができました！それでは、自己管理型Vector Search Indexを作成しましょう。
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-self-managed-prep-3.png?raw=true" style="float: right; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC データセットの準備ができました。ドキュメントページを小さなセクションに分割し、エンベッディングを計算し、Delta Lakeテーブルとして保存しました。
# MAGIC
# MAGIC 次に、このテーブルからデータを取り込むためにDatabricks Vector Searchを設定します。
# MAGIC
# MAGIC Vector search indexは、エンベッディングデータを提供するためにVector searchエンドポイントを使用します（Vector Search APIエンドポイントと考えることができます）。<br/>
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

# DBTITLE 1,エンドポイントを使って自己管理型ベクターサーチインデックスを作成します。
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c

#The table we'd like to index
source_table_fullname = f"{catalog}.{db}.{embed_table_name}"
# Where we want to store our index
vs_index_fullname = f"{catalog}.{db}.{embed_table_name}_vs_index"

if index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname):
  print(f"Deleting index {vs_index_fullname} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}...")
  vsc.delete_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)

# if not index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname):
print(f"Creating index {vs_index_fullname} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}...")
vsc.create_delta_sync_index(
  endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
  index_name=vs_index_fullname,
  source_table_name=source_table_fullname,
  pipeline_type="TRIGGERED",
  primary_key="id",
  embedding_dimension=1024, #Match your model embedding size (e5)
  embedding_vector_column="embedding"
)
# else:
  #同期をトリガーして、テーブルに保存された新しいデータでベクターサーチのコンテンツを更新します。
  # vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).sync()

#インデックスの準備ができ、すべてエンベッディングが作成され、インデックスが作成されるのを待ちましょう。
wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
print(f"index {vs_index_fullname} on table {source_table_fullname} is ready")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 類似コンテンツの検索
# MAGIC
# MAGIC これだけです。Databricksは自動的にDelta Live Tableの新しいエントリーを取り込み、同期します。
# MAGIC
# MAGIC データセットのサイズとモデルのサイズによっては、インデックスの作成に数秒かかることがあります。
# MAGIC
# MAGIC 試しに類似コンテンツを検索してみましょう。
# MAGIC
# MAGIC *Note:`similarity_search` は filters パラメータもサポートしています。これは、RAGシステムにセキュリティレイヤーを追加するのに便利です。誰がエンドポイントへのアクセスを行うかに基づいて、機密性の高いコンテンツをフィルタリングすることができます（例えば、ユーザー情報に基づいて特定の部署をフィルタリングするなど）。*

# COMMAND ----------

import mlflow.deployments
deploy_client = mlflow.deployments.get_deploy_client("databricks")

question = "電気代に関して最も費用対効果が高い製品は？"
response = deploy_client.predict(
  endpoint=embedding_endpoint_name, 
  inputs={"inputs": [question]}
)
embeddings = [e for e in response.predictions]

results = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).similarity_search(
  query_vector=embeddings[0],
  columns=["query", "response"],
  num_results=10)
docs = results.get('result', {}).get('data_array', [])
docs

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 次のステップ RAGを使ったチャットボットモデルのデプロイ
# MAGIC
# MAGIC Databricks Lakehouse AIを使用すると、数行のコードと設定だけで、ドキュメントの取り込みと準備、その上でのVector Searchインデックスのデプロイを簡単に行うことができます。
# MAGIC
# MAGIC これにより、データプロジェクトが簡素化、高速化され、次のステップである、プロンプトのオーグメンテーションによるリアルタイムチャットボットのエンドポイントの作成に集中できるようになります。
# MAGIC
# MAGIC [02-Deploy-RAG-Chatbot-Model]($./02-Deploy-RAG-Chatbot-Model) チャットボットエンドポイントを作成し、デプロイするためのノートブックを開いてください。
