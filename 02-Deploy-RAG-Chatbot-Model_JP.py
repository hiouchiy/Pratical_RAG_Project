# Databricks notebook source
# MAGIC %md 
# MAGIC ### 環境
# MAGIC - Runtime: 14.2 ML GPU
# MAGIC - Node type: g5.8xlarge (Single Node)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC # 2/ Creating the chatbot with Retrieval Augmented Generation (RAG)
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-self-managed-flow-2.png?raw=true" style="float: right; margin-left: 10px"  width="900px;">
# MAGIC
# MAGIC ベクターサーチインデックスの準備ができました！
# MAGIC
# MAGIC それでは、RAGを実行するための新しいModel Serving Endpointを作成し、デプロイしてみましょう。
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

# DBTITLE 1,必要な外部ライブラリのインストール 
# MAGIC %pip install mlflow==2.9.0 langchain==0.0.344 databricks-vectorsearch==0.22 databricks-sdk==0.12.0 mlflow[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,コンフィグ(環境に合わせて修正してください）
# MAGIC %run ./config

# COMMAND ----------

# DBTITLE 1,カタログ初期化及びデモ用のインポートとヘルパーのインストール
# MAGIC %run ./_resources/00-init $reset_all_data=false

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
# MAGIC `databricks secrets create-scope --scope dbdemos`
# MAGIC - サービスプリンシパルのシークレットを保存します。これはモデルエンドポイントが認証するために使われます。これがデモ/テストである場合、あなたの[PAT token](https://docs.databricks.com/en/dev-tools/auth/pat.html)を利用できます。<br>
# MAGIC `databricks secrets put --scope dbdemos --key rag_sp_token`
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

# DBTITLE 1,Vector Searchインデックスへのアクセス権を持っていることを確認してください。
index_name=f"{catalog}.{db}.{embed_table_name}_vs_index"
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

test_demo_permissions(
  host, 
  secret_scope=databricks_token_secrets_scope, 
  secret_key=databricks_token_secrets_key, 
  vs_endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME, 
  index_name=index_name, 
  embedding_endpoint_name=None
)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Langchain リトリーバー
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-self-managed-model-1.png?raw=true" style="float: right" width="500px">
# MAGIC
# MAGIC まずはLangchain・リトリーバーを作りましょう。
# MAGIC
# MAGIC Langchainリトリーバーは以下のことを行います：
# MAGIC
# MAGIC * 入力質問のエンベッディングを作成します。 (with Databricks `bge-large-en`)
# MAGIC * ベクトル検索インデックスを呼び出して類似文書を検索し、プロンプトを補強します。
# MAGIC
# MAGIC DatabricksのLangchainラッパーは、全ての基礎となるロジックとAPIコールを処理し、1ステップで簡単に実行できます。

# COMMAND ----------

# DBTITLE 1,モデルの認証設定
# url used to send the request to your model from the serverless endpoint
os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get(databricks_token_secrets_scope, databricks_token_secrets_key)

host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
os.environ['DATABRICKS_HOST'] = host

print(host)
print(os.environ['DATABRICKS_TOKEN'])

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### チャットモデルの構築 （llama-2-70b-chat ファウンデーションモデルへのクエリ）
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-self-managed-model-3.png?raw=true" style="float: right" width="500px">
# MAGIC
# MAGIC 私たちのチャットボットは llama2 ファウンデーションモデルを使って回答を提供します。
# MAGIC
# MAGIC ビルトイン[Foundation endpoint](/ml/endpoints) (using the `/serving-endpoints/databricks-llama-2-70b-chat/invocations` API)モデルがすぐに利用可能で, Databricks Langchain Chatモデルラッパーを利用することで簡単にチェーンを構築することができます。
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

##############################################
# chat_modelモデルの定義(カスタムモデルを使用)
##############################################
chat_model = ChatDatabricks(
  endpoint=instruct_endpoint_name, 
  max_tokens = 2000, 
  temprature = 0.1)

print(f"Test chat model: {chat_model.predict('リビングが３０平米なのですが、どの製品がベスト？')}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### RAGチェーンの構築
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-self-managed-model-2.png?raw=true" style="float: right" width="600px">
# MAGIC
# MAGIC それでは、エンベッディングを取得と類似文書を検索しプロンプトを補強するレトリーバーと、回答を作成するチャットモデルを1つのLangchainチェーンに統合しましょう。
# MAGIC
# MAGIC 私たちは、適切な回答をするために、アシスタントにカスタムlangchainテンプレートを使用します。
# MAGIC
# MAGIC 時間をかけて様々なテンプレートを試し、アシスタントの口調や性格をあなたの要求に合うように調整してください。
# MAGIC
# MAGIC

# COMMAND ----------

import mlflow
from mlflow.pyfunc import PythonModel

class ChatbotRAGOrchestratorApp(mlflow.pyfunc.PythonModel):

    def __init__(self):
        from databricks.vector_search.client import VectorSearchClient
        import mlflow.deployments
        from langchain.chat_models import ChatDatabricks
        import os

        #Get the vector search index
        vsc = VectorSearchClient(
          workspace_url=os.environ["DATABRICKS_HOST"], 
          personal_access_token=os.environ["DATABRICKS_TOKEN"])
        self.vs_index = vsc.get_index(
            endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
            index_name=index_name
        )

        # Get the client for embedding endpoint
        self.deploy_client = mlflow.deployments.get_deploy_client("databricks")

        # Get the client for instruct endpoint
        self.chat_model = ChatDatabricks(endpoint=instruct_endpoint_name, max_tokens = 2000)

    #Send a request to our Vector Search Index to retrieve similar content.
    def find_relevant_doc(self, question, num_results = 10, relevant_threshold = 0.7):

        response = self.deploy_client.predict(
          endpoint=embedding_endpoint_name, 
          inputs={"inputs": [question]})
        embeddings = [e for e in response.predictions]

        results = self.vs_index.similarity_search(
            query_vector=embeddings[0],
            columns=["query", "response"],
            num_results=num_results)
        
        docs = results.get('result', {}).get('data_array', [])
        #Filter on the relevancy score. Below 0.7 means we don't have good relevant content

        returned_docs = []
        for doc in docs:
          if doc[-1] > relevant_threshold:
            returned_docs.append({"query": doc[0], "response": doc[1]})
        # if len(docs) > 0 and docs[0][-1] > relevant_threshold :
        #   return {"query": docs[0][0], "response": docs[0][1]}
        return returned_docs

    def predict(self, context, model_input):
        answers = []
        for question in model_input:
          #Build the prompt
          # prompt = "[INST] <<SYS>>あなたはエアコンメーカーのコールセンターアシスタントです。あなたはエアコン製品に関する仕様、機能、トラブルシューティングなどの質問に回答します。"
          prompt = "[INST] <<SYS>>【参考情報】を元に、【ユーザーからの質問】にできるだけ正確に答えてください。なお、【参考情報】をそのまま出力するのではなく、ユーザーからの質問に沿う形へと適切に微調整してから出力してください。わからない場合や、質問が適切でない場合、有益な参考情報が無い場合は、そのように答えてください。<</SYS>>\n\n 【参考情報】: "
          
          docs = self.find_relevant_doc(question)

          ref_info = ""
          for doc in docs:
            ref_info = ref_info + doc['response'] + "\n\n"

          #Add docs from our knowledge base to the prompt
          if len(docs) > 0:
            prompt += f"\n\n{ref_info}"

          #Final instructions
          # prompt += f"\n\n <</SYS>>参考情報を参照しながら以下の【質問】に答えてください。なお、参考情報をそのまま出力するのではなく、ユーザーからの質問に沿う形に微調整してから出力してください。わからない場合や、質問が適切でない場合、有益な参考情報が無い場合は、そのように答えてください。詳細な回答のみしてください。メモやコメントは不要です。\n\n  【質問】: {question}[/INST]"
          prompt += f"\n\n  【ユーザーからの質問】: {question}[/INST]"

          response = self.chat_model.predict(prompt)

          answers.append({"answer": response, "prompt": prompt})
        return answers

# COMMAND ----------

proxy_model = ChatbotRAGOrchestratorApp()
results = proxy_model.predict(None, ["8畳間の和室に適した製品はどれ？その根拠も一緒に教えて。"])
print(results[0]["answer"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### モデルをUnityカタログレジストリに保存
# MAGIC
# MAGIC モデルの準備ができたので、Unity Catalogスキーマに登録します

# COMMAND ----------

from mlflow.models import infer_signature

mlflow.set_registry_uri("databricks-uc")
model_name = f"{catalog}.{db}.{registered_model_name}"

with mlflow.start_run(run_name="chatbot_rag_ja") as run:
    chatbot = ChatbotRAGOrchestratorApp()
    
    #Let's try our model calling our Gateway API: 
    signature = infer_signature(["some", "data"], results)

    mlflow.pyfunc.log_model(
      artifact_path="model", 
      python_model=chatbot, 
      signature=signature, 
      registered_model_name=model_name,
      pip_requirements=[
            "mlflow==2.9.0",
            "langchain==0.0.344",
            "databricks-vectorsearch==0.22",
            "databricks-sdk==0.12.0",
            "mlflow[databricks]"]
    ) 
print(run.info.run_id)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### チャットモデルをサーバーレスモデルのエンドポイントとしてデプロイする
# MAGIC
# MAGIC モデルはUnity Catalogに保存されます。最後のステップは、Model Servingとしてデプロイすることです。
# MAGIC
# MAGIC これでアシスタントのフロントエンドからリクエストを送信できるようになります。

# COMMAND ----------

# サービングエンドポイントの作成または更新
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput

serving_endpoint_name = f"japan_demo_rag_endpoint_{catalog}_{db}"[:63]
latest_model_version = get_latest_model_version(model_name)

w = WorkspaceClient()
endpoint_config = EndpointCoreConfigInput(
    name=serving_endpoint_name,
    served_models=[
        ServedModelInput(
            model_name=model_name,
            model_version=latest_model_version,
            workload_size="Small",
            scale_to_zero_enabled=True,
            environment_vars={
                "DATABRICKS_TOKEN": "{{secrets/"+databricks_token_secrets_scope+"/"+databricks_token_secrets_key+"}}",
                "DATABRICKS_HOST": "{{secrets/"+databricks_host_secrets_scope+"/"+databricks_host_secrets_key+"}}"
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
    "dataframe_split": {
      "columns": [      
        "8畳間の和室に適した製品はどれ？その根拠も一緒に教えて。"    
      ],    
      "data": [  ]  
    }
  }, 
  headers={
    'Authorization': f'Bearer {dbutils.secrets.get(scope=databricks_token_secrets_scope, key=databricks_token_secrets_key)}', 
    'Content-Type': 'application/json'
  }
).json()

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

#display_gradio_app("databricks-demos-chatbot")

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
