# Databricks notebook source
# MAGIC %pip install -U -qqqq databricks-agents mlflow mlflow-skinny databricks-vectorsearch langchain==0.2.11 langchain_core==0.2.23 langchain_community==0.2.10 openai

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os

API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
os.environ["DATABRICKS_HOST"] = API_ROOT
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DATABRICKS_TOKEN"] = API_TOKEN

# COMMAND ----------

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
            columns=["id", "query", "response", "url"],
            num_results=num_results)
        
        docs = results.get('result', {}).get('data_array', [])

        #関連性スコアでフィルタリングします。0.7以下は、関連性の高いコンテンツがないことを意味する
        returned_docs = []
        for doc in docs:
          if doc[-1] > relevant_threshold:
            returned_docs.append(
                # {"id": doc[0], "query": doc[1], "response": doc[2], "url": doc[3]}
                {
                    "page_content": doc[2],
                    "metadata": {
                        "url": doc[3],
                        "id": doc[0]
                    },
                    "id": doc[0]
                }
            )

        return returned_docs
    

    def _build_prompt(self, docs, question):
        """
        プロンプトの構築
        """

        context = ""
        for doc in docs:
          context = context + doc['page_content'] + "\n\n"

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
        with mlflow.start_span(name="_find_relevant_doc", span_type="RETRIEVER") as span:
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
        with mlflow.start_span(name="generate_answer", span_type="LLM") as span:
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

############
# Required to:
# 1. Enable the RAG Studio Review App to properly display retrieved chunks
# 2. Enable evaluation suite to measure the retriever
############

mlflow.models.set_retriever_schema(
    primary_key="id",
    text_column="response",
    doc_uri="url",  # Review App uses `doc_uri` to display chunks from the same document in a single view
)

mlflow.models.set_model(model=AirbricksRAGAgentApp())

# COMMAND ----------

input_example = {
  "messages": [{"role": "user", "content": "Zenith ZR-450のタッチスクリーン操作パネルの反応が鈍いです。どうしたら良いですか？"}]
}

rag_model = AirbricksRAGAgentApp()
# rag_model.predict(None, model_input=input_example)

# COMMAND ----------


