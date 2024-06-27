# Databricks notebook source
# MAGIC %md
# MAGIC ## AI エージェント (Chainアプリ)の実装
# MAGIC
# MAGIC それでは、フロントエンドアプリから質問を取得し、FAQデータ（ベクトル検索）とカード会員マスタ（特徴量サービング）から関連情報を検索し、プロンプトを拡張するレトリーバーと、回答を作成するチャットモデルを1つのチェーンに統合しましょう。
# MAGIC
# MAGIC 必要に応じて、様々なテンプレートを試し、AIアシスタントの口調や性格を皆様の要求に合うように調整してください。
# MAGIC
# MAGIC RAGエージェント・アプリにおける処理フローは以下のようになります：
# MAGIC
# MAGIC 1. ユーザーから質問とユーザーIDが RAGチェーン・アプリ のエンドポイントへ送信される
# MAGIC 1. RAGチェーン・アプリにて、ユーザーIDをキーとして、ユーザーマスタから当該ユーザーの属性情報を取得
# MAGIC 1. RAGチェーン・アプリにて、質問に関連した情報を抜き出すべく、ベクトルインデックスに対して類似検索を実施
# MAGIC 1. RAGチェーン・アプリにて、ユーザー属性情報、質問関連情報、質問を組み合わせてプロンプトを作成
# MAGIC 1. RAGチェーン・アプリにて、プロンプトを文章生成LLM（DBRX）のエンドポイントへ送信
# MAGIC 1. ユーザーにLLMの出力を返す

# COMMAND ----------

# MAGIC %pip install -U -qqqq databricks-agents==0.1.0 mlflow==2.14.0 langchain==0.2.1 langchain_core==0.2.5 langchain_community==0.2.4 databricks-vectorsearch==0.38 openai
# MAGIC dbutils.library.restartPython() 

# COMMAND ----------

import os
from typing import Dict
import time

import pandas as pd

import mlflow
from mlflow.pyfunc import PythonModel
import mlflow.deployments

from databricks.vector_search.client import VectorSearchClient

from langchain.chat_models import ChatDatabricks
from langchain_core.messages import SystemMessage
from langchain_core.prompts.chat import HumanMessagePromptTemplate, ChatPromptTemplate

from openai import OpenAI

# COMMAND ----------

model_config = mlflow.models.ModelConfig(development_config="rag_chain_config.yaml")

# COMMAND ----------

class ChatbotRAGOrchestratorApp(PythonModel):

    def __init__(self):
        """
        コンストラクタ
        """

        try:
            # サービングエンドポイントのホストに"DB_MODEL_SERVING_HOST_URL"が自動設定されるので、その内容をDATABRICKS_HOSTにも設定
            os.environ["DATABRICKS_HOST"] = os.environ["DB_MODEL_SERVING_HOST_URL"]
        except:
            pass

        vsc = VectorSearchClient(disable_notice=True)
        self.vs_index = vsc.get_index(
            endpoint_name=model_config.get("vector_search_endpoint_name"),
            index_name=model_config.get("vector_search_index_name")
        )

        # 特徴量サービングアクセス用クライアントの取得
        self.deploy_client = mlflow.deployments.get_deploy_client("databricks")

        # LLM基盤モデルのエンドポイントのクライアントを取得
        self.chat_model = ChatDatabricks(
            endpoint=model_config.get("llm_endpoint_name"), 
            max_tokens = 2000)
        # self.openai_client = OpenAI(
        #     api_key=os.environ.get("DATABRICKS_TOKEN"),
        #     base_url=os.environ.get("DATABRICKS_HOST") + "/serving-endpoints",
        # )

        # システムプロンプトを準備
        self.SYSTEM_MESSAGE = SystemMessage(content="【ユーザー情報】と【参考情報】のみを参考にしながら【質問】にできるだけ正確に答えてください。わからない場合や、質問が適切でない場合は、分からない旨を答えてください。【参考情報】に記載されていない事実を答えるのはやめてください。")

        # ヒューマンプロンプトテンプレートを準備
        human_template = """【ユーザー情報】
名前：{name}
ランク：{rank}
生年月日：{birthday}
入会日：{since}


【参考情報】
{context}

【質問】
{question}"""
        self.HUMAN_MESSAGE = HumanMessagePromptTemplate.from_template(human_template)
        
        # chatプロンプトテンプレートの準備
        self.CHAT_PROMPT = ChatPromptTemplate.from_messages([self.SYSTEM_MESSAGE, self.HUMAN_MESSAGE])
        
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
          endpoint=model_config.get("user_info_endpoint_name"),
          inputs={"dataframe_records": [{"id": user_id}]},
        )
        name = result['outputs'][0]['name']
        rank = result['outputs'][0]['type']
        birthday = result['outputs'][0]['birthday']
        since = result['outputs'][0]['since']

        return name, rank, birthday, since
    
    def _build_prompt(self, name, rank, birthday, since, docs, question):
        """
        プロンプトの構築
        """

        context = ""
        for doc in docs:
          context = context + doc['response'] + "\n\n"

        prompt = self.CHAT_PROMPT.format_prompt(
          name=name, 
          rank=rank, 
          birthday=birthday, 
          since=since, 
          context=context, 
          question=question
        ).to_messages()

        return prompt

    @mlflow.trace(name="predict_rag")
    def predict(self, context, model_input, params=None):
        """
        推論メイン関数
        """

        if isinstance(model_input, pd.DataFrame):
            model_input = model_input.to_dict(orient="records")[0]
        
        # カード会員マスタから当該ユーザーの属性情報を取得
        with mlflow.start_span(name="_get_user_info") as span:
            userId = params["id"][0] if params else "111"
            name, rank, birthday, since = self._get_user_info(userId)
            span.set_inputs({"userId": userId})
            span.set_outputs({"name": name, "rank": rank, "birthday": birthday, "since": since})

        # FAQデータからベクター検索を用いて質問と類似している情報を検索
        with mlflow.start_span(name="_find_relevant_doc") as span:
            question = model_input["messages"][-1]["content"]
            docs = self._find_relevant_doc(question, rank)
            span.set_inputs({"question": question, "rank": rank})
            span.set_outputs({"docs": docs})

        # プロンプトの構築
        with mlflow.start_span(name="_build_prompt") as span:
            prompt = self._build_prompt(name, rank, birthday, since, docs, question)
            span.set_inputs({"question": question, "docs": docs, "name": name, "rank": rank, "birthday": birthday, "since": since})
            span.set_outputs({"prompt": prompt})

        # LLMに回答を生成させる
        with mlflow.start_span(name="generate_answer") as span:
            response = self.chat_model.invoke(prompt)
            span.set_inputs({"question": question, "prompt": prompt})
            span.set_outputs({"answer": response})
        
        
        # 回答データを整形して返す
        # return asdict(
        #     ChatCompletionResponse(
        #         choices=[ChainCompletionChoice(message=response.content)]
        #     )
        # )
        # ChatCompletionResponseの形式で返さないと後々エラーとなる。
        # TODO:もっとスマートな方法がないかは要確認
        return {
            "id": response.id,
            "created": int(time.time()),
            "choices": [
                {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response.content
                },
                "finish_reason": None, 
                }
            ],
            "usage": {
                "prompt_tokens": response.response_metadata["prompt_tokens"],
                "completion_tokens": response.response_metadata["completion_tokens"],
                "total_tokens": response.response_metadata["total_tokens"],
            }
        }

mlflow.models.set_model(model=ChatbotRAGOrchestratorApp())

# COMMAND ----------

# MAGIC %md
# MAGIC ## AI Agentをテスト

# COMMAND ----------

input_example = {'messages': [
  {'role': 'user', 'content': '私のランクの特典を全て教えてください。'}, 
  {'role': 'assistant', 'content': '吉田\u3000太郎さんの現在のランクはゴールドです。ゴールドランクの特典は以下の通りです。\n\n・ポイント還元: 全ての購入に対し2.0%のポイント還元。\n・誕生日ボーナス: 誕生日月に利用金額の5%をポイントで追加付与。\n・会員限定セールへの招待: 年に2回、会員限定セールへの優先招待。\n・旅行保険: 海外旅行保険最高5,000万円、国内旅行傷害保険最高3,000万円の補償。\n・ショッピング保険: 年間最高200万円までの商品購入保険。\n・空港ラウンジ利用: 世界100カ国以上、1,000か所以上の空港ラウンジを年間無制限で無料利用。\n・コンシェルジュサービス: 24時間対応の個人コンシェルジュが、各種予約手配をサポート。\n・高額購入時の分割手数料免除: 年間10万円以上の購入に対し、分割払い手数料を全額免除。\n・ホテル・レストラン優待: 提携ホテルやレストランで最大20%の割引。\n\nまた、ゴールドランクになるための条件は、過去6カ月間で3,000ポイント以上、かつ25回以上ポイントを獲得することです。'}, 
  {'role': 'user', 'content': '私の現在のランクには空港ラウンジ特典はついていますか？'}
]}

# rag_model = ChatbotRAGOrchestratorApp()
# rag_model.predict(None, model_input=input_example, params={"id": ["222"]})

# COMMAND ----------

input_example = {"messages": [{"role": "user", "content": "現在のランクから一つ上のランクに行くためにはどういった条件が必要ですか？"}]}

# rag_model = ChatbotRAGOrchestratorApp()
# rag_model.predict(None, model_input=input_example)

# COMMAND ----------


