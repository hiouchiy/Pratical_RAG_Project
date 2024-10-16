# Databricks notebook source
# MAGIC %pip install -U -qqqq databricks-agents mlflow mlflow-skinny databricks-vectorsearch langchain==0.2.11 langchain_core==0.2.23 langchain_community==0.2.10 

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from operator import itemgetter
import mlflow
import os

from databricks.vector_search.client import VectorSearchClient

from langchain_community.chat_models import ChatDatabricks
from langchain_community.vectorstores import DatabricksVectorSearch

from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
)

## Enable MLflow Tracing
mlflow.langchain.autolog()


############
# Helper functions
############
# Return the string contents of the most recent message from the user
def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]


# Return the chat history, which is is everything before the last question
def extract_chat_history(chat_messages_array):
    return chat_messages_array[:-1]


# FIT AND FINISH: We should not require a value here.
model_config = mlflow.models.ModelConfig(development_config='rag_chain_config.yaml')

############
# Connect to the Vector Search Index
############
vs_client = VectorSearchClient(disable_notice=True)
vs_index = vs_client.get_index(
    endpoint_name=model_config.get("vector_search_endpoint_name"),
    index_name=model_config.get("vector_search_index_name")
)

############
# Turn the Vector Search index into a LangChain retriever
############
vector_search_as_retriever = DatabricksVectorSearch(
    vs_index,
    text_column="response",
    columns=[
        "id",
        "response",
        "url",
    ],
).as_retriever(search_kwargs={"k": 5, "query_type": "ann"})

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


############
# Method to format the docs returned by the retriever into the prompt
############
def format_context(docs):
    chunk_template = "Passage: {chunk_text}\n"
    chunk_contents = [
        chunk_template.format(
            chunk_text=d.page_content,
            document_uri=d.metadata["url"],
        )
        for d in docs
    ]
    return "".join(chunk_contents)

############
# Prompt Template for generation
############
prompt = ChatPromptTemplate.from_messages(
    [
        (  # System prompt contains the instructions
            "system",
            "【参考情報】のみを参考にしながら【質問】にできるだけ正確に答えてください。わからない場合や、質問が適切でない場合は、分からない旨を答えてください。【参考情報】に記載されていない事実を答えるのはやめてください。",
        ),
        # User's question
        ("user", """【参考情報】
{context}

【質問】
{question}"""),
    ]
)

############
# FM for generation
############
model = ChatDatabricks(
    endpoint=model_config.get("llm_endpoint_name"),
    extra_params={"temperature": 0.01, "max_tokens": 1500},
)

############
# RAG Chain
############
chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
        "context": itemgetter("messages")
        | RunnableLambda(extract_user_query_string)
        | vector_search_as_retriever
        | RunnableLambda(format_context),
    }
    | prompt
    | model
    | StrOutputParser()
)

## Tell MLflow logging where to find your chain.
# `mlflow.models.set_model(model=...)` function specifies the LangChain chain to use for evaluation and deployment.  This is required to log this chain to MLflow with `mlflow.langchain.log_model(...)`.

mlflow.models.set_model(model=chain)

# COMMAND ----------

input_example = {
  "messages": [{"role": "user", "content": "Zenith ZR-450のタッチスクリーン操作パネルの反応が鈍いです。どうしたら良いですか？"}]
}

# chain.invoke(input_example)

# COMMAND ----------


