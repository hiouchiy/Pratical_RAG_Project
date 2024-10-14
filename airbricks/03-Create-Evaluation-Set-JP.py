# Databricks notebook source
# MAGIC %md # アプリログのレビューから評価セットを作成する
# MAGIC
# MAGIC このステップでは、ステークホルダーがReview Appを使用して提供したフィードバックを元に、評価セットをブートストラップします。評価セットには*質問だけ*を含めることもできるので、ステークホルダーがアプリとチャットしただけでフィードバックを提供していなくても、このステップを実行できます。
# MAGIC
# MAGIC Agent Evaluationの評価セットスキーマを理解するために、[ドキュメント](https://docs.databricks.com/generative-ai/agent-evaluation/evaluation-set.html#evaluation-set-schema)を参照してください。以下で参照されているフィールドが説明されています。
# MAGIC
# MAGIC このステップの終わりには、次の内容を含む評価セットが作成されます。
# MAGIC
# MAGIC 1. 👍 が付いたリクエスト:
# MAGIC    - `request`: ユーザーが入力した内容
# MAGIC    - `expected_response`: ユーザーが応答を編集した場合はその内容、そうでなければモデルが生成した応答。
# MAGIC 2. 👎 が付いたリクエスト:
# MAGIC    - `request`: ユーザーが入力した内容
# MAGIC    - `expected_response`: ユーザーが応答を編集した場合はその内容、そうでなければnull。
# MAGIC 3. フィードバックがないリクエスト（例: 👍 も 👎 もなし）
# MAGIC    - `request`: ユーザーが入力した内容
# MAGIC
# MAGIC 上記のすべてにおいて、ユーザーが `retrieved_context` からチャンクに 👍 を付けた場合、そのチャンクの `doc_uri` が質問の `expected_retrieved_context` に含まれます。

# COMMAND ----------

# MAGIC %pip install -U -qqqq databricks-agents mlflow mlflow-skinny databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

# MAGIC %run ./z_eval_set_utilities

# COMMAND ----------

import pandas as pd

import mlflow

# COMMAND ----------

# MAGIC %md ## リクエストと評価ログテーブルを取得する
# MAGIC
# MAGIC これらのテーブルは、生の推論テーブルからのデータで約1時間ごとに更新されます。
# MAGIC
# MAGIC TODO: Add docs link to the schemas

# COMMAND ----------

w = WorkspaceClient()

model_name = f"{catalog}.{dbName}.{registered_model_name}"

active_deployments = agents.list_deployments()
active_deployment = next(
    (item for item in active_deployments if item.model_name == model_name), None
)

endpoint = w.serving_endpoints.get(active_deployment.endpoint_name)

try:
    endpoint_config = endpoint.config.auto_capture_config
except AttributeError as e:
    endpoint_config = endpoint.pending_config.auto_capture_config

inference_table_name = endpoint_config.state.payload_table.name
inference_table_catalog = endpoint_config.catalog_name
inference_table_schema = endpoint_config.schema_name

# Cleanly formatted tables
assessment_log_table_name = f"{inference_table_catalog}.{inference_table_schema}.`{inference_table_name}_assessment_logs`"
request_log_table_name = f"{inference_table_catalog}.{inference_table_schema}.`{inference_table_name}_request_logs`"

print(f"Assessment logs: {assessment_log_table_name}")
print(f"Request logs: {request_log_table_name}")


assessment_log_df = _dedup_assessment_log(spark.table(assessment_log_table_name))
request_log_df = spark.table(request_log_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## リクエストと評価のログをETLして、評価セットのスキーマにログインする
# MAGIC
# MAGIC 注：リクエストと評価のログの列一式は、このテーブルに残します。これらの列は、問題のデバッグに使用できます。

# COMMAND ----------

requests_with_feedback_df = create_potential_evaluation_set(request_log_df, assessment_log_df)

requests_with_feedback_df.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow Tracing を使用して潜在的な評価セットを検査する
# MAGIC
# MAGIC 表示されたテーブルの `trace` 列をクリックして、トレースを表示します。これらのレコードを検査する必要があります。

# COMMAND ----------

display(requests_with_feedback_df.select(
    F.col("request_id"),
    F.col("request"),
    F.col("response"),
    F.col("trace"),
    F.col("expected_response"),
    F.col("expected_retrieved_context"),
    F.col("is_correct"),
))

# COMMAND ----------

# MAGIC %md
# MAGIC # 結果の評価セットをデルタテーブルに保存

# COMMAND ----------

eval_set = requests_with_feedback_df[["request", "request_id", "expected_response", "expected_retrieved_context", "source_user", "source_tag"]]

eval_set.write.format("delta").saveAsTable(EVALUATION_SET_FQN)
