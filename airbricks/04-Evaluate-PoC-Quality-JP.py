# Databricks notebook source
# MAGIC %pip install -U -qqqq databricks-agents mlflow mlflow-skinny databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import pandas as pd
from databricks import agents

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

# MAGIC %md
# MAGIC # 前のステップで読み込んだ評価セットを読み込みます。

# COMMAND ----------

df = spark.table(EVALUATION_SET_FQN)
eval_df = df.toPandas()
display(eval_df)

# COMMAND ----------

# ステークホルダーからフィードバックを収集しておらず、手動で作成した質問セットを使用してとりあえずプログラムを動かしたい場合は、以下のデータを使用できます。

eval_data  = [
    {
      "request_id": "999",
      "request": "この会社で最も電力効率の良い大型エアコンはどれですか？",
      "expected_retrieved_context": [
        {
            "doc_uri": "https://example.com/6396",
        },
        {
            "doc_uri": "https://example.com/9509",
        },
        {
            "doc_uri": "https://example.com/9447",
        }
      ],
      "expected_response": "この会社で最も電力効率の良い大型エアコンとしては、以下の3つのモデルが候補になります：\n\n	1.	EcoSmart TY-700（EER 6.2）: 電力効率が最も高い大型エアコンです。\n	2.	Mirage MX-800（EER 6.0）: 超大型空間に対応可能で、省エネルギー性能が高いです。\n	3.	Solaris SX-550（EER 5.9）: ソーラーパネルを活用するため、非常に効率的です。"
    }
]

# Uncomment this row to use the above data instead of your evaluation set
eval_df = pd.DataFrame(eval_data)

# COMMAND ----------

# MAGIC %md
# MAGIC # POCアプリケーションを評価

# COMMAND ----------

# MAGIC %md
# MAGIC ## POCアプリケーションのMLflow実行を取得する

# COMMAND ----------

user_account_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
runs = mlflow.search_runs(experiment_names=[f"/Users/{user_account_name}/airbricks_rag_experiment"], filter_string=f"run_name = 'airbricks_rag_chatbot'", output_format="list")

if len(runs) != 1:
    raise ValueError(f"Found {len(runs)} run with name airbricks_rag_chatbot.  Ensure the run name is accurate and try again.")

poc_run = runs[0]

# COMMAND ----------

# MAGIC %md
# MAGIC ## POCのアプリに適したPython環境をロードする
# MAGIC
# MAGIC TODO: replace this with env_manager=virtualenv once that works

# COMMAND ----------

pip_requirements = mlflow.pyfunc.get_model_dependencies(f"runs:/{poc_run.info.run_id}/chain")

# COMMAND ----------

# MAGIC %pip install -r $pip_requirements

# COMMAND ----------

# MAGIC %md
# MAGIC ## POCアプリで評価を実行する

# COMMAND ----------

with mlflow.start_run(run_id=poc_run.info.run_id):
    # Evaluate
    eval_results = mlflow.evaluate(
        data=eval_df,
        model=f"runs:/{poc_run.info.run_id}/chain",  # replace `chain` with artifact_path that you used when calling log_model.  By default, this is `chain`.
        model_type="databricks-agent",
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 評価結果を確認する
# MAGIC
# MAGIC MLflow UIへの上記のリンクを使用して評価結果を探索できます。データを直接使用したい場合は、以下のセルを参照してください。

# COMMAND ----------

# Summary metrics across the entire evaluation set
eval_results.metrics

# COMMAND ----------

# Evaluation results including LLM judge scores/rationales for each row in your evaluation set
per_question_results_df = eval_results.tables["eval_results"]

# You can click on a row in the `trace` column to view the detailed MLflow trace
display(per_question_results_df)

# COMMAND ----------


