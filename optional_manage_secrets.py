# Databricks notebook source
# MAGIC %md
# MAGIC サーバレス

# COMMAND ----------

import time
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

scope_name = 'YOUR_ID'
key_name = 'databricks_token'
token = 'YOUR_TOKEN'

w.secrets.create_scope(scope=scope_name)
w.secrets.put_secret(scope=scope_name, key=key_name, string_value=token)

# COMMAND ----------

w.secrets.list_scopes()

# COMMAND ----------

w.secrets.list_secrets(scope=scope_name)

# COMMAND ----------

# cleanup
w.secrets.delete_secret(scope=scope_name, key=key_name)
w.secrets.delete_scope(scope=scope_name)

# COMMAND ----------


