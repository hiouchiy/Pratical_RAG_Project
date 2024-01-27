# Databricks notebook source
# MAGIC %pip install -U langchain_community mlflow[genai]==2.9.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from langchain_community.chat_models import ChatDatabricks
from langchain_core.messages import HumanMessage
from mlflow.deployments import get_deploy_client
from requests.exceptions import HTTPError

def regster_or_replace_openai_model_as_external_model(endpoint_name: str, entity_name: str, model_type: str):

  client = get_deploy_client("databricks")

  try:
    client.get_endpoint(endpoint=endpoint_name)
    client.delete_endpoint(endpoint=endpoint_name)
  except HTTPError as e:
    print("There is no endpoint.")

  client.create_endpoint(
      name=endpoint_name,
      config={
          "served_entities": [
              {
                  "name": entity_name,
                  "external_model": {
                      "name": model_type,
                      "provider": "openai",
                      "task": "llm/v1/chat",
                      "openai_config": {
                          "openai_api_key": "{{secrets/hiouchiy/openai_api_key}}",
                      },
                  },
              }
          ],
      },
  )


# COMMAND ----------

regster_or_replace_openai_model_as_external_model(
  endpoint_name="japan-practical-demo-openai-gpt35", 
  entity_name="openai_gpt-35", 
  model_type="gpt-3.5-turbo"
)

regster_or_replace_openai_model_as_external_model(
  endpoint_name="japan-practical-demo-openai-gpt4", 
  entity_name="openai_gpt-4", 
  model_type="gpt-4"
)

# COMMAND ----------

chat = ChatDatabricks(endpoint="japan-practical-demo-openai-gpt35", temperature=0.1)
print(chat([HumanMessage(content="hello")]))

chat = ChatDatabricks(endpoint="japan-practical-demo-openai-gpt35", temperature=0.1)
print(chat([HumanMessage(content="hello")]))
