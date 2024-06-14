# 実践的RAGサンプル
このレポジトリは、Databricks上での生成AI活用のサンプルとして、実践的なビジネスシナリオに則ったデモプログラムを提供しています。

## 環境
Databricks（AWS/Azure）

## サンプル
- ドリー食品ホールディングス株式会社（[コード](./dollyfoodsholdings)）
  - 経理部が決算レポートを作成するために **Databricks Genie** を使ってデータ分析をします。
- メダリオンカード株式会社（[コード](./medallioncard)）
  - カード会員向けのFAQボットをRAGベースで開発します。
- 株式会社エアブリックス（[コード](./airbricks)）
  - コールセンターのオペレータ向けの情報検索ツールをRAGベースで開発します。
- その他
  - [optional-create_endpoint_for_elyza_13b](./optional-create_endpoint_for_elyza_13b)
    - 日本語LLM「ELYZA-13b」をDatabricks上にエンドポイントとしてデプロイするサンプルコード
  - [optional-create_endopint_for_openai_gpt3.5_and_gpt4](./optional-create_endopint_for_openai_gpt3.5_and_gpt4)
    - OpenAI GPT-3.5とGPT-4へアクセスするエンドポイントをDatabricks上に作成するサンプルコード
