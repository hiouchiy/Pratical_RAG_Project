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

## 事前準備
###  1. シークレットの登録
Model Serving Endpointから Vector Search インデックスなど他のエンドポイントへ接続するために、Databricksのアクセストークンを必要とします。 ([参照先](https://docs.databricks.com/en/security/secrets/secrets.html)).  <br/>
**Note: 共有のデモ・ワークスペースを使用していて、シークレットが設定されていることを確認した場合は、以下の手順を実行せず、その値を上書きしないでください。**<br/>

- ラップトップで[Databricks CLI](https://docs.databricks.com/en/dev-tools/cli/install.html) をセットアップ<br/>
`pip install databricks-cli` <br/>
- CLIを設定。ワークスペースのURLとプロフィールページのPersonal Access Token(PAT)トークンが必要。PATトークンの発行の仕方は[こちら](https://docs.databricks.com/ja/dev-tools/auth/pat.html#databricks-personal-access-tokens-for-workspace-users)をご参照ください。<br>
`databricks configure`
- medallioncard および airbricks スコープを作成<br/>
`databricks secrets create-scope medallioncard`<br/>
`databricks secrets create-scope airbricks`<br/>
- サービスプリンシパルのシークレットを保存。これはモデルエンドポイントが認証するために使われます。これがデモ/テストである場合、あなたの[PAT token](https://docs.databricks.com/en/dev-tools/auth/pat.html)を利用できます。<br>
`databricks secrets put-secret medallioncard databricks_token`<br/>
`databricks secrets put-secret airbricks databricks_token`<br/>
