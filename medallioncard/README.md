# メダリオンカード株式会社　RAGを用いたカード会員向けのFAQボット

## 事前準備
###  シークレットの登録
Model Serving Endpointから Vector Search インデックスなど他のエンドポイントへ接続するために、Databricksのアクセストークンを必要とします。 ([参照先](https://docs.databricks.com/en/security/secrets/secrets.html)).  <br/>
**Note: 共有のデモ・ワークスペースを使用していて、シークレットが設定されていることを確認した場合は、以下の手順を実行せず、その値を上書きしないでください。**<br/>

- ラップトップで[Databricks CLI](https://docs.databricks.com/en/dev-tools/cli/install.html) をセットアップ<br/>
`pip install databricks-cli` <br/>
- CLIを設定。ワークスペースのURLとプロフィールページのPersonal Access Token(PAT)トークンが必要。PATトークンの発行の仕方は[こちら](https://docs.databricks.com/ja/dev-tools/auth/pat.html#databricks-personal-access-tokens-for-workspace-users)をご参照ください。<br>
`databricks configure`
- medallioncard および airbricks スコープを作成<br/>
`databricks secrets create-scope medallioncard`<br/>
- サービスプリンシパルのシークレットを保存。これはモデルエンドポイントが認証するために使われます。これがデモ/テストである場合、あなたの[PAT token](https://docs.databricks.com/en/dev-tools/auth/pat.html)を利用できます。<br>
`databricks secrets put-secret medallioncard databricks_token`<br/>
