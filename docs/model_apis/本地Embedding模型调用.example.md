# Ollama Embedding 本地配置模板

此文件只作配置结构示例；没有连接或验证任何部署。复制到忽略的本地指南后，
按实际服务填写地址和维度，并重新执行小样本验证。

| 项目 | 示例配置 |
|---|---|
| 实测向量维度 | `4096` |

上表是所选模型的预期示例值，不是你部署环境的实测结论。

~~~python
EMBEDDING_URL = "http://127.0.0.1:11434/api/embed"
EMBEDDING_MODEL = "qwen3-embedding:8b"
QUERY_INSTRUCTION = "Given a technical question, retrieve relevant passages that answer the question"

request_options = {"truncate": False, "keep_alive": "10m"}
~~~

文档输入使用正文；问题使用 `Instruct: {instruction}\nQuery: {query}`。
该代码块供配置解析，不是完整的模型调用程序。部署、模型下载与访问控制不由此模板自动完成。
