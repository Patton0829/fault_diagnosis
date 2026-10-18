"""第一步：读取模型配置，验证千问或 GLM 的流式聊天接口。"""

import argparse
import os
import sys
from time import perf_counter

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    OpenAIError,
)

from qa_platform.settings import ENV_PATH, PROJECT_ROOT as PROJECT_ROOT


# 模型名是字典的键；每个模型有自己的地址、认证方式和附加参数。
# 配置来自 models/ 下的调用示例，千问密钥仅用于千问服务。
MODEL_CONFIGS = {
    "qwen3.8-flash": {
        "base_url": "https://maas.qianwenaiapi.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "extra_body": {"enable_thinking": True},
    },
    "qwen3.8-max": {
        "base_url": "https://maas.qianwenaiapi.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "extra_body": {"enable_thinking": True},
    },
    "glm": {
        "base_url": "http://115.190.191.103:8081/v1",
        "api_key_env": None,
        "extra_body": None,
    },
}
DEFAULT_MODEL = "qwen3.8-flash"
QUESTION = "讲解下机器学习中梯度下降的原理。"


def main() -> int:
    """按顺序检查配置、发送请求、读取回复，并返回退出状态码。"""
    parser = argparse.ArgumentParser(description="验证千问或 GLM 的流式聊天调用")
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS),
        default=DEFAULT_MODEL,
        help="选择聊天模型（默认：qwen3.8-flash）",
    )
    parser.add_argument(
        "--prompt",
        default=QUESTION,
        help="本次提问的内容，省略时使用代码中的 QUESTION",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="只检查本地配置，不向模型发送请求",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="输出 Token 上限，不是中文字数（默认：4096）",
    )
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens 必须是大于 0 的整数")
    if not args.prompt.strip():
        parser.error("--prompt 不能为空或只有空白字符")

    model_config = MODEL_CONFIGS[args.model]
    api_key_env = model_config["api_key_env"]
    if api_key_env:
        if not ENV_PATH.is_file():
            print(f"[错误] 配置文件不存在：{ENV_PATH}", file=sys.stderr)
            return 1

        # .env 的内容被载入当前进程；已存在的系统环境变量优先。
        load_dotenv(dotenv_path=ENV_PATH, override=False, encoding="utf-8")
        api_key = os.getenv(api_key_env, "").strip()
        if not api_key:
            print(f"[错误] 请配置 {api_key_env}。", file=sys.stderr)
            return 1
    else:
        # GLM 文档无需真实密钥；SDK 要求提供一个值，因此使用占位字符串。
        api_key = "not-needed"

    print(f"[配置] Python：{sys.executable}")
    print(f"[配置] 模型：{args.model}")
    print(f"[配置] 服务地址：{model_config['base_url']}")
    print(f"[配置] 输出 Token 上限：{args.max_tokens}")
    if api_key_env:
        print(f"[配置] 文件：{ENV_PATH}")
        print(f"[配置] {api_key_env} 已配置（不显示密钥）", flush=True)
    else:
        print("[配置] 按文档无需服务密钥，SDK 使用占位值。", flush=True)
    if args.check_config:
        print("[成功] 本地配置检查通过，未发送模型请求。")
        return 0

    print(f"\n[提问] {args.prompt}")
    print("[回复] ", end="", flush=True)
    started = perf_counter()
    has_content = False
    finish_reason = None

    try:
        # with 会在结束时关闭连接；关闭自动重试，便于观察单次调用。
        with OpenAI(
            api_key=api_key,
            base_url=model_config["base_url"],
            timeout=60.0,
            max_retries=0,
        ) as client:
            with client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": args.prompt}],
                stream=True,
                max_tokens=args.max_tokens,
                extra_body=model_config["extra_body"],
            ) as stream:
                for chunk in stream:
                    # 部分片段只携带统计信息，没有 choices。
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
                    # delta.content 是这次新到达的正文，不是完整回答。
                    content = choice.delta.content
                    if content:
                        print(content, end="", flush=True)
                        has_content = True
    except AuthenticationError:
        print(
            f"\n[错误] {args.model} 服务认证失败，请检查其认证配置。", file=sys.stderr
        )
        return 1
    except APITimeoutError:
        print("\n[错误] 请求超时，请检查服务状态或网络。", file=sys.stderr)
        return 1
    except APIConnectionError:
        print("\n[错误] 连接失败，请检查服务地址和网络。", file=sys.stderr)
        return 1
    except APIStatusError as exc:
        # 只显示 HTTP 状态码，避免把服务返回的敏感内容写进日志。
        print(f"\n[错误] 服务返回 HTTP {exc.status_code}。", file=sys.stderr)
        return 1
    except OpenAIError as exc:
        print(f"\n[错误] SDK 调用失败：{type(exc).__name__}。", file=sys.stderr)
        return 1

    print()
    print(f"[结束] finish_reason={finish_reason or '未返回'}", flush=True)
    # length 表示已收到服务响应，但输出触及上限；不能当作完整答案。
    # 先检查它，也能识别额度耗尽时没有产生正文的情况。
    if finish_reason == "length":
        print(
            f"[截断] 请求已收到响应，但回答达到了 max_tokens={args.max_tokens} 的输出上限。\n"
            "[建议] 提高 --max-tokens 的值，或在问题中要求简短回答。",
            file=sys.stderr,
        )
        return 2
    if not has_content:
        print("[错误] 未收到回复正文，请检查服务响应或输出额度。", file=sys.stderr)
        return 1
    if finish_reason != "stop":
        print("[错误] 服务未正常结束，具体原因见上方 finish_reason。", file=sys.stderr)
        return 1

    print(f"[成功] 流式调用完成，耗时 {perf_counter() - started:.2f} 秒。")
    return 0


# 直接运行这个文件时执行 main；被其他文件导入时不会发起请求。
if __name__ == "__main__":
    raise SystemExit(main())
