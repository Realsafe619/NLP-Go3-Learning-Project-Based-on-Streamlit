# utils/llm_api.py

import streamlit as st
from dashscope import Generation
import dashscope

def call_qwen_api(messages, model_name='qwen-max', stream=False):
    """
    调用通义千问 API

    Args:
        messages (list): 对话历史，格式为 [{"role": "user", "content": "..."}, ...]
        model_name (str): 要使用的模型名称，默认 'qwen-max'
        stream (bool): 是否启用流式输出

    Returns:
        tuple: (success: bool, response_content_or_error_message: str)
    """
    try:
        # 从 Streamlit secrets 中获取 API Key
        api_key = st.secrets["secrets"]["DASHSCOPE_API_KEY"]
        dashscope.api_key = api_key

        if stream:
            # 流式调用 (适合逐字输出效果)
            responses = Generation.call(
                model=model_name,
                messages=messages,
                result_format='message',  # 以 message 形式返回
                stream=True,              # 设置 stream=True
                incremental_output=True   # 增量输出
            )
            full_content = ""
            for response in responses:
                if response.status_code == 200:
                    chunk_content = response.output.choices[0]['message']['content']
                    if chunk_content: # 避免空内容
                        full_content += chunk_content
                else:
                    return False, f"API Error: {response.code} - {response.message}"
            return True, full_content

        else:
            # 非流式调用 (一次性获取完整回复)
            response = Generation.call(
                model=model_name,
                messages=messages,
                result_format='message', # 以 message 形式返回
            )

            if response.status_code == 200:
                content = response.output.choices[0]['message']['content']
                return True, content
            else:
                return False, f"API Error: {response.code} - {response.message}"

    except KeyError:
        # 如果 secrets.toml 中没有找到 DASHSCOPE_API_KEY
        return False, "错误：API Key 未在 secrets.toml 中配置。"
    except Exception as e:
        # 捕获其他可能的异常
        return False, f"调用 API 时发生未知错误: {e}"

# --- 可选：定义一个更通用的 LLM 接口类 ---
# class LLMInterface:
#     @staticmethod
#     # ... 可以在这里添加其他模型的调用方法 ...
#     pass