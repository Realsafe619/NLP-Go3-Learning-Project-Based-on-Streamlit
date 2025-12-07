# test_qwen.py (临时测试文件)

import streamlit as st
from utils.llm_api import call_qwen_api # 导入封装好的函数

st.set_page_config(page_title="Qwen API Test", layout="wide")
st.title("🤖 通义千问 (Qwen) API 接入测试")

# 初始化 session_state 用于存储对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 用户输入
if prompt := st.chat_input("向 Qwen 提问..."):
    # 将用户输入添加到历史记录
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 调用 Qwen API
    with st.spinner("Qwen 正在思考..."):
        success, response = call_qwen_api(st.session_state.messages) # 传入整个对话历史

    if success:
        # 将 Qwen 的回复添加到历史记录
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
    else:
        # 如果 API 调用失败，显示错误信息
        st.error(response) # response 变量此时包含错误信息
        # 可选：将错误信息也加入历史，方便调试
        # st.session_state.messages.append({"role": "assistant", "content": response})