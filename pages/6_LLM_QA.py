# pages/6_LLM_QA.py

import streamlit as st
from utils.user_auth import is_authenticated, get_current_username
from utils.llm_api import call_qwen_api # 导入我们封装好的 Qwen API 函数

# --- 页面配置 ---
st.set_page_config(
    page_title="大模型问答",
    page_icon="🤖",
    layout="wide"
)

# --- 权限检查 ---
if not is_authenticated():
    st.error("⚠️ 请先登录以访问此功能。")
    st.stop() # 如果未登录，停止执行后续代码

# --- 页面标题 ---
st.title("🤖 大模型问答助手")

# --- 初始化 session_state ---
# 用于存储对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []
# 用于存储当前实验上下文 (可选，但符合你的设计思路)
if "current_experiment_context" not in st.session_state:
    st.session_state.current_experiment_context = ""

# --- 实验上下文选择 (可选功能) ---
# 这个功能允许用户选择当前讨论的是哪个实验
# 你可以根据实际需要扩展这个功能，例如动态获取当前页面信息
st.sidebar.header("实验上下文设置")
experiments = ["无特定实验", "HW01: 梯度下降 & 词云", "HW02: 表征学习", "HW03: LSTM 文本生成", "其他 NLP 实验"]
selected_experiment = st.sidebar.selectbox("选择当前讨论的实验 (可选)", experiments)

# 根据选择设置上下文前缀
context_prefix = ""
if selected_experiment != "无特定实验":
    context_prefix = f"你正在参与关于'{selected_experiment}'的讨论。用户的后续问题将围绕此实验展开。请基于此背景回答。"

# --- 显示历史对话 ---
# 遍历历史消息并按角色显示
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- 用户输入区域 ---
# 用户输入框，按下回车或点击发送后触发
if prompt := st.chat_input("向大模型提问..."):
    # 1. 将用户输入添加到历史记录
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)

    # 2. 在界面上显示用户输入
    with st.chat_message("user"):
        st.write(prompt)

    # 3. 准备发送给 LLM 的消息列表
    # 包含上下文前缀（如果设置了）
    messages_to_send = []
    if context_prefix:
        messages_to_send.append({"role": "system", "content": context_prefix})
    # 添加之前的所有对话历史
    messages_to_send.extend(st.session_state.messages)

    # 4. 调用封装好的 Qwen API
    with st.spinner("大模型正在思考..."):
        success, response = call_qwen_api(messages_to_send) # 传入包含上下文和历史的完整消息列表

    # 5. 处理 API 响应
    if success:
        # 5a. 将 LLM 的回复添加到历史记录
        assistant_message = {"role": "assistant", "content": response}
        st.session_state.messages.append(assistant_message)

        # 5b. 在界面上显示 LLM 回复
        with st.chat_message("assistant"):
            st.write(response)
    else:
        # 5c. 如果 API 调用失败，显示错误信息
        error_message = {"role": "assistant", "content": f"❌ API 调用失败: {response}"}
        st.session_state.messages.append(error_message)
        with st.chat_message("assistant"):
            st.error(response) # 显示具体的错误信息

# --- 可选：添加一个清除对话历史的按钮 ---
st.sidebar.divider()
if st.sidebar.button("🗑️ 清除对话历史"):
    st.session_state.messages = [] # 清空历史记录
    st.rerun() # 刷新页面以反映变化

# --- 显示当前用户 (可选) ---
st.sidebar.success(f"已登录: {get_current_username()}")