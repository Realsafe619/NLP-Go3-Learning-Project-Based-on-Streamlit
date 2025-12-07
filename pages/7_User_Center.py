# pages/7_User_Center.py

import streamlit as st
from utils.user_auth import is_authenticated, get_current_username, logout
from utils.data_manager import list_user_datasets, list_user_models, list_user_results

# --- 页面配置 ---
st.set_page_config(
    page_title="用户中心",
    page_icon="👤",
    layout="wide"
)

# --- 权限检查 ---
if not is_authenticated():
    st.error("⚠️ 请先登录以访问用户中心。")
    st.stop()

# --- 获取当前用户名 ---
current_user = get_current_username()

# --- 侧边栏 ---
st.sidebar.header(f"欢迎, {current_user}!")
if st.sidebar.button("登出", type="secondary"):
    logout()
    st.rerun()

# --- 主页面标题 ---
st.title("👤 用户中心")

# --- 用户概览 ---
st.header("账户概览")
st.write(f"**当前用户**: {current_user}")
# 可以在这里添加更多用户信息，例如注册时间等 (如果在 config.json 中有存储)

# --- 用户数据管理 ---
st.header("数据管理")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📊 我的数据集")
    datasets = list_user_datasets(current_user)
    if datasets:
        for ds in datasets:
            st.write(f"- {ds}")
    else:
        st.info("暂无上传的数据集。")

with col2:
    st.subheader("🧠 我的模型")
    models = list_user_models(current_user, model_type='keras') # 这里假设模型扩展名是 keras，根据实际情况调整
    models_w2v = list_user_models(current_user, model_type='word2vec') # 例如 Word2Vec
    all_models = models + models_w2v
    if all_models:
        for m in all_models:
            st.write(f"- {m}")
    else:
        st.info("暂无训练的模型。")

with col3:
    st.subheader("📋 我的结果")
    results = list_user_results(current_user, format='json') # 假设结果是 json 格式
    results_txt = list_user_results(current_user, format='txt') # 例如生成的文本
    all_results = results + results_txt
    if all_results:
        for r in all_results:
            st.write(f"- {r}")
    else:
        st.info("暂无生成的结果。")

# --- 可选：用户设置或偏好 ---
st.header("账户设置 (示例)")
st.write("在此处可以添加用户偏好设置，例如默认模型、主题等。")
# 这里可以使用 data_manager 中的 save_user_config 和 get_user_config 函数
# 例如：
# user_config = get_user_config(current_user)
# if user_config:
#     default_model = st.selectbox("选择默认模型", ["Model A", "Model B"], index=user_config.get("default_model_index", 0))
#     if st.button("保存设置"):
#         user_config["default_model_index"] = ["Model A", "Model B"].index(default_model)
#         save_user_config(current_user, user_config)
#         st.success("设置已保存！")

# --- 信息提示 ---
st.divider()
st.info("💡 提示：在其他实验页面上传的数据、训练的模型和生成的结果都会自动保存到你的个人空间中。")
