# streamlit_app.py

import streamlit as st
from utils.user_auth import is_authenticated, login, register, logout, get_current_username

st.set_page_config(
    page_title="NLP 课程大作业",
    page_icon="🤖",
    layout="wide"
)

# --- 侧边栏导航 (仅在登录后显示) ---
def show_navigation():
    st.sidebar.title(f"欢迎, {get_current_username()}!")
    st.sidebar.markdown("---")
    
    # 实验入口
    st.sidebar.header("实验入口")
    st.sidebar.page_link("pages/1_HW01_Gradient_WordCloud.py", label="HW01: 梯度下降 & 词云")
    st.sidebar.page_link("pages/2_HW02_Representation_Learning.py", label="HW02: 表征学习")
    st.sidebar.page_link("pages/3_HW03_LSTM_Text_Generation.py", label="HW03: LSTM 文本生成")
    st.sidebar.page_link("pages/4_HW04_Some_Task.py", label="HW04: ")
    st.sidebar.page_link("pages/5_HW05_Some_Task.py", label="HW05: ")
    # 功能入口
    st.sidebar.header("功能入口")
    st.sidebar.page_link("pages/6_LLM_QA.py", label="大模型问答")
    st.sidebar.page_link("pages/7_User_Center.py", label="用户中心")
    st.sidebar.page_link("pages/8_NLP_Applications.py", label="NLP 应用任务")

    # 登出按钮
    if st.sidebar.button("登出"):
        logout()
        st.rerun() # 重新运行应用以更新状态

# --- 主界面 ---
def main():
    if is_authenticated():
        # 用户已登录，显示导航和内容
        show_navigation()
        st.title("欢迎来到 NLP 课程 Web 展示平台")
        st.markdown("---")
        st.markdown("### 请选择左侧菜单中的实验或功能开始探索。")
    else:
        # 用户未登录，显示登录/注册界面
        st.title("NLP 课程大作业 - 用户登录")
        
        tab1, tab2 = st.tabs(["登录", "注册"])

        with tab1:
            st.subheader("登录")
            login_username = st.text_input("用户名", key="login_user")
            login_password = st.text_input("密码", type="password", key="login_pass")
            if st.button("登录"):
                success, message = login(login_username, login_password)
                if success:
                    st.success(message)
                    st.rerun() # 登录成功后刷新页面以显示导航
                else:
                    st.error(message)

        with tab2:
            st.subheader("注册")
            reg_username = st.text_input("新用户名", key="reg_user")
            reg_password = st.text_input("新密码", type="password", key="reg_pass")
            reg_password_confirm = st.text_input("确认密码", type="password", key="reg_pass_confirm")
            
            if st.button("注册"):
                if reg_password != reg_password_confirm:
                    st.error("两次输入的密码不一致。")
                else:
                    success, message = register(reg_username, reg_password)
                    if success:
                        st.success(message)
                        # 注册成功后，可以选择自动跳转到登录或手动刷新
                        # st.rerun() # 如果想自动跳转到登录页，取消注释这行
                    else:
                        st.error(message)

if __name__ == "__main__":
    main()