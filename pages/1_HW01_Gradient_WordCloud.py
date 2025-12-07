# 1_HW01_Gradient_WordCloud.py

# --- 导入必要的库 ---
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt # 修改：导入 pyplot 并命名为 plt
import matplotlib # 修改：同时导入 matplotlib 本身
from mpl_toolkits.mplot3d import Axes3D
import time
from wordcloud import WordCloud
from PIL import Image
import tempfile
import os

#标题
st.set_page_config(
    page_title="HW01: 梯度下降与词云图",
    page_icon="📈",
    layout="wide"
)


# 设置中文字体和负号显示 (全局设置一次即可)
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['savefig.dpi'] = 150
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['grid.alpha'] = 0.3

# --- 主标题 ---
st.title("📊 HW01: 梯度下降可视化 & 交互式词云图")

# --- 创建选项卡 ---
tab1, tab2 = st.tabs(["📈 梯度下降", "☁️ 词云图"])

# ==================== Tab 1: 梯度下降可视化 ====================
with tab1:
    st.header("梯度下降可视化演示")

    # --- 左侧边栏控件 ---
    with st.sidebar:
        st.subheader("参数设置")
        dimension = st.selectbox("选择维度", ("二维", "三维"))
        
        function_type = st.selectbox(
            "选择目标函数",
            (
                "二次函数 f(x,y) = x² + y²",
                "Rosenbrock函数 f(x,y) = (a-x)² + b(y-x²)²",
                "Himmelblau函数 f(x,y) = (x²+y-11)² + (x+y²-7)²"
            )
        )

        learning_rate = st.slider("学习率 (步长)", 0.001, 0.1, 0.01)
        max_iterations = st.slider("最大迭代次数", 10, 200, 50)

        st.subheader("初始点设置")
        x0 = st.number_input("x₀", value=1.0)
        y0 = st.number_input("y₀", value=1.0)

    # --- 定义目标函数和梯度函数 ---
    def get_function_and_gradient(func_type):
        if func_type == "二次函数 f(x,y) = x² + y²":
            def f(x, y):
                return x**2 + y**2
            
            def grad_f(x, y):
                dx = 2 * x
                dy = 2 * y
                return np.array([dx, dy])
            
            x_range = np.linspace(-2, 2, 100)
            y_range = np.linspace(-2, 2, 100)
            return f, grad_f, x_range, y_range
        
        elif func_type == "Rosenbrock函数 f(x,y) = (a-x)² + b(y-x²)²":
            a, b = 1, 100
            
            def f(x, y):
                return (a - x)**2 + b * (y - x**2)**2
            
            def grad_f(x, y):
                dx = -2*(a - x) - 4*b*x*(y - x**2)
                dy = 2*b*(y - x**2)
                return np.array([dx, dy])
            
            x_range = np.linspace(-2, 2, 100)
            y_range = np.linspace(-1, 3, 100)
            return f, grad_f, x_range, y_range
        
        elif func_type == "Himmelblau函数 f(x,y) = (x²+y-11)² + (x+y²-7)²":
            def f(x, y):
                return (x**2 + y - 11)**2 + (x + y**2 - 7)**2
            
            def grad_f(x, y):
                dx = 2*(x**2 + y - 11)*2*x + 2*(x + y**2 - 7)
                dy = 2*(x**2 + y - 11) + 2*(x + y**2 - 7)*2*y
                return np.array([dx, dy])
            
            x_range = np.linspace(-5, 5, 100)
            y_range = np.linspace(-5, 5, 100)
            return f, grad_f, x_range, y_range

    # 获取函数和梯度
    f, grad_f, x_range, y_range = get_function_and_gradient(function_type)

    # --- 显示公式 ---
    st.subheader("函数定义")
    st.latex(f"f(x, y) = {function_type.split(' = ')[1]}")

    st.subheader("梯度公式")
    if function_type == "二次函数 f(x,y) = x² + y²":
        st.latex(r"\nabla f(x, y) = \begin{bmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{bmatrix} = \begin{bmatrix} 2x \\ 2y \end{bmatrix}")
    elif function_type == "Rosenbrock函数 f(x,y) = (a-x)² + b(y-x²)²":
        st.latex(r"\nabla f(x, y) = \begin{bmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{bmatrix} = \begin{bmatrix} -2(1-x) - 400x(y-x^2) \\ 200(y-x^2) \end{bmatrix}")
    elif function_type == "Himmelblau函数 f(x,y) = (x²+y-11)² + (x+y²-7)²":
        st.latex(r"\nabla f(x, y) = \begin{bmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{bmatrix} = \begin{bmatrix} 4x(x^2+y-11) + 2(x+y^2-7) \\ 2(x^2+y-11) + 4y(x+y^2-7) \end{bmatrix}")

    # --- 执行梯度下降 ---
    def gradient_descent(f, grad_f, x0, y0, lr, max_iters):
        x, y = x0, y0
        path = [(x, y)]
        values = [f(x, y)]
        
        for _ in range(max_iters):
            grad = grad_f(x, y)
            x_new = x - lr * grad[0]
            y_new = y - lr * grad[1]
            
            if abs(x_new - x) < 1e-6 and abs(y_new - y) < 1e-6:
                break
                
            x, y = x_new, y_new
            path.append((x, y))
            values.append(f(x, y))
        
        return np.array(path), np.array(values)

    path, values = gradient_descent(f, grad_f, x0, y0, learning_rate, max_iterations)

    # --- 根据维度显示结果 ---
    if dimension == "二维":
        st.subheader("二维梯度下降过程")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        X, Y = np.meshgrid(x_range, y_range)
        Z = f(X, Y)
        
        contour = ax.contour(X, Y, Z, levels=20, alpha=0.6)
        ax.clabel(contour, inline=True, fontsize=8)
        
        path_x, path_y = path[:, 0], path[:, 1]
        ax.plot(path_x, path_y, 'ro-', markersize=5, linewidth=2, label='梯度下降路径')
        ax.scatter(path_x[0], path_y[0], color='green', s=100, label='起始点', zorder=5)
        ax.scatter(path_x[-1], path_y[-1], color='red', s=100, label='终点', zorder=5)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Gradient Descent Process - {function_type.split(" = ")[0]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
    else:
        st.subheader("三维梯度下降过程")
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        X, Y = np.meshgrid(x_range, y_range)
        Z = f(X, Y)
        ax.plot_surface(X, Y, Z, alpha=0.5, cmap='viridis')
        
        path_z = np.array([f(p[0], p[1]) for p in path])
        ax.plot(path[:, 0], path[:, 1], path_z, 'ro-', markersize=5, linewidth=2, label='梯度下降路径')
        ax.scatter(path[0, 0], path[0, 1], f(path[0, 0], path[0, 1]), color='green', s=100, label='起始点', zorder=5)
        ax.scatter(path[-1, 0], path[-1, 1], f(path[-1, 0], path[-1, 1]), color='red', s=100, label='终点', zorder=5)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x, y)')
        ax.set_title(f'Gradient Descent Process - {function_type.split(" = ")[0]}')
        ax.legend()
        
        st.pyplot(fig)

    # --- 显示结果信息 ---
    st.subheader("梯度下降结果")
    col1, col2 = st.columns(2)
    col1.write(f"**起始点**: ({x0:.4f}, {y0:.4f})")
    col2.write(f"**终点**: ({path[-1][0]:.4f}, {path[-1][1]:.4f})")
    col1.write(f"**起始函数值**: {values[0]:.6f}")
    col2.write(f"**终点函数值**: {values[-1]:.6f}")
    col1.write(f"**迭代次数**: {len(path)}")
    col2.write(f"**函数值下降**: {values[0] - values[-1]:.6f}")

    # --- 动画效果 ---
    st.subheader("梯度下降动画")
    animate = st.button("播放动画")

    if animate:
        placeholder = st.empty()
        if dimension == "二维":
            fig, ax = plt.subplots(figsize=(10, 6))
            X, Y = np.meshgrid(x_range, y_range)
            Z = f(X, Y)
            contour = ax.contour(X, Y, Z, levels=20, alpha=0.6)
            ax.clabel(contour, inline=True, fontsize=8)
            
            path_x, path_y = path[:, 0], path[:, 1]
            ax.scatter(path_x[0], path_y[0], color='green', s=100, label='起始点', zorder=5)
            
            line, = ax.plot([], [], 'ro-', markersize=5, linewidth=2)
            current_point, = ax.plot([], [], 'bo', markersize=8, zorder=6)
            
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'Gradient Descent Animation - {function_type.split(" = ")[0]}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            for i in range(1, len(path)):
                line.set_data(path_x[:i+1], path_y[:i+1])
                current_point.set_data([path_x[i]], [path_y[i]])
                ax.set_title(f'Gradient Descent Animation - {function_type.split(" = ")[0]} (Iteration {i})') 
                placeholder.pyplot(fig)
                time.sleep(0.2)
        
        else:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            X, Y = np.meshgrid(x_range, y_range)
            Z = f(X, Y)
            ax.plot_surface(X, Y, Z, alpha=0.5, cmap='viridis')
            ax.scatter(path[0, 0], path[0, 1], f(path[0, 0], path[0, 1]), color='green', s=100, label='起始点', zorder=5)
            
            path_z = np.array([f(p[0], p[1]) for p in path])
            line, = ax.plot([], [], [], 'ro-', markersize=5, linewidth=2)
            current_point, = ax.plot([], [], [], 'bo', markersize=8, zorder=6)
            
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('f(x, y)')
            ax.set_title(f'Gradient Descent Animation - {function_type.split(" = ")[0]}')
            ax.legend()
            
            for i in range(1, len(path)):
                line.set_data_3d(path[:i+1, 0], path[:i+1, 1], path_z[:i+1])
                current_point.set_data_3d([path[i, 0]], [path[i, 1]], [path_z[i]])
                ax.set_title(f'Gradient Descent Animation - {function_type.split(" = ")[0]} (Iteration {i})')
                placeholder.pyplot(fig)
                time.sleep(0.2)

# ==================== Tab 2: 词云图生成 ====================
with tab2:
    st.header("交互式词云图生成器")

    # --- 获取用户输入 ---
    uploaded_file = st.file_uploader("上传一个文本文件 (.txt)", type=["txt"])
    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")
    else:
        text_input = st.text_area("或者在此处粘贴/输入你的文本:", height=150)
        if text_input:
            text = text_input
        else:
            text = ""
            st.warning("⚠️ 请上传文件或输入文本以生成词云。")

    # --- 参数设置 ---
    st.subheader("词云图参数设置")

    # 背景颜色
    background_color = st.color_picker("选择词云背景颜色", "#5cb3cc")  # 默认青色

    # 遮罩图片
    mask_image = st.file_uploader("上传遮罩图片 (可选，如心形、圆形等)", type=["jpg", "jpeg", "png"])
    mask = None
    if mask_image is not None:
        mask_img = Image.open(mask_image)
        mask = np.array(mask_img)

    # 停用词
    st.subheader("设置停用词 (可选)")
    stop_words_method = st.radio("选择停用词来源", ("上传停用词文件", "手动输入停用词"))

    stop_words = None

    if stop_words_method == "上传停用词文件":
        uploaded_stopwords = st.file_uploader("上传停用词文件 (.txt)", type=["txt"])
        if uploaded_stopwords is not None:
            stop_words_content = uploaded_stopwords.read().decode("utf-8")
            stop_words = set(stop_words_content.splitlines())
            st.success(f"✅ 成功加载 {len(stop_words)} 个停用词。")

    elif stop_words_method == "手动输入停用词":
        stop_words_text = st.text_area("请在下方输入停用词，每行一个词:", height=100)
        if stop_words_text:
            stop_words = set(stop_words_text.splitlines())
            stop_words = {word.strip() for word in stop_words if word.strip()}
            st.success(f"✅ 成功设置 {len(stop_words)} 个停用词。")

    # 字体路径 (优化：只检查一次)
    font_path = "C:/Windows/Fonts/simhei.ttf"  # 中文字体
    if not st.session_state.get('font_checked', False):
        if not os.path.exists(font_path):
            st.warning(f"⚠️ 字体文件 {font_path} 未找到，将使用默认字体。")
            font_path = None
        st.session_state.font_checked = True

    # --- 生成并显示词云 ---
    if text:
        try:
            wc = WordCloud(
                font_path=font_path,
                background_color=background_color,
                max_words=200,
                width=800,
                height=400,
                mask=mask,
                stopwords=stop_words,
                colormap='viridis'
            )
            wc.generate(text)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

            # 下载按钮
            if st.button("📥 下载词云图"):
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                wc.to_file(temp_file.name)
                with open(temp_file.name, "rb") as f:
                    st.download_button(
                        label="点击下载",
                        data=f,
                        file_name="wordcloud.png",
                        mime="image/png"
                    )
                temp_file.close()
                os.unlink(temp_file.name)

        except Exception as e:
            st.error(f"❌ 生成词云时出错: {e}")
    else:
        st.info("ℹ️ 请提供文本内容以开始生成词云。")