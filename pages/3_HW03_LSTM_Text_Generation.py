import streamlit as st
import numpy as np
import re
import jieba
import io
import tensorflow

# 尝试不同的导入方式来兼容不同版本的TensorFlow/Keras
try:
    # TensorFlow 2.x 方式
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.utils import to_categorical
except ImportError:
    try:
        # 较旧版本的方式
        from keras.models import Sequential
        from keras.layers import Embedding, LSTM, Dense
        from keras.preprocessing.text import Tokenizer
        from keras.utils import to_categorical
    except ImportError:
        # 最新版本的方式
        from keras.src.models import Sequential
        from keras.src.layers import Embedding, LSTM, Dense
        try:
            from keras.src.preprocessing.text import Tokenizer
        except ImportError:
            from tensorflow.keras.preprocessing.text import Tokenizer
        from keras.src.utils import to_categorical

st.title("HW03: LSTM文本生成")

# 初始化session state
if 'text_loaded' not in st.session_state:
    st.session_state.text_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None

# 文本清洗函数
def clean_text(text):
    # 转换为小写
    text = text.lower()
    # 使用正则表达式移除标点符号（保留中文字符、字母、数字和空格）
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
    # 移除多余的空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 准备数据函数
def prepare_data(texts, sequence_length=5):
    # 构建tokenizer
    vocab_size = 10000
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts([texts])
    
    # 创建整数序列
    int_sequences = tokenizer.texts_to_sequences([texts])[0]
    
    # 创建X和y
    X = []
    y = []
    
    for i in range(len(int_sequences) - sequence_length):
        # 取前sequence_length个词作为输入
        input_seq = int_sequences[i:i + sequence_length]
        # 下一词作为标签
        target_word = int_sequences[i + sequence_length]
        X.append(input_seq)
        y.append(target_word)
    
    X = np.array(X)
    y = np.array(y)
    
    # 转换为独热编码
    y = to_categorical(y, num_classes=vocab_size)
    
    return X, y, tokenizer, vocab_size

# 构建模型函数
def build_model(vocab_size, sequence_length):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=sequence_length))
    model.add(LSTM(150, return_sequences=True))
    model.add(LSTM(150))
    model.add(Dense(vocab_size, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# 文本生成函数
def generate_text(model, tokenizer, seed_text, length, sequence_length):
    result = []
    for _ in range(length):
        # 将种子文本转换成序列
        encoded = tokenizer.texts_to_sequences([seed_text])[0]
        
        # 若长度不够则填充到指定长度
        if len(encoded) < sequence_length:
            encoded = np.pad(encoded, (sequence_length - len(encoded), 0), 'constant')
        # 若太长，取最后sequence_length个词
        else:
            encoded = encoded[-sequence_length:]
        
        # 转换为模型输入格式
        encoded = np.array(encoded).reshape(1, -1)
        
        # 预测下一个词的分布概率
        pred_probs = model.predict(encoded, verbose=0)[0]
        
        # 选择概率最高的词（贪心策略）
        pred_index = np.argmax(pred_probs)
        
        # 转换回单词
        next_word = tokenizer.index_word.get(pred_index, '<UNK>')
        
        # 添加到结果
        result.append(next_word)
        
        # 更新文本种子：添加新词，保持长度为seq_length
        seed_text += " " + next_word
        words = seed_text.split()
        if len(words) > sequence_length:
            seed_text = ' '.join(words[-sequence_length:])
            
    return ' '.join(result)

# 页面内容
tab1, tab2 = st.tabs(["直接输入文本", "上传TXT文件"])

with tab1:
    text_input = st.text_area("请输入用于训练的文本:", height=200)
    
    if st.button("加载并处理文本"):
        if text_input.strip():
            # 清洗文本
            cleaned_text = clean_text(text_input.strip())
            st.session_state.processed_text = cleaned_text
            
            # 分词
            with st.spinner("正在进行分词..."):
                texts = jieba.lcut(cleaned_text)
                # 过滤掉标点符号和空字符串
                texts = [word for word in texts if word.strip() and word not in '，。！？、；：""''（）【】']
            
            st.success(f"✅ 分词完成，共 {len(texts)} 个词")
            st.write("前20个词预览：", texts[:20])
            
            # 准备数据
            with st.spinner("正在准备训练数据..."):
                sequence_length = 5
                X, y, tokenizer, vocab_size = prepare_data(texts, sequence_length)
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.tokenizer = tokenizer
                st.session_state.vocab_size = vocab_size
                st.session_state.sequence_length = sequence_length
                
            st.success(f"✅ 数据准备完成，词汇表大小: {vocab_size}")
            st.write(f"X shape: {X.shape}")
            st.write(f"y shape: {y.shape}")
            st.session_state.text_loaded = True
        else:
            st.warning("请输入文本内容!")

with tab2:
    uploaded_file = st.file_uploader("选择一个TXT文件", type="txt")
    
    if uploaded_file is not None:
        # 读取文件内容
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        file_content = stringio.read()
        
        st.success("文件上传成功!")
        st.text_area("文件内容预览:", value=file_content[:500]+"..." if len(file_content)>500 else file_content, height=200)
        
        if st.button("处理上传的文件"):
            # 清洗文本
            cleaned_text = clean_text(file_content)
            st.session_state.processed_text = cleaned_text
            
            # 分词
            with st.spinner("正在进行分词..."):
                texts = jieba.lcut(cleaned_text)
                # 过滤掉标点符号和空字符串
                texts = [word for word in texts if word.strip() and word not in '，。！？、；：""''（）【】']
            
            st.success(f"✅ 分词完成，共 {len(texts)} 个词")
            st.write("前20个词预览：", texts[:20])
            
            # 准备数据
            with st.spinner("正在准备训练数据..."):
                sequence_length = 5
                X, y, tokenizer, vocab_size = prepare_data(texts, sequence_length)
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.tokenizer = tokenizer
                st.session_state.vocab_size = vocab_size
                st.session_state.sequence_length = sequence_length
                
            st.success(f"✅ 数据准备完成，词汇表大小: {vocab_size}")
            st.write(f"X shape: {X.shape}")
            st.write(f"y shape: {y.shape}")
            st.session_state.text_loaded = True

if st.session_state.text_loaded:
    if st.button("训练模型"):
        with st.spinner("正在训练模型..."):
            # 构建模型
            model = build_model(st.session_state.vocab_size, st.session_state.sequence_length)
            st.session_state.model = model
            
            # 训练模型
            history = model.fit(
                st.session_state.X, 
                st.session_state.y, 
                batch_size=32, 
                epochs=10, 
                verbose=1
            )
            
            st.session_state.model_trained = True
            st.success("✅ 模型训练完成!")

if st.session_state.model_trained:
    st.subheader("文本生成测试")
    seed_text = st.text_input("请输入种子文本:", "中国共产党")
    gen_length = st.slider("生成文本长度:", 5, 20, 10)
    
    if st.button("生成文本"):
        with st.spinner("正在生成文本..."):
            generated = generate_text(
                st.session_state.model, 
                st.session_state.tokenizer, 
                seed_text, 
                gen_length, 
                st.session_state.sequence_length
            )
            
            st.write(f"种子文本: {seed_text}")
            st.write(f"生成文本: {generated}")

# 提供默认示例文本
st.subheader("示例文本")
if st.button("使用示例文本"):
    example_text = """我们正在使用 LSTM 模型 来生成 文本
这是一个简单的例子，用于演示如何构建一个文本生成模型。
你可以替换这里的文本为你自己的数据集，比如小说、诗歌或对话记录。
中国共产党领导是中国特色社会主义最本质的特征。
全党全国各族人民要紧密团结在党中央周围，为实现中华民族伟大复兴的中国梦而奋斗。"""
    
    st.session_state.example_text = example_text
    st.text_area("示例文本:", value=example_text, height=200, key="example_display")