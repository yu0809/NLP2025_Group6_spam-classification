# -*- coding: utf-8 -*-
"""
ResNet 文本分类 - 简易推理脚本
输入一句话即可返回 Spam/Ham 及概率，便于快速验证模型。
"""

import os
import sys
import pickle
import importlib.util
import numpy as np

# 减少 TensorFlow 日志输出
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# 避免某些环境下 Matplotlib 写权限报错（Python 启动钩子可能会导入它）
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(CURRENT_DIR, ".mpl_cache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.layers import (
        Embedding,
        Conv1D,
        GlobalMaxPooling1D,
        GlobalAveragePooling1D,
        Dense,
        Dropout,
        BatchNormalization,
        Add,
        Activation,
        Input,
        Bidirectional,
        LSTM,
        Concatenate,
        Reshape,
        Multiply,
        Lambda,
    )
except Exception as e:  # pragma: no cover - 仅在环境缺依赖时提示
    print("无法导入 TensorFlow/Keras，请先安装依赖：pip install tensorflow")
    print(f"具体错误: {e}")
    sys.exit(1)

MODEL_PATH = os.path.join(CURRENT_DIR, "resnet_model.h5")
TOKENIZER_PATH = os.path.join(CURRENT_DIR, "resnet_tokenizer.pkl")
PREPROCESSED_PATH = os.path.join(CURRENT_DIR, "spam_preprocessed.csv")

MAX_WORDS = 10000
MAX_LEN = 100


def load_tokenizer():
    """优先从 pickle 读取 tokenizer；缺失时用预处理数据重建。"""
    if os.path.exists(TOKENIZER_PATH):
        with open(TOKENIZER_PATH, "rb") as f:
            data = pickle.load(f)
        tokenizer = data.get("tokenizer") if isinstance(data, dict) else data
        if not hasattr(tokenizer, "word_index"):
            raise ValueError("resnet_tokenizer.pkl 不包含有效的 tokenizer。")

        # 如果文件里带有 max_words / max_length，用它们校准当前脚本设置
        if isinstance(data, dict):
            global MAX_WORDS, MAX_LEN
            MAX_WORDS = int(data.get("max_words", MAX_WORDS))
            MAX_LEN = int(data.get("max_length", MAX_LEN))

        return tokenizer

    if not os.path.exists(PREPROCESSED_PATH):
        raise FileNotFoundError(
            "未找到 tokenizer 文件，也未找到预处理数据 spam_preprocessed.csv，"
            "请先生成预处理数据或提供 resnet_tokenizer.pkl。"
        )

    # 懒加载 pandas 以减少启动开销
    import pandas as pd

    df = pd.read_csv(PREPROCESSED_PATH)
    df_clean = df.dropna(subset=["message_clean"]).copy()
    df_clean.loc[:, "message_clean"] = df_clean["message_clean"].astype(str)
    df_clean.loc[:, "message_clean"] = df_clean["message_clean"].replace("nan", "").fillna("")
    df_clean = df_clean[df_clean["message_clean"].str.strip() != ""].copy()

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(df_clean["message_clean"])

    # 缓存一份，方便下次直接加载
    try:
        with open(TOKENIZER_PATH, "wb") as f:
            pickle.dump(tokenizer, f)
    except Exception:
        pass

    return tokenizer


def _load_arch_module():
    """动态加载 08_resnet_inference.py，复用完整的模型结构定义。"""
    infer_path = os.path.join(CURRENT_DIR, "08_resnet_inference.py")
    if not os.path.exists(infer_path):
        return None

    spec = importlib.util.spec_from_file_location("resnet_infer", infer_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ==== Fallback: 内置模型结构（当 08_resnet_inference.py 缺失时使用，保持与训练一致） ====
def se_block(x, filters, block_name=""):
    se = GlobalAveragePooling1D(name=f"{block_name}_se_gap")(x)
    se = Reshape((1, filters), name=f"{block_name}_se_reshape")(se)
    se = Dense(filters // 16, activation="relu", name=f"{block_name}_se_fc1")(se)
    se = Dense(filters, activation="sigmoid", name=f"{block_name}_se_fc2")(se)
    x = Multiply(name=f"{block_name}_se_multiply")([x, se])
    return x


def multi_scale_conv_block(x, filters, block_name=""):
    branch1 = Conv1D(filters, 1, padding="same", name=f"{block_name}_ms_conv1x1")(x)
    branch1 = BatchNormalization(name=f"{block_name}_ms_bn1")(branch1)
    branch1 = Activation("relu", name=f"{block_name}_ms_relu1")(branch1)

    branch2 = Conv1D(filters, 3, padding="same", name=f"{block_name}_ms_conv3x3")(x)
    branch2 = BatchNormalization(name=f"{block_name}_ms_bn2")(branch2)
    branch2 = Activation("relu", name=f"{block_name}_ms_relu2")(branch2)

    branch3 = Conv1D(filters, 5, padding="same", name=f"{block_name}_ms_conv5x5")(x)
    branch3 = BatchNormalization(name=f"{block_name}_ms_bn3")(branch3)
    branch3 = Activation("relu", name=f"{block_name}_ms_relu3")(branch3)

    branch4 = Conv1D(filters, 3, padding="same", name=f"{block_name}_ms_conv3x3_1")(x)
    branch4 = BatchNormalization(name=f"{block_name}_ms_bn4_1")(branch4)
    branch4 = Activation("relu", name=f"{block_name}_ms_relu4_1")(branch4)
    branch4 = Conv1D(filters, 5, padding="same", name=f"{block_name}_ms_conv5x5_2")(branch4)
    branch4 = BatchNormalization(name=f"{block_name}_ms_bn4_2")(branch4)
    branch4 = Activation("relu", name=f"{block_name}_ms_relu4_2")(branch4)

    x = Concatenate(axis=-1, name=f"{block_name}_ms_concat")([branch1, branch2, branch3, branch4])
    return x


def attention_pooling(x, name=""):
    attention = Dense(1, activation="tanh", name=f"{name}_att_dense")(x)
    attention = Activation("softmax", name=f"{name}_att_softmax")(attention)
    x_weighted = Multiply(name=f"{name}_att_multiply")([x, attention])
    x_pooled = Lambda(lambda t: tf.reduce_sum(t, axis=1), name=f"{name}_att_sum")(x_weighted)
    return x_pooled


def multi_head_self_attention(x, num_heads=4, head_dim=32, name=""):
    input_dim = int(x.shape[-1])
    total_dim = num_heads * head_dim

    q = Dense(total_dim, name=f"{name}_att_q")(x)
    k = Dense(total_dim, name=f"{name}_att_k")(x)
    v = Dense(total_dim, name=f"{name}_att_v")(x)

    def reshape_and_transpose(tensor):
        tensor = tf.reshape(tensor, (-1, tf.shape(tensor)[1], num_heads, head_dim))
        return tf.transpose(tensor, [0, 2, 1, 3])

    q = Lambda(reshape_and_transpose, name=f"{name}_att_q_reshape")(q)
    k = Lambda(reshape_and_transpose, name=f"{name}_att_k_reshape")(k)
    v = Lambda(reshape_and_transpose, name=f"{name}_att_v_reshape")(v)

    def compute_scores(qk):
        q_tensor, k_tensor = qk
        return tf.matmul(q_tensor, k_tensor, transpose_b=True) / tf.sqrt(float(head_dim))

    scores = Lambda(compute_scores, name=f"{name}_att_scores")([q, k])
    att_weights = Activation("softmax", name=f"{name}_att_weights")(scores)

    def apply_attention(av):
        att_weights_tensor, v_tensor = av
        return tf.matmul(att_weights_tensor, v_tensor)

    att_output = Lambda(apply_attention, name=f"{name}_att_output")([att_weights, v])

    def transpose_and_reshape(tensor):
        tensor = tf.transpose(tensor, [0, 2, 1, 3])
        return tf.reshape(tensor, (-1, tf.shape(tensor)[1], total_dim))

    att_output = Lambda(transpose_and_reshape, name=f"{name}_att_concat")(att_output)
    output = Dense(input_dim, name=f"{name}_att_proj")(att_output)
    output = Add(name=f"{name}_att_residual")([x, output])
    output = BatchNormalization(name=f"{name}_att_norm")(output)
    return output


def residual_block(x, filters, kernel_size=3, stride=1, block_name="", use_se=False):
    shortcut = x

    x = Conv1D(filters, kernel_size, strides=stride, padding="same", name=f"{block_name}_conv1")(x)
    x = BatchNormalization(name=f"{block_name}_bn1")(x)
    x = Activation("relu", name=f"{block_name}_relu1")(x)
    x = Dropout(0.2, name=f"{block_name}_dropout1")(x)

    x = Conv1D(filters, kernel_size, strides=1, padding="same", name=f"{block_name}_conv2")(x)
    x = BatchNormalization(name=f"{block_name}_bn2")(x)

    if use_se:
        x = se_block(x, filters, block_name=f"{block_name}_se")

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, strides=stride, padding="same", name=f"{block_name}_shortcut_conv")(shortcut)
        shortcut = BatchNormalization(name=f"{block_name}_shortcut_bn")(shortcut)

    x = Add(name=f"{block_name}_add")([x, shortcut])
    x = Activation("relu", name=f"{block_name}_relu2")(x)
    return x


def create_resnet_model_fallback(vocab_size, embedding_dim=128, max_length=100, enhanced=True):
    """与训练时一致的模型结构，供缺失 inference 文件时使用。"""
    inputs = Input(shape=(max_length,), name="input")

    embedding_dim_enhanced = embedding_dim * 2 if enhanced else embedding_dim
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim_enhanced, input_length=max_length, name="embedding")(inputs)

    if enhanced:
        x = multi_head_self_attention(x, num_heads=4, head_dim=32, name="att1")
        x = Dropout(0.2, name="att1_dropout")(x)

    if enhanced:
        x = multi_scale_conv_block(x, filters=32, block_name="initial_ms")
        x = Conv1D(128, 1, padding="same", name="initial_proj")(x)
    else:
        x = Conv1D(64, 7, strides=2, padding="same", name="initial_conv")(x)

    x = BatchNormalization(name="initial_bn")(x)
    x = Activation("relu", name="initial_relu")(x)
    x = Dropout(0.2, name="initial_dropout")(x)

    x = residual_block(x, filters=128, kernel_size=3, stride=1, block_name="res_block1", use_se=enhanced)
    x = residual_block(x, filters=128, kernel_size=3, stride=1, block_name="res_block2", use_se=enhanced)

    if enhanced:
        x = residual_block(x, filters=128, kernel_size=3, stride=1, block_name="res_block2_5", use_se=True)

    x = residual_block(x, filters=256, kernel_size=3, stride=2, block_name="res_block3", use_se=enhanced)
    x = residual_block(x, filters=256, kernel_size=3, stride=1, block_name="res_block4", use_se=enhanced)

    if enhanced:
        x = residual_block(x, filters=256, kernel_size=3, stride=1, block_name="res_block4_5", use_se=True)

    final_filters = 512 if enhanced else 256
    x = residual_block(x, filters=final_filters, kernel_size=3, stride=2, block_name="res_block5", use_se=enhanced)
    x = residual_block(x, filters=final_filters, kernel_size=3, stride=1, block_name="res_block6", use_se=enhanced)

    if enhanced:
        x = residual_block(x, filters=final_filters, kernel_size=3, stride=1, block_name="res_block6_5", use_se=True)
        x = residual_block(x, filters=final_filters, kernel_size=3, stride=1, block_name="res_block7", use_se=True)

    if enhanced:
        lstm_out = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3), name="bilstm")(x)
        x = Concatenate(axis=-1, name="cnn_lstm_concat")([x, lstm_out])
        x = Conv1D(final_filters, 1, padding="same", name="fusion_proj")(x)
        x = BatchNormalization(name="fusion_bn")(x)
        x = Activation("relu", name="fusion_relu")(x)

    if enhanced:
        x_pooled_att = attention_pooling(x, name="att_pool")
        x_pooled_max = GlobalMaxPooling1D(name="global_max_pool")(x)
        x_pooled_avg = GlobalAveragePooling1D(name="global_avg_pool")(x)
        x = Concatenate(axis=-1, name="pool_concat")([x_pooled_att, x_pooled_max, x_pooled_avg])
    else:
        x = GlobalMaxPooling1D(name="global_max_pool")(x)

    if enhanced:
        x = Dense(256, activation="relu", name="dense_1")(x)
        x = BatchNormalization(name="fc_bn1")(x)
        x = Dropout(0.5, name="fc_dropout1")(x)

        x = Dense(128, activation="relu", name="dense_2")(x)
        x = BatchNormalization(name="fc_bn2")(x)
        x = Dropout(0.4, name="fc_dropout2")(x)

        x = Dense(64, activation="relu", name="dense_3")(x)
        x = BatchNormalization(name="fc_bn3")(x)
        x = Dropout(0.3, name="fc_dropout3")(x)
    else:
        x = Dense(128, activation="relu", name="dense_1")(x)
        x = BatchNormalization(name="fc_bn1")(x)
        x = Dropout(0.5, name="fc_dropout1")(x)

        x = Dense(64, activation="relu", name="dense_2")(x)
        x = BatchNormalization(name="fc_bn2")(x)
        x = Dropout(0.3, name="fc_dropout2")(x)

    outputs = Dense(1, activation="sigmoid", name="output")(x)

    model_name = "Enhanced_ResNet_Text_Classifier" if enhanced else "ResNet_Text_Classifier"
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model


def load_model(tokenizer):
    """
    重建模型结构并加载权重，避免自定义 Lambda 函数反序列化失败。

    参数:
        tokenizer: 已拟合的 Tokenizer，用于确定 vocab_size。
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "未找到模型文件 resnet_model.h5，请先运行 08_resnet_text.py 训练并保存模型。"
        )

    arch = _load_arch_module()
    vocab_size = len(tokenizer.word_index) + 1

    if arch and hasattr(arch, "create_resnet_model"):
        model = arch.create_resnet_model(
            vocab_size=vocab_size,
            embedding_dim=128,
            max_length=MAX_LEN,
            enhanced=True,
        )
    else:
        # Fallback: 使用内置结构
        model = create_resnet_model_fallback(
            vocab_size=vocab_size,
            embedding_dim=128,
            max_length=MAX_LEN,
            enhanced=True,
        )

    # 即便模型文件保存了完整结构，也能用作权重文件加载
    model.load_weights(MODEL_PATH)
    return model


def predict_text(model, tokenizer, text):
    """对单条文本进行预测，返回 (label, prob)。"""
    if not isinstance(text, str):
        text = str(text)
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding="post", truncating="post")
    prob = float(model.predict(padded, verbose=0)[0][0])
    label = "Spam" if prob > 0.5 else "Ham"
    return label, prob


def _decode_tokens(token_ids, tokenizer):
    """将ID序列还原为token列表（去掉padding）。"""
    tokens = []
    for tid in token_ids:
        if tid == 0:
            continue
        if hasattr(tokenizer, "index_word"):
            tokens.append(tokenizer.index_word.get(int(tid), f"<UNK_{tid}>"))
        elif hasattr(tokenizer, "word_index"):
            # 兜底反查
            reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
            tokens.append(reverse_word_index.get(int(tid), f"<UNK_{tid}>"))
        else:
            tokens.append(f"<UNK_{tid}>")
    return tokens


def gradient_x_embedding_importance(model, x_ids, layer_name="embedding"):
    """
    Gradient×Embedding token 重要性（参考 08_resnet_text.py 的实现）。
    返回每个位置的分数以及预测概率。
    """
    from tensorflow.keras.models import Model

    emb_layer = model.get_layer(layer_name)
    probe = Model(inputs=model.inputs, outputs=[emb_layer.output, model.output])

    x = tf.convert_to_tensor(x_ids[None, :], dtype=tf.int32)
    with tf.GradientTape() as tape:
        emb_out, y = probe(x, training=False)
        tape.watch(emb_out)
        score = y[0, 0]

    grads = tape.gradient(score, emb_out)
    token_scores = tf.reduce_sum(tf.abs(grads * emb_out), axis=-1)[0].numpy()
    pred_prob = float(y.numpy()[0, 0])
    return token_scores, pred_prob


def get_token_importance(model, tokenizer, text, top_k=8):
    """获取当前输入的 Top-K 重要 token 及相对分数，返回列表。"""
    import re
    
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding="post", truncating="post")
    x_ids = padded[0]

    token_scores, pred_prob = gradient_x_embedding_importance(model, x_ids, layer_name="embedding")
    nonpad_mask = x_ids != 0
    scores = token_scores[nonpad_mask]
    tokens = _decode_tokens(x_ids[nonpad_mask], tokenizer)

    if len(scores) == 0:
        return []

    # 尝试从原始文本中提取词，用于映射 OOV token
    # tokenizer 通常会将文本转为小写并按空格分词
    text_lower = text.lower().strip()
    # 使用正则表达式提取词（字母数字组合）
    original_words = re.findall(r'\b\w+\b', text_lower)
    
    # 创建显示 token 的映射
    # 由于 tokenizer 的处理可能复杂，我们采用简单策略：
    # 如果 token 是 OOV 且位置在原始词范围内，显示原始词
    oov_id = tokenizer.word_index.get("<OOV>", 1)
    display_tokens = []
    
    for i, (token_id, decoded_token) in enumerate(zip(x_ids[nonpad_mask], tokens)):
        if decoded_token == "<OOV>" or token_id == oov_id:
            # 尝试从原始文本中找到对应的词
            if i < len(original_words):
                display_tokens.append(f"{original_words[i]} (OOV)")
            else:
                display_tokens.append("<OOV>")
        else:
            display_tokens.append(decoded_token)

    # 归一化到 0-1，便于直观比较
    scores_norm = scores / (scores.max() + 1e-9)
    idx = np.argsort(scores_norm)[::-1][:top_k]

    result = []
    for i in idx:
        result.append((display_tokens[i], scores_norm[i]))
    return result


def show_token_importance(model, tokenizer, text, top_k=8):
    """打印当前输入的 Top-K 重要 token 及相对分数。"""
    result = get_token_importance(model, tokenizer, text, top_k)
    if not result:
        print("无有效 token，无法计算重要性。")
        return
    
    print("Top 词语重要性 (Gradient×Embedding):")
    for token, score in result:
        print(f"  {token}: {score:.3f}")


def create_gui():
    """创建图形界面"""
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
    
    # 创建主窗口
    root = tk.Tk()
    root.title("ResNet 垃圾邮件分类器")
    root.geometry("750x650")
    root.resizable(True, True)
    
    # 设置样式
    style = ttk.Style()
    try:
        style.theme_use('clam')
    except:
        pass  # 如果主题不可用，使用默认主题
    
    # 全局变量存储模型和tokenizer
    model = None
    tokenizer = None
    loading_status = {"status": "未加载"}
    
    # 加载模型
    def load_models():
        nonlocal model, tokenizer
        try:
            status_label.config(text="正在加载模型...", foreground="blue")
            root.update()
            tokenizer = load_tokenizer()
            model = load_model(tokenizer)
            loading_status["status"] = "已加载"
            status_label.config(text="✓ 模型已加载，可以开始预测", foreground="green")
            predict_btn.config(state="normal")
        except Exception as e:
            loading_status["status"] = "加载失败"
            status_label.config(text=f"✗ 加载失败: {str(e)[:50]}...", foreground="red")
            messagebox.showerror("加载错误", f"加载模型或 tokenizer 失败:\n{e}")
            predict_btn.config(state="disabled")
    
    # 预测函数
    def perform_prediction():
        if model is None or tokenizer is None:
            messagebox.showwarning("警告", "请先加载模型")
            return
        
        text = input_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("警告", "请输入要分类的文本")
            return
        
        try:
            # 更新状态
            status_label.config(text="正在预测...", foreground="blue")
            root.update()
            
            # 执行预测
            label, prob = predict_text(model, tokenizer, text)
            confidence = prob if label == "Spam" else 1 - prob
            
            # 更新结果显示
            result_label.config(text=f"分类: {label}")
            prob_label.config(text=f"垃圾邮件概率: {prob:.4f}")
            conf_label.config(text=f"置信度: {confidence*100:.2f}%")
            
            # 根据结果设置颜色
            if label == "Spam":
                result_label.config(foreground="red")
            else:
                result_label.config(foreground="green")
            
            # 获取并显示token重要性
            importance_list = get_token_importance(model, tokenizer, text, top_k=10)
            importance_text.delete("1.0", tk.END)
            if importance_list:
                importance_text.insert("1.0", "Top 词语重要性 (Gradient×Embedding):\n\n")
                for i, (token, score) in enumerate(importance_list, 1):
                    importance_text.insert(tk.END, f"{i}. {token}: {score:.3f}\n")
            else:
                importance_text.insert("1.0", "无有效 token，无法计算重要性。")
            
            status_label.config(text="✓ 预测完成", foreground="green")
            
        except Exception as e:
            status_label.config(text="✗ 预测失败", foreground="red")
            messagebox.showerror("预测错误", f"预测过程中发生错误:\n{e}")
    
    # 清空输入
    def clear_input():
        input_text.delete("1.0", tk.END)
        result_label.config(text="分类: -")
        prob_label.config(text="垃圾邮件概率: -")
        conf_label.config(text="置信度: -")
        importance_text.delete("1.0", tk.END)
    
    # 创建界面布局
    # 标题
    title_frame = ttk.Frame(root, padding="10")
    title_frame.pack(fill=tk.X)
    title_label = ttk.Label(title_frame, text="ResNet 垃圾邮件分类器", 
                           font=("Arial", 16, "bold"))
    title_label.pack()
    
    # 状态栏
    status_frame = ttk.Frame(root, padding="10")
    status_frame.pack(fill=tk.X)
    status_label = ttk.Label(status_frame, text="点击'加载模型'按钮开始", 
                             font=("Arial", 10))
    status_label.pack(side=tk.LEFT)
    load_btn = ttk.Button(status_frame, text="加载模型", command=load_models)
    load_btn.pack(side=tk.RIGHT, padx=5)
    
    # 输入区域
    input_frame = ttk.LabelFrame(root, text="输入文本", padding="10")
    input_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    input_text = scrolledtext.ScrolledText(input_frame, height=8, wrap=tk.WORD,
                                           font=("Arial", 11))
    input_text.pack(fill=tk.BOTH, expand=True)
    
    # 按钮区域
    button_frame = ttk.Frame(root, padding="10")
    button_frame.pack(fill=tk.X)
    predict_btn = ttk.Button(button_frame, text="预测", command=perform_prediction, 
                            state="disabled")
    predict_btn.pack(side=tk.LEFT, padx=5)
    clear_btn = ttk.Button(button_frame, text="清空", command=clear_input)
    clear_btn.pack(side=tk.LEFT, padx=5)
    
    # 结果显示区域
    result_frame = ttk.LabelFrame(root, text="预测结果", padding="10")
    result_frame.pack(fill=tk.X, padx=10, pady=5)
    
    result_label = ttk.Label(result_frame, text="分类: -", font=("Arial", 12, "bold"))
    result_label.pack(anchor=tk.W, pady=2)
    prob_label = ttk.Label(result_frame, text="垃圾邮件概率: -", font=("Arial", 11))
    prob_label.pack(anchor=tk.W, pady=2)
    conf_label = ttk.Label(result_frame, text="置信度: -", font=("Arial", 11))
    conf_label.pack(anchor=tk.W, pady=2)
    
    # Token重要性显示区域
    importance_frame = ttk.LabelFrame(root, text="词语重要性分析", padding="10")
    importance_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    importance_text = scrolledtext.ScrolledText(importance_frame, height=8, wrap=tk.WORD,
                                                font=("Courier", 10))
    importance_text.pack(fill=tk.BOTH, expand=True)
    
    # 绑定快捷键
    def on_ctrl_enter(event):
        perform_prediction()
    
    def on_enter(event):
        if model is not None and tokenizer is not None:
            perform_prediction()
    
    input_text.bind("<Control-Return>", on_ctrl_enter)
    root.bind("<Return>", lambda e: None)  # 防止回车键触发默认行为
    
    # 自动加载模型
    root.after(100, load_models)
    
    # 设置窗口居中
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    # 运行主循环
    root.mainloop()


def main():
    """主函数：根据是否在GUI环境选择运行方式"""
    # 检查是否支持GUI（有DISPLAY环境变量或macOS）
    try:
        import tkinter as tk
        # 尝试创建测试窗口
        test_root = tk.Tk()
        test_root.withdraw()  # 隐藏测试窗口
        test_root.destroy()
        # 如果成功，使用GUI模式
        create_gui()
    except Exception:
        # 如果GUI不可用，回退到命令行模式
        print("=" * 50)
        print("ResNet 垃圾邮件分类器 - 简易推理")
        print("=" * 50)

        try:
            tokenizer = load_tokenizer()
            model = load_model(tokenizer)
        except Exception as e:
            print(f"加载模型或 tokenizer 失败: {e}")
            sys.exit(1)

        print("模型与 tokenizer 已加载，首次预测可能需要几秒钟初始化。\n")
        print("输入文本后回车即可预测；输入 exit/quit/q 退出。")

        while True:
            try:
                user_input = input("\n请输入文本: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n已退出")
                break

            if user_input.lower() in {"exit", "quit", "q"}:
                print("已退出")
                break

            if not user_input:
                print("输入为空，请重新输入。")
                continue

            label, prob = predict_text(model, tokenizer, user_input)
            confidence = prob if label == "Spam" else 1 - prob
            print(f"分类: {label} | 垃圾邮件概率: {prob:.4f} | 置信度: {confidence*100:.2f}%")
            show_token_importance(model, tokenizer, user_input)


if __name__ == "__main__":
    main()
