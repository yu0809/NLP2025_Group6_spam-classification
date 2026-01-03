# -*- coding: utf-8 -*-
"""
LSTM 模型 (使用GloVe词向量)
使用LSTM神经网络和GloVe词嵌入进行垃圾邮件检测。
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 尝试导入TensorFlow/Keras（兼容不同版本）
keras_imported = False
try:
    # 优先尝试从 tensorflow.keras 导入（TensorFlow 2.x）
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (LSTM, 
                                         Embedding, 
                                         BatchNormalization,
                                         Dense, 
                                         Dropout, 
                                         Bidirectional,
                                         GlobalMaxPool1D)
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
    print(f"✓ 成功导入 TensorFlow {tf.__version__}")
    keras_imported = True
except (ImportError, AttributeError, ModuleNotFoundError) as e1:
    try:
        # 如果失败，尝试从 keras 导入（独立Keras或旧版本）
        import keras
        from keras.models import Sequential
        from keras.layers import (LSTM, 
                                  Embedding, 
                                  BatchNormalization,
                                  Dense, 
                                  Dropout, 
                                  Bidirectional,
                                  GlobalMaxPool1D)
        from keras.preprocessing.text import Tokenizer
        from keras.preprocessing.sequence import pad_sequences
        from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
        print(f"✓ 成功导入 Keras {keras.__version__}")
        keras_imported = True
    except (ImportError, AttributeError, ModuleNotFoundError) as e2:
        print("=" * 60)
        print("错误：无法导入Keras/TensorFlow库")
        print("=" * 60)
        print("尝试从 tensorflow.keras 导入时出错:")
        print(f"  {type(e1).__name__}: {e1}")
        print("\n尝试从 keras 导入时出错:")
        print(f"  {type(e2).__name__}: {e2}")
        print("\n解决方案：")
        print("  1. 安装TensorFlow: pip install tensorflow")
        print("  2. 或安装独立Keras: pip install keras")
        print("  3. 检查虚拟环境是否正确激活")
        print("=" * 60)
        sys.exit(1)

if not keras_imported:
    print("错误：Keras/TensorFlow导入失败")
    sys.exit(1)

# 定义word_tokenize函数，处理NLTK资源缺失问题
def safe_word_tokenize(text):
    """安全的词切分函数，如果NLTK不可用则使用简单split"""
    try:
        from nltk.tokenize import word_tokenize
        try:
            return word_tokenize(str(text))
        except LookupError:
            # NLTK资源缺失，使用简单split
            return str(text).split()
    except ImportError:
        # NLTK未安装，使用简单split
        return str(text).split()

# 尝试下载NLTK数据（如果需要）
try:
    import nltk
    import ssl
    try:
        # 尝试禁用SSL验证（仅用于下载NLTK数据）
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        # 尝试下载punkt_tab（新版本）或punkt（旧版本）
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                print("正在尝试下载NLTK punkt数据...")
                try:
                    nltk.download('punkt_tab', quiet=True)
                except:
                    try:
                        nltk.download('punkt', quiet=True)
                    except:
                        print("警告：无法下载NLTK punkt数据，将使用简单的split方法")
    except Exception:
        pass
except ImportError:
    pass

# 使用安全的word_tokenize
word_tokenize = safe_word_tokenize


# 1. 加载数据

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
preprocessed_file = os.path.join(current_dir, 'spam_preprocessed.csv')

# 检查预处理文件是否存在
if not os.path.exists(preprocessed_file):
    print(f"错误：未找到预处理文件 {preprocessed_file}")
    print("请先运行 01_data_loading_and_preprocessing.py 生成预处理文件")
    sys.exit(1)

# 加载预处理后的数据
df = pd.read_csv(preprocessed_file)

# 数据清理：处理NaN值
print(f"原始数据量: {len(df)}")
print(f"message_clean中的NaN数量: {df['message_clean'].isna().sum()}")
print(f"target_encoded中的NaN数量: {df['target_encoded'].isna().sum()}")

# 移除包含NaN的行，使用.copy()避免SettingWithCopyWarning
df_clean = df.dropna(subset=['message_clean', 'target_encoded']).copy()
print(f"清理后数据量: {len(df_clean)}")

# 确保message_clean都是字符串类型
df_clean.loc[:, 'message_clean'] = df_clean['message_clean'].astype(str)
df_clean.loc[:, 'message_clean'] = df_clean['message_clean'].replace('nan', '')
df_clean.loc[:, 'message_clean'] = df_clean['message_clean'].fillna('')

# 移除空字符串
df_clean = df_clean[df_clean['message_clean'].str.strip() != ''].copy()

texts = df_clean['message_clean']
target = df_clean['target_encoded']

print(f"最终数据量: {len(texts)}")


# 2. 文本序列化和填充

# 创建tokenizer并拟合文本
word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(texts)

vocab_length = len(word_tokenizer.word_index) + 1
print(f"词汇表大小: {vocab_length}")


# 将文本转换为序列并填充
def embed(corpus): 
    return word_tokenizer.texts_to_sequences(corpus)

longest_train = max(texts, key=lambda sentence: len(word_tokenize(sentence)))
length_long_sentence = len(word_tokenize(longest_train))
print(f"最长句子长度: {length_long_sentence}")

train_padded_sentences = pad_sequences(
    embed(texts), 
    length_long_sentence, 
    padding='post'
)
print(f"填充后的序列形状: {train_padded_sentences.shape}")


# 3. 加载GloVe词向量
# **注意**: 需要下载GloVe词向量文件。可以从 https://nlp.stanford.edu/projects/glove/ 下载 `glove.6B.100d.txt`

# 加载GloVe词向量
embeddings_dictionary = dict()
embedding_dim = 100

# 使用绝对路径查找GloVe文件
glove_path = os.path.join(current_dir, "glove.6B.100d.txt")

if not os.path.exists(glove_path):
    # 尝试其他可能的位置
    alternative_paths = [
        os.path.join(current_dir, "glove", "glove.6B.100d.txt"),
        os.path.expanduser("~/Downloads/glove.6B.100d.txt"),
    ]
    for alt_path in alternative_paths:
        if os.path.exists(alt_path):
            glove_path = alt_path
            break

try:
    with open(glove_path, 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            records = line.split()
            if len(records) > 1:  # 确保行有效
                word = records[0]
                vector_dimensions = np.asarray(records[1:], dtype='float32')
                embeddings_dictionary[word] = vector_dimensions
    print(f"成功加载 {len(embeddings_dictionary)} 个词向量")
except FileNotFoundError:
    print(f"警告: 未找到GloVe文件 {glove_path}")
    print("请下载GloVe词向量文件: https://nlp.stanford.edu/projects/glove/")
    print("下载后请将文件放在项目目录下，命名为 glove.6B.100d.txt")
    print("程序将继续运行，但将使用随机初始化的词向量")
except Exception as e:
    print(f"加载GloVe文件时出错: {e}")
    print("程序将继续运行，但将使用随机初始化的词向量")


# 创建嵌入矩阵
embedding_matrix = np.zeros((vocab_length, embedding_dim))

for word, index in word_tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

print(f"嵌入矩阵形状: {embedding_matrix.shape}")
print(f"非零向量数量: {np.count_nonzero(np.sum(embedding_matrix, axis=1))}")


# 4. 构建LSTM模型

def glove_lstm():
    model = Sequential()
    
    model.add(Embedding(
        input_dim=embedding_matrix.shape[0], 
        output_dim=embedding_matrix.shape[1], 
        weights=[embedding_matrix], 
        input_length=length_long_sentence,
        trainable=False  # 冻结GloVe权重
    ))
    
    model.add(Bidirectional(LSTM(
        length_long_sentence, 
        return_sequences=True, 
        recurrent_dropout=0.2
    )))
    
    model.add(GlobalMaxPool1D())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(length_long_sentence, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(length_long_sentence, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

model = glove_lstm()
model.summary()


# 5. 训练模型

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    train_padded_sentences, 
    target, 
    test_size=0.25,
    random_state=42
)

print(f"训练集: {len(X_train)}")
print(f"测试集: {len(X_test)}")


# 创建模型并设置回调
model = glove_lstm()

model_file = os.path.join(current_dir, 'lstm_model.h5')
checkpoint = ModelCheckpoint(
    model_file, 
    monitor='val_loss', 
    verbose=1, 
    save_best_only=True
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, 
    verbose=1, 
    patience=5,                        
    min_lr=0.001
)

# 训练模型
history = model.fit(
    X_train, 
    y_train, 
    epochs=7,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1,
    callbacks=[reduce_lr, checkpoint]
)


# 6. 可视化训练过程

def plot_learning_curves(history, arr, output_file=None):
    """
    绘制学习曲线并保存为图片
    
    参数:
    history: 训练历史对象
    arr: 要绘制的指标列表，格式为 [['loss', 'val_loss'], ['accuracy', 'val_accuracy']]
    output_file: 输出文件路径
    """
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    for idx in range(2):
        ax[idx].plot(history.history[arr[idx][0]], label=arr[idx][0], linewidth=2)
        ax[idx].plot(history.history[arr[idx][1]], label=arr[idx][1], linewidth=2)
        ax[idx].legend(fontsize=14)
        ax[idx].set_xlabel('Epoch', fontsize=14)
        ax[idx].set_ylabel('Value', fontsize=14)
        ax[idx].set_title(f'{arr[idx][0]} vs {arr[idx][1]}', fontsize=16, fontweight='bold')
        ax[idx].grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"学习曲线已保存为 {output_file}")
    else:
        try:
            plt.show()
        except Exception as e:
            print(f"显示图表时出错（可忽略）: {e}")
    
    plt.close()

learning_curves_file = os.path.join(current_dir, "learning_curves_lstm.png")
plot_learning_curves(history, [['loss', 'val_loss'], ['accuracy', 'val_accuracy']], 
                     output_file=learning_curves_file)


# 7. 模型评估

# 预测
print("\n开始预测...")
y_preds = (model.predict(X_test, verbose=0) > 0.5).astype("int32")
y_pred_prob = model.predict(X_test, verbose=0)[:, 0]

# 混淆矩阵可视化函数
def conf_matrix(cm, labels=['Ham', 'Spam'], output_file=None, title='混淆矩阵'):
    """
    绘制混淆矩阵并保存为图片
    
    参数:
    cm: 混淆矩阵数组
    labels: 类别标签列表
    output_file: 输出文件路径
    title: 图表标题
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': '数量'})
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('真实值', fontsize=12)
    plt.xlabel('预测值', fontsize=12)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存为 {output_file}")
    else:
        try:
            plt.show()
        except Exception as e:
            print(f"显示图表时出错（可忽略）: {e}")
    
    plt.close()

# 计算所有评价指标
accuracy = metrics.accuracy_score(y_test, y_preds)
auc = metrics.roc_auc_score(y_test, y_pred_prob)
precision = metrics.precision_score(y_test, y_preds)
recall = metrics.recall_score(y_test, y_preds)
f1 = metrics.f1_score(y_test, y_preds)

print(f"\n模型性能指标:")
print(f"  准确率 (Accuracy): {accuracy:.4f}")
print(f"  AUC: {auc:.4f}")
print(f"  精确率 (Precision): {precision:.4f}")
print(f"  召回率 (Recall): {recall:.4f}")
print(f"  F1分数: {f1:.4f}")

# 分类报告
print(f"\n分类报告:")
print(metrics.classification_report(y_test, y_preds, target_names=['Ham', 'Spam']))

conf_matrix_file = os.path.join(current_dir, "confusion_matrix_lstm.png")
conf_matrix(metrics.confusion_matrix(y_test, y_preds),
            labels=['Ham', 'Spam'],
            output_file=conf_matrix_file,
            title='LSTM (GloVe) 混淆矩阵')

# ROC曲线
def plot_roc_curve(y_true, y_pred_prob, output_file=None, title='ROC曲线'):
    """绘制ROC曲线"""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC曲线 (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='随机猜测')
    plt.xlabel('假正率 (False Positive Rate)', fontsize=12)
    plt.ylabel('真正率 (True Positive Rate)', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"ROC曲线已保存为 {output_file}")
    else:
        try:
            plt.show()
        except Exception as e:
            print(f"显示图表时出错（可忽略）: {e}")
    plt.close()

roc_file = os.path.join(current_dir, "roc_curve_lstm.png")
plot_roc_curve(y_test, y_pred_prob, output_file=roc_file, title='LSTM (GloVe) ROC曲线')

