# -*- coding: utf-8 -*-
"""
ResNet (残差网络) 文本分类
使用带残差连接的一维卷积神经网络进行垃圾邮件检测，结合词嵌入技术。
ResNet通过残差连接解决深层网络的梯度消失问题，可能获得更好的性能。
"""

import os
import sys
import html
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
    from tensorflow.keras.layers import (Embedding, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D,
                                         Dense, Dropout, BatchNormalization, Add, Activation)
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback
    from tensorflow.keras.optimizers import Adam
    print(f"✓ 成功导入 TensorFlow {tf.__version__}")
    keras_imported = True
except (ImportError, AttributeError, ModuleNotFoundError) as e1:
    try:
        # 如果失败，尝试从 keras 导入（独立Keras或旧版本）
        import keras
        from keras.models import Sequential
        from keras.layers import (Embedding, Conv1D, GlobalMaxPooling1D, 
                                 Dense, Dropout, BatchNormalization, Add, Activation)
        from keras.preprocessing.text import Tokenizer
        from keras.preprocessing.sequence import pad_sequences
        from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback
        from keras.optimizers import Adam
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


# 1. 加载数据

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 创建输出文件夹
output_dir = os.path.join(current_dir, 'resnet_explanation')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"✓ 创建输出文件夹: {output_dir}")
else:
    print(f"✓ 使用现有输出文件夹: {output_dir}")

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

print("\n开始文本序列化...")

# 设置最大词汇量和序列长度
max_words = 10000  # 词汇表大小
max_length = 100   # 序列最大长度

# 创建tokenizer并拟合文本
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

# 将文本转换为序列
sequences = tokenizer.texts_to_sequences(texts)

# 填充序列到相同长度
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

vocab_size = len(tokenizer.word_index) + 1
print(f"词汇表大小: {vocab_size}")
print(f"填充后的序列形状: {padded_sequences.shape}")


# 3. 构建ResNet模型

def se_block(x, filters, block_name=''):
    """
    SE-Net注意力块（Squeeze-and-Excitation）
    通过通道注意力机制增强重要特征
    
    参数:
    x: 输入张量 (batch, seq_len, filters)
    filters: 通道数
    block_name: 块名称
    """
    from tensorflow.keras.layers import GlobalAveragePooling1D, Reshape, Multiply
    
    # Squeeze: 全局平均池化
    se = GlobalAveragePooling1D(name=f'{block_name}_se_gap')(x)
    # Reshape for Dense layers
    se = Reshape((1, filters), name=f'{block_name}_se_reshape')(se)
    
    # Excitation: 两个全连接层
    se = Dense(filters // 16, activation='relu', name=f'{block_name}_se_fc1')(se)
    se = Dense(filters, activation='sigmoid', name=f'{block_name}_se_fc2')(se)
    
    # Scale: 将注意力权重应用到原始特征
    x = Multiply(name=f'{block_name}_se_multiply')([x, se])
    return x

def multi_scale_conv_block(x, filters, block_name=''):
    """
    多尺度卷积块（类似Inception）
    并行使用不同kernel size的卷积，捕获不同尺度的特征
    
    参数:
    x: 输入张量
    filters: 每个分支的卷积核数量
    block_name: 块名称
    """
    from tensorflow.keras.layers import Concatenate
    
    # 分支1: 1x1卷积（点卷积）
    branch1 = Conv1D(filters, 1, padding='same', name=f'{block_name}_ms_conv1x1')(x)
    branch1 = BatchNormalization(name=f'{block_name}_ms_bn1')(branch1)
    branch1 = Activation('relu', name=f'{block_name}_ms_relu1')(branch1)
    
    # 分支2: 3x3卷积
    branch2 = Conv1D(filters, 3, padding='same', name=f'{block_name}_ms_conv3x3')(x)
    branch2 = BatchNormalization(name=f'{block_name}_ms_bn2')(branch2)
    branch2 = Activation('relu', name=f'{block_name}_ms_relu2')(branch2)
    
    # 分支3: 5x5卷积
    branch3 = Conv1D(filters, 5, padding='same', name=f'{block_name}_ms_conv5x5')(x)
    branch3 = BatchNormalization(name=f'{block_name}_ms_bn3')(branch3)
    branch3 = Activation('relu', name=f'{block_name}_ms_relu3')(branch3)
    
    # 分支4: 3x3卷积 + 5x5卷积（级联）
    branch4 = Conv1D(filters, 3, padding='same', name=f'{block_name}_ms_conv3x3_1')(x)
    branch4 = BatchNormalization(name=f'{block_name}_ms_bn4_1')(branch4)
    branch4 = Activation('relu', name=f'{block_name}_ms_relu4_1')(branch4)
    branch4 = Conv1D(filters, 5, padding='same', name=f'{block_name}_ms_conv5x5_2')(branch4)
    branch4 = BatchNormalization(name=f'{block_name}_ms_bn4_2')(branch4)
    branch4 = Activation('relu', name=f'{block_name}_ms_relu4_2')(branch4)
    
    # 拼接所有分支
    x = Concatenate(axis=-1, name=f'{block_name}_ms_concat')([branch1, branch2, branch3, branch4])
    
    return x

def attention_pooling(x, name=''):
    """
    注意力池化层
    使用注意力机制替代简单的全局最大池化
    
    参数:
    x: 输入张量 (batch, seq_len, filters)
    name: 层名称前缀
    """
    from tensorflow.keras.layers import Dense, Dot, Lambda, Multiply
    
    # 计算注意力权重
    attention = Dense(1, activation='tanh', name=f'{name}_att_dense')(x)  # (batch, seq_len, 1)
    attention = Activation('softmax', name=f'{name}_att_softmax')(attention)  # 归一化
    
    # 加权求和: 使用Multiply然后求和
    # x: (batch, seq_len, filters), attention: (batch, seq_len, 1)
    x_weighted = Multiply(name=f'{name}_att_multiply')([x, attention])
    # 沿序列维度求和: (batch, seq_len, filters) -> (batch, filters)
    x_pooled = Lambda(lambda t: tf.reduce_sum(t, axis=1), name=f'{name}_att_sum')(x_weighted)
    
    return x_pooled

def multi_head_self_attention(x, num_heads=4, head_dim=32, name=''):
    """
    多头自注意力机制
    让模型能够关注序列中不同位置的重要信息
    
    参数:
    x: 输入张量 (batch, seq_len, filters)
    num_heads: 注意力头数
    head_dim: 每个头的维度
    name: 层名称前缀
    """
    from tensorflow.keras.layers import Dense, Reshape, Lambda
    
    input_dim = int(x.shape[-1])
    total_dim = num_heads * head_dim
    
    # Query, Key, Value投影
    q = Dense(total_dim, name=f'{name}_att_q')(x)
    k = Dense(total_dim, name=f'{name}_att_k')(x)
    v = Dense(total_dim, name=f'{name}_att_v')(x)
    
    # 重塑为多头形式并转置: (batch, num_heads, seq_len, head_dim)
    def reshape_and_transpose(tensor):
        # tensor: (batch, seq_len, total_dim)
        # reshape: (batch, seq_len, num_heads, head_dim)
        # transpose: (batch, num_heads, seq_len, head_dim)
        tensor = tf.reshape(tensor, (-1, tf.shape(tensor)[1], num_heads, head_dim))
        return tf.transpose(tensor, [0, 2, 1, 3])
    
    q = Lambda(reshape_and_transpose, name=f'{name}_att_q_reshape')(q)
    k = Lambda(reshape_and_transpose, name=f'{name}_att_k_reshape')(k)
    v = Lambda(reshape_and_transpose, name=f'{name}_att_v_reshape')(v)
    
    # 计算注意力分数: Q @ K^T / sqrt(head_dim)
    def compute_scores(qk):
        q_tensor, k_tensor = qk
        return tf.matmul(q_tensor, k_tensor, transpose_b=True) / tf.sqrt(float(head_dim))
    
    scores = Lambda(compute_scores, name=f'{name}_att_scores')([q, k])
    att_weights = Activation('softmax', name=f'{name}_att_weights')(scores)
    
    # 应用注意力权重: att_weights @ V
    def apply_attention(av):
        att_weights_tensor, v_tensor = av
        return tf.matmul(att_weights_tensor, v_tensor)
    
    att_output = Lambda(apply_attention, name=f'{name}_att_output')([att_weights, v])
    
    # 转置并重塑回: (batch, seq_len, total_dim)
    def transpose_and_reshape(tensor):
        # tensor: (batch, num_heads, seq_len, head_dim)
        # transpose: (batch, seq_len, num_heads, head_dim)
        # reshape: (batch, seq_len, total_dim)
        tensor = tf.transpose(tensor, [0, 2, 1, 3])
        return tf.reshape(tensor, (-1, tf.shape(tensor)[1], total_dim))
    
    att_output = Lambda(transpose_and_reshape, name=f'{name}_att_concat')(att_output)
    
    # 输出投影
    output = Dense(input_dim, name=f'{name}_att_proj')(att_output)
    
    # 残差连接和层归一化
    output = Add(name=f'{name}_att_residual')([x, output])
    output = BatchNormalization(name=f'{name}_att_norm')(output)
    
    return output

def residual_block(x, filters, kernel_size=3, stride=1, block_name='', use_se=False):
    """
    增强的残差块（Residual Block with optional SE-Net）
    
    参数:
    x: 输入张量
    filters: 卷积核数量
    kernel_size: 卷积核大小
    stride: 步长
    block_name: 块名称
    use_se: 是否使用SE-Net注意力
    """
    # 主路径
    shortcut = x
    
    # 第一个卷积层
    x = Conv1D(filters, kernel_size, strides=stride, padding='same', 
               name=f'{block_name}_conv1')(x)
    x = BatchNormalization(name=f'{block_name}_bn1')(x)
    x = Activation('relu', name=f'{block_name}_relu1')(x)
    
    # Dropout
    x = Dropout(0.2, name=f'{block_name}_dropout1')(x)
    
    # 第二个卷积层
    x = Conv1D(filters, kernel_size, strides=1, padding='same', 
               name=f'{block_name}_conv2')(x)
    x = BatchNormalization(name=f'{block_name}_bn2')(x)
    
    # 可选：SE-Net注意力
    if use_se:
        x = se_block(x, filters, block_name=f'{block_name}_se')
    
    # 如果维度不匹配，需要调整shortcut
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, strides=stride, padding='same', 
                         name=f'{block_name}_shortcut_conv')(shortcut)
        shortcut = BatchNormalization(name=f'{block_name}_shortcut_bn')(shortcut)
    
    # 残差连接
    x = Add(name=f'{block_name}_add')([x, shortcut])
    x = Activation('relu', name=f'{block_name}_relu2')(x)
    
    return x

def create_resnet_model(vocab_size, embedding_dim=128, max_length=100, enhanced=True):
    """
    创建ResNet文本分类模型（增强版）
    
    参数:
    vocab_size: 词汇表大小
    embedding_dim: 词嵌入维度
    max_length: 序列最大长度
    enhanced: 是否使用增强功能（多头注意力、SE-Net、多尺度卷积等）
    """
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Bidirectional, LSTM, Concatenate
    
    # 输入层
    inputs = Input(shape=(max_length,), name='input')
    
    # 词嵌入层（增加维度以支持更复杂的特征）
    embedding_dim_enhanced = embedding_dim * 2 if enhanced else embedding_dim
    x = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim_enhanced,
        input_length=max_length,
        name='embedding'
    )(inputs)
    
    if enhanced:
        # 增强版：添加多头自注意力
        print("  使用多头自注意力机制...")
        x = multi_head_self_attention(x, num_heads=4, head_dim=32, name='att1')
        x = Dropout(0.2, name='att1_dropout')(x)
    
    # 初始卷积层（使用多尺度卷积）
    if enhanced:
        print("  使用多尺度卷积块...")
        x = multi_scale_conv_block(x, filters=32, block_name='initial_ms')
        # 将多尺度输出投影到统一维度
        x = Conv1D(128, 1, padding='same', name='initial_proj')(x)
    else:
        x = Conv1D(64, 7, strides=2, padding='same', name='initial_conv')(x)
    
    x = BatchNormalization(name='initial_bn')(x)
    x = Activation('relu', name='initial_relu')(x)
    x = Dropout(0.2, name='initial_dropout')(x)
    
    # 残差块1 (128 filters)
    x = residual_block(x, filters=128, kernel_size=3, stride=1, block_name='res_block1', use_se=enhanced)
    x = residual_block(x, filters=128, kernel_size=3, stride=1, block_name='res_block2', use_se=enhanced)
    
    if enhanced:
        # 添加额外的残差块
        x = residual_block(x, filters=128, kernel_size=3, stride=1, block_name='res_block2_5', use_se=True)
    
    # 残差块2 (256 filters)
    x = residual_block(x, filters=256, kernel_size=3, stride=2, block_name='res_block3', use_se=enhanced)
    x = residual_block(x, filters=256, kernel_size=3, stride=1, block_name='res_block4', use_se=enhanced)
    
    if enhanced:
        # 添加额外的残差块
        x = residual_block(x, filters=256, kernel_size=3, stride=1, block_name='res_block4_5', use_se=True)
    
    # 残差块3 (512 filters) - 增强版使用更大的filters
    final_filters = 512 if enhanced else 256
    x = residual_block(x, filters=final_filters, kernel_size=3, stride=2, block_name='res_block5', use_se=enhanced)
    x = residual_block(x, filters=final_filters, kernel_size=3, stride=1, block_name='res_block6', use_se=enhanced)
    
    if enhanced:
        # 添加额外的残差块
        x = residual_block(x, filters=final_filters, kernel_size=3, stride=1, block_name='res_block6_5', use_se=True)
        x = residual_block(x, filters=final_filters, kernel_size=3, stride=1, block_name='res_block7', use_se=True)
    
    # 可选的LSTM层（增强版）
    if enhanced:
        print("  添加双向LSTM层...")
        # 双向LSTM捕获序列依赖
        lstm_out = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
                                 name='bilstm')(x)
        # 将LSTM输出与CNN输出融合
        x = Concatenate(axis=-1, name='cnn_lstm_concat')([x, lstm_out])
        # 投影到统一维度
        x = Conv1D(final_filters, 1, padding='same', name='fusion_proj')(x)
        x = BatchNormalization(name='fusion_bn')(x)
        x = Activation('relu', name='fusion_relu')(x)
    
    # 池化层（增强版使用注意力池化）
    if enhanced:
        print("  使用注意力池化...")
        x_pooled_att = attention_pooling(x, name='att_pool')
        x_pooled_max = GlobalMaxPooling1D(name='global_max_pool')(x)
        x_pooled_avg = GlobalAveragePooling1D(name='global_avg_pool')(x)
        # 融合多种池化方式
        x = Concatenate(axis=-1, name='pool_concat')([x_pooled_att, x_pooled_max, x_pooled_avg])
    else:
        x = GlobalMaxPooling1D(name='global_max_pool')(x)
    
    # 全连接层（增强版使用更深的网络）
    if enhanced:
        x = Dense(256, activation='relu', name='dense_1')(x)
        x = BatchNormalization(name='fc_bn1')(x)
        x = Dropout(0.5, name='fc_dropout1')(x)
        
        x = Dense(128, activation='relu', name='dense_2')(x)
        x = BatchNormalization(name='fc_bn2')(x)
        x = Dropout(0.4, name='fc_dropout2')(x)
        
        x = Dense(64, activation='relu', name='dense_3')(x)
        x = BatchNormalization(name='fc_bn3')(x)
        x = Dropout(0.3, name='fc_dropout3')(x)
    else:
        x = Dense(128, activation='relu', name='dense_1')(x)
        x = BatchNormalization(name='fc_bn1')(x)
        x = Dropout(0.5, name='fc_dropout1')(x)
        
        x = Dense(64, activation='relu', name='dense_2')(x)
        x = BatchNormalization(name='fc_bn2')(x)
        x = Dropout(0.3, name='fc_dropout2')(x)
    
    # 输出层
    outputs = Dense(1, activation='sigmoid', name='output')(x)
    
    # 创建模型
    model_name = 'Enhanced_ResNet_Text_Classifier' if enhanced else 'ResNet_Text_Classifier'
    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    
    # 编译模型（增强版使用更小的学习率）
    lr = 0.0005 if enhanced else 0.001
    try:
        optimizer = Adam(learning_rate=lr)
    except TypeError:
        optimizer = Adam(lr=lr)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

print("\n构建增强版ResNet模型（增强功能已启用）...")
print("增强功能包括：")
print("  ✓ 多头自注意力机制")
print("  ✓ SE-Net通道注意力")
print("  ✓ 多尺度卷积块")
print("  ✓ 注意力池化")
print("  ✓ 双向LSTM层")
print("  ✓ 更深的网络结构")
print("  ✓ 更大的embedding维度")
model = create_resnet_model(vocab_size, embedding_dim=128, max_length=max_length, enhanced=True)
model.summary()


# 4. 训练模型

# 划分训练集、验证集和测试集
# 为了能够映射回原始文本，我们需要保存索引
indices = np.arange(len(padded_sequences))

# 将target转换为数组格式，方便用于stratify
target_array = target.values if hasattr(target, 'values') else np.array(target)

# 第一步：先划分出测试集（25%）
train_val_idx, test_idx = train_test_split(
    indices,
    test_size=0.25,
    random_state=42,
    stratify=target_array  # 保持类别比例
)

# 第二步：从训练+验证集中划分出训练集和验证集
# 验证集占剩余数据的20%（即总数据的15%），训练集占剩余数据的80%（即总数据的60%）
train_val_target = target_array[train_val_idx]  # 获取train_val_idx对应的标签
train_idx, val_idx = train_test_split(
    train_val_idx,
    test_size=0.2,  # 占train_val_idx的20%，即总数据的15%
    random_state=42,
    stratify=train_val_target  # 保持类别比例
)

# 提取数据
X_train = padded_sequences[train_idx]
X_val = padded_sequences[val_idx]
X_test = padded_sequences[test_idx]

y_train = target.iloc[train_idx] if hasattr(target, 'iloc') else target[train_idx]
y_val = target.iloc[val_idx] if hasattr(target, 'iloc') else target[val_idx]
y_test = target.iloc[test_idx] if hasattr(target, 'iloc') else target[test_idx]

print(f"\n数据划分完成:")
print(f"  训练集: {len(X_train)} ({len(X_train)/len(padded_sequences)*100:.1f}%)")
print(f"  验证集: {len(X_val)} ({len(X_val)/len(padded_sequences)*100:.1f}%)")
print(f"  测试集: {len(X_test)} ({len(X_test)/len(padded_sequences)*100:.1f}%)")

# 选择典型样本用于动态可视化（垃圾邮件和正常邮件各选几个）
print("\n选择典型样本用于动态可视化...")
# 从验证集中选择典型样本（用于训练过程可视化，保持测试集独立性）
val_indices = np.arange(len(X_val))
y_val_array = y_val.values if hasattr(y_val, 'values') else np.array(y_val)
spam_indices = val_indices[y_val_array == 1]
ham_indices = val_indices[y_val_array == 0]

# 选择几个典型样本（垃圾邮件和正常邮件各选5个）
np.random.seed(42)
selected_spam_indices = np.random.choice(spam_indices, min(5, len(spam_indices)), replace=False)
selected_ham_indices = np.random.choice(ham_indices, min(5, len(ham_indices)), replace=False)
selected_indices = np.concatenate([selected_spam_indices, selected_ham_indices])

# 获取对应的文本和标签
selected_X = X_val[selected_indices]
selected_labels = y_val_array[selected_indices]
# 通过val_idx找到原始文本索引
original_indices = val_idx[selected_indices]
selected_texts = [texts.iloc[i] if hasattr(texts, 'iloc') else texts[i] for i in original_indices]

print(f"已选择 {len(selected_indices)} 个典型样本用于可视化（来自验证集）")
print(f"  垃圾邮件样本: {len(selected_spam_indices)} 个")
print(f"  正常邮件样本: {len(selected_ham_indices)} 个")

# 在训练前，用初始模型（随机权重）找出预测错误的样本
print("\n使用初始模型（随机权重）找出预测错误的样本...")
initial_model = create_resnet_model(vocab_size, embedding_dim=128, max_length=max_length, enhanced=True)
initial_predictions = initial_model.predict(X_val, verbose=0).flatten()
initial_pred_class = (initial_predictions > 0.5).astype(int)

# 找出预测错误的样本
wrong_predictions_mask = initial_pred_class != y_val_array
wrong_indices = np.where(wrong_predictions_mask)[0]

print(f"初始模型预测错误的样本数: {len(wrong_indices)} / {len(X_val)}")

# ========= 新增：强制挑选 2 个展示样本（Ham + Spam），且"初始预测明显错" =========
def pick_two_explain_samples(wrong_indices, y_test_array, initial_predictions, need_each_class=True):
    """
    从 initial_model 的错误样本里挑 2 个用于展示：
    - 一个 Ham（label=0）+ 一个 Spam（label=1）
    - 优先挑"错得更离谱"的（更容易展示从错到对的过程）
    """
    if len(wrong_indices) == 0:
        return None

    wrong_probs = initial_predictions[wrong_indices]
    wrong_labels = y_test_array[wrong_indices]

    # 对于真实 spam(1)：越接近0越"错得离谱"；对真实 ham(0)：越接近1越"错得离谱"
    differences = np.where(wrong_labels == 1, 1 - wrong_probs, wrong_probs)

    # 按离谱程度从大到小排序
    ranked = wrong_indices[np.argsort(differences)[::-1]]

    if not need_each_class:
        return ranked[:2] if len(ranked) >= 2 else ranked

    # 分别找 ham/spam 各 1 个
    ham_pick = None
    spam_pick = None
    for idx in ranked:
        if ham_pick is None and y_test_array[idx] == 0:
            ham_pick = idx
        if spam_pick is None and y_test_array[idx] == 1:
            spam_pick = idx
        if ham_pick is not None and spam_pick is not None:
            break

    if ham_pick is None or spam_pick is None:
        return None
    return np.array([ham_pick, spam_pick], dtype=int)

# 选两个解释样本（验证集索引空间 0..len(X_val)-1）
explain_indices = pick_two_explain_samples(wrong_indices, y_val_array, initial_predictions, need_each_class=True)

if explain_indices is None:
    print("警告：未能从初始错误预测中同时选到 Ham+Spam，将放宽条件从全验证集选择。")
    # 从全验证集选：同样按"错得离谱"挑 Ham/Spam（即使初始没错也尽量挑边界样本）
    all_idx = np.arange(len(X_val))
    all_probs = initial_predictions
    all_labels = y_val_array
    diffs_all = np.where(all_labels == 1, 1 - all_probs, all_probs)
    ranked_all = all_idx[np.argsort(diffs_all)[::-1]]
    # 强制各一类
    ham_pick = next((i for i in ranked_all if all_labels[i] == 0), None)
    spam_pick = next((i for i in ranked_all if all_labels[i] == 1), None)
    explain_indices = np.array([ham_pick, spam_pick], dtype=int)

# explain 两个样本（固定用于解释：token重要性、动态HTML）
explain_X = X_val[explain_indices]
explain_y = y_val_array[explain_indices]
explain_original_indices = val_idx[explain_indices]
explain_texts = [texts.iloc[i] if hasattr(texts, 'iloc') else texts[i] for i in explain_original_indices]
explain_initial_probs = initial_predictions[explain_indices]

print("\n✅ 已固定 2 个解释展示样本（强制 Ham+Spam）:")
for j, (txt, yy, p) in enumerate(zip(explain_texts, explain_y, explain_initial_probs), 1):
    lab = "Ham" if yy == 0 else "Spam"
    pred = "Spam" if p > 0.5 else "Ham"
    print(f"  样本{j}: 真={lab}, 初始预测={pred} (p_spam={p:.4f})  文本: {txt[:80]}...")

# 保留原来的 learning_samples 用于其他可视化（可选）
# 但解释性(token重要性/动态HTML)只对 explain 两个做，避免慢+不稳定
if len(wrong_indices) > 0:
    # 选择预测概率与真实标签差异最大的几个样本（这些是最难学习的）
    wrong_probs = initial_predictions[wrong_indices]
    wrong_labels = y_val_array[wrong_indices]
    # 计算预测概率与真实标签的差异（对于垃圾邮件，差异是1-prob；对于正常邮件，差异是prob）
    differences = np.where(wrong_labels == 1, 1 - wrong_probs, wrong_probs)
    # 选择差异最大的几个（最难学习的）
    top_wrong_indices = wrong_indices[np.argsort(differences)[-min(6, len(wrong_indices)):]]
    
    # 确保有垃圾邮件和正常邮件各几个
    top_wrong_labels = y_val_array[top_wrong_indices]
    spam_wrong = top_wrong_indices[top_wrong_labels == 1]
    ham_wrong = top_wrong_indices[top_wrong_labels == 0]
    
    # 选择3个垃圾邮件和3个正常邮件（如果都有的话）
    np.random.seed(42)
    if len(spam_wrong) > 0:
        selected_spam_wrong = np.random.choice(spam_wrong, min(3, len(spam_wrong)), replace=False)
    else:
        selected_spam_wrong = np.array([])
    
    if len(ham_wrong) > 0:
        selected_ham_wrong = np.random.choice(ham_wrong, min(3, len(ham_wrong)), replace=False)
    else:
        selected_ham_wrong = np.array([])
    
    learning_samples_indices = np.concatenate([selected_spam_wrong, selected_ham_wrong]) if len(selected_spam_wrong) > 0 and len(selected_ham_wrong) > 0 else (selected_spam_wrong if len(selected_spam_wrong) > 0 else selected_ham_wrong)
    
    learning_samples_X = X_val[learning_samples_indices]
    learning_samples_labels = y_val_array[learning_samples_indices]
    learning_samples_original_indices = val_idx[learning_samples_indices]
    learning_samples_texts = [texts.iloc[i] if hasattr(texts, 'iloc') else texts[i] for i in learning_samples_original_indices]
    learning_samples_initial_probs = initial_predictions[learning_samples_indices]
    
    print(f"\n选择了 {len(learning_samples_indices)} 个初始预测错误的样本用于详细跟踪（可选）:")
    for i, (text, label, prob) in enumerate(zip(learning_samples_texts, learning_samples_labels, learning_samples_initial_probs)):
        label_str = "垃圾邮件" if label == 1 else "正常邮件"
        pred_str = "垃圾邮件" if prob > 0.5 else "正常邮件"
        print(f"  样本 {i+1} ({label_str}): 初始预测={pred_str} (概率={prob:.4f})")
        print(f"    文本: {text[:80]}...")
else:
    print("警告：初始模型没有预测错误的样本，将使用随机选择的样本")
    learning_samples_indices = selected_indices[:6]
    learning_samples_X = X_val[learning_samples_indices]
    learning_samples_labels = y_val_array[learning_samples_indices]
    learning_samples_original_indices = val_idx[learning_samples_indices]
    learning_samples_texts = [texts.iloc[i] if hasattr(texts, 'iloc') else texts[i] for i in learning_samples_original_indices]
    learning_samples_initial_probs = initial_predictions[learning_samples_indices]

# 删除初始模型以释放内存
del initial_model

# 自定义回调类：记录每个epoch的预测结果
class TrainingVisualizationCallback(Callback):
    """记录训练过程中每个epoch的预测结果，用于动态可视化（并对 explain 样本做 token 重要性）"""

    def __init__(self, X_sample, y_sample, texts_sample,
                 X_test_full, y_test_full,
                 explain_X, explain_y, explain_texts):
        super().__init__()
        self.X_sample = X_sample
        self.y_sample = y_sample
        self.texts_sample = texts_sample

        self.X_test_full = X_test_full
        self.y_test_full = y_test_full

        # 仅用于解释的 2 个样本（强制 Ham+Spam）
        self.explain_X = explain_X
        self.explain_y = explain_y
        self.explain_texts = explain_texts

        self.epoch_predictions = []          # 典型样本预测（每epoch）
        self.epoch_confusion_matrices = []   # 每epoch混淆矩阵
        self.epoch_metrics = []              # 每epoch指标

        # explain 样本：预测概率 & token重要性（包含 epoch0 初始）
        self.explain_epoch_probs = []        # shape: (epochs+1, 2)
        self.explain_epoch_token_scores = [] # list of epochs+1, each: [scores_for_sample1, scores_for_sample2]

        self._probe = None
        self._emb_layer = None

    def _build_probe_once(self):
        if self._probe is not None:
            return
        from tensorflow.keras.models import Model
        self._emb_layer = self.model.get_layer("embedding")
        self._probe = Model(inputs=self.model.inputs,
                            outputs=[self._emb_layer.output, self.model.output])

    def _grad_x_emb_scores(self, x_ids_1d):
        """对单个样本计算 Gradient×Embedding，返回非padding部分scores"""
        self._build_probe_once()

        x = tf.convert_to_tensor(x_ids_1d[None, :], dtype=tf.int32)
        with tf.GradientTape() as tape:
            emb_out, y = self._probe(x, training=False)
            tape.watch(emb_out)
            score = y[0, 0]
        grads = tape.gradient(score, emb_out)
        token_scores_full = tf.reduce_sum(tf.abs(grads * emb_out), axis=-1)[0].numpy()

        nonpad_mask = (x_ids_1d != 0)
        L = int(np.sum(nonpad_mask))
        return token_scores_full[:L].tolist(), float(y.numpy()[0, 0])

    def _record_one_epoch(self, epoch_idx, logs=None):
        # 典型样本预测
        sample_pred = self.model.predict(self.X_sample, verbose=0).flatten()
        self.epoch_predictions.append(sample_pred)

        # 测试集混淆矩阵
        test_pred_prob = self.model.predict(self.X_test_full, verbose=0).flatten()
        test_pred_class = (test_pred_prob > 0.5).astype("int32")
        cm = metrics.confusion_matrix(self.y_test_full, test_pred_class)

        acc = metrics.accuracy_score(self.y_test_full, test_pred_class)
        f1 = metrics.f1_score(self.y_test_full, test_pred_class)

        self.epoch_confusion_matrices.append(cm)
        self.epoch_metrics.append({
            "epoch": epoch_idx,
            "accuracy": acc,
            "f1": f1,
            "val_loss": 0 if logs is None else float(logs.get("val_loss", 0)),
            "val_accuracy": 0 if logs is None else float(logs.get("val_accuracy", 0)),
        })

        # explain 两个样本：预测+token重要性
        epoch_scores = []
        epoch_probs = []
        for s in range(len(self.explain_X)):
            scores, prob = self._grad_x_emb_scores(self.explain_X[s])
            epoch_scores.append(scores)
            epoch_probs.append(prob)

        self.explain_epoch_token_scores.append(epoch_scores)
        self.explain_epoch_probs.append(epoch_probs)

        # 打印 explain 样本当前是否正确（方便你观察"前错后对"）
        pred_cls = (np.array(epoch_probs) > 0.5).astype(int)
        correct = np.sum(pred_cls == self.explain_y)
        print(f"  [Explain] Epoch {epoch_idx}: acc={acc:.4f}, f1={f1:.4f}, explain正确 {correct}/{len(self.explain_y)}")

    def on_train_begin(self, logs=None):
        # 记录 epoch0（初始权重）
        self._record_one_epoch(epoch_idx=0, logs={"val_loss": np.nan, "val_accuracy": np.nan})

    def on_epoch_end(self, epoch, logs=None):
        # epoch 从 0 开始，这里我们用 epoch+1 表示第几个训练后状态
        self._record_one_epoch(epoch_idx=epoch + 1, logs=logs)

# 创建可视化回调（使用验证集进行训练过程可视化）
viz_callback = TrainingVisualizationCallback(
    X_sample=selected_X,
    y_sample=selected_labels,
    texts_sample=selected_texts,
    X_test_full=X_val,  # 使用验证集进行训练过程可视化
    y_test_full=y_val_array,  # 使用验证集标签
    explain_X=explain_X,
    explain_y=explain_y,
    explain_texts=explain_texts
)

# 设置模型保存路径
model_file = os.path.join(current_dir, 'resnet_model.h5')
checkpoint = ModelCheckpoint(
    model_file,
    monitor='val_loss',
    verbose=1,
    save_best_only=True
)

# 学习率衰减
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1,
    min_lr=1e-6
)

# 早停
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    restore_best_weights=True
)

# 训练模型
print("\n开始训练ResNet模型...")
print("使用验证集进行模型选择和早停，测试集仅用于最终评估")
history = model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=32,
    validation_data=(X_val, y_val),  # 使用验证集而不是测试集
    verbose=1,
    callbacks=[checkpoint, reduce_lr, early_stopping, viz_callback]
)
print("✓ 模型训练完成")

# 检查 explain 样本的学习过程是否符合"先错后对"的模式
if 'explain_y' in locals() and 'explain_initial_probs' in locals() and hasattr(viz_callback, 'explain_epoch_probs'):
    print("\n检查 explain 样本的学习过程（确保'先错后对'模式）...")
    n_epochs = len(viz_callback.explain_epoch_probs)
    
    for sample_idx in range(len(explain_y)):
        true_label = explain_y[sample_idx]
        initial_prob = explain_initial_probs[sample_idx]
        initial_pred = 1 if initial_prob > 0.5 else 0
        initial_correct = (initial_pred == true_label)
        
        # 获取每个epoch的预测
        epoch_predictions = []
        for epoch_idx in range(1, n_epochs):  # 跳过epoch0（初始）
            if epoch_idx < len(viz_callback.explain_epoch_probs):
                epoch_prob = viz_callback.explain_epoch_probs[epoch_idx][sample_idx]
                epoch_pred = 1 if epoch_prob > 0.5 else 0
                epoch_correct = (epoch_pred == true_label)
                epoch_predictions.append(epoch_correct)
        
        # 检查是否符合"先错后对"模式
        # 要求：前K个epoch错，后M个epoch对（K=2, M=2）
        K = 2  # 前K个epoch应该错
        M = 2  # 后M个epoch应该对
        
        if len(epoch_predictions) >= K + M:
            first_K_wrong = all(not epoch_predictions[i] for i in range(min(K, len(epoch_predictions))))
            last_M_correct = all(epoch_predictions[i] for i in range(max(0, len(epoch_predictions) - M), len(epoch_predictions)))
            
            if not initial_correct and first_K_wrong and last_M_correct:
                print(f"  ✓ 样本 {sample_idx + 1} ({'Spam' if true_label == 1 else 'Ham'}): 符合'先错后对'模式")
            else:
                print(f"  ⚠ 样本 {sample_idx + 1} ({'Spam' if true_label == 1 else 'Ham'}): 不符合'先错后对'模式")
                print(f"     初始预测: {'正确' if initial_correct else '错误'}")
                print(f"     前{K}个epoch: {'正确' if not first_K_wrong else '错误'}")
                print(f"     后{M}个epoch: {'正确' if last_M_correct else '错误'}")
        else:
            print(f"  ⚠ 样本 {sample_idx + 1}: epoch数不足，无法检查'先错后对'模式")


# 5. 可视化训练过程

def plot_learning_curves(history, output_file=None):
    """
    绘制学习曲线并保存为图片
    
    参数:
    history: 训练历史对象
    output_file: 输出文件路径
    """
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    
    # Loss曲线
    ax[0].plot(history.history['loss'], label='训练损失', linewidth=2)
    ax[0].plot(history.history['val_loss'], label='验证损失', linewidth=2)
    ax[0].legend(fontsize=14)
    ax[0].set_xlabel('Epoch', fontsize=14)
    ax[0].set_ylabel('Loss', fontsize=14)
    ax[0].set_title('Loss vs Val Loss', fontsize=16, fontweight='bold')
    ax[0].grid(True, alpha=0.3)
    
    # Accuracy曲线
    ax[1].plot(history.history['accuracy'], label='训练准确率', linewidth=2)
    ax[1].plot(history.history['val_accuracy'], label='验证准确率', linewidth=2)
    ax[1].legend(fontsize=14)
    ax[1].set_xlabel('Epoch', fontsize=14)
    ax[1].set_ylabel('Accuracy', fontsize=14)
    ax[1].set_title('Accuracy vs Val Accuracy', fontsize=16, fontweight='bold')
    ax[1].grid(True, alpha=0.3)
    
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

learning_curves_file = os.path.join(output_dir, "learning_curves_resnet.png")
plot_learning_curves(history, output_file=learning_curves_file)

# 5.5 动态可视化训练过程
def plot_training_evolution(viz_callback, selected_texts, selected_labels, output_dir, 
                            X_test, y_test_array, model):
    """
    绘制训练过程的动态可视化，展示模型如何逐步学会分类
    
    参数:
    viz_callback: 包含训练历史记录的回调对象
    selected_texts: 典型样本的文本列表
    selected_labels: 典型样本的真实标签
    output_dir: 输出目录
    X_test: 测试集数据
    y_test_array: 测试集标签
    model: 训练好的模型（已restore_best_weights）
    """
    if not viz_callback.epoch_predictions:
        print("警告：没有记录到训练历史，跳过动态可视化")
        return
    
    epochs = list(range(1, len(viz_callback.epoch_predictions) + 1))
    n_samples = len(selected_texts)
    
    # 图1: 典型样本的预测概率随epoch的变化
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # 上子图：所有样本的预测概率变化
    ax1 = axes[0]
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, n_samples))
    
    for i in range(n_samples):
        probs = [pred[i] for pred in viz_callback.epoch_predictions]
        label_type = "垃圾邮件" if selected_labels[i] == 1 else "正常邮件"
        # 截断文本用于显示
        text_preview = selected_texts[i][:50] + "..." if len(selected_texts[i]) > 50 else selected_texts[i]
        ax1.plot(epochs, probs, marker='o', linewidth=2, markersize=6, 
                color=colors[i], label=f'{label_type}: {text_preview}', alpha=0.7)
    
    ax1.axhline(y=0.5, color='r', linestyle='--', linewidth=1, alpha=0.5, label='分类阈值 (0.5)')
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('预测概率 (Spam概率)', fontsize=14)
    ax1.set_title('典型样本预测概率随训练过程的变化', fontsize=16, fontweight='bold', pad=20)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # 下子图：按类别分组显示平均预测概率
    ax2 = axes[1]
    spam_indices = np.where(selected_labels == 1)[0]
    ham_indices = np.where(selected_labels == 0)[0]
    
    if len(spam_indices) > 0:
        spam_probs_avg = [np.mean([pred[i] for i in spam_indices]) 
                          for pred in viz_callback.epoch_predictions]
        ax2.plot(epochs, spam_probs_avg, marker='s', linewidth=3, markersize=8, 
                color='red', label='垃圾邮件平均预测概率', alpha=0.8)
    
    if len(ham_indices) > 0:
        ham_probs_avg = [np.mean([pred[i] for i in ham_indices]) 
                        for pred in viz_callback.epoch_predictions]
        ax2.plot(epochs, ham_probs_avg, marker='^', linewidth=3, markersize=8, 
                color='green', label='正常邮件平均预测概率', alpha=0.8)
    
    ax2.axhline(y=0.5, color='black', linestyle='--', linewidth=1, alpha=0.5, label='分类阈值 (0.5)')
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('平均预测概率', fontsize=14)
    ax2.set_title('按类别分组的平均预测概率变化', fontsize=16, fontweight='bold', pad=20)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    evolution_file = os.path.join(output_dir, "training_evolution_samples.png")
    plt.savefig(evolution_file, dpi=300, bbox_inches='tight')
    print(f"典型样本预测概率变化图已保存为 {evolution_file}")
    plt.close()
    
    # 图2: 混淆矩阵随epoch的变化（展示最后一个epoch）
    n_epochs_done = len(viz_callback.epoch_confusion_matrices)  # 这里包含 epoch0 + epoch1..epochN
    if n_epochs_done > 0 and viz_callback.epoch_metrics:
        # epoch_confusion_matrices 包含 epoch0（索引0），所以最后一个训练epoch是 n_epochs_done - 1
        last_training_epoch = n_epochs_done - 1  # 最后一个训练epoch（不包含epoch0）
        
        # 找 best epoch（val_loss 最小的 epoch）
        valid_rows = [m for m in viz_callback.epoch_metrics if m["epoch"] >= 1 and not np.isnan(m["val_loss"])]
        if valid_rows:
            best_row = min(valid_rows, key=lambda x: x["val_loss"])
            best_epoch = int(best_row["epoch"])
            best_acc = best_row["accuracy"]
        else:
            best_epoch = last_training_epoch
            best_row = next((m for m in viz_callback.epoch_metrics if m["epoch"] == best_epoch), None)
            best_acc = best_row["accuracy"] if best_row else 0.0
        
        # 选择几个关键epoch进行展示：开始、中间、最后（训练epoch，不包含epoch0）
        key_epochs = [1, max(1, last_training_epoch // 2), last_training_epoch]
        key_epochs = sorted(list(set(key_epochs)))  # 去重并排序

        # 需要展示的面板：关键epoch + 最终模型（best epoch，因为 restore_best_weights=True）
        panels = []
        for ep in key_epochs:
            if ep <= last_training_epoch:
                mrow = next((m for m in viz_callback.epoch_metrics if m["epoch"] == ep), None)
                acc_show = mrow["accuracy"] if mrow else 0.0
                panels.append((f"Epoch {ep}", ep, viz_callback.epoch_confusion_matrices[ep], acc_show))
        
        # 添加最终模型（使用 best epoch 的混淆矩阵，因为 restore_best_weights=True）
        panels.append(("Final Model (Best)", best_epoch, viz_callback.epoch_confusion_matrices[best_epoch], best_acc))

        fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5))
        if len(panels) == 1:
            axes = [axes]
        
        for i, (tag, ep, cm_, acc_show) in enumerate(panels):
            sns.heatmap(cm_, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                        xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'],
                        cbar=False)
            axes[i].set_title(f'{tag}\nAcc={acc_show:.3f}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Pred')
            axes[i].set_ylabel('True')

        plt.suptitle(f'混淆矩阵随训练过程的变化（训练停止epoch={last_training_epoch}）',
                     fontsize=16, fontweight='bold', y=1.05)
        plt.tight_layout()
        cm_evolution_file = os.path.join(output_dir, "training_evolution_confusion_matrix.png")
        plt.savefig(cm_evolution_file, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵变化图已保存: {cm_evolution_file}")
        plt.close()
    
    # 图3: 整体指标随epoch的变化
    if viz_callback.epoch_metrics:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        epochs_list = [m['epoch'] for m in viz_callback.epoch_metrics]
        accuracies = [m['accuracy'] for m in viz_callback.epoch_metrics]
        f1_scores = [m['f1'] for m in viz_callback.epoch_metrics]
        val_losses = [m['val_loss'] for m in viz_callback.epoch_metrics]
        val_accuracies = [m['val_accuracy'] for m in viz_callback.epoch_metrics]
        
        # 测试准确率
        axes[0, 0].plot(epochs_list, accuracies, marker='o', linewidth=2, color='blue')
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('测试准确率', fontsize=12)
        axes[0, 0].set_title('测试准确率变化', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 1])
        
        # F1分数
        axes[0, 1].plot(epochs_list, f1_scores, marker='s', linewidth=2, color='green')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('F1分数', fontsize=12)
        axes[0, 1].set_title('F1分数变化', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1])
        
        # 验证损失
        axes[1, 0].plot(epochs_list, val_losses, marker='^', linewidth=2, color='red')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('验证损失', fontsize=12)
        axes[1, 0].set_title('验证损失变化', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 验证准确率
        axes[1, 1].plot(epochs_list, val_accuracies, marker='d', linewidth=2, color='purple')
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('验证准确率', fontsize=12)
        axes[1, 1].set_title('验证准确率变化', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 1])
        
        plt.suptitle('模型性能指标随训练过程的变化', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        metrics_file = os.path.join(output_dir, "training_evolution_metrics.png")
        plt.savefig(metrics_file, dpi=300, bbox_inches='tight')
        print(f"性能指标变化图已保存为 {metrics_file}")
        plt.close()
    
    # 图4: 预测概率分布随epoch的变化（直方图）
    if len(viz_callback.epoch_predictions) > 0:
        key_epochs = [0, len(viz_callback.epoch_predictions) // 2, len(viz_callback.epoch_predictions) - 1]
        key_epochs = sorted(list(set(key_epochs)))
        
        fig, axes = plt.subplots(1, len(key_epochs), figsize=(6 * len(key_epochs), 5))
        if len(key_epochs) == 1:
            axes = [axes]
        
        for idx, epoch_idx in enumerate(key_epochs):
            # 对完整测试集的预测（需要重新计算或从历史中获取）
            # 这里我们使用典型样本的预测来展示趋势
            probs = viz_callback.epoch_predictions[epoch_idx]
            
            axes[idx].hist(probs[selected_labels == 1], bins=20, alpha=0.6, 
                          color='red', label='垃圾邮件', density=True)
            axes[idx].hist(probs[selected_labels == 0], bins=20, alpha=0.6, 
                          color='green', label='正常邮件', density=True)
            axes[idx].axvline(x=0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
            axes[idx].set_xlabel('预测概率', fontsize=12)
            axes[idx].set_ylabel('密度', fontsize=12)
            axes[idx].set_title(f'Epoch {epoch_idx + 1} 预测概率分布', fontsize=14, fontweight='bold')
            axes[idx].legend(fontsize=10)
            axes[idx].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('预测概率分布随训练过程的变化', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        dist_file = os.path.join(output_dir, "training_evolution_distribution.png")
        plt.savefig(dist_file, dpi=300, bbox_inches='tight')
        print(f"预测概率分布变化图已保存为 {dist_file}")
        plt.close()
    
    print("\n✓ 动态可视化图表生成完成！")

print("\n生成训练过程动态可视化...")
# 使用验证集进行训练过程可视化（保持测试集独立性）
plot_training_evolution(viz_callback, selected_texts, selected_labels, output_dir, 
                       X_val, y_val_array, model)

# 5.6 详细展示初始预测错误样本的学习过程
def plot_individual_learning_process(viz_callback, learning_texts, learning_labels, 
                                     initial_probs, tokenizer, output_dir):
    """
    详细展示初始预测错误的样本如何逐步学习到正确分类
    
    参数:
    viz_callback: 包含训练历史记录的回调对象
    learning_texts: 学习样本的文本列表
    learning_labels: 学习样本的真实标签
    initial_probs: 初始预测概率
    tokenizer: 用于将序列转换回文本的tokenizer
    output_dir: 输出目录
    """
    if not viz_callback.epoch_learning_predictions:
        print("警告：没有记录到学习样本的训练历史，跳过详细可视化")
        return
    
    epochs = list(range(0, len(viz_callback.epoch_learning_predictions) + 1))  # 包括初始epoch (0)
    n_samples = len(learning_texts)
    
    # 为每个样本创建详细的学习过程图
    for sample_idx in range(n_samples):
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 2, 1], hspace=0.3)
        
        # 上部分：文本内容和标签信息
        ax_text = fig.add_subplot(gs[0])
        ax_text.axis('off')
        true_label = "垃圾邮件 (Spam)" if learning_labels[sample_idx] == 1 else "正常邮件 (Ham)"
        text_content = learning_texts[sample_idx]
        
        # 显示文本（如果太长则截断）
        display_text = text_content if len(text_content) <= 200 else text_content[:200] + "..."
        ax_text.text(0.05, 0.9, f"样本 {sample_idx + 1} - 真实标签: {true_label}", 
                    fontsize=14, fontweight='bold', transform=ax_text.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax_text.text(0.05, 0.7, "文本内容:", fontsize=12, fontweight='bold', 
                    transform=ax_text.transAxes)
        ax_text.text(0.05, 0.5, display_text, fontsize=10, 
                    transform=ax_text.transAxes, wrap=True,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # 中间部分：预测概率变化曲线
        ax_prob = fig.add_subplot(gs[1])
        
        # 添加初始预测概率
        all_probs = [initial_probs[sample_idx]] + [pred[sample_idx] for pred in viz_callback.epoch_learning_predictions]
        all_epochs = [0] + epochs[1:]
        
        # 绘制预测概率曲线
        ax_prob.plot(all_epochs, all_probs, marker='o', linewidth=3, markersize=8, 
                    color='blue', label='预测概率 (Spam概率)')
        
        # 标注每个epoch的预测是否正确
        for epoch_idx, prob in enumerate(all_probs):
            pred_class = 1 if prob > 0.5 else 0
            is_correct = pred_class == learning_labels[sample_idx]
            color = 'green' if is_correct else 'red'
            marker = 'o' if is_correct else 'x'  # 使用标准marker样式：'o'表示正确，'x'表示错误
            ax_prob.scatter(all_epochs[epoch_idx], prob, s=200, c=color, 
                           marker=marker, zorder=5, edgecolors='black', linewidths=1)
        
        # 添加分类阈值线
        ax_prob.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='分类阈值 (0.5)')
        
        # 添加目标区域（根据真实标签）
        if learning_labels[sample_idx] == 1:
            ax_prob.axhspan(0.5, 1.0, alpha=0.1, color='red', label='目标区域 (Spam)')
        else:
            ax_prob.axhspan(0.0, 0.5, alpha=0.1, color='green', label='目标区域 (Ham)')
        
        ax_prob.set_xlabel('Epoch', fontsize=14)
        ax_prob.set_ylabel('预测概率 (Spam概率)', fontsize=14)
        ax_prob.set_title(f'样本 {sample_idx + 1} 的学习过程：从错误到正确', 
                         fontsize=16, fontweight='bold', pad=20)
        ax_prob.legend(fontsize=12, loc='best')
        ax_prob.grid(True, alpha=0.3)
        ax_prob.set_ylim([0, 1])
        ax_prob.set_xlim([-0.5, max(all_epochs) + 0.5])
        
        # 下部分：预测状态表格
        ax_table = fig.add_subplot(gs[2])
        ax_table.axis('off')
        
        # 创建表格数据
        table_data = []
        table_data.append(['Epoch', '预测概率', '预测类别', '是否正确', '与目标差距'])
        
        for epoch_idx, prob in enumerate(all_probs):
            pred_class = 1 if prob > 0.5 else 0
            pred_class_str = "Spam" if pred_class == 1 else "Ham"
            is_correct = pred_class == learning_labels[sample_idx]
            is_correct_str = "✓ 正确" if is_correct else "✗ 错误"
            target = 1.0 if learning_labels[sample_idx] == 1 else 0.0
            gap = abs(prob - target)
            table_data.append([f"Epoch {all_epochs[epoch_idx]}", f"{prob:.4f}", 
                              pred_class_str, is_correct_str, f"{gap:.4f}"])
        
        table = ax_table.table(cellText=table_data[1:], colLabels=table_data[0],
                              cellLoc='center', loc='center',
                              colWidths=[0.15, 0.2, 0.15, 0.2, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 设置表格样式
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 根据是否正确设置行颜色
        for i in range(1, len(table_data)):
            is_correct = "正确" in table_data[i][3]
            color = '#C8E6C9' if is_correct else '#FFCDD2'
            for j in range(len(table_data[0])):
                table[(i, j)].set_facecolor(color)
        
        plt.suptitle(f'样本 {sample_idx + 1} 详细学习过程分析', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # 保存图片
        individual_file = os.path.join(output_dir, f"learning_process_sample_{sample_idx + 1}.png")
        plt.savefig(individual_file, dpi=300, bbox_inches='tight')
        print(f"样本 {sample_idx + 1} 学习过程图已保存为 {individual_file}")
        plt.close()
    
    # 创建汇总图：所有学习样本的对比
    fig, axes = plt.subplots(n_samples, 1, figsize=(16, 4 * n_samples))
    if n_samples == 1:
        axes = [axes]
    
    for sample_idx in range(n_samples):
        all_probs = [initial_probs[sample_idx]] + [pred[sample_idx] for pred in viz_callback.epoch_learning_predictions]
        all_epochs = [0] + epochs[1:]
        
        axes[sample_idx].plot(all_epochs, all_probs, marker='o', linewidth=2, markersize=6, 
                            color='blue', label='预测概率')
        axes[sample_idx].axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        # 标注正确/错误
        for epoch_idx, prob in enumerate(all_probs):
            pred_class = 1 if prob > 0.5 else 0
            is_correct = pred_class == learning_labels[sample_idx]
            color = 'green' if is_correct else 'red'
            marker = 'o' if is_correct else 'x'
            axes[sample_idx].scatter(all_epochs[epoch_idx], prob, s=100, c=color, 
                                   marker=marker, zorder=5, edgecolors='black', linewidths=0.5)
        
        true_label_str = "垃圾邮件" if learning_labels[sample_idx] == 1 else "正常邮件"
        text_preview = learning_texts[sample_idx][:60] + "..." if len(learning_texts[sample_idx]) > 60 else learning_texts[sample_idx]
        axes[sample_idx].set_title(f'样本 {sample_idx + 1} ({true_label_str}): {text_preview}', 
                                  fontsize=12, fontweight='bold')
        axes[sample_idx].set_ylabel('预测概率', fontsize=10)
        axes[sample_idx].grid(True, alpha=0.3)
        axes[sample_idx].set_ylim([0, 1])
        axes[sample_idx].legend(fontsize=9)
    
    axes[-1].set_xlabel('Epoch', fontsize=12)
    plt.suptitle('所有初始预测错误样本的学习过程对比', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    summary_file = os.path.join(output_dir, "learning_process_summary.png")
    plt.savefig(summary_file, dpi=300, bbox_inches='tight')
    print(f"学习过程汇总图已保存为 {summary_file}")
    plt.close()
    
    print(f"\n✓ 已生成 {n_samples} 个样本的详细学习过程可视化！")

def plot_token_level_analysis(viz_callback, learning_texts, learning_labels, 
                              learning_X, tokenizer, output_dir):
    """
    展示token级别的分析，包括token序列和在不同epoch的变化
    
    参数:
    viz_callback: 包含训练历史记录的回调对象
    learning_texts: 学习样本的文本列表
    learning_labels: 学习样本的真实标签
    learning_X: 学习样本的序列数据
    tokenizer: tokenizer对象
    output_dir: 输出目录
    """
    if not viz_callback.epoch_learning_predictions:
        return
    
    n_samples = len(learning_texts)
    
    # 为每个样本创建token级别的可视化
    for sample_idx in range(n_samples):
        # 将序列转换回token
        sequence = learning_X[sample_idx].copy()
        # 移除padding (0值)
        sequence = sequence[sequence != 0]
        
        # 将token ID转换回单词
        reverse_word_index = {}
        if hasattr(tokenizer, 'word_index'):
            reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
        
        tokens = []
        for token_id in sequence:
            if token_id in reverse_word_index:
                tokens.append(reverse_word_index[token_id])
            elif token_id == 0:
                continue  # 跳过padding
            else:
                tokens.append(f'<UNK_{token_id}>')
        
        # 如果token太多，只显示前50个
        if len(tokens) > 50:
            tokens = tokens[:50]
            tokens.append('...')
        
        # 如果无法从tokenizer恢复，使用原始文本分词
        if len(tokens) == 0 or all(t.startswith('<UNK_') for t in tokens):
            # 使用原始文本，简单分词
            text = learning_texts[sample_idx]
            tokens = text.split()[:50]
            if len(text.split()) > 50:
                tokens.append('...')
        
        # 创建图表
        fig, axes = plt.subplots(2, 1, figsize=(18, 10))
        
        # 上部分：token序列展示
        ax_tokens = axes[0]
        ax_tokens.axis('off')
        
        true_label = "垃圾邮件 (Spam)" if learning_labels[sample_idx] == 1 else "正常邮件 (Ham)"
        ax_tokens.text(0.05, 0.95, f"样本 {sample_idx + 1} - Token级别分析", 
                      fontsize=16, fontweight='bold', transform=ax_tokens.transAxes)
        ax_tokens.text(0.05, 0.90, f"真实标签: {true_label}", 
                      fontsize=14, transform=ax_tokens.transAxes)
        ax_tokens.text(0.05, 0.85, f"Token数量: {len(sequence)}", 
                      fontsize=12, transform=ax_tokens.transAxes)
        
        # 显示token序列（每行显示10个token）
        token_text = "Token序列: "
        for i, token in enumerate(tokens):
            if i > 0 and i % 10 == 0:
                token_text += "\n" + " " * 13
            token_text += f"{token}  "
        
        ax_tokens.text(0.05, 0.70, token_text, fontsize=10, 
                      transform=ax_tokens.transAxes,
                      family='monospace',
                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3),
                      verticalalignment='top')
        
        # 下部分：预测概率变化（与之前类似，但更详细）
        ax_prob = axes[1]
        
        all_probs = [pred[sample_idx] for pred in viz_callback.epoch_learning_predictions]
        all_epochs = list(range(1, len(all_probs) + 1))
        
        # 绘制预测概率曲线
        ax_prob.plot(all_epochs, all_probs, marker='o', linewidth=3, markersize=8, 
                    color='blue', label='预测概率 (Spam概率)')
        
        # 标注每个epoch的预测是否正确
        for epoch_idx, prob in enumerate(all_probs):
            pred_class = 1 if prob > 0.5 else 0
            is_correct = pred_class == learning_labels[sample_idx]
            color = 'green' if is_correct else 'red'
            marker = 'o' if is_correct else 'x'  # 使用标准marker样式：'o'表示正确，'x'表示错误
            ax_prob.scatter(all_epochs[epoch_idx], prob, s=200, c=color, 
                           marker=marker, zorder=5, edgecolors='black', linewidths=1)
            
            # 添加epoch标签
            ax_prob.annotate(f'E{epoch_idx+1}', 
                           (all_epochs[epoch_idx], prob),
                           textcoords="offset points", 
                           xytext=(0,15), 
                           ha='center', fontsize=8)
        
        # 添加分类阈值线和目标区域
        ax_prob.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='分类阈值 (0.5)')
        
        if learning_labels[sample_idx] == 1:
            ax_prob.axhspan(0.5, 1.0, alpha=0.1, color='red', label='目标区域 (Spam)')
        else:
            ax_prob.axhspan(0.0, 0.5, alpha=0.1, color='green', label='目标区域 (Ham)')
        
        ax_prob.set_xlabel('Epoch', fontsize=14)
        ax_prob.set_ylabel('预测概率 (Spam概率)', fontsize=14)
        ax_prob.set_title(f'样本 {sample_idx + 1} Token级别学习过程', 
                         fontsize=16, fontweight='bold', pad=20)
        ax_prob.legend(fontsize=12, loc='best')
        ax_prob.grid(True, alpha=0.3)
        ax_prob.set_ylim([0, 1])
        ax_prob.set_xlim([0.5, max(all_epochs) + 0.5])
        
        plt.tight_layout()
        
        # 保存图片
        token_file = os.path.join(output_dir, f"token_analysis_sample_{sample_idx + 1}.png")
        plt.savefig(token_file, dpi=300, bbox_inches='tight')
        print(f"样本 {sample_idx + 1} Token分析图已保存为 {token_file}")
        plt.close()
    
    print(f"✓ Token级别分析完成！")

print("\n生成初始预测错误样本的详细学习过程可视化...")
# 使用 explain 两个样本进行可视化
if 'explain_texts' in locals() and len(explain_texts) > 0:
    # 创建适配的回调数据（将 explain_epoch_probs 转换为 epoch_learning_predictions 格式）
    class AdaptedVizCallback:
        def __init__(self, original_callback):
            # 将 explain_epoch_probs 转换为 epoch_learning_predictions 格式
            self.epoch_learning_predictions = []
            for epoch_probs in original_callback.explain_epoch_probs:
                self.epoch_learning_predictions.append(np.array(epoch_probs))
    
    adapted_viz = AdaptedVizCallback(viz_callback)
    
    plot_individual_learning_process(
        adapted_viz, 
        explain_texts, 
        explain_y,
        explain_initial_probs,
        tokenizer,
        output_dir
    )
    
    # 生成token级别的分析
    print("\n生成Token级别分析...")
    plot_token_level_analysis(
        adapted_viz,
        explain_texts,
        explain_y,
        explain_X,
        tokenizer,
        output_dir
    )
else:
    print("警告：没有找到 explain 样本，跳过详细学习过程可视化")


# 6. 模型评估（使用测试集进行最终评估）

# 预测（使用独立的测试集，确保评估的客观性）
print("\n开始预测（使用测试集进行最终评估）...")
y_pred_prob = model.predict(X_test, verbose=0)
y_pred_class = (y_pred_prob > 0.5).astype("int32")

# 确保y_test是数组格式
y_test_eval = y_test.values if hasattr(y_test, 'values') else np.array(y_test)

# 评估指标
accuracy = metrics.accuracy_score(y_test_eval, y_pred_class)
auc = metrics.roc_auc_score(y_test_eval, y_pred_prob)
precision = metrics.precision_score(y_test_eval, y_pred_class)
recall = metrics.recall_score(y_test_eval, y_pred_class)
f1 = metrics.f1_score(y_test_eval, y_pred_class)

print(f"\n模型性能指标:")
print(f"  准确率 (Accuracy): {accuracy:.4f}")
print(f"  AUC: {auc:.4f}")
print(f"  精确率 (Precision): {precision:.4f}")
print(f"  召回率 (Recall): {recall:.4f}")
print(f"  F1分数: {f1:.4f}")


# 7. 混淆矩阵可视化

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

conf_matrix_file = os.path.join(output_dir, "confusion_matrix_resnet.png")
conf_matrix(metrics.confusion_matrix(y_test_eval, y_pred_class),
            labels=['Ham', 'Spam'],
            output_file=conf_matrix_file,
            title='ResNet 文本分类 混淆矩阵')

# 分类报告
print(f"\n分类报告:")
print(metrics.classification_report(y_test_eval, y_pred_class, target_names=['Ham', 'Spam']))

# ROC曲线
def plot_roc_curve(y_true, y_pred_prob, output_file=None, title='ROC曲线'):
    """绘制ROC曲线"""
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    auc_score = roc_auc_score(y_true, y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC曲线 (AUC = {auc_score:.4f})')
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

roc_file = os.path.join(output_dir, "roc_curve_resnet.png")
plot_roc_curve(y_test_eval, y_pred_prob, output_file=roc_file, title='ResNet 文本分类 ROC曲线')

print("\n" + "=" * 60)
print("ResNet模型训练和评估完成！")
print("=" * 60)

# 7.4. 筛选最佳展示样本：找出从错误到正确的学习样本
print("\n" + "=" * 60)
print("7.4. 筛选最佳展示样本（从错误到正确的学习过程）")
print("=" * 60)

def find_best_learning_samples(viz_callback, learning_samples_X, learning_samples_labels, 
                                learning_samples_texts, learning_samples_initial_probs):
    """
    找出最佳的展示样本：前几个epoch预测错误，后面epoch预测正确的样本
    分别找出一个ham和一个spam样本
    
    参数:
    viz_callback: 包含训练历史记录的回调对象
    learning_samples_X: 学习样本的序列数据
    learning_samples_labels: 学习样本的真实标签
    learning_samples_texts: 学习样本的原始文本
    learning_samples_initial_probs: 初始预测概率
    
    返回:
    best_ham_idx: 最佳ham样本的索引（如果没有则返回None）
    best_spam_idx: 最佳spam样本的索引（如果没有则返回None）
    """
    if not viz_callback.epoch_learning_predictions or len(learning_samples_X) == 0:
        print("警告：没有训练历史数据，无法筛选样本")
        return None, None
    
    n_samples = len(learning_samples_X)
    n_epochs = len(viz_callback.epoch_learning_predictions)
    
    # 分析每个样本的学习过程
    sample_scores = []  # 存储每个样本的评分信息
    
    for sample_idx in range(n_samples):
        true_label = learning_samples_labels[sample_idx]
        
        # 获取初始预测
        initial_prob = learning_samples_initial_probs[sample_idx]
        initial_pred = 1 if initial_prob > 0.5 else 0
        initial_correct = (initial_pred == true_label)
        
        # 获取每个epoch的预测
        epoch_predictions = []
        epoch_correct = []
        for epoch_idx in range(n_epochs):
            epoch_prob = viz_callback.epoch_learning_predictions[epoch_idx][sample_idx]
            epoch_pred = 1 if epoch_prob > 0.5 else 0
            epoch_is_correct = (epoch_pred == true_label)
            epoch_predictions.append(epoch_pred)
            epoch_correct.append(epoch_is_correct)
        
        # 找出第一个预测正确的epoch
        first_correct_epoch = None
        for epoch_idx, is_correct in enumerate(epoch_correct):
            if is_correct:
                first_correct_epoch = epoch_idx
                break
        
        # 计算错误epoch的数量
        wrong_epochs = sum(1 for is_correct in epoch_correct if not is_correct)
        
        # 计算最终是否正确
        final_correct = epoch_correct[-1] if epoch_correct else False
        
        # 评分标准：
        # 1. 初始预测必须错误
        # 2. 最终预测必须正确
        # 3. 错误epoch数越多越好（说明学习过程明显）
        # 4. 第一个正确epoch越靠后越好（说明学习过程有挑战性）
        
        score = 0
        if not initial_correct and final_correct:
            # 基础分数：错误epoch数
            score += wrong_epochs * 10
            # 额外分数：如果第一个正确epoch在中间或后面
            if first_correct_epoch is not None:
                # 第一个正确epoch越靠后，分数越高（最多加50分）
                score += (first_correct_epoch / n_epochs) * 50
            # 如果整个过程中有波动（先错后对），额外加分
            if wrong_epochs > 0 and first_correct_epoch is not None:
                score += 20
        
        sample_scores.append({
            'idx': sample_idx,
            'label': true_label,
            'score': score,
            'initial_correct': initial_correct,
            'final_correct': final_correct,
            'wrong_epochs': wrong_epochs,
            'first_correct_epoch': first_correct_epoch,
            'text': learning_samples_texts[sample_idx]
        })
    
    # 分别找出ham和spam中最好的样本
    ham_samples = [s for s in sample_scores if s['label'] == 0 and s['score'] > 0]
    spam_samples = [s for s in sample_scores if s['label'] == 1 and s['score'] > 0]
    
    best_ham_idx = None
    best_spam_idx = None
    
    if ham_samples:
        # 按分数排序，选择最好的
        ham_samples.sort(key=lambda x: x['score'], reverse=True)
        best_ham_idx = ham_samples[0]['idx']
        print(f"\n✓ 找到最佳Ham样本（索引 {best_ham_idx}）:")
        print(f"  初始预测: {'正确' if ham_samples[0]['initial_correct'] else '错误'}")
        print(f"  最终预测: {'正确' if ham_samples[0]['final_correct'] else '错误'}")
        print(f"  错误epoch数: {ham_samples[0]['wrong_epochs']}/{n_epochs}")
        print(f"  第一个正确epoch: {ham_samples[0]['first_correct_epoch'] + 1 if ham_samples[0]['first_correct_epoch'] is not None else '无'}")
        print(f"  文本预览: {ham_samples[0]['text'][:80]}...")
    else:
        print("\n⚠ 未找到符合要求的Ham样本（初始错误→最终正确）")
    
    if spam_samples:
        # 按分数排序，选择最好的
        spam_samples.sort(key=lambda x: x['score'], reverse=True)
        best_spam_idx = spam_samples[0]['idx']
        print(f"\n✓ 找到最佳Spam样本（索引 {best_spam_idx}）:")
        print(f"  初始预测: {'正确' if spam_samples[0]['initial_correct'] else '错误'}")
        print(f"  最终预测: {'正确' if spam_samples[0]['final_correct'] else '错误'}")
        print(f"  错误epoch数: {spam_samples[0]['wrong_epochs']}/{n_epochs}")
        print(f"  第一个正确epoch: {spam_samples[0]['first_correct_epoch'] + 1 if spam_samples[0]['first_correct_epoch'] is not None else '无'}")
        print(f"  文本预览: {spam_samples[0]['text'][:80]}...")
    else:
        print("\n⚠ 未找到符合要求的Spam样本（初始错误→最终正确）")
    
    return best_ham_idx, best_spam_idx

# 筛选最佳展示样本
best_ham_idx = None
best_spam_idx = None

if 'learning_samples_X' in locals() and len(learning_samples_X) > 0 and hasattr(viz_callback, 'epoch_learning_predictions'):
    best_ham_idx, best_spam_idx = find_best_learning_samples(
        viz_callback,
        learning_samples_X,
        learning_samples_labels,
        learning_samples_texts,
        learning_samples_initial_probs
    )
    
    # 如果找到了两个样本，创建新的展示样本列表
    if best_ham_idx is not None and best_spam_idx is not None:
        print(f"\n✓ 已选择两个最佳展示样本：Ham样本（索引{best_ham_idx}）和Spam样本（索引{best_spam_idx}）")
        # 创建新的展示样本（确保样本1是Ham，样本2是Spam）
        show_X = np.array([learning_samples_X[best_ham_idx], learning_samples_X[best_spam_idx]])
        show_y = np.array([learning_samples_labels[best_ham_idx], learning_samples_labels[best_spam_idx]])
        show_texts = [learning_samples_texts[best_ham_idx], learning_samples_texts[best_spam_idx]]
        show_indices = [best_ham_idx, best_spam_idx]  # 保存原始索引，用于获取初始概率
        show_initial_probs = [learning_samples_initial_probs[best_ham_idx], learning_samples_initial_probs[best_spam_idx]]
        
        # 验证样本顺序：确保样本1是Ham（label=0），样本2是Spam（label=1）
        if show_y[0] != 0:
            print(f"⚠ 警告：样本1的真实标签是{show_y[0]}（期望是0=Ham），正在调整顺序...")
            # 交换顺序
            show_X = np.array([show_X[1], show_X[0]])
            show_y = np.array([show_y[1], show_y[0]])
            show_texts = [show_texts[1], show_texts[0]]
            show_indices = [show_indices[1], show_indices[0]]
            show_initial_probs = [show_initial_probs[1], show_initial_probs[0]]
            print(f"✓ 已调整：样本1现在是Ham（label={show_y[0]}），样本2是Spam（label={show_y[1]}）")
    elif best_ham_idx is not None:
        print(f"\n⚠ 只找到Ham样本，将使用该样本和第一个学习样本")
        show_X = np.array([learning_samples_X[best_ham_idx], learning_samples_X[0]])
        show_y = np.array([learning_samples_labels[best_ham_idx], learning_samples_labels[0]])
        show_texts = [learning_samples_texts[best_ham_idx], learning_samples_texts[0]]
        show_indices = [best_ham_idx, 0]
        show_initial_probs = [learning_samples_initial_probs[best_ham_idx], learning_samples_initial_probs[0]]
    elif best_spam_idx is not None:
        print(f"\n⚠ 只找到Spam样本，将使用该样本和第一个学习样本")
        show_X = np.array([learning_samples_X[0], learning_samples_X[best_spam_idx]])
        show_y = np.array([learning_samples_labels[0], learning_samples_labels[best_spam_idx]])
        show_texts = [learning_samples_texts[0], learning_samples_texts[best_spam_idx]]
        show_indices = [0, best_spam_idx]
        show_initial_probs = [learning_samples_initial_probs[0], learning_samples_initial_probs[best_spam_idx]]
    else:
        print(f"\n⚠ 未找到符合要求的样本，将使用前2个学习样本")
        show_X = learning_samples_X[:2]
        show_y = learning_samples_labels[:2]
        show_texts = learning_samples_texts[:2]
        show_indices = [0, 1]
        show_initial_probs = learning_samples_initial_probs[:2]
else:
    print("⚠ 没有学习样本数据，将使用典型样本")
    show_X = selected_X[:2] if len(selected_X) >= 2 else selected_X
    show_y = selected_labels[:2] if len(selected_labels) >= 2 else selected_labels
    show_texts = selected_texts[:2] if len(selected_texts) >= 2 else selected_texts
    show_indices = list(range(len(show_X)))
    show_initial_probs = [0.5] * len(show_X)

print(f"\n将使用 {len(show_X)} 个样本进行详细的可解释性分析")

# 7.5. 模型可解释性分析：Token/位置重要性（Gradient×Embedding）和 Grad-CAM-1D
print("\n" + "=" * 60)
print("7.5. 模型可解释性分析（Gradient×Embedding & Grad-CAM-1D）")
print("=" * 60)

def decode_tokens_from_ids(token_ids, tokenizer):
    """
    从token ID序列解码回token字符串列表
    
    参数:
    token_ids: token ID数组
    tokenizer: Keras Tokenizer对象
    
    返回:
    tokens: token字符串列表
    """
    tokens = []
    for tid in token_ids:
        if tid == 0:  # padding
            continue
        # 使用index_word将ID转换为词
        if hasattr(tokenizer, 'index_word') and int(tid) in tokenizer.index_word:
            tokens.append(tokenizer.index_word[int(tid)])
        elif hasattr(tokenizer, 'word_index'):
            # 如果没有index_word，尝试反向查找
            reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
            tokens.append(reverse_word_index.get(int(tid), f"<UNK_{int(tid)}>"))
        else:
            tokens.append(f"<UNK_{int(tid)}>")
    return tokens

def gradient_x_embedding_importance(model, x_ids, layer_name="embedding"):
    """
    计算每个token位置的重要性分数（使用 Gradient × Embedding）
    
    参数:
    model: 训练好的模型
    x_ids: token ID序列（1D数组）
    layer_name: embedding层的名称
    
    返回:
    token_scores: 每个位置的重要性分数（长度=max_length）
    pred_prob: 预测的spam概率
    """
    # 导入Model（兼容不同Keras版本）
    try:
        from tensorflow.keras.models import Model
    except ImportError:
        from keras.models import Model
    
    # 获取embedding层和最终输出
    emb_layer = model.get_layer(layer_name)
    probe = Model(inputs=model.inputs,
                  outputs=[emb_layer.output, model.output])
    
    # 转换为tensor（添加batch维度）
    x = tf.convert_to_tensor(x_ids[None, :], dtype=tf.int32)
    
    with tf.GradientTape() as tape:
        emb_out, y = probe(x, training=False)  # emb_out: (1, L, D), y: (1, 1)
        tape.watch(emb_out)
        score = y[0, 0]  # spam概率
    
    # 计算梯度
    grads = tape.gradient(score, emb_out)  # (1, L, D)
    
    # Gradient × Embedding，沿embedding维度聚合
    token_scores = tf.reduce_sum(tf.abs(grads * emb_out), axis=-1)[0]  # (L,)
    token_scores = token_scores.numpy()
    
    pred_prob = float(y.numpy()[0, 0])
    return token_scores, pred_prob

def save_highlight_html(tokens, scores, pred_prob, true_label, out_path, title="Token Importance"):
    """
    生成HTML文件，用颜色高亮显示token重要性
    
    参数:
    tokens: token字符串列表（不含padding）
    scores: 同长度的重要性分数数组
    pred_prob: 预测的spam概率
    true_label: 真实标签（0或1）
    out_path: 输出HTML文件路径
    title: HTML标题
    """
    scores = np.array(scores, dtype=float)
    if scores.max() > 0:
        scores = scores / (scores.max() + 1e-9)
    
    # 用红色强度表示重要性（越红越重要）
    spans = []
    for tok, s in zip(tokens, scores):
        alpha = 0.15 + 0.85 * float(s)
        tok_esc = html.escape(tok)
        spans.append(f"<span style='background: rgba(255,0,0,{alpha:.3f}); padding:2px 4px; margin:2px; border-radius:4px; display:inline-block'>{tok_esc}</span>")
    
    label_str = "Spam" if true_label == 1 else "Ham"
    html_str = f"""
    <html><head><meta charset="utf-8"><title>{html.escape(title)}</title></head>
    <body style="font-family: Arial, sans-serif; line-height: 2;">
      <h2>{html.escape(title)}</h2>
      <p><b>真实标签:</b> {label_str} &nbsp;&nbsp; <b>预测(Spam概率):</b> {pred_prob:.4f}</p>
      <div style="padding:10px; border:1px solid #ddd; border-radius:8px;">
        {''.join(spans)}
      </div>
      <p style="color:#666;margin-top:10px;">颜色强度表示重要性（Gradient×Embedding）。</p>
    </body></html>
    """
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_str)

def plot_token_scores_bar(tokens, scores, out_png, top_k=20, title="Top Token Importance"):
    """
    绘制Top K重要token的条形图
    
    参数:
    tokens: token字符串列表
    scores: 重要性分数数组
    out_png: 输出PNG文件路径
    top_k: 显示前K个重要token
    title: 图表标题
    """
    scores = np.array(scores, dtype=float)
    # 取top_k
    idx = np.argsort(scores)[::-1][:top_k]
    top_tokens = [tokens[i] for i in idx]
    top_scores = scores[idx]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_tokens))[::-1], top_scores[::-1], color='coral', alpha=0.7)
    plt.yticks(range(len(top_tokens))[::-1], top_tokens[::-1])
    plt.xlabel("重要性分数", fontsize=12)
    plt.ylabel("Token", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

def gradcam_1d(model, x_ids, target_layer="res_block6_relu2"):
    """
    对某个Conv/Activation层输出做Grad-CAM-1D
    
    参数:
    model: 训练好的模型
    x_ids: token ID序列（1D数组）
    target_layer: 目标层的名称（通常是最后一个残差块的激活层）
    
    返回:
    cam: 序列位置的重要性分数（长度可能因下采样而缩短）
    pred_prob: 预测的spam概率
    """
    # 导入Model（兼容不同Keras版本）
    try:
        from tensorflow.keras.models import Model
    except ImportError:
        from keras.models import Model
    
    try:
        layer = model.get_layer(target_layer)
        cam_model = Model(inputs=model.inputs, outputs=[layer.output, model.output])
        
        x = tf.convert_to_tensor(x_ids[None, :], dtype=tf.int32)
        with tf.GradientTape() as tape:
            fmap, y = cam_model(x, training=False)      # fmap: (1, L', C)
            score = y[0, 0]
        grads = tape.gradient(score, fmap)             # (1, L', C)
        
        # 计算权重（对通道维度求平均）
        weights = tf.reduce_mean(grads, axis=1)        # (1, C)
        # 加权求和得到CAM
        cam = tf.reduce_sum(fmap * weights[:, None, :], axis=-1)[0]  # (L',)
        cam = tf.nn.relu(cam)  # 只保留正贡献
        cam = cam.numpy()
        
        # 归一化
        if cam.max() > 0:
            cam = cam / (cam.max() + 1e-9)
        
        pred_prob = float(y.numpy()[0, 0])
        return cam, pred_prob
    except Exception as e:
        print(f"Grad-CAM-1D计算出错（可能层名称不匹配）: {e}")
        return None, None

def plot_cam_curve(cam, out_png, title="Grad-CAM-1D over sequence"):
    """
    绘制Grad-CAM-1D重要性曲线
    
    参数:
    cam: 序列位置的重要性分数数组
    out_png: 输出PNG文件路径
    title: 图表标题
    """
    plt.figure(figsize=(12, 3))
    plt.plot(cam, linewidth=2, color='steelblue')
    plt.fill_between(range(len(cam)), cam, alpha=0.3, color='steelblue')
    plt.ylim([0, 1.05])
    plt.xlabel("序列位置（经过卷积/下采样后）", fontsize=12)
    plt.ylabel("重要性分数", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

# 执行可解释性分析
print("\n开始生成模型可解释性可视化...")

# 1. Gradient×Embedding 分析
print("\n【方法1】Gradient×Embedding Token重要性分析...")
for i in range(len(explain_X)):
    try:
        x_ids = explain_X[i]
        true_y = int(explain_y[i])
        
        # 计算token重要性
        scores_full, pred_prob = gradient_x_embedding_importance(model, x_ids, layer_name="embedding")
        
        # 去掉padding对应的score，并解码token
        nonpad_mask = x_ids != 0
        nonpad_ids = x_ids[nonpad_mask]
        tokens = decode_tokens_from_ids(nonpad_ids, tokenizer)
        scores = scores_full[:len(tokens)]
        
        # 生成HTML高亮文件
        html_path = os.path.join(output_dir, f"explain_gradxemb_{i+1}.html")
        save_highlight_html(tokens, scores, pred_prob, true_y, html_path,
                          title=f"样本 {i+1} Gradient×Embedding 高亮显示")
        
        # 生成Top词条形图
        png_path = os.path.join(output_dir, f"explain_gradxemb_{i+1}_top.png")
        plot_token_scores_bar(tokens, scores, png_path, top_k=20,
                            title=f"样本 {i+1} Top 20 重要Token (预测概率={pred_prob:.3f})")
        
        print(f"  样本 {i+1}: HTML={html_path}, PNG={png_path}")
    except Exception as e:
        print(f"  样本 {i+1} 分析出错: {e}")
        import traceback
        traceback.print_exc()

print("✓ Gradient×Embedding 解释可视化完成（HTML高亮 + Top词条形图）")

# 2. Grad-CAM-1D 分析
print("\n【方法2】Grad-CAM-1D 序列位置重要性分析...")
# 尝试不同的目标层（从后往前）
target_layers = ["res_block6_relu2", "res_block5_relu2", "res_block4_relu2", "initial_relu"]
target_layer = None
for layer_name in target_layers:
    try:
        model.get_layer(layer_name)
        target_layer = layer_name
        print(f"使用目标层: {target_layer}")
        break
    except:
        continue

if target_layer is None:
    print("警告：未找到合适的Grad-CAM目标层，跳过Grad-CAM-1D分析")
else:
    for i in range(min(3, len(explain_X))):
        try:
            cam, pred_prob = gradcam_1d(model, explain_X[i], target_layer=target_layer)
            if cam is not None:
                out = os.path.join(output_dir, f"explain_gradcam1d_{i+1}.png")
                true_y = int(explain_y[i])
                label_str = "Spam" if true_y == 1 else "Ham"
                plot_cam_curve(cam, out, 
                             title=f"样本 {i+1} ({label_str}) Grad-CAM-1D (预测概率={pred_prob:.3f})")
                print(f"  样本 {i+1}: {out}")
        except Exception as e:
            print(f"  样本 {i+1} Grad-CAM-1D分析出错: {e}")
            import traceback
            traceback.print_exc()
    
    print("✓ Grad-CAM-1D 序列重要性曲线生成完成")

# 3. 生成动态HTML：展示每个epoch的token重要性变化
print("\n【方法3】生成动态HTML：展示训练过程中token重要性的变化...")

def create_dynamic_token_importance_html(viz_callback, learning_samples_X, learning_samples_labels, 
                                         learning_samples_texts, tokenizer, output_path, 
                                         initial_probs=None):
    """
    创建动态HTML，展示每个epoch的token重要性变化
    
    参数:
    viz_callback: 包含训练历史记录的回调对象
    learning_samples_X: 学习样本的序列数据（现在应该是 explain_X）
    learning_samples_labels: 学习样本的真实标签（现在应该是 explain_y）
    learning_samples_texts: 学习样本的原始文本（现在应该是 explain_texts）
    tokenizer: tokenizer对象
    output_path: 输出HTML文件路径
    initial_probs: 初始预测概率列表（可选，现在应该是 explain_initial_probs）
    """
    import json
    
    if not viz_callback.explain_epoch_token_scores or len(viz_callback.explain_epoch_token_scores) == 0:
        print("警告：没有记录到token重要性历史，跳过动态HTML生成")
        return
    
    n_samples = len(learning_samples_X)
    n_epochs_total = len(viz_callback.explain_epoch_token_scores)  # 包含 epoch0
    n_epochs = n_epochs_total - 1  # 训练 epoch 数（不含 epoch0）
    
    # 解码所有样本的tokens（只解码一次）
    all_tokens = []
    for sample_idx in range(n_samples):
        x_ids = learning_samples_X[sample_idx]
        nonpad_mask = x_ids != 0
        nonpad_ids = x_ids[nonpad_mask]
        tokens = decode_tokens_from_ids(nonpad_ids, tokenizer)
        all_tokens.append(tokens)
    
    # 准备数据：为每个样本、每个epoch准备token重要性分数
    # 格式：data[sample_idx][epoch_idx] = [scores列表]
    data_js = {}
    predictions_js = {}  # 存储每个epoch的预测概率
    
    for sample_idx in range(n_samples):
        data_js[sample_idx] = {}
        predictions_js[sample_idx] = []
        
        # 遍历所有epoch（包括epoch0）
        for epoch_idx in range(len(viz_callback.explain_epoch_token_scores)):
            if epoch_idx < len(viz_callback.explain_epoch_token_scores):
                token_scores = viz_callback.explain_epoch_token_scores[epoch_idx][sample_idx]
                # 确保长度匹配
                if len(token_scores) == len(all_tokens[sample_idx]):
                    data_js[sample_idx][epoch_idx] = token_scores
                else:
                    # 如果长度不匹配，用0填充或截断
                    scores_padded = token_scores[:len(all_tokens[sample_idx])] + [0] * max(0, len(all_tokens[sample_idx]) - len(token_scores))
                    data_js[sample_idx][epoch_idx] = scores_padded[:len(all_tokens[sample_idx])]
            else:
                data_js[sample_idx][epoch_idx] = [0] * len(all_tokens[sample_idx])
        
        # 获取每个epoch的预测概率（包括epoch0）
        for epoch_idx in range(len(viz_callback.explain_epoch_probs)):
            if epoch_idx < len(viz_callback.explain_epoch_probs):
                pred_prob = float(viz_callback.explain_epoch_probs[epoch_idx][sample_idx])
                predictions_js[sample_idx].append(pred_prob)
            else:
                # 如果缺失，使用初始概率
                if initial_probs is not None and sample_idx < len(initial_probs):
                    predictions_js[sample_idx].append(float(initial_probs[sample_idx]))
                else:
                    predictions_js[sample_idx].append(0.5)
    
    # 生成HTML内容
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Token重要性动态变化 - 训练过程可视化</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }}
        .controls {{
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 15px;
            flex-wrap: wrap;
        }}
        .control-group {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        label {{
            font-weight: bold;
            color: #555;
        }}
        select, input[type="range"] {{
            padding: 5px 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }}
        input[type="range"] {{
            width: 200px;
        }}
        .epoch-info {{
            font-size: 16px;
            color: #2c3e50;
            font-weight: bold;
        }}
        .sample-container {{
            margin-bottom: 40px;
            padding: 20px;
            border: 2px solid #ddd;
            border-radius: 8px;
            background-color: #fafafa;
        }}
        .sample-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #3498db;
        }}
        .sample-title {{
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .sample-meta {{
            display: flex;
            gap: 20px;
            font-size: 14px;
        }}
        .label-badge {{
            padding: 5px 12px;
            border-radius: 4px;
            font-weight: bold;
            color: white;
        }}
        .label-spam {{
            background-color: #e74c3c;
        }}
        .label-ham {{
            background-color: #27ae60;
        }}
        .pred-prob {{
            padding: 5px 12px;
            border-radius: 4px;
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        .text-display {{
            padding: 15px;
            background-color: white;
            border-radius: 5px;
            line-height: 2;
            min-height: 60px;
            word-wrap: break-word;
        }}
        .token {{
            padding: 3px 6px;
            margin: 2px;
            border-radius: 4px;
            display: inline-block;
            transition: all 0.3s ease;
            cursor: pointer;
        }}
        .token:hover {{
            transform: scale(1.1);
            z-index: 10;
            position: relative;
        }}
        .legend {{
            margin-top: 15px;
            padding: 10px;
            background-color: #ecf0f1;
            border-radius: 5px;
            font-size: 12px;
        }}
        .legend-item {{
            display: inline-block;
            margin-right: 15px;
        }}
        .legend-color {{
            display: inline-block;
            width: 20px;
            height: 20px;
            border-radius: 3px;
            margin-right: 5px;
            vertical-align: middle;
        }}
        .play-controls {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 10px;
        }}
        button {{
            padding: 8px 16px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }}
        button:hover {{
            background-color: #2980b9;
        }}
        button:disabled {{
            background-color: #bdc3c7;
            cursor: not-allowed;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Token重要性动态变化 - 训练过程可视化</h1>
        
        <div class="controls">
            <div class="control-group">
                <label>选择样本:</label>
                <select id="sampleSelect" onchange="updateDisplay()">
"""
    
    # 添加样本选择选项
    for i in range(n_samples):
        label_str = "Spam" if learning_samples_labels[i] == 1 else "Ham"
        html_content += f'                    <option value="{i}">样本 {i+1} ({label_str})</option>\n'
    
    html_content += """                </select>
            </div>
            <div class="control-group">
                <label>Epoch:</label>
                <input type="range" id="epochSlider" min="0" max=""" + str(n_epochs) + f""" value="0" 
                       oninput="updateEpoch(this.value)" onchange="updateEpoch(this.value)">
                <span class="epoch-info" id="epochDisplay">Epoch 0 (初始)</span>
            </div>
            <div class="play-controls">
                <button onclick="playAnimation()" id="playBtn">▶ 播放动画</button>
                <button onclick="stopAnimation()" id="stopBtn" disabled>⏸ 停止</button>
                <label style="margin-left: 20px;">
                    <input type="number" id="speedInput" value="500" min="100" max="2000" step="100" style="width: 60px;">
                    <span>ms/帧</span>
                </label>
            </div>
        </div>
"""
    
    # 为每个样本创建显示区域
    for i in range(n_samples):
        label_str = "Spam" if learning_samples_labels[i] == 1 else "Ham"
        label_class = "label-spam" if learning_samples_labels[i] == 1 else "label-ham"
        text_preview = learning_samples_texts[i][:100] + "..." if len(learning_samples_texts[i]) > 100 else learning_samples_texts[i]
        
        html_content += f"""
        <div class="sample-container" id="sample{i}" style="display: {'block' if i == 0 else 'none'};">
            <div class="sample-header">
                <div class="sample-title">样本 {i+1}</div>
                <div class="sample-meta">
                    <span class="label-badge {label_class}">真实标签: {label_str}</span>
                    <span class="pred-prob" id="predProb{i}">预测概率: 0.0000</span>
                </div>
            </div>
            <div class="text-display" id="textDisplay{i}">
                <!-- Token将在这里动态插入 -->
            </div>
            <div class="legend">
                <div class="legend-item">
                    <span class="legend-color" style="background: rgba(255,0,0,0.15);"></span>
                    <span>低重要性</span>
                </div>
                <div class="legend-item">
                    <span class="legend-color" style="background: rgba(255,0,0,0.5);"></span>
                    <span>中等重要性</span>
                </div>
                <div class="legend-item">
                    <span class="legend-color" style="background: rgba(255,0,0,0.85);"></span>
                    <span>高重要性</span>
                </div>
            </div>
        </div>
"""
    
    # 添加JavaScript代码
    html_content += f"""
    </div>
    
    <script>
        // 数据
        const tokensData = {json.dumps(all_tokens, ensure_ascii=False)};
        const importanceData = {json.dumps(data_js, ensure_ascii=False)};
        const predictionsData = {json.dumps(predictions_js, ensure_ascii=False)};
        const labels = {json.dumps([int(l) for l in learning_samples_labels], ensure_ascii=False)};
        const nEpochs = {n_epochs};
        
        let currentSample = 0;
        let currentEpoch = 0;
        let animationInterval = null;
        
        function updateDisplay() {{
            currentSample = parseInt(document.getElementById('sampleSelect').value);
            // 显示当前样本，隐藏其他
            for (let i = 0; i < {n_samples}; i++) {{
                document.getElementById(`sample${{i}}`).style.display = (i === currentSample) ? 'block' : 'none';
            }}
            updateEpoch(currentEpoch);
        }}
        
        function updateEpoch(epoch) {{
            currentEpoch = parseInt(epoch);
            document.getElementById('epochSlider').value = currentEpoch;
            document.getElementById('epochDisplay').textContent = 
                currentEpoch === 0 ? 'Epoch 0 (初始)' : `Epoch ${{currentEpoch}}`;
            
            const tokens = tokensData[currentSample];
            // 直接使用对应 epoch 的数据（包括 epoch0）
            let scores = importanceData[currentSample][currentEpoch] || new Array(tokens.length).fill(0);
            
            // 确保scores长度与tokens匹配
            if (scores.length < tokens.length) {{
                scores = scores.concat(new Array(tokens.length - scores.length).fill(0));
            }} else {{
                scores = scores.slice(0, tokens.length);
            }}
            const predProb = predictionsData[currentSample][currentEpoch] || 0.5;
            
            // 更新预测概率显示
            document.getElementById(`predProb${{currentSample}}`).textContent = 
                `预测概率: ${{predProb.toFixed(4)}}`;
            
            // 归一化分数
            const maxScore = Math.max(...scores, 1e-9);
            const normalizedScores = scores.map(s => s / maxScore);
            
            // 生成token HTML
            let html = '';
            for (let i = 0; i < tokens.length; i++) {{
                const token = tokens[i];
                const score = normalizedScores[i] || 0;
                const alpha = 0.15 + 0.85 * score;
                const escapedToken = escapeHtml(token);
                html += `<span class="token" style="background: rgba(255,0,0,${{alpha.toFixed(3)}});" 
                         title="重要性: ${{score.toFixed(3)}}">${{escapedToken}}</span>`;
            }}
            
            document.getElementById(`textDisplay${{currentSample}}`).innerHTML = html;
        }}
        
        function escapeHtml(text) {{
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }}
        
        function playAnimation() {{
            if (animationInterval) return;
            
            document.getElementById('playBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            
            const speed = parseInt(document.getElementById('speedInput').value);
            let epoch = 0;
            
            animationInterval = setInterval(() => {{
                updateEpoch(epoch);
                epoch++;
                if (epoch > nEpochs) {{
                    stopAnimation();
                }}
            }}, speed);
        }}
        
        function stopAnimation() {{
            if (animationInterval) {{
                clearInterval(animationInterval);
                animationInterval = null;
            }}
            document.getElementById('playBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
        }}
        
        // 初始化显示
        updateDisplay();
    </script>
</body>
</html>
"""
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"动态HTML已保存为: {output_path}")

# 生成动态HTML（使用 explain 两个样本）
if 'explain_X' in locals() and len(explain_X) > 0 and hasattr(viz_callback, 'explain_epoch_token_scores'):
    try:
        dynamic_html_path = os.path.join(output_dir, "explain_dynamic_token_importance.html")
        
        create_dynamic_token_importance_html(
            viz_callback,
            explain_X,
            explain_y,
            explain_texts,
            tokenizer,
            dynamic_html_path,
            initial_probs=explain_initial_probs if 'explain_initial_probs' in locals() else None
        )
        
        print("✓ 动态Token重要性HTML生成完成！")
    except Exception as e:
        print(f"生成动态HTML时出错: {e}")
        import traceback
        traceback.print_exc()
else:
    print("警告：缺少必要的数据，跳过动态HTML生成")

print("\n" + "=" * 60)
print("模型可解释性分析完成！")
print("=" * 60)

# 8. 特征重要性分析：展示哪些词汇/特征更容易导致垃圾邮件判断
print("\n" + "=" * 60)
print("8. ResNet特征重要性分析")
print("=" * 60)

def analyze_feature_importance(model, tokenizer, texts, labels, vocab_size, top_n=50):
    """
    分析ResNet模型的特征重要性
    通过多种方法展示哪些词汇更容易导致垃圾邮件判断
    
    参数:
    model: 训练好的ResNet模型
    tokenizer: 文本tokenizer
    texts: 文本数据
    labels: 标签数据
    vocab_size: 词汇表大小
    top_n: 显示前N个重要特征
    """
    from collections import Counter
    import re
    
    print("\n分析特征重要性...")
    
    # 方法1: 词频统计分析（最直观的方法）
    print("\n【方法1】词频统计分析：垃圾邮件 vs 正常邮件")
    print("-" * 60)
    
    # 分离垃圾邮件和正常邮件
    labels_array = labels.values if hasattr(labels, 'values') else np.array(labels)
    spam_texts = [texts.iloc[i] if hasattr(texts, 'iloc') else texts[i] 
                  for i in range(len(texts)) if labels_array[i] == 1]
    ham_texts = [texts.iloc[i] if hasattr(texts, 'iloc') else texts[i] 
                 for i in range(len(texts)) if labels_array[i] == 0]
    
    # 统计词频
    def get_word_freq(text_list):
        """统计词频"""
        all_words = []
        for text in text_list:
            if pd.isna(text):
                continue
            # 简单分词（按空格）
            words = str(text).lower().split()
            # 移除标点符号
            words = [re.sub(r'[^\w\s]', '', word) for word in words if word]
            all_words.extend(words)
        return Counter(all_words)
    
    spam_word_freq = get_word_freq(spam_texts)
    ham_word_freq = get_word_freq(ham_texts)
    
    # 计算垃圾邮件倾向度（Spam Score）
    # Score = (spam_freq / total_spam) / (ham_freq / total_ham + epsilon)
    all_words = set(spam_word_freq.keys()) | set(ham_word_freq.keys())
    total_spam = sum(spam_word_freq.values())
    total_ham = sum(ham_word_freq.values())
    epsilon = 1e-6
    
    word_spam_scores = {}
    for word in all_words:
        spam_count = spam_word_freq.get(word, 0)
        ham_count = ham_word_freq.get(word, 0)
        spam_prob = spam_count / (total_spam + epsilon)
        ham_prob = ham_count / (total_ham + epsilon)
        
        # 计算垃圾邮件倾向度
        if ham_prob > 0:
            spam_score = spam_prob / (ham_prob + epsilon)
        else:
            spam_score = spam_prob * 1000  # 如果正常邮件中没有，给高分
        
        # 只考虑出现频率足够高的词（至少出现3次）
        if spam_count + ham_count >= 3:
            word_spam_scores[word] = {
                'score': spam_score,
                'spam_count': spam_count,
                'ham_count': ham_count,
                'spam_freq': spam_count / (total_spam + epsilon),
                'ham_freq': ham_count / (total_ham + epsilon)
            }
    
    # 排序并获取Top N
    sorted_words = sorted(word_spam_scores.items(), key=lambda x: x[1]['score'], reverse=True)
    top_spam_words = sorted_words[:top_n]
    
    print(f"\n【最可能导致垃圾邮件判断的词汇】（前{top_n}个，按垃圾邮件倾向度排序）:")
    print(f"{'排名':<6} {'词汇':<20} {'倾向度':<12} {'Spam频次':<12} {'Ham频次':<12} {'Spam占比':<12}")
    print("-" * 80)
    for i, (word, info) in enumerate(top_spam_words, 1):
        spam_pct = info['spam_freq'] * 100
        print(f"{i:<6} {word:<20} {info['score']:>10.2f} {info['spam_count']:>10} {info['ham_count']:>10} {spam_pct:>10.2f}%")
    
    # 方法2: 词嵌入权重分析
    print("\n【方法2】词嵌入层权重分析")
    print("-" * 60)
    
    try:
        # 获取嵌入层
        embedding_layer = None
        for layer in model.layers:
            if 'embedding' in layer.name.lower():
                embedding_layer = layer
                break
        
        if embedding_layer is not None:
            embedding_weights = embedding_layer.get_weights()[0]  # shape: (vocab_size, embedding_dim)
            
            # 计算每个词嵌入向量的L2范数（作为重要性指标）
            word_importance = {}
            reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
            
            for word, word_id in tokenizer.word_index.items():
                if word_id < len(embedding_weights):
                    # 计算嵌入向量的L2范数
                    embedding_vec = embedding_weights[word_id]
                    importance = np.linalg.norm(embedding_vec)
                    word_importance[word] = importance
            
            # 排序并获取Top N
            sorted_embedding = sorted(word_importance.items(), key=lambda x: x[1], reverse=True)
            top_embedding_words = sorted_embedding[:top_n]
            
            print(f"\n【嵌入层权重最大的词汇】（前{top_n}个，可能对模型决策影响较大）:")
            print(f"{'排名':<6} {'词汇':<20} {'嵌入权重(L2范数)':<20}")
            print("-" * 50)
            for i, (word, importance) in enumerate(top_embedding_words, 1):
                print(f"{i:<6} {word:<20} {importance:>15.4f}")
        else:
            print("未找到嵌入层")
    except Exception as e:
        print(f"嵌入层分析出错: {e}")
    
    # 方法3: 分类常见垃圾邮件特征词汇
    print("\n【方法3】常见垃圾邮件特征词汇分类统计")
    print("-" * 60)
    
    # 定义常见的垃圾邮件特征类别
    spam_categories = {
        '金融/投资': ['free', 'money', 'cash', 'prize', 'win', 'winner', 'million', 'dollar', 
                    'investment', 'profit', 'earn', 'income', 'wealth', 'rich', 'bank'],
        '广告/促销': ['offer', 'deal', 'discount', 'sale', 'special', 'limited', 'save', 
                    'buy', 'order', 'shop', 'store', 'promotion', 'advertisement'],
        '时间紧迫性': ['urgent', 'now', 'today', 'immediately', 'hurry', 'limited time', 
                    'expire', 'deadline', 'act now', 'call now'],
        '诱导性词汇': ['congratulations', 'guaranteed', 'risk-free', 'secret', 'exclusive', 
                    'amazing', 'incredible', 'miracle', 'unbelievable'],
        '行动号召': ['click', 'call', 'reply', 'text', 'visit', 'register', 'subscribe', 
                    'download', 'claim', 'apply']
    }
    
    category_stats = {}
    for category, keywords in spam_categories.items():
        category_count_spam = 0
        category_count_ham = 0
        category_words_found = []
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            spam_matches = sum(1 for text in spam_texts if keyword_lower in str(text).lower())
            ham_matches = sum(1 for text in ham_texts if keyword_lower in str(text).lower())
            
            if spam_matches > 0 or ham_matches > 0:
                category_count_spam += spam_matches
                category_count_ham += ham_matches
                category_words_found.append((keyword, spam_matches, ham_matches))
        
        if category_count_spam + category_count_ham > 0:
            spam_ratio = category_count_spam / (category_count_spam + category_count_ham) * 100
            category_stats[category] = {
                'spam_count': category_count_spam,
                'ham_count': category_count_ham,
                'spam_ratio': spam_ratio,
                'words': category_words_found
            }
    
    # 按垃圾邮件占比排序
    sorted_categories = sorted(category_stats.items(), 
                              key=lambda x: x[1]['spam_ratio'], reverse=True)
    
    print(f"\n【各类特征词汇在垃圾邮件中的出现情况】:")
    print(f"{'特征类别':<20} {'Spam出现':<12} {'Ham出现':<12} {'Spam占比':<12} {'典型词汇':<30}")
    print("-" * 90)
    for category, stats in sorted_categories:
        # 显示前3个典型词汇
        top_words = sorted(stats['words'], key=lambda x: x[1], reverse=True)[:3]
        words_str = ', '.join([f"{w[0]}({w[1]})" for w in top_words])
        print(f"{category:<20} {stats['spam_count']:>10} {stats['ham_count']:>10} "
              f"{stats['spam_ratio']:>10.1f}% {words_str:<30}")
    
    # 可视化：Top垃圾邮件特征词汇
    print("\n生成特征重要性可视化图表...")
    
    # 图1: Top垃圾邮件特征词汇
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    # 上部分：Top 20垃圾邮件倾向词汇
    top_20_words = [w[0] for w in top_spam_words[:20]]
    top_20_scores = [w[1]['score'] for w in top_spam_words[:20]]
    
    ax1 = axes[0]
    bars = ax1.barh(range(len(top_20_words)), top_20_scores, color='coral', alpha=0.7)
    ax1.set_yticks(range(len(top_20_words)))
    ax1.set_yticklabels(top_20_words)
    ax1.set_xlabel('垃圾邮件倾向度', fontsize=12)
    ax1.set_title('Top 20 最可能导致垃圾邮件判断的词汇', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()
    
    # 添加数值标签
    for i, (bar, score) in enumerate(zip(bars, top_20_scores)):
        ax1.text(bar.get_width() + max(top_20_scores) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.1f}', va='center', fontsize=9)
    
    # 下部分：特征类别统计
    categories_list = [cat[0] for cat in sorted_categories]
    spam_ratios = [cat[1]['spam_ratio'] for cat in sorted_categories]
    
    ax2 = axes[1]
    bars2 = ax2.barh(range(len(categories_list)), spam_ratios, color='steelblue', alpha=0.7)
    ax2.set_yticks(range(len(categories_list)))
    ax2.set_yticklabels(categories_list)
    ax2.set_xlabel('垃圾邮件占比 (%)', fontsize=12)
    ax2.set_title('各类特征词汇在垃圾邮件中的占比', fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()
    ax2.set_xlim([0, 100])
    
    # 添加数值标签
    for i, (bar, ratio) in enumerate(zip(bars2, spam_ratios)):
        ax2.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                f'{ratio:.1f}%', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    feature_importance_file = os.path.join(output_dir, "resnet_feature_importance.png")
    plt.savefig(feature_importance_file, dpi=300, bbox_inches='tight')
    print(f"特征重要性分析图已保存为 {feature_importance_file}")
    plt.close()
    
    # 保存详细结果到CSV
    import csv
    csv_file = os.path.join(output_dir, "resnet_feature_importance.csv")
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['排名', '词汇', '垃圾邮件倾向度', 'Spam出现次数', 'Ham出现次数', 
                        'Spam频率', 'Ham频率', 'Spam占比(%)'])
        for i, (word, info) in enumerate(top_spam_words, 1):
            spam_pct = info['spam_freq'] * 100
            writer.writerow([i, word, f"{info['score']:.4f}", info['spam_count'], 
                           info['ham_count'], f"{info['spam_freq']:.6f}", 
                           f"{info['ham_freq']:.6f}", f"{spam_pct:.2f}"])
    print(f"特征重要性详细数据已保存为 {csv_file}")
    
    return top_spam_words, category_stats

# 执行特征重要性分析
try:
    top_features, category_stats = analyze_feature_importance(
        model, tokenizer, texts, target, vocab_size, top_n=50
    )
    print("\n✓ 特征重要性分析完成！")
except Exception as e:
    print(f"特征重要性分析出错: {e}")
    import traceback
    traceback.print_exc()

