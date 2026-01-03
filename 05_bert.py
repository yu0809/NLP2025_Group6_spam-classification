# -*- coding: utf-8 -*-
"""
BERT 模型
使用BERT (Bidirectional Encoder Representations from Transformers) 进行垃圾邮件检测。
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

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 尝试导入TensorFlow和Transformers
try:
import tensorflow as tf
    print(f"✓ 成功导入 TensorFlow {tf.__version__}")
except ImportError as e:
    print("=" * 60)
    print("错误：无法导入TensorFlow")
    print("=" * 60)
    print("请安装TensorFlow：")
    print("  pip install tensorflow")
    print("=" * 60)
    sys.exit(1)

# 检查Keras版本兼容性
try:
    import keras
    keras_version = keras.__version__
    print(f"检测到 Keras {keras_version}")
    
    # 如果是Keras 3，尝试导入tf-keras
    if keras_version.startswith('3.'):
        try:
            import tf_keras
            print("✓ 检测到 tf-keras，使用 tf_keras 替代 keras")
            # 设置环境变量让transformers使用tf-keras
            os.environ['TF_USE_LEGACY_KERAS'] = '1'
        except ImportError:
            print("=" * 60)
            print("警告：检测到 Keras 3，但未安装 tf-keras")
            print("=" * 60)
            print("Transformers 库需要 tf-keras 来兼容 Keras 3")
            print("请运行以下命令安装：")
            print("  pip install tf-keras")
            print("=" * 60)
            print("继续尝试运行，如果失败请安装 tf-keras")
except ImportError:
    pass

# 尝试导入Transformers并检查版本
try:
    import transformers
    transformers_version = transformers.__version__
    print(f"✓ 成功导入 Transformers {transformers_version}")
    
    # 检查版本兼容性
    from packaging import version
    if version.parse(transformers_version) >= version.parse("5.0.0"):
        print("=" * 60)
        print("警告：检测到 Transformers 5.0+ 版本")
        print("=" * 60)
        print("TensorFlow BERT模型在Transformers 5.0+中存在兼容性问题")
        print("建议降级到兼容版本：")
        print("  pip install 'transformers<5.0'")
        print("  或者")
        print("  pip install transformers==4.30.0")
        print("\n继续尝试运行，如果失败请降级transformers库")
        print("=" * 60)
    
    from transformers import BertTokenizer, TFBertModel
except ImportError:
    try:
        from packaging import version
    except ImportError:
        print("警告：无法检查版本，请安装 packaging: pip install packaging")
        from transformers import BertTokenizer, TFBertModel
    else:
        from transformers import BertTokenizer, TFBertModel
except (ImportError, ValueError) as e:
    print("=" * 60)
    print("错误：无法导入Transformers或BERT模型")
    print("=" * 60)
    if "tf-keras" in str(e) or "Keras 3" in str(e):
        print("Keras 3 兼容性问题：")
        print("  请安装 tf-keras: pip install tf-keras")
    else:
        print(f"详细错误: {e}")
        print("\n请安装transformers：")
        print("  pip install transformers")
    print("=" * 60)
    sys.exit(1)

from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint


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


# 2. BERT编码

# 加载BERT tokenizer
print("\n正在加载BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("✓ BERT tokenizer加载完成")

def bert_encode(data, maximum_length):
    input_ids = []
    attention_masks = []

    for text in data:
        encoded = tokenizer.encode_plus(
            text, 
            add_special_tokens=True,
            max_length=maximum_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
        
    return np.array(input_ids), np.array(attention_masks)


# 编码文本
max_length = 60
train_input_ids, train_attention_masks = bert_encode(texts, max_length)

print(f"输入ID形状: {train_input_ids.shape}")
print(f"注意力掩码形状: {train_attention_masks.shape}")


# 3. 构建BERT模型

# 加载预训练的BERT模型
print("\n正在加载预训练的BERT模型...")
bert_model = None

# 尝试多种方式加载模型
loading_methods = [
    # 方法1: 标准加载
    lambda: TFBertModel.from_pretrained('bert-base-uncased'),
    # 方法2: 禁用safetensors
    lambda: TFBertModel.from_pretrained('bert-base-uncased', use_safetensors=False),
    # 方法3: 使用本地文件（如果已下载）
    lambda: TFBertModel.from_pretrained('bert-base-uncased', local_files_only=False),
]

for i, load_method in enumerate(loading_methods, 1):
    try:
        print(f"尝试方法 {i}...")
        bert_model = load_method()
        print("✓ BERT模型加载完成")
        break
    except (TypeError, ValueError, Exception) as e:
        if i == len(loading_methods):
            # 所有方法都失败了
            error_msg = str(e)
            print("=" * 60)
            print("错误：无法加载TensorFlow版本的BERT模型")
            print("=" * 60)
            print(f"详细错误: {error_msg}")
            print("\n这是Transformers新版本与TensorFlow BERT的已知兼容性问题")
            print("\n推荐解决方案：降级transformers库到兼容版本")
            print("  运行以下命令：")
            print("  pip install 'transformers<5.0'")
            print("  或者")
            print("  pip install transformers==4.30.0")
            print("\n降级后重新运行此脚本")
            print("=" * 60)
            sys.exit(1)
        else:
            print(f"方法 {i} 失败，尝试下一个方法...")
            continue

if bert_model is None:
    print("错误：所有加载方法都失败了")
    sys.exit(1)


# 创建自定义层来包装BERT模型
class BertLayer(tf.keras.layers.Layer):
    def __init__(self, bert_model, **kwargs):
        super(BertLayer, self).__init__(**kwargs)
        self.bert_model = bert_model
    
    def call(self, inputs):
        input_ids, attention_mask = inputs
        # 确保输入是Tensor类型
        input_ids = tf.cast(input_ids, tf.int32)
        attention_mask = tf.cast(attention_mask, tf.int32)
        
        # 尝试不同的调用方式
        try:
            # 方法1: 字典格式
            bert_output = self.bert_model({'input_ids': input_ids, 'attention_mask': attention_mask})
        except (TypeError, ValueError):
            try:
                # 方法2: 关键字参数
                bert_output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
            except (TypeError, ValueError):
                # 方法3: 位置参数
                bert_output = self.bert_model(input_ids, attention_mask=attention_mask)
        
        # 提取pooled output
        if isinstance(bert_output, tuple):
            return bert_output[1]  # pooled output
        elif hasattr(bert_output, 'pooler_output') and bert_output.pooler_output is not None:
            return bert_output.pooler_output
        elif hasattr(bert_output, 'last_hidden_state'):
            # 使用CLS token (第一个token)
            return bert_output.last_hidden_state[:, 0, :]
        else:
            # 如果都不存在，尝试直接使用输出
            return bert_output
    
    def get_config(self):
        config = super().get_config()
        return config

def create_model(bert_model):
    input_ids = tf.keras.Input(shape=(max_length,), dtype='int32', name='input_ids')
    attention_masks = tf.keras.Input(shape=(max_length,), dtype='int32', name='attention_mask')

    # 使用自定义层包装BERT模型
    bert_layer = BertLayer(bert_model, name='bert_layer')
    bert_output = bert_layer([input_ids, attention_masks])
    
    # 添加分类层
    output = tf.keras.layers.Dense(32, activation='relu')(bert_output)
    output = tf.keras.layers.Dropout(0.2)(output)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(output)
    
    model = tf.keras.models.Model(inputs=[input_ids, attention_masks], outputs=output)
    # 兼容不同版本的Keras，尝试使用learning_rate，如果失败则使用lr
    try:
        optimizer = Adam(learning_rate=1e-5)
    except TypeError:
        optimizer = Adam(lr=1e-5)
    model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = create_model(bert_model)
model.summary()


# 4. 训练模型

# 划分训练集和测试集
X_train_ids, X_test_ids, X_train_masks, X_test_masks, y_train, y_test = train_test_split(
    train_input_ids, train_attention_masks, target,
    test_size=0.2,
    random_state=42
)

print(f"\n训练集: {len(X_train_ids)}")
print(f"测试集: {len(X_test_ids)}")

# 设置模型保存路径
model_file = os.path.join(current_dir, 'bert_model.h5')
checkpoint = ModelCheckpoint(
    model_file,
    monitor='val_loss',
    verbose=1,
    save_best_only=True
)

# 训练模型
print("\n开始训练BERT模型...")
history = model.fit(
    [X_train_ids, X_train_masks],
    y_train,
    validation_data=([X_test_ids, X_test_masks], y_test),
    epochs=3,
    batch_size=10,
    verbose=1,
    callbacks=[checkpoint]
)
print("✓ 模型训练完成")


# 5. 可视化训练过程

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

learning_curves_file = os.path.join(current_dir, "learning_curves_bert.png")
plot_learning_curves(history, [['loss', 'val_loss'], ['accuracy', 'val_accuracy']], 
                     output_file=learning_curves_file)

# 6. 模型评估

# 预测
print("\n开始预测...")
y_pred_prob = model.predict([X_test_ids, X_test_masks], verbose=0)
y_pred_class = (y_pred_prob > 0.5).astype("int32")

# 计算所有评价指标
accuracy = metrics.accuracy_score(y_test, y_pred_class)
auc = metrics.roc_auc_score(y_test, y_pred_prob)
precision = metrics.precision_score(y_test, y_pred_class)
recall = metrics.recall_score(y_test, y_pred_class)
f1 = metrics.f1_score(y_test, y_pred_class)

print(f"\n模型性能指标:")
print(f"  准确率 (Accuracy): {accuracy:.4f}")
print(f"  AUC: {auc:.4f}")
print(f"  精确率 (Precision): {precision:.4f}")
print(f"  召回率 (Recall): {recall:.4f}")
print(f"  F1分数: {f1:.4f}")

# 分类报告
print(f"\n分类报告:")
print(metrics.classification_report(y_test, y_pred_class, target_names=['Ham', 'Spam']))

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

conf_matrix_file = os.path.join(current_dir, "confusion_matrix_bert.png")
conf_matrix(metrics.confusion_matrix(y_test, y_pred_class),
            labels=['Ham', 'Spam'],
            output_file=conf_matrix_file,
            title='BERT 混淆矩阵')

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

roc_file = os.path.join(current_dir, "roc_curve_bert.png")
plot_roc_curve(y_test, y_pred_prob, output_file=roc_file, title='BERT ROC曲线')

