# -*- coding: utf-8 -*-

# Naive Bayes 模型

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import metrics

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# 1. 加载预处理后的数据

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
preprocessed_file = os.path.join(current_dir, 'spam_preprocessed.csv')
preprocessing_script = os.path.join(current_dir, '01_data_loading_and_preprocessing.py')

# 检查预处理文件是否存在，如果不存在则运行预处理脚本
if not os.path.exists(preprocessed_file):
    print(f"未找到预处理文件 {preprocessed_file}")
    print("正在运行数据预处理脚本...")
    if os.path.exists(preprocessing_script):
        import subprocess
        result = subprocess.run([sys.executable, preprocessing_script], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print("错误：预处理脚本运行失败")
            print(result.stderr)
            sys.exit(1)
        print("预处理完成！")
    else:
        print(f"错误：找不到预处理脚本 {preprocessing_script}")
        print("请先运行 01_data_loading_and_preprocessing.py 生成预处理文件")
        sys.exit(1)

# 加载数据
df = pd.read_csv(preprocessed_file)

# 数据清理：处理NaN值
print(f"原始数据量: {len(df)}")
print(f"message_clean中的NaN数量: {df['message_clean'].isna().sum()}")
print(f"target_encoded中的NaN数量: {df['target_encoded'].isna().sum()}")

# 移除包含NaN的行
df_clean = df.dropna(subset=['message_clean', 'target_encoded'])
print(f"清理后数据量: {len(df_clean)}")

# 确保message_clean都是字符串类型，并将NaN填充为空字符串（如果还有的话）
df_clean['message_clean'] = df_clean['message_clean'].astype(str)
df_clean['message_clean'] = df_clean['message_clean'].replace('nan', '')
df_clean['message_clean'] = df_clean['message_clean'].fillna('')

# 移除空字符串（可选，但建议保留，因为空字符串也是有效的文档）
# df_clean = df_clean[df_clean['message_clean'].str.strip() != '']

x = df_clean['message_clean']
y = df_clean['target_encoded']

print(f"最终数据量: {len(x)}")
print(f"标签量: {len(y)}")

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.25)
print(f"训练集: {len(x_train)}")
print(f"测试集: {len(x_test)}")


# 2. 混淆矩阵可视化函数

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


# 3. Naive Bayes with DTM (Document Term Matrix)

# 确保训练和测试数据都是字符串类型且没有NaN
x_train = x_train.astype(str).replace('nan', '').fillna('')
x_test = x_test.astype(str).replace('nan', '').fillna('')

# 使用CountVectorizer创建文档-词矩阵
vect = CountVectorizer()
vect.fit(x_train)
x_train_dtm = vect.transform(x_train)
x_test_dtm = vect.transform(x_test)


# 创建并训练Naive Bayes模型
nb = MultinomialNB()
nb.fit(x_train_dtm, y_train)


# 预测和评估
y_pred_class = nb.predict(x_test_dtm)
y_pred_prob = nb.predict_proba(x_test_dtm)[:, 1]

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

conf_matrix_file = os.path.join(current_dir, "confusion_matrix_nb_dtm.png")
conf_matrix(metrics.confusion_matrix(y_test, y_pred_class), 
            labels=['Ham', 'Spam'], 
            output_file=conf_matrix_file,
            title='Naive Bayes (DTM) 混淆矩阵')

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

roc_file = os.path.join(current_dir, "roc_curve_nb_dtm.png")
plot_roc_curve(y_test, y_pred_prob, output_file=roc_file, title='Naive Bayes (DTM) ROC曲线')


# 4. Naive Bayes with TF-IDF

# 使用Pipeline组合CountVectorizer、TfidfTransformer和MultinomialNB
pipe = Pipeline([
    ('bow', CountVectorizer()), 
    ('tfid', TfidfTransformer()),  
    ('model', MultinomialNB())
])


# 训练模型
pipe.fit(x_train, y_train)

# 预测
y_pred_class = pipe.predict(x_test)
y_pred_prob = pipe.predict_proba(x_test)[:, 1]

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

conf_matrix_file_tfidf = os.path.join(current_dir, "confusion_matrix_nb_tfidf.png")
conf_matrix(metrics.confusion_matrix(y_test, y_pred_class),
            labels=['Ham', 'Spam'],
            output_file=conf_matrix_file_tfidf,
            title='Naive Bayes (TF-IDF) 混淆矩阵')

# ROC曲线
roc_file_tfidf = os.path.join(current_dir, "roc_curve_nb_tfidf.png")
plot_roc_curve(y_test, y_pred_prob, output_file=roc_file_tfidf, title='Naive Bayes (TF-IDF) ROC曲线')

