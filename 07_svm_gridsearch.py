# -*- coding: utf-8 -*-
"""
SVM (支持向量机) + 网格搜索优化
使用支持向量机进行垃圾邮件检测，通过网格搜索找到最优超参数。
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import time

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


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

# 移除包含NaN的行
df_clean = df.dropna(subset=['message_clean', 'target_encoded']).copy()
print(f"清理后数据量: {len(df_clean)}")

# 确保message_clean都是字符串类型
df_clean.loc[:, 'message_clean'] = df_clean['message_clean'].astype(str)
df_clean.loc[:, 'message_clean'] = df_clean['message_clean'].replace('nan', '')
df_clean.loc[:, 'message_clean'] = df_clean['message_clean'].fillna('')

# 移除空字符串
df_clean = df_clean[df_clean['message_clean'].str.strip() != ''].copy()

x = df_clean['message_clean']
y = df_clean['target_encoded']

print(f"最终数据量: {len(x)}")
print(f"标签量: {len(y)}")

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.25)
print(f"训练集: {len(x_train)}")
print(f"测试集: {len(x_test)}")

# 确保训练和测试数据都是字符串类型且没有NaN
x_train = x_train.astype(str).replace('nan', '').fillna('')
x_test = x_test.astype(str).replace('nan', '').fillna('')


# 2. 特征提取 - TF-IDF向量化

print("\n开始特征提取...")
# 使用TF-IDF向量化，限制特征数量以提高效率
vectorizer = TfidfVectorizer(
    max_features=5000,  # 限制特征数量
    ngram_range=(1, 2),  # 使用1-gram和2-gram
    min_df=2,  # 最小文档频率
    max_df=0.95,  # 最大文档频率
    sublinear_tf=True  # 使用对数缩放
)

X_train_tfidf = vectorizer.fit_transform(x_train)
X_test_tfidf = vectorizer.transform(x_test)

print(f"TF-IDF特征矩阵形状: {X_train_tfidf.shape}")
print(f"特征提取完成")


# 3. 网格搜索优化SVM超参数

print("\n开始网格搜索优化SVM超参数...")
print("这可能需要几分钟时间，请耐心等待...")

# 定义参数网格
# 为了节省时间，使用较小的参数范围
param_grid = {
    'C': [0.1, 1, 10],  # 正则化参数
    'gamma': ['scale', 'auto', 0.001, 0.01],  # RBF核参数
    'kernel': ['linear', 'rbf']  # 核函数类型
}

# 创建基础SVM模型
base_svm = SVC(probability=True, random_state=42)

# 使用网格搜索，使用3折交叉验证
print("正在进行网格搜索（3折交叉验证）...")
start_time = time.time()

grid_search = GridSearchCV(
    base_svm,
    param_grid,
    cv=3,  # 3折交叉验证
    scoring='roc_auc',  # 使用AUC作为评分标准
    n_jobs=-1,  # 使用所有CPU核心
    verbose=1  # 显示进度
)

grid_search.fit(X_train_tfidf, y_train)

elapsed_time = time.time() - start_time
print(f"\n网格搜索完成，耗时: {elapsed_time:.2f} 秒")

# 显示最佳参数
print("\n最佳超参数:")
print(f"  C: {grid_search.best_params_['C']}")
print(f"  gamma: {grid_search.best_params_['gamma']}")
print(f"  kernel: {grid_search.best_params_['kernel']}")
print(f"  最佳交叉验证AUC: {grid_search.best_score_:.4f}")


# 4. 使用最佳模型进行预测

print("\n使用最佳模型进行预测...")
best_svm = grid_search.best_estimator_

# 预测
y_pred_class = best_svm.predict(X_test_tfidf)
y_pred_prob = best_svm.predict_proba(X_test_tfidf)[:, 1]

# 评估
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


# 5. 混淆矩阵可视化

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

conf_matrix_file = os.path.join(current_dir, "confusion_matrix_svm.png")
conf_matrix(metrics.confusion_matrix(y_test, y_pred_class),
            labels=['Ham', 'Spam'],
            output_file=conf_matrix_file,
            title=f'SVM (GridSearch) 混淆矩阵\nC={grid_search.best_params_["C"]}, kernel={grid_search.best_params_["kernel"]}')

# 分类报告
print(f"\n分类报告:")
print(metrics.classification_report(y_test, y_pred_class, target_names=['Ham', 'Spam']))

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

roc_file = os.path.join(current_dir, "roc_curve_svm.png")
plot_roc_curve(y_test, y_pred_prob, output_file=roc_file, title='SVM (GridSearch) ROC曲线')


# 6. 特征重要性分析（仅对线性核）

if grid_search.best_params_['kernel'] == 'linear':
    print("\n分析特征重要性（线性核）...")
    # 获取特征重要性（线性SVM的系数）
    # 将稀疏矩阵转换为密集数组
    coef = best_svm.coef_[0]
    if hasattr(coef, 'toarray'):
        # 如果是稀疏矩阵，转换为密集数组
        coef = coef.toarray().flatten()
    else:
        # 如果是numpy数组，直接使用
        coef = np.asarray(coef).flatten()
    
    feature_importance = np.abs(coef)
    feature_names = vectorizer.get_feature_names_out()
    
    # 获取最重要的特征
    top_n = 20
    top_indices = np.argsort(feature_importance)[-top_n:][::-1]
    top_features = [(feature_names[i], float(feature_importance[i])) for i in top_indices]
    
    print(f"\n最重要的 {top_n} 个特征（用于垃圾邮件检测）:")
    for i, (feature, importance) in enumerate(top_features, 1):
        print(f"  {i:2d}. {feature:20s} (重要性: {importance:.4f})")

print("\n" + "=" * 60)
print("SVM模型训练和评估完成！")
print("=" * 60)

