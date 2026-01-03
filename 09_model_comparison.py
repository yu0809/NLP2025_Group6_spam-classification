# -*- coding: utf-8 -*-
"""
模型性能对比
统一对比所有模型的性能指标，生成对比表格和可视化。
"""

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
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 尝试导入其他模型库
try:
    import xgboost as xgb
    xgb_available = True
except ImportError:
    xgb_available = False
    print("警告：XGBoost未安装，将跳过XGBoost模型")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    svm_available = True
except ImportError:
    svm_available = False
    print("警告：SVM相关库未安装，将跳过SVM模型")

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
preprocessed_file = os.path.join(current_dir, 'spam_preprocessed.csv')

# 检查预处理文件是否存在
if not os.path.exists(preprocessed_file):
    print(f"错误：未找到预处理文件 {preprocessed_file}")
    print("请先运行 01_data_loading_and_preprocessing.py 生成预处理文件")
    sys.exit(1)

# 加载数据
print("=" * 60)
print("开始模型性能对比")
print("=" * 60)

df = pd.read_csv(preprocessed_file)
df_clean = df.dropna(subset=['message_clean', 'target_encoded']).copy()
df_clean.loc[:, 'message_clean'] = df_clean['message_clean'].astype(str)
df_clean.loc[:, 'message_clean'] = df_clean['message_clean'].replace('nan', '')
df_clean = df_clean[df_clean['message_clean'].str.strip() != ''].copy()

x = df_clean['message_clean']
y = df_clean['target_encoded']

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.25)
x_train = x_train.astype(str).replace('nan', '').fillna('')
x_test = x_test.astype(str).replace('nan', '').fillna('')

print(f"\n数据准备完成:")
print(f"  训练集: {len(x_train)}")
print(f"  测试集: {len(x_test)}")

# 存储所有模型的结果
results = []


# 1. Naive Bayes (DTM)
print("\n" + "-" * 60)
print("1. 训练 Naive Bayes (DTM)...")
print("-" * 60)
try:
    vect = CountVectorizer()
    vect.fit(x_train)
    x_train_dtm = vect.transform(x_train)
    x_test_dtm = vect.transform(x_test)
    
    nb = MultinomialNB()
    nb.fit(x_train_dtm, y_train)
    
    y_pred = nb.predict(x_test_dtm)
    y_pred_prob = nb.predict_proba(x_test_dtm)[:, 1]
    
    results.append({
        '模型': 'Naive Bayes (DTM)',
        '准确率': metrics.accuracy_score(y_test, y_pred),
        '精确率': metrics.precision_score(y_test, y_pred),
        '召回率': metrics.recall_score(y_test, y_pred),
        'F1分数': metrics.f1_score(y_test, y_pred),
        'AUC': metrics.roc_auc_score(y_test, y_pred_prob)
    })
    print("✓ 完成")
except Exception as e:
    print(f"✗ 失败: {e}")


# 2. Naive Bayes (TF-IDF)
print("\n" + "-" * 60)
print("2. 训练 Naive Bayes (TF-IDF)...")
print("-" * 60)
try:
    pipe = Pipeline([
        ('bow', CountVectorizer()),
        ('tfid', TfidfTransformer()),
        ('model', MultinomialNB())
    ])
    pipe.fit(x_train, y_train)
    
    y_pred = pipe.predict(x_test)
    y_pred_prob = pipe.predict_proba(x_test)[:, 1]
    
    results.append({
        '模型': 'Naive Bayes (TF-IDF)',
        '准确率': metrics.accuracy_score(y_test, y_pred),
        '精确率': metrics.precision_score(y_test, y_pred),
        '召回率': metrics.recall_score(y_test, y_pred),
        'F1分数': metrics.f1_score(y_test, y_pred),
        'AUC': metrics.roc_auc_score(y_test, y_pred_prob)
    })
    print("✓ 完成")
except Exception as e:
    print(f"✗ 失败: {e}")


# 3. XGBoost
if xgb_available:
    print("\n" + "-" * 60)
    print("3. 训练 XGBoost...")
    print("-" * 60)
    try:
        pipe = Pipeline([
            ('bow', CountVectorizer()),
            ('tfid', TfidfTransformer()),
            ('model', xgb.XGBClassifier(
                learning_rate=0.1,
                max_depth=7,
                n_estimators=80,
                use_label_encoder=False,
                eval_metric='auc',
                random_state=42,
            ))
        ])
        pipe.fit(x_train, y_train)
        
        y_pred = pipe.predict(x_test)
        y_pred_prob = pipe.predict_proba(x_test)[:, 1]
        
        results.append({
            '模型': 'XGBoost (TF-IDF)',
            '准确率': metrics.accuracy_score(y_test, y_pred),
            '精确率': metrics.precision_score(y_test, y_pred),
            '召回率': metrics.recall_score(y_test, y_pred),
            'F1分数': metrics.f1_score(y_test, y_pred),
            'AUC': metrics.roc_auc_score(y_test, y_pred_prob)
        })
        print("✓ 完成")
    except Exception as e:
        print(f"✗ 失败: {e}")
else:
    print("\n跳过 XGBoost（未安装）")


# 4. SVM (GridSearch)
if svm_available:
    print("\n" + "-" * 60)
    print("4. 训练 SVM (GridSearch)...")
    print("-" * 60)
    print("注意：SVM网格搜索可能需要几分钟时间...")
    try:
        # 使用TF-IDF向量化
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        X_train_tfidf = vectorizer.fit_transform(x_train)
        X_test_tfidf = vectorizer.transform(x_test)
        
        # 简化的参数网格（加快速度）
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['linear', 'rbf']
        }
        
        base_svm = SVC(probability=True, random_state=42)
        grid_search = GridSearchCV(
            base_svm,
            param_grid,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train_tfidf, y_train)
        best_svm = grid_search.best_estimator_
        
        y_pred = best_svm.predict(X_test_tfidf)
        y_pred_prob = best_svm.predict_proba(X_test_tfidf)[:, 1]
        
        results.append({
            '模型': 'SVM (GridSearch)',
            '准确率': metrics.accuracy_score(y_test, y_pred),
            '精确率': metrics.precision_score(y_test, y_pred),
            '召回率': metrics.recall_score(y_test, y_pred),
            'F1分数': metrics.f1_score(y_test, y_pred),
            'AUC': metrics.roc_auc_score(y_test, y_pred_prob)
        })
        print("✓ 完成")
    except Exception as e:
        print(f"✗ 失败: {e}")
else:
    print("\n跳过 SVM（未安装）")


# 5. 深度学习模型说明
print("\n" + "=" * 60)
print("深度学习模型说明")
print("=" * 60)
print("以下模型需要单独运行，训练时间较长：")
print("  - LSTM: 运行 04_lstm.py")
print("  - BERT: 运行 05_bert.py")
print("  - CNN: 运行 08_cnn_text.py")
print("\n这些模型的训练结果会保存在各自的脚本中。")
print("如需完整对比，请先运行这些脚本，然后查看各自的输出结果。")
print("=" * 60)


# 6. 创建对比表格
print("\n" + "=" * 60)
print("模型性能对比结果")
print("=" * 60)

if results:
    df_results = pd.DataFrame(results)
    df_results = df_results.round(4)
    
    # 按AUC排序
    df_results = df_results.sort_values('AUC', ascending=False)
    
    print("\n性能指标对比表:")
    print(df_results.to_string(index=False))
    
    # 添加说明
    print("\n注意：此对比表仅包含传统机器学习模型。")
    print("深度学习模型（LSTM、BERT、CNN）需要单独运行对应的脚本。")
    print("完整模型列表：")
    print("  1. Naive Bayes (DTM)")
    print("  2. Naive Bayes (TF-IDF)")
    print("  3. XGBoost")
    print("  4. SVM (GridSearch)")
    print("  5. LSTM - 运行 04_lstm.py 查看结果")
    print("  6. BERT - 运行 05_bert.py 查看结果")
    print("  7. CNN - 运行 08_cnn_text.py 查看结果")
    
    # 保存结果到CSV
    results_file = os.path.join(current_dir, 'model_comparison_results.csv')
    df_results.to_csv(results_file, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: {results_file}")
    
    # 可视化对比
    print("\n生成可视化对比图...")
    
    # 1. 指标对比柱状图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics_list = ['准确率', '精确率', '召回率', 'F1分数', 'AUC']
    
    for idx, metric in enumerate(metrics_list):
        ax = axes[idx // 3, idx % 3]
        bars = ax.bar(df_results['模型'], df_results[metric], color=plt.cm.Blues(np.linspace(0.4, 0.9, len(df_results))))
        ax.set_title(f'{metric}对比', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric, fontsize=12)
        ax.set_ylim([0, 1])
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=10)
    
    # 综合雷达图（使用最后一个子图位置）
    ax = axes[1, 2]
    ax.axis('off')
    ax.text(0.5, 0.5, '综合性能对比\n（详见各指标图表）', 
           ha='center', va='center', fontsize=14, 
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    comparison_file = os.path.join(current_dir, 'model_comparison.png')
    plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
    print(f"对比图已保存为: {comparison_file}")
    plt.close()
    
    # 2. 热力图对比
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_data = df_results.set_index('模型')[['准确率', '精确率', '召回率', 'F1分数', 'AUC']]
    sns.heatmap(metrics_data, annot=True, fmt='.4f', cmap='YlOrRd', 
                cbar_kws={'label': '分数'}, ax=ax)
    ax.set_title('模型性能热力图对比', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    heatmap_file = os.path.join(current_dir, 'model_comparison_heatmap.png')
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    print(f"热力图已保存为: {heatmap_file}")
    plt.close()
    
    # 3. 找出最佳模型
    print("\n" + "=" * 60)
    print("最佳模型分析")
    print("=" * 60)
    best_auc = df_results.loc[df_results['AUC'].idxmax()]
    best_f1 = df_results.loc[df_results['F1分数'].idxmax()]
    best_acc = df_results.loc[df_results['准确率'].idxmax()]
    
    print(f"\n最佳AUC模型: {best_auc['模型']} (AUC = {best_auc['AUC']:.4f})")
    print(f"最佳F1模型: {best_f1['模型']} (F1 = {best_f1['F1分数']:.4f})")
    print(f"最佳准确率模型: {best_acc['模型']} (准确率 = {best_acc['准确率']:.4f})")
    
    print("\n" + "=" * 60)
    print("模型对比完成！")
    print("=" * 60)
else:
    print("错误：没有成功训练任何模型")

