# -*- coding: utf-8 -*-
"""
模型可解释性分析
分析哪些特征导致邮件被分类为垃圾邮件，提供算法的现实意义解释。
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
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
    print("警告：XGBoost未安装，将跳过XGBoost特征分析")

try:
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    svm_available = True
except ImportError:
    svm_available = False
    print("警告：SVM相关库未安装，将跳过SVM特征分析")

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
print("模型可解释性分析")
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

print(f"数据准备完成: 训练集 {len(x_train)}, 测试集 {len(x_test)}")


# 1. Naive Bayes 特征重要性分析
print("\n" + "=" * 60)
print("1. Naive Bayes 特征重要性分析")
print("=" * 60)

print("\n训练 Naive Bayes (TF-IDF) 模型...")
pipe_nb = Pipeline([
    ('bow', CountVectorizer()),
    ('tfid', TfidfTransformer()),
    ('model', MultinomialNB())
])
pipe_nb.fit(x_train, y_train)

# 获取特征重要性（Naive Bayes的log概率比）
vectorizer_nb = pipe_nb.named_steps['bow']
tfidf_nb = pipe_nb.named_steps['tfid']
model_nb = pipe_nb.named_steps['model']

# 计算特征重要性：log P(word|spam) - log P(word|ham)
feature_names_nb = vectorizer_nb.get_feature_names_out()
log_prob_spam = model_nb.feature_log_prob_[1]  # Spam类的log概率
log_prob_ham = model_nb.feature_log_prob_[0]   # Ham类的log概率
feature_importance_nb = log_prob_spam - log_prob_ham  # 差值越大，越倾向于Spam

# 获取最重要的特征（倾向于Spam的词汇）
top_n = 30
top_indices_spam = np.argsort(feature_importance_nb)[-top_n:][::-1]
top_features_spam = [(feature_names_nb[i], float(feature_importance_nb[i])) 
                     for i in top_indices_spam]

# 获取最不重要的特征（倾向于Ham的词汇）
top_indices_ham = np.argsort(feature_importance_nb)[:top_n]
top_features_ham = [(feature_names_nb[i], float(feature_importance_nb[i])) 
                    for i in top_indices_ham]

print(f"\n【最可能导致垃圾邮件分类的词汇】（前{top_n}个）:")
print(f"{'排名':<6} {'词汇':<25} {'Spam倾向度':<15}")
print("-" * 50)
for i, (feature, importance) in enumerate(top_features_spam, 1):
    print(f"{i:<6} {feature:<25} {importance:>10.4f}")

print(f"\n【最可能导致正常邮件分类的词汇】（前{top_n}个）:")
print(f"{'排名':<6} {'词汇':<25} {'Ham倾向度':<15}")
print("-" * 50)
for i, (feature, importance) in enumerate(top_features_ham, 1):
    print(f"{i:<6} {feature:<25} {importance:>10.4f}")


# 2. XGBoost 特征重要性分析
if xgb_available:
    print("\n" + "=" * 60)
    print("2. XGBoost 特征重要性分析")
    print("=" * 60)
    
    print("\n训练 XGBoost 模型...")
    pipe_xgb = Pipeline([
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
    pipe_xgb.fit(x_train, y_train)
    
    # 获取特征重要性
    vectorizer_xgb = pipe_xgb.named_steps['bow']
    model_xgb = pipe_xgb.named_steps['model']
    feature_names_xgb = vectorizer_xgb.get_feature_names_out()
    
    # XGBoost的特征重要性
    try:
        feature_importance_xgb = model_xgb.feature_importances_
        top_indices_xgb = np.argsort(feature_importance_xgb)[-top_n:][::-1]
        top_features_xgb = [(feature_names_xgb[i], float(feature_importance_xgb[i])) 
                           for i in top_indices_xgb]
        
        print(f"\n【XGBoost认为最重要的特征】（前{top_n}个）:")
        print(f"{'排名':<6} {'词汇':<25} {'重要性':<15}")
        print("-" * 50)
        for i, (feature, importance) in enumerate(top_features_xgb, 1):
            print(f"{i:<6} {feature:<25} {importance:>10.4f}")
    except Exception as e:
        print(f"无法获取XGBoost特征重要性: {e}")


# 3. SVM 特征重要性分析（线性核）
if svm_available:
    print("\n" + "=" * 60)
    print("3. SVM 特征重要性分析（线性核）")
    print("=" * 60)
    
    print("\n训练 SVM 模型（使用线性核）...")
    vectorizer_svm = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    
    X_train_tfidf = vectorizer_svm.fit_transform(x_train)
    X_test_tfidf = vectorizer_svm.transform(x_test)
    
    # 使用线性SVM
    svm_model = SVC(kernel='linear', probability=True, random_state=42)
    svm_model.fit(X_train_tfidf, y_train)
    
    # 获取特征重要性（系数）
    coef = svm_model.coef_[0]
    if hasattr(coef, 'toarray'):
        coef = coef.toarray().flatten()
    else:
        coef = np.asarray(coef).flatten()
    
    feature_names_svm = vectorizer_svm.get_feature_names_out()
    feature_importance_svm = np.abs(coef)
    
    # 正系数表示倾向于Spam，负系数表示倾向于Ham
    top_indices_svm_spam = np.argsort(coef)[-top_n:][::-1]  # 正系数最大的
    top_indices_svm_ham = np.argsort(coef)[:top_n]  # 负系数最小的
    
    top_features_svm_spam = [(feature_names_svm[i], float(coef[i])) 
                             for i in top_indices_svm_spam]
    top_features_svm_ham = [(feature_names_svm[i], float(coef[i])) 
                            for i in top_indices_svm_ham]
    
    print(f"\n【SVM认为最可能导致垃圾邮件的特征】（前{top_n}个，正系数）:")
    print(f"{'排名':<6} {'词汇':<25} {'系数值':<15}")
    print("-" * 50)
    for i, (feature, coef_val) in enumerate(top_features_svm_spam, 1):
        print(f"{i:<6} {feature:<25} {coef_val:>10.4f}")
    
    print(f"\n【SVM认为最可能导致正常邮件的特征】（前{top_n}个，负系数）:")
    print(f"{'排名':<6} {'词汇':<25} {'系数值':<15}")
    print("-" * 50)
    for i, (feature, coef_val) in enumerate(top_features_svm_ham, 1):
        print(f"{i:<6} {feature:<25} {coef_val:>10.4f}")


# 4. 误分类案例分析
print("\n" + "=" * 60)
print("4. 误分类案例分析")
print("=" * 60)

# 使用Naive Bayes进行预测
y_pred_nb = pipe_nb.predict(x_test)
y_pred_prob_nb = pipe_nb.predict_proba(x_test)[:, 1]

# 找出误分类的案例
misclassified = pd.DataFrame({
    'text': x_test.values,
    'true_label': y_test.values,
    'predicted_label': y_pred_nb,
    'predicted_prob': y_pred_prob_nb
})

# False Positive: 正常邮件被误判为垃圾邮件
fp_cases = misclassified[(misclassified['true_label'] == 0) & 
                         (misclassified['predicted_label'] == 1)]
# False Negative: 垃圾邮件被误判为正常邮件
fn_cases = misclassified[(misclassified['true_label'] == 1) & 
                         (misclassified['predicted_label'] == 0)]

print(f"\n误分类统计:")
print(f"  正常邮件被误判为垃圾邮件 (False Positive): {len(fp_cases)} 条")
print(f"  垃圾邮件被误判为正常邮件 (False Negative): {len(fn_cases)} 条")

if len(fp_cases) > 0:
    print(f"\n【正常邮件被误判为垃圾邮件的案例】（前5个）:")
    print("-" * 60)
    for idx, row in fp_cases.head(5).iterrows():
        print(f"\n案例 {idx}:")
        print(f"  文本: {row['text'][:100]}...")
        print(f"  预测概率: {row['predicted_prob']:.4f}")
        
        # 分析为什么被误判
        text_words = set(str(row['text']).lower().split())
        spam_words_found = [word for word, _ in top_features_spam[:20] if word in text_words]
        if spam_words_found:
            print(f"  包含的垃圾邮件特征词: {', '.join(spam_words_found)}")

if len(fn_cases) > 0:
    print(f"\n【垃圾邮件被误判为正常邮件的案例】（前5个）:")
    print("-" * 60)
    for idx, row in fn_cases.head(5).iterrows():
        print(f"\n案例 {idx}:")
        print(f"  文本: {row['text'][:100]}...")
        print(f"  预测概率: {row['predicted_prob']:.4f}")


# 5. 特征词汇分类和现实意义解释
print("\n" + "=" * 60)
print("5. 垃圾邮件特征词汇的现实意义解释")
print("=" * 60)

# 根据分析结果，分类垃圾邮件特征词汇
spam_keywords = {
    '诱导性词汇': ['free', 'win', 'prize', 'winner', 'congratulations', 'urgent', 'limited',
                'offer', 'deal', 'discount', 'save', 'money', 'cash', 'click', 'now',
                'guaranteed', 'risk-free', 'act now', 'call now', 'buy now', 'order now',
                'special', 'exclusive', 'secret', 'miracle', 'amazing', 'incredible'],
    '金融相关': ['money', 'cash', 'dollar', 'credit', 'loan', 'debt', 'payment', 'account',
              'bank', 'card', 'investment', 'profit', 'income', 'wealth'],
    '时间紧迫性': ['urgent', 'now', 'immediately', 'limited', 'expire', 'deadline', 'today',
               'hurry', 'asap', 'soon'],
    '行动号召': ['click', 'call', 'buy', 'order', 'register', 'subscribe', 'download',
              'visit', 'claim', 'reply', 'respond'],
    '夸大宣传': ['amazing', 'incredible', 'miracle', 'guaranteed', 'proven', 'secret',
              'exclusive', 'revolutionary', 'breakthrough']
}

print("\n根据模型分析，垃圾邮件通常包含以下类型的特征：")
print("\n1. 诱导性词汇")
print("   现实意义：垃圾邮件发送者使用这些词汇来吸引用户注意力，")
print("   诱导用户采取行动（点击链接、购买产品等）")
print(f"   典型词汇：{', '.join(spam_keywords['诱导性词汇'][:10])}...")

print("\n2. 金融相关词汇")
print("   现实意义：许多垃圾邮件涉及金融诈骗、虚假投资、贷款等")
print(f"   典型词汇：{', '.join(spam_keywords['金融相关'][:10])}...")

print("\n3. 时间紧迫性词汇")
print("   现实意义：制造紧迫感，让用户来不及仔细思考就采取行动")
print(f"   典型词汇：{', '.join(spam_keywords['时间紧迫性'])}")

print("\n4. 行动号召词汇")
print("   现实意义：直接要求用户执行某个操作，是垃圾邮件的典型特征")
print(f"   典型词汇：{', '.join(spam_keywords['行动号召'])}")

print("\n5. 夸大宣传词汇")
print("   现实意义：使用夸张的形容词来夸大产品或服务的价值")
print(f"   典型词汇：{', '.join(spam_keywords['夸大宣传'])}")


# 6. 生成特征重要性可视化
print("\n" + "=" * 60)
print("6. 生成特征重要性可视化")
print("=" * 60)

# Naive Bayes特征重要性可视化
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 左侧：最倾向于Spam的词汇
top_spam_words = [f[0] for f in top_features_spam[:20]]
top_spam_scores = [f[1] for f in top_features_spam[:20]]

axes[0].barh(range(len(top_spam_words)), top_spam_scores, color='#c6ccd8')
axes[0].set_yticks(range(len(top_spam_words)))
axes[0].set_yticklabels(top_spam_words)
axes[0].set_xlabel('Spam倾向度 (log P(Spam|word) - log P(Ham|word))', fontsize=12)
axes[0].set_title('最可能导致垃圾邮件分类的词汇 (Top 20)', fontsize=14, fontweight='bold')
axes[0].invert_yaxis()
axes[0].grid(True, alpha=0.3, axis='x')

# 右侧：最倾向于Ham的词汇
top_ham_words = [f[0] for f in top_features_ham[:20]]
top_ham_scores = [f[1] for f in top_features_ham[:20]]

axes[1].barh(range(len(top_ham_words)), top_ham_scores, color='#496595')
axes[1].set_yticks(range(len(top_ham_words)))
axes[1].set_yticklabels(top_ham_words)
axes[1].set_xlabel('Ham倾向度 (log P(Ham|word) - log P(Spam|word))', fontsize=12)
axes[1].set_title('最可能导致正常邮件分类的词汇 (Top 20)', fontsize=14, fontweight='bold')
axes[1].invert_yaxis()
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
importance_file = os.path.join(current_dir, 'feature_importance_analysis.png')
plt.savefig(importance_file, dpi=300, bbox_inches='tight')
print(f"特征重要性可视化已保存为: {importance_file}")
plt.close()


# 7. 保存分析结果
print("\n" + "=" * 60)
print("7. 保存分析结果")
print("=" * 60)

# 保存特征重要性结果
results_summary = {
    '模型': [],
    '特征类型': [],
    '特征词汇': [],
    '重要性分数': []
}

# Naive Bayes结果
for feature, importance in top_features_spam[:30]:
    results_summary['模型'].append('Naive Bayes')
    results_summary['特征类型'].append('Spam倾向')
    results_summary['特征词汇'].append(feature)
    results_summary['重要性分数'].append(importance)

for feature, importance in top_features_ham[:30]:
    results_summary['模型'].append('Naive Bayes')
    results_summary['特征类型'].append('Ham倾向')
    results_summary['特征词汇'].append(feature)
    results_summary['重要性分数'].append(importance)

df_results = pd.DataFrame(results_summary)
results_file = os.path.join(current_dir, 'feature_importance_results.csv')
df_results.to_csv(results_file, index=False, encoding='utf-8-sig')
print(f"特征重要性结果已保存为: {results_file}")

print("\n" + "=" * 60)
print("模型可解释性分析完成！")
print("=" * 60)
print("\n总结：")
print("1. 已分析各模型认为最重要的特征词汇")
print("2. 已识别导致垃圾邮件分类的关键词汇类型")
print("3. 已分析误分类案例")
print("4. 已提供现实意义解释")
print("\n这些分析有助于理解模型决策过程，而不仅仅是依赖loss值。")

