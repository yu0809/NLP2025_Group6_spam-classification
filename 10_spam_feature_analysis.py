# -*- coding: utf-8 -*-
"""
垃圾邮件特征分析
分析垃圾邮件的文本特征，包括诱导性词汇、语法错误等。
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(current_dir, 'spam.csv')

# 检查数据文件是否存在
if not os.path.exists(data_file):
    print(f"错误：未找到数据文件 {data_file}")
    sys.exit(1)

# 加载原始数据
df = pd.read_csv(data_file, encoding="latin-1")
df = df.dropna(how="any", axis=1)
df.columns = ['target', 'message']

print("=" * 60)
print("垃圾邮件特征分析")
print("=" * 60)
print(f"总数据量: {len(df)}")
print(f"Ham数量: {len(df[df['target'] == 'ham'])}")
print(f"Spam数量: {len(df[df['target'] == 'spam'])}")


# 1. 诱导性词汇分析
print("\n" + "=" * 60)
print("1. 诱导性词汇分析")
print("=" * 60)

# 定义常见的诱导性词汇（英文）
inducement_words = [
    'free', 'win', 'prize', 'winner', 'congratulations', 'urgent', 'limited',
    'offer', 'deal', 'discount', 'save', 'money', 'cash', 'click', 'now',
    'guaranteed', 'risk-free', 'act now', 'call now', 'buy now', 'order now',
    'special', 'exclusive', 'secret', 'miracle', 'amazing', 'incredible'
]

def count_inducement_words(text):
    """统计文本中的诱导性词汇数量"""
    if pd.isna(text):
        return 0
    text_lower = str(text).lower()
    count = sum(1 for word in inducement_words if word in text_lower)
    return count

df['inducement_count'] = df['message'].apply(count_inducement_words)

# 统计
spam_inducement = df[df['target'] == 'spam']['inducement_count']
ham_inducement = df[df['target'] == 'ham']['inducement_count']

print(f"\n诱导性词汇统计:")
print(f"  Spam平均包含: {spam_inducement.mean():.2f} 个诱导性词汇")
print(f"  Ham平均包含: {ham_inducement.mean():.2f} 个诱导性词汇")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 柱状图对比
categories = ['Ham', 'Spam']
means = [ham_inducement.mean(), spam_inducement.mean()]
bars = axes[0].bar(categories, means, color=['#496595', '#c6ccd8'], alpha=0.8)
axes[0].set_title('平均诱导性词汇数量', fontsize=14, fontweight='bold')
axes[0].set_ylabel('平均数量', fontsize=12)
axes[0].grid(True, alpha=0.3, axis='y')
for bar, mean in zip(bars, means):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{mean:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# 分布直方图
axes[1].hist(ham_inducement, bins=20, alpha=0.6, label='Ham', color='#496595', edgecolor='black')
axes[1].hist(spam_inducement, bins=20, alpha=0.6, label='Spam', color='#c6ccd8', edgecolor='black')
axes[1].set_title('诱导性词汇数量分布', fontsize=14, fontweight='bold')
axes[1].set_xlabel('诱导性词汇数量', fontsize=12)
axes[1].set_ylabel('频数', fontsize=12)
axes[1].legend(fontsize=12)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
inducement_file = os.path.join(current_dir, 'spam_inducement_analysis.png')
plt.savefig(inducement_file, dpi=300, bbox_inches='tight')
print(f"诱导性词汇分析图已保存为: {inducement_file}")
plt.close()


# 2. 最常见的垃圾邮件词汇
print("\n" + "=" * 60)
print("2. 垃圾邮件高频词汇分析")
print("=" * 60)

spam_texts = ' '.join(df[df['target'] == 'spam']['message'].astype(str).str.lower())
ham_texts = ' '.join(df[df['target'] == 'ham']['message'].astype(str).str.lower())

# 简单的词汇提取（移除标点）
spam_words = re.findall(r'\b[a-z]+\b', spam_texts)
ham_words = re.findall(r'\b[a-z]+\b', ham_texts)

# 统计词频
spam_word_freq = Counter(spam_words)
ham_word_freq = Counter(ham_words)

# 找出spam中频率高但ham中频率低的词
spam_specific_words = {}
for word, count in spam_word_freq.most_common(100):
    spam_freq = count / len(spam_words)
    ham_freq = ham_word_freq.get(word, 0) / len(ham_words) if len(ham_words) > 0 else 0
    if spam_freq > ham_freq * 2:  # spam中的频率至少是ham的2倍
        spam_specific_words[word] = {
            'spam_freq': spam_freq,
            'ham_freq': ham_freq,
            'ratio': spam_freq / (ham_freq + 1e-6)
        }

# 排序并显示前20个
top_spam_words = sorted(spam_specific_words.items(), key=lambda x: x[1]['ratio'], reverse=True)[:20]

print(f"\n垃圾邮件特征词汇（前20个）:")
print(f"{'排名':<6} {'词汇':<20} {'Spam频率':<15} {'Ham频率':<15} {'比率':<10}")
print("-" * 70)
for i, (word, info) in enumerate(top_spam_words, 1):
    print(f"{i:<6} {word:<20} {info['spam_freq']:<15.6f} {info['ham_freq']:<15.6f} {info['ratio']:<10.2f}")


# 3. 消息长度分析
print("\n" + "=" * 60)
print("3. 消息长度特征分析")
print("=" * 60)

df['message_length'] = df['message'].apply(lambda x: len(str(x)))
df['word_count'] = df['message'].apply(lambda x: len(str(x).split()))

spam_length = df[df['target'] == 'spam']['message_length']
ham_length = df[df['target'] == 'ham']['message_length']
spam_words_count = df[df['target'] == 'spam']['word_count']
ham_words_count = df[df['target'] == 'ham']['word_count']

print(f"\n消息长度统计:")
print(f"  Spam平均字符数: {spam_length.mean():.1f}")
print(f"  Ham平均字符数: {ham_length.mean():.1f}")
print(f"  Spam平均单词数: {spam_words_count.mean():.1f}")
print(f"  Ham平均单词数: {ham_words_count.mean():.1f}")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 字符数分布
axes[0].hist(ham_length, bins=50, alpha=0.6, label='Ham', color='#496595', edgecolor='black', density=True)
axes[0].hist(spam_length, bins=50, alpha=0.6, label='Spam', color='#c6ccd8', edgecolor='black', density=True)
axes[0].set_title('消息字符数分布', fontsize=14, fontweight='bold')
axes[0].set_xlabel('字符数', fontsize=12)
axes[0].set_ylabel('密度', fontsize=12)
axes[0].legend(fontsize=12)
axes[0].set_xlim(0, 500)
axes[0].grid(True, alpha=0.3, axis='y')

# 单词数分布
axes[1].hist(ham_words_count, bins=50, alpha=0.6, label='Ham', color='#496595', edgecolor='black', density=True)
axes[1].hist(spam_words_count, bins=50, alpha=0.6, label='Spam', color='#c6ccd8', edgecolor='black', density=True)
axes[1].set_title('消息单词数分布', fontsize=14, fontweight='bold')
axes[1].set_xlabel('单词数', fontsize=12)
axes[1].set_ylabel('密度', fontsize=12)
axes[1].legend(fontsize=12)
axes[1].set_xlim(0, 100)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
length_file = os.path.join(current_dir, 'spam_length_analysis.png')
plt.savefig(length_file, dpi=300, bbox_inches='tight')
print(f"消息长度分析图已保存为: {length_file}")
plt.close()


# 4. 大写字母比例分析（可能的语法错误指标）
print("\n" + "=" * 60)
print("4. 大写字母比例分析（可能的语法错误指标）")
print("=" * 60)

def uppercase_ratio(text):
    """计算大写字母比例"""
    if pd.isna(text) or len(str(text)) == 0:
        return 0
    text_str = str(text)
    letters = [c for c in text_str if c.isalpha()]
    if len(letters) == 0:
        return 0
    uppercase = sum(1 for c in letters if c.isupper())
    return uppercase / len(letters)

df['uppercase_ratio'] = df['message'].apply(uppercase_ratio)

spam_upper = df[df['target'] == 'spam']['uppercase_ratio']
ham_upper = df[df['target'] == 'ham']['uppercase_ratio']

print(f"\n大写字母比例统计:")
print(f"  Spam平均大写比例: {spam_upper.mean():.4f}")
print(f"  Ham平均大写比例: {ham_upper.mean():.4f}")

# 可视化
plt.figure(figsize=(10, 6))
plt.hist(ham_upper, bins=30, alpha=0.6, label='Ham', color='#496595', edgecolor='black', density=True)
plt.hist(spam_upper, bins=30, alpha=0.6, label='Spam', color='#c6ccd8', edgecolor='black', density=True)
plt.title('大写字母比例分布', fontsize=16, fontweight='bold')
plt.xlabel('大写字母比例', fontsize=12)
plt.ylabel('密度', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
upper_file = os.path.join(current_dir, 'spam_uppercase_analysis.png')
plt.savefig(upper_file, dpi=300, bbox_inches='tight')
print(f"大写字母比例分析图已保存为: {upper_file}")
plt.close()


# 5. 特殊字符分析（URL、数字等）
print("\n" + "=" * 60)
print("5. 特殊特征分析（URL、数字等）")
print("=" * 60)

def count_urls(text):
    """统计URL数量"""
    if pd.isna(text):
        return 0
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', str(text))
    return len(urls)

def count_numbers(text):
    """统计数字数量"""
    if pd.isna(text):
        return 0
    numbers = re.findall(r'\d+', str(text))
    return len(numbers)

df['url_count'] = df['message'].apply(count_urls)
df['number_count'] = df['message'].apply(count_numbers)

spam_urls = df[df['target'] == 'spam']['url_count']
ham_urls = df[df['target'] == 'ham']['url_count']
spam_numbers = df[df['target'] == 'spam']['number_count']
ham_numbers = df[df['target'] == 'ham']['number_count']

print(f"\n特殊特征统计:")
print(f"  Spam平均URL数: {spam_urls.mean():.2f}")
print(f"  Ham平均URL数: {ham_urls.mean():.2f}")
print(f"  Spam平均数字数: {spam_numbers.mean():.2f}")
print(f"  Ham平均数字数: {ham_numbers.mean():.2f}")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# URL数量
categories = ['Ham', 'Spam']
url_means = [ham_urls.mean(), spam_urls.mean()]
bars1 = axes[0].bar(categories, url_means, color=['#496595', '#c6ccd8'], alpha=0.8)
axes[0].set_title('平均URL数量', fontsize=14, fontweight='bold')
axes[0].set_ylabel('平均数量', fontsize=12)
axes[0].grid(True, alpha=0.3, axis='y')
for bar, mean in zip(bars1, url_means):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{mean:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# 数字数量
num_means = [ham_numbers.mean(), spam_numbers.mean()]
bars2 = axes[1].bar(categories, num_means, color=['#496595', '#c6ccd8'], alpha=0.8)
axes[1].set_title('平均数字数量', fontsize=14, fontweight='bold')
axes[1].set_ylabel('平均数量', fontsize=12)
axes[1].grid(True, alpha=0.3, axis='y')
for bar, mean in zip(bars2, num_means):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{mean:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
special_file = os.path.join(current_dir, 'spam_special_features.png')
plt.savefig(special_file, dpi=300, bbox_inches='tight')
print(f"特殊特征分析图已保存为: {special_file}")
plt.close()


# 6. 生成特征分析报告
print("\n" + "=" * 60)
print("特征分析总结")
print("=" * 60)

summary = {
    '特征': [
        '诱导性词汇数量',
        '消息字符数',
        '消息单词数',
        '大写字母比例',
        'URL数量',
        '数字数量'
    ],
    'Spam平均值': [
        f"{spam_inducement.mean():.2f}",
        f"{spam_length.mean():.1f}",
        f"{spam_words_count.mean():.1f}",
        f"{spam_upper.mean():.4f}",
        f"{spam_urls.mean():.2f}",
        f"{spam_numbers.mean():.2f}"
    ],
    'Ham平均值': [
        f"{ham_inducement.mean():.2f}",
        f"{ham_length.mean():.1f}",
        f"{ham_words_count.mean():.1f}",
        f"{ham_upper.mean():.4f}",
        f"{ham_urls.mean():.2f}",
        f"{ham_numbers.mean():.2f}"
    ]
}

df_summary = pd.DataFrame(summary)
print("\n特征对比表:")
print(df_summary.to_string(index=False))

# 保存报告
summary_file = os.path.join(current_dir, 'spam_feature_summary.csv')
df_summary.to_csv(summary_file, index=False, encoding='utf-8-sig')
print(f"\n特征分析报告已保存为: {summary_file}")

print("\n" + "=" * 60)
print("垃圾邮件特征分析完成！")
print("=" * 60)

