# -*- coding: utf-8 -*-


# 数据加载和预处理

import re
import string
import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import nltk
import ssl

from sklearn.preprocessing import LabelEncoder

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义备用英文停用词列表（如果NLTK下载失败时使用）
DEFAULT_STOPWORDS = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll",
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
    'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
    'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during',
    'before', 'after', 'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
    'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't",
    'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
    'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
    'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
    "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",
    'won', "won't", 'wouldn', "wouldn't", 'u', 'im', 'c'
]

# 尝试下载必要的NLTK数据（处理SSL证书问题）
def download_nltk_data():
    """下载NLTK数据，处理SSL证书问题"""
    try:
        # 尝试禁用SSL验证（仅用于下载NLTK数据）
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        # 下载punkt
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("正在下载NLTK punkt数据...")
            try:
                nltk.download('punkt', quiet=True)
                print("punkt数据下载成功")
            except Exception as e:
                print(f"punkt数据下载失败（将使用备用方法）: {e}")

        # 下载stopwords
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("正在下载NLTK stopwords数据...")
            try:
                nltk.download('stopwords', quiet=True)
                print("stopwords数据下载成功")
            except Exception as e:
                print(f"stopwords数据下载失败（将使用备用停用词列表）: {e}")
    except Exception as e:
        print(f"NLTK数据下载过程出错: {e}")

download_nltk_data()

# 获取停用词列表
try:
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    print("使用NLTK停用词列表")
except (LookupError, ImportError, Exception) as e:
    print(f"无法加载NLTK停用词，使用备用停用词列表: {e}")
    stop_words = DEFAULT_STOPWORDS

# 添加额外的停用词
more_stopwords = ['u', 'im', 'c']
stop_words = list(set(stop_words + more_stopwords))


# 1. 加载数据

# 定义颜色调色板
primary_blue = "#496595"
primary_blue2 = "#85a1c1"
primary_blue3 = "#3f4d63"
primary_grey = "#c6ccd8"
primary_black = "#202022"
primary_bgcolor = "#f4f0ea"
primary_green = "#2ca02c"


# 加载数据
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df.dropna(how="any", axis=1)
df.columns = ['target', 'message']
df['message_len'] = df['message'].apply(lambda x: len(x.split(' ')))
print(df.head())


# 2. 探索性数据分析 (EDA)

# 查看目标分布
balance_counts = df.groupby('target')['target'].agg('count').values

plt.figure(figsize=(10, 6))
categories = ['ham', 'spam']
colors = [primary_blue, primary_grey]
bars = plt.bar(categories, balance_counts, color=colors, alpha=0.8)
plt.title('数据集目标分布', fontsize=16, fontweight='bold')
plt.ylabel('数量', fontsize=12)
plt.xlabel('类别', fontsize=12)

# 在柱状图上添加数值标签
for i, (bar, count) in enumerate(zip(bars, balance_counts)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
try:
    plt.show()
except Exception as e:
    print(f"显示图表时出错（可忽略）: {e}")
plt.close()


# 消息长度分布
ham_df = df[df['target'] == 'ham']['message_len'].value_counts().sort_index()
spam_df = df[df['target'] == 'spam']['message_len'].value_counts().sort_index()

plt.figure(figsize=(12, 6))
plt.plot(ham_df.index, ham_df.values, label='ham', color=primary_blue, linewidth=2, alpha=0.7)
plt.fill_between(ham_df.index, ham_df.values, alpha=0.3, color=primary_blue)
plt.plot(spam_df.index, spam_df.values, label='spam', color=primary_grey, linewidth=2, alpha=0.7)
plt.fill_between(spam_df.index, spam_df.values, alpha=0.3, color=primary_grey)
plt.title('不同类别消息长度分布', fontsize=16, fontweight='bold')
plt.xlabel('消息长度（单词数）', fontsize=12)
plt.ylabel('频数', fontsize=12)
plt.xlim(0, 70)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
try:
    plt.show()
except Exception as e:
    print(f"显示图表时出错（可忽略）: {e}")
plt.close()


# 3. 数据预处理

# 文本清理函数
def clean_text(text):
    '''将文本转为小写，移除方括号中的文本，移除链接，移除标点符号和包含数字的单词'''
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

df['message_clean'] = df['message'].apply(clean_text)
print(df.head())


# 移除停用词
def remove_stopwords(text):
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    return text
    
df['message_clean'] = df['message_clean'].apply(remove_stopwords)
print(df.head())


# 词干提取
try:
    stemmer = nltk.SnowballStemmer("english")
    use_stemmer = True
    print("使用词干提取")
except Exception as e:
    print(f"无法初始化词干提取器（将跳过词干提取）: {e}")
    use_stemmer = False
    stemmer = None

def stemm_text(text):
    if use_stemmer and stemmer:
        text = ' '.join(stemmer.stem(word) for word in text.split(' ') if word.strip())
    return text

if use_stemmer:
    df['message_clean'] = df['message_clean'].apply(stemm_text)
    print(df.head())
else:
    print("跳过词干提取步骤")


# 完整的预处理函数
def preprocess_data(text):
    # 清理标点、URL等
    text = clean_text(text)
    # 移除停用词
    text = ' '.join(word for word in text.split(' ') if word not in stop_words and word.strip())
    # 词干提取
    if use_stemmer and stemmer:
        text = ' '.join(stemmer.stem(word) for word in text.split(' ') if word.strip())
    return text

df['message_clean'] = df['message_clean'].apply(preprocess_data)
print(df.head())


# 目标编码
le = LabelEncoder()
le.fit(df['target'])
df['target_encoded'] = le.transform(df['target'])
print(df.head())


# 4. 词云可视化

# HAM消息词云
try:
    ham_texts = ' '.join(text for text in df.loc[df['target'] == 'ham', 'message_clean'] if pd.notna(text) and text.strip())
    if ham_texts:
        wc = WordCloud(
            background_color='white', 
            max_words=200,
        )
        wc.generate(ham_texts)
        plt.figure(figsize=(18,10))
        plt.title('HAM消息高频词', fontdict={'size': 22, 'verticalalignment': 'bottom'})
        plt.imshow(wc)
        plt.axis("off")
        try:
            plt.show()
        except Exception as e:
            print(f"显示图表时出错（可忽略）: {e}")
        plt.close()
except Exception as e:
    print(f"生成HAM词云时出错（可忽略）: {e}")


# SPAM消息词云
try:
    spam_texts = ' '.join(text for text in df.loc[df['target'] == 'spam', 'message_clean'] if pd.notna(text) and text.strip())
    if spam_texts:
        wc = WordCloud(
            background_color='white', 
            max_words=200,
        )
        wc.generate(spam_texts)
        plt.figure(figsize=(18,10))
        plt.title('SPAM消息高频词', fontdict={'size': 22, 'verticalalignment': 'bottom'})
        plt.imshow(wc)
        plt.axis("off")
        try:
            plt.show()
        except Exception as e:
            print(f"显示图表时出错（可忽略）: {e}")
        plt.close()
except Exception as e:
    print(f"生成SPAM词云时出错（可忽略）: {e}")


# 保存预处理后的数据
try:
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(current_dir, 'spam_preprocessed.csv')
    df.to_csv(output_file, index=False)
    print("=" * 50)
    print(f"预处理完成！数据已保存到 {output_file}")
    print(f"数据形状: {df.shape}")
    print(f"包含列: {list(df.columns)}")
    print("=" * 50)
except Exception as e:
    print(f"错误：保存文件失败 - {e}")
    import traceback
    traceback.print_exc()

