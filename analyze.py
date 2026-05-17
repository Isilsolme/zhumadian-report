#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
驻马店市政府工作报告文本分析
需要先运行 crawl.py 爬取数据，生成 data 目录
"""

import os
import re
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import jieba
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import warnings
warnings.filterwarnings("ignore")

try:
    from snownlp import SnowNLP
    HAS_SNOW = True
except ImportError:
    HAS_SNOW = False

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 输出图片目录
PLOT_DIR = "output"
os.makedirs(PLOT_DIR, exist_ok=True)

# ========== 辅助函数 ==========
def robust_extract_year(text):
    """从标题中提取规范年份，处理中文数字混写"""
    if not text:
        return None
    normalized = unicodedata.normalize("NFKC", text)
    trans = str.maketrans({
        "零": "0", "〇": "0", "○": "0", "O": "0", "o": "0",
        "一": "1", "二": "2", "三": "3", "四": "4",
        "五": "5", "六": "6", "七": "7", "八": "8", "九": "9",
    })
    normalized = normalized.translate(trans)
    m = re.search(r"(19\d{2}|20\d{2})", normalized)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d{4})\s*年", normalized)
    if m:
        y = int(m.group(1))
        if 1990 <= y <= 2030:
            return y
    return None


def read_reports(report_dir="data"):
    """读取所有报告文件，返回DataFrame"""
    data = []
    for fname in os.listdir(report_dir):
        if not fname.endswith(".txt"):
            continue
        filepath = os.path.join(report_dir, fname)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        parts = fname.replace(".txt", "").split("_", 1)
        file_year = int(parts[0]) if parts[0].isdigit() else None
        lines = content.split("\n")
        title_line = None
        for line in lines[:10]:
            if line.startswith("标题："):
                title_line = line.replace("标题：", "").strip()
                break
        correct_year = robust_extract_year(title_line) if title_line else None
        final_year = correct_year if correct_year else file_year
        sep = "=" * 60
        main_content = content.split(sep)[-1].strip() if sep in content else content
        data.append({
            "year": final_year,
            "title": title_line if title_line else "",
            "filepath": filepath,
            "content": main_content,
            "full_text": content
        })
    return pd.DataFrame(data).sort_values("year").reset_index(drop=True)


# ---------- 关键词 ----------
AI_KEYWORDS = [
    "人工智能", "AI", "大数据", "云计算", "物联网", "5G", "数字化", "智能",
    "智慧城市", "机器人", "自动化", "数字经济", "算法", "芯片", "半导体",
    "智能制造", "数字转型", "工业互联网", "区块链", "算力", "新基建", "数字政府"
]
ENV_KEYWORDS = [
    "环保", "生态", "绿色", "低碳", "碳达峰", "碳中和", "污染", "减排",
    "节能", "新能源", "清洁能源", "可持续发展", "环境", "绿化",
    "空气质量", "pm2.5", "蓝天保卫战", "污染防治", "双碳", "能耗",
    "生态修复", "绿色发展"
]
LIVELIHOOD_KEYWORDS = [
    "就业", "教育", "医疗", "养老", "社保", "住房", "保障房", "脱贫",
    "扶贫", "乡村振兴", "收入", "工资", "物价", "消费", "交通", "公共设施",
    "社会保障", "健康", "学区", "托幼", "养老院", "民生实事"
]
ECONOMY_KEYWORDS = [
    "GDP", "经济", "增长", "投资", "消费", "出口", "产业", "制造业",
    "服务业", "财政", "税收", "金融", "小微企业", "营商环境", "市场",
    "产业链", "供应链", "新质生产力", "专精特新", "招商引资", "实体经济",
    "规上工业", "消费升级"
]


def calc_density(text, keywords):
    """计算关键词词频密度（每千字）"""
    words = jieba.lcut(text)
    filtered = [w for w in words if len(w) > 1 and not w.isdigit() and w.strip()]
    word_count = Counter(filtered)
    total = sum(word_count.get(kw, 0) for kw in keywords)
    return total / len(text) * 1000 if len(text) > 0 else 0


def extract_metrics(df):
    """提取各报告指标及TF-IDF加权密度"""
    metrics_list = []
    for _, row in df.iterrows():
        text = row["content"]
        words = jieba.lcut(text)
        clean_words = [w for w in words if len(w) > 1 and not w.isdigit() and w.strip()]
        word_counts = Counter(clean_words)
        total_chars = len(text)
        total_sentences = len([s for s in re.split(r"[。！？]", text) if s.strip()])
        metrics = {
            "year": row["year"],
            "title": row["title"],
            "total_chars": total_chars,
            "total_sentences": total_sentences,
            "avg_sentence_length": total_chars / total_sentences if total_sentences else 0,
            "total_words": len(clean_words),
            "unique_words": len(word_counts),
            "lexical_diversity": len(word_counts) / len(clean_words) if clean_words else 0,
            "ai_density": calc_density(text, AI_KEYWORDS),
            "env_density": calc_density(text, ENV_KEYWORDS),
            "livelihood_density": calc_density(text, LIVELIHOOD_KEYWORDS),
            "economy_density": calc_density(text, ECONOMY_KEYWORDS),
        }
        metrics_list.append(metrics)
    metrics_df = pd.DataFrame(metrics_list)

    # TF-IDF 加权密度
    tfidf_vec = TfidfVectorizer(tokenizer=jieba.lcut, max_df=0.85, min_df=2,
                                stop_words=["的", "了", "在", "是", "和", "也", "有", "不", "这", "那"])
    tfidf_mat = tfidf_vec.fit_transform(df["content"].tolist())
    feat_names = tfidf_vec.get_feature_names_out()

    for label, kw_list in [
        ("ai", AI_KEYWORDS), ("env", ENV_KEYWORDS),
        ("livelihood", LIVELIHOOD_KEYWORDS), ("economy", ECONOMY_KEYWORDS)
    ]:
        col = f"{label}_density_tfidf"
        scores = []
        for i in range(tfidf_mat.shape[0]):
            row_dict = dict(zip(feat_names, tfidf_mat[i].toarray().flatten()))
            scores.append(sum(row_dict.get(k, 0) for k in kw_list))
        metrics_df[col] = scores
    return metrics_df, tfidf_mat


# ========== 可视化函数（均保存至 PLOT_DIR） ==========

def plot_trends(metrics_df):
    """四领域关注度趋势图"""
    fields = [
        ("ai_density_tfidf", "人工智能与科技关注度 (TF-IDF)", "#1f77b4"),
        ("env_density_tfidf", "环保与绿色关注度 (TF-IDF)", "#2ca02c"),
        ("livelihood_density_tfidf", "民生与社会保障关注度 (TF-IDF)", "#ff7f0e"),
        ("economy_density_tfidf", "经济发展关注度 (TF-IDF)", "#d62728"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, (col, title, color) in enumerate(fields):
        ax = axes[i]
        ax.plot(metrics_df["year"], metrics_df[col], marker="o", color=color, linewidth=2)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("年份")
        ax.set_ylabel("TF-IDF 加权密度")
        ax.grid(alpha=0.3)
        sns.regplot(x=metrics_df["year"], y=metrics_df[col], ax=ax, scatter=False,
                    color="gray", line_kws={"linestyle": "--", "alpha": 0.7})
    plt.tight_layout()
    plt.suptitle("驻马店市政府工作报告中各领域关注度变化趋势 (TF-IDF 加权)", fontsize=16, y=1.02)
    plt.savefig(os.path.join(PLOT_DIR, "trends.png"), dpi=150, bbox_inches="tight")
    plt.show()


def plot_consistency(df, tfidf_mat):
    """相邻年份余弦相似度"""
    sim_mat = cosine_similarity(tfidf_mat)
    years = df["year"].values
    adj = [sim_mat[i, i+1] for i in range(len(years)-1)]
    plt.figure(figsize=(10, 4))
    plt.plot(years[1:], adj, marker="o", color="purple", linewidth=2)
    plt.axhline(y=np.mean(adj), color="gray", linestyle="--", alpha=0.5, label="均值")
    plt.title("相邻年份政府工作报告文本一致性（余弦相似度）")
    plt.xlabel("年份（后一年）")
    plt.ylabel("余弦相似度")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "consistency.png"), dpi=150, bbox_inches="tight")
    plt.show()


def plot_basic_stats(metrics_df):
    """篇幅、句长、词汇丰富度"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar(metrics_df["year"], metrics_df["total_chars"] / 1000, color="steelblue", alpha=0.8)
    axes[0].set_title("报告篇幅变化")
    axes[0].set_xlabel("年份")
    axes[0].set_ylabel("字数 (千字)")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].plot(metrics_df["year"], metrics_df["avg_sentence_length"], marker="s", color="coral", linewidth=2)
    axes[1].set_title("平均句子长度变化")
    axes[1].set_xlabel("年份")
    axes[1].set_ylabel("平均每句字数")
    axes[1].grid(alpha=0.3)

    axes[2].plot(metrics_df["year"], metrics_df["lexical_diversity"], marker="^", color="seagreen", linewidth=2)
    axes[2].set_title("词汇丰富度变化")
    axes[2].set_xlabel("年份")
    axes[2].set_ylabel("词汇多样性指数")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "basic_stats.png"), dpi=150, bbox_inches="tight")
    plt.show()


def plot_radar(metrics_df):
    """2014年前后政策关注点对比雷达图"""
    early = metrics_df[metrics_df["year"] < 2014].mean(numeric_only=True)
    late = metrics_df[metrics_df["year"] >= 2014].mean(numeric_only=True)
    categories = ["人工智能", "环境保护", "民生保障", "经济发展"]
    cols = ["ai_density_tfidf", "env_density_tfidf", "livelihood_density_tfidf", "economy_density_tfidf"]
    early_vals = [early[c] for c in cols]
    late_vals = [late[c] for c in cols]
    max_vals = [max(e, l) for e, l in zip(early_vals, late_vals)]
    norm_early = [e / m if m > 0 else 0 for e, m in zip(early_vals, max_vals)]
    norm_late = [l / m if m > 0 else 0 for l, m in zip(late_vals, max_vals)]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    norm_early += norm_early[:1]
    norm_late += norm_late[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, norm_early, "o-", linewidth=2, label="2014年之前均值", color="#1f77b4")
    ax.fill(angles, norm_early, alpha=0.25, color="#1f77b4")
    ax.plot(angles, norm_late, "o-", linewidth=2, label="2014年及之后均值", color="#ff7f0e")
    ax.fill(angles, norm_late, alpha=0.25, color="#ff7f0e")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title("政策关注点转变对比 (TF-IDF 加权指标)", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    plt.savefig(os.path.join(PLOT_DIR, "radar.png"), dpi=150, bbox_inches="tight")
    plt.show()


def plot_sentiment(df):
    """SnowNLP 情感分析"""
    if not HAS_SNOW:
        print("SnowNLP 未安装，跳过情感分析。")
        return

    def split_sentences(text):
        chunks = []
        for block in re.split(r"\n{2,}", text):
            block = block.strip()
            if not block:
                continue
            for part in re.split(r"[。！？!?；;]\s*", block):
                part = part.strip()
                if len(part) >= 8:
                    chunks.append(part)
        return chunks

    def sentiment_snownlp(text):
        segments = split_sentences(text)
        if not segments:
            return np.nan
        scores = []
        for seg in segments:
            try:
                scores.append(SnowNLP(seg).sentiments)
            except Exception:
                continue
        return float(np.mean(scores)) if scores else np.nan

    df["sentiment"] = df["content"].apply(sentiment_snownlp)
    valid = df["sentiment"].dropna()
    if len(valid) == 0:
        print("未能生成有效情感分数。")
        return

    print(f"SnowNLP 有效样本数: {len(valid)}")
    print(f"平均情感分数: {valid.mean():.4f}")
    print(f"分数范围: {valid.min():.4f} - {valid.max():.4f}")

    plt.figure(figsize=(10, 4))
    plt.plot(df["year"], df["sentiment"], marker="o", color="darkred")
    plt.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    plt.title("政府工作报告情感倾向（SnowNLP 句子级平均）")
    plt.xlabel("年份")
    plt.ylabel("情感得分(0-1)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "sentiment.png"), dpi=150, bbox_inches="tight")
    plt.show()


def plot_wordcloud(df):
    """TF-IDF 权重词云"""
    def clean_text(text):
        text = re.sub(r"[^\u4e00-\u9fa5]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    stopwords = {
    '的', '了', '和', '是', '就', '都', '而', '及', '与', '着',
    '或', '一个', '没有', '我们', '你们', '他们', '自己',
    '这', '那', '这些', '那些', '这', '那样',
    '在', '有', '被', '把', '让', '向', '对', '从', '到',
    '年', '月', '日', '时', '上', '下', '前', '后', '中',
    '等', '其', '之', '为', '所', '但', '却', '只',
    '要', '将', '会', '可', '能', '可以', '能够',
    '进行', '通过', '根据', '关于', '对于', '以及',
    '不仅', '而且', '虽然', '但是', '因为', '所以',
    '中国', '大道', '习近平', '集聚', '行动', '企业家', 
    '搞好', '风险', '各位', '万吨', '责任制',
    '代表', '努力提高', '思想', '学校', '卫生', '上市', 
    '平稳', '平台', '攻坚', '新型','万户', '利用', '部署', 
    '水库', '二期', '中小学', '补贴', '用地','如果', '那么', 
    '这样', '那样','努力实现', '切实加强', '构建', '进行', 
    '通过', '根据', '关于', '对于', '以及','不仅', '而且', 
    '虽然', '但是', '因为', '所以', '如果', '那么', '这样', 
    '那样','一年', '三年', '五年', '年均', '按照', '实行', 
    '注重', '进度', '进展', '同时', '累计','全部', '左右', 
    '结果', '压力', '挑战', '重视', '努力', '责任', '东西', 
    '部门', '时期', '阶段', '过程', '计划', '年度计划', '各种', 
    '部分', '系统', '专业', '服务体系','推进', '落实', '实现', 
    '开展', '推动', '加快', '加强', '不断', '逐步', '快速',
    '大幅', '高度', '全力', '重点', '主要', '基本', '重大', 
    '重要', '关键', '核心','第一', '第二', '第三', '首先', 
    '其次', '最后', '此外','发展', '建设', '推进', '加强', 
    '大力', '全面', '深入', '加快', '积极', '努力', '坚持',
    '切实', '落实', '抓好', '做好', '推动', '促进', '提高',
    '提升', '完善', '强化', '确保', '实现', '达到', '完成',
    '增长', '增加', '减少', '下降', '提高', '降低', '保持',
    '稳定', '保障', '改善', '改进', '优化', '调整', '改革',
    '创新', '建立', '健全', '制定', '出台', '实施', '执行',
    '贯彻', '服务', '管理', '监督', '检查', '指导', '协调',
    '支持', '帮助', '扶持', '发布', '保险', '示范区', '法治', 
    '一切', '供给', '实事', '万元', '规上', '进一步', '不断', 
    '继续','我市', '全市', '重点', '主要', '工作', '报告', '政府'
    }

    cleaned_texts = [clean_text(c) for c in df["content"]]
    tfidf_vec = TfidfVectorizer(
        tokenizer=jieba.lcut, max_df=0.85, min_df=2,
        stop_words=list(stopwords)
    )
    tfidf_mat = tfidf_vec.fit_transform(cleaned_texts)
    avg_tfidf = np.array(tfidf_mat.mean(axis=0)).flatten()
    feat_names = tfidf_vec.get_feature_names_out()
    word_freq = {}
    for i, word in enumerate(feat_names):
        if len(word) > 1 and word not in stopwords and not word.isdigit():
            word_freq[word] = avg_tfidf[i]

    wc = WordCloud(font_path="simhei.ttf", width=800, height=400,
                   background_color="white", collocations=False)
    wc.generate_from_frequencies(word_freq)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("驻马店市政府工作报告 TF-IDF 权重词云")
    plt.savefig(os.path.join(PLOT_DIR, "wordcloud.png"), dpi=150, bbox_inches="tight")
    plt.show()


def main():
    df = read_reports("data")
    print(f"成功读取 {len(df)} 份报告，年份范围: {df['year'].min()} - {df['year'].max()}")

    metrics_df, tfidf_mat = extract_metrics(df)

    plot_trends(metrics_df)
    plot_consistency(df, tfidf_mat)
    plot_basic_stats(metrics_df)
    plot_radar(metrics_df)
    plot_sentiment(df)
    plot_wordcloud(df)

    print(f"所有图表已保存至 {PLOT_DIR}/ 目录。")


if __name__ == "__main__":
    main()