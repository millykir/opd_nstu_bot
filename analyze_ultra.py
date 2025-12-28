import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime
import os
import warnings
import numpy as np

# --- КОНФИГУРАЦИЯ ---
LOG_FILE = 'chat_qa_log.txt'
OUTPUT_DIR = 'analytics_ultra_report'

# ЧЕРНЫЙ СПИСОК (Михаил, Захар, Егор)
EXCLUDED_IDS = {'6753772275', '814358254', '1270577551'}

# Настройки стиля
warnings.simplefilter(action='ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

STOPWORDS = {
    'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой', 'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много', 'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 'между', 'привет', 'здравствуйте', 'спасибо', 'пожалуйста', 'бот', 'подскажи', 'скажи'
}

def parse_log_file(filepath):
    print(f"🔮 Запуск нейро-анализа логов: {filepath}...")
    data = []
    current_entry = {}
    
    patterns = {
        'user_id': re.compile(r'^UserID:\s*(.*)'),
        'timestamp': re.compile(r'^Время сообщения:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'),
        'latency': re.compile(r'^Задержка \(сек\):\s*([\d\.]+)'),
    }
    
    if not os.path.exists(filepath): return pd.DataFrame()

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    q_buf, a_buf = [], []
    
    for line in lines:
        line = line.strip()
        if line.startswith('----------'):
            if current_entry and ('question' in current_entry or q_buf):
                current_entry['question'] = " ".join(q_buf).strip()
                current_entry['answer'] = " ".join(a_buf).strip()
                if not current_entry['question']: current_entry['question'] = "<Empty>"
                data.append(current_entry)
            current_entry = {}
            q_buf, a_buf = [], []
            continue

        if line.startswith('Q:'):
            q_buf.append(line[2:].strip())
        elif line.startswith('A:'):
            a_buf.append(line[2:].strip())
        else:
            for k, p in patterns.items():
                m = p.match(line)
                if m:
                    val = m.group(1).strip()
                    if k == 'latency': 
                        try: current_entry[k] = float(val)
                        except: pass
                    elif k == 'timestamp':
                        try: current_entry['datetime'] = datetime.strptime(val, '%Y-%m-%d %H:%M:%S')
                        except: pass
                    else: current_entry[k] = val

    if current_entry and q_buf:
        current_entry['question'] = " ".join(q_buf).strip()
        current_entry['answer'] = " ".join(a_buf).strip()
        data.append(current_entry)

    df = pd.DataFrame(data)
    
    # --- ТОТАЛЬНАЯ ФИЛЬТРАЦИЯ ---
    if 'user_id' in df.columns:
        before = len(df)
        df = df[~df['user_id'].astype(str).isin(EXCLUDED_IDS)]
        print(f"🚮 Исключено {before - len(df)} записей (Тесты/Разработчики).")
        
    return df

def get_ngrams(text_series, n=2, top_k=10):
    ngrams_list = []
    valid_texts = text_series.dropna().astype(str)
    for text in valid_texts:
        if text == "<Empty>": continue
        text = re.sub(r'[^\w\s]', '', text.lower())
        words = [w for w in text.split() if w not in STOPWORDS and len(w) > 3]
        if len(words) >= n:
            ngrams_list.extend(zip(*[words[i:] for i in range(n)]))
    return Counter(ngrams_list).most_common(top_k)

def generate_report():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    df = parse_log_file(LOG_FILE)
    if df.empty:
        print("❌ Данных нет.")
        return

    df_time = df.dropna(subset=['datetime']).copy()
    print(f"🔬 Анализ {len(df_time)} валидных сессий...")

    # 1. RAG HEALTH (DONUT)
    plt.figure(figsize=(8, 8))
    fail_phrases = ['к сожалению', 'не нашел', 'не смог найти', 'обратитесь к', 'попробуйте переформулировать']
    is_fail = df['answer'].fillna('').str.lower().apply(lambda x: any(p in x for p in fail_phrases))
    success_rate = (1 - is_fail.mean()) * 100
    
    colors = ['#00b894', '#d63031']
    plt.pie([success_rate, 100-success_rate], labels=['Успешный ответ', 'Нет в базе'], 
            autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops={'width': 0.4})
    plt.title('RAG System Health Index', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/1_rag_donut_pro.png', dpi=300)
    print("✅ RAG Donut Chart готов.")

    # 2. ACTIVITY WAVE
    if not df_time.empty:
        plt.figure(figsize=(12, 6))
        df_time['date'] = df_time['datetime'].dt.date
        daily = df_time['date'].value_counts().sort_index()
        
        plt.fill_between(daily.index, daily.values, color='#0984e3', alpha=0.3)
        plt.plot(daily.index, daily.values, color='#0984e3', linewidth=2)
        plt.title('Волны активности студентов (Timeline)', fontsize=16)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/2_timeline_wave.png', dpi=300)
        print("✅ Timeline Wave готов.")

    # 3. LATENCY VIOLIN
    if 'latency' in df_time.columns:
        plt.figure(figsize=(10, 6))
        clean_latency = df_time[df_time['latency'] < 40]['latency']
        sns.violinplot(x=clean_latency, color='#6c5ce7')
        plt.title('ДНК Скорости: Плотность задержки (Violin)', fontsize=16)
        plt.xlabel('Секунды')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/3_latency_violin.png', dpi=300)
        print("✅ Latency Violin готов.")

    # 4. ENGAGEMENT HEXBIN
    if not df_time.empty and 'user_id' in df_time.columns:
        user_stats = df_time.groupby('user_id').agg(
            msg_count=('question', 'count'),
            first_msg=('datetime', 'min'),
            last_msg=('datetime', 'max')
        )
        user_stats['lifetime_days'] = (user_stats['last_msg'] - user_stats['first_msg']).dt.total_seconds() / 86400
        
        plt.figure(figsize=(10, 6))
        plt.hexbin(user_stats['lifetime_days'], user_stats['msg_count'], gridsize=15, cmap='Purples', mincnt=1)
        plt.colorbar(label='Плотность студентов')
        plt.title('Кластеризация: Жизненный цикл vs Активность', fontsize=16)
        plt.xlabel('Дней с ботом')
        plt.ylabel('Сообщений отправлено')
        plt.savefig(f'{OUTPUT_DIR}/4_engagement_hex.png', dpi=300)
        print("✅ Engagement Hexbin готов.")

    # 5. SEMANTIC CORE
    bigrams = get_ngrams(df['question'], n=2, top_k=10)
    if bigrams:
        plt.figure(figsize=(10, 8))
        phrases, counts = zip(*bigrams)
        phrases = [" ".join(p).upper() for p in phrases]
        
        sns.barplot(x=list(counts), y=list(phrases), palette='Spectral')
        plt.title('Семантическое ядро (Боли студентов)', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/5_semantics_pro.png', dpi=300)
        print("✅ Semantic Bars готов.")

    # 6. SCALABILITY
    if not df_time.empty:
        df_time['hour_bucket'] = df_time['datetime'].dt.floor('h')
        load = df_time.groupby('hour_bucket').agg(Load=('question', 'count'), Latency=('latency', 'mean')).dropna()
        
        plt.figure(figsize=(10, 6))
        sns.regplot(data=load, x='Load', y='Latency', scatter_kws={'s':60, 'alpha':0.6}, line_kws={'color':'red'})
        plt.title('Стресс-тест: Масштабируемость архитектуры', fontsize=16)
        plt.savefig(f'{OUTPUT_DIR}/6_scalability.png', dpi=300)
        print("✅ Scalability Test готов.")

    df.to_excel(f'{OUTPUT_DIR}/ULTIMATE_DATA.xlsx', index=False)
    print(f"\n🏆 ГЕНИАЛЬНЫЙ ОТЧЕТ СОЗДАН: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_report()