import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from datetime import datetime
import os
import warnings

# --- НАСТРОЙКИ ---
LOG_FILE = 'chat_qa_log.txt'
OUTPUT_DIR = 'analytics_simple_report'

# ЧЕРНЫЙ СПИСОК (Тестировщики: Михаил, Захар, Егор)
BANNED_IDS = {'6753772275', '814358254', '1270577551'}

warnings.simplefilter(action='ignore')
plt.style.use('seaborn-v0_8-darkgrid')

def parse_log_file(filepath):
    data = []
    current_entry = {}
    
    patterns = {
        'user_id': re.compile(r'^UserID:\s*(.*)'),
        'timestamp': re.compile(r'^Время сообщения:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'),
        'latency': re.compile(r'^Задержка \(сек\):\s*([\d\.]+)'),
    }
    
    if not os.path.exists(filepath):
        return pd.DataFrame()

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    buffer_q = []
    buffer_a = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('----------'):
            if current_entry:
                current_entry['question'] = " ".join(buffer_q).strip()
                current_entry['answer'] = " ".join(buffer_a).strip()
                data.append(current_entry)
            current_entry = {}
            buffer_q, buffer_a = [], []
            continue

        if line.startswith('Q:'):
            buffer_q.append(line[2:].strip())
        elif line.startswith('A:'):
            buffer_a.append(line[2:].strip())
        else:
            for key, pattern in patterns.items():
                match = pattern.match(line)
                if match:
                    val = match.group(1).strip()
                    if key == 'latency':
                        try: current_entry[key] = float(val)
                        except: pass
                    elif key == 'timestamp':
                        try: current_entry['datetime'] = datetime.strptime(val, '%Y-%m-%d %H:%M:%S')
                        except: pass
                    else:
                        current_entry[key] = val

    if current_entry:
        current_entry['question'] = " ".join(buffer_q).strip()
        current_entry['answer'] = " ".join(buffer_a).strip()
        data.append(current_entry)

    df = pd.DataFrame(data)
    
    # --- ЖЕСТКАЯ ЗАЧИСТКА ---
    if 'user_id' in df.columns:
        total_rows = len(df)
        # Удаляем строки, где user_id совпадает с черным списком
        df = df[~df['user_id'].astype(str).isin(BANNED_IDS)]
        deleted = total_rows - len(df)
        print(f"🔥 УНИЧТОЖЕНО {deleted} ЗАПИСЕЙ ТЕСТИРОВЩИКОВ (Захар, Егор, Михаил).")
        
    return df

def generate_report():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    print("1. Чтение и парсинг логов...")
    df = parse_log_file(LOG_FILE)
    if df.empty: return

    df_time = df.dropna(subset=['datetime']).copy()

    # 1. TIMELINE
    if not df_time.empty:
        plt.figure(figsize=(12, 6))
        df_time['date'] = df_time['datetime'].dt.date
        daily_counts = df_time['date'].value_counts().sort_index()
        sns.barplot(x=daily_counts.index, y=daily_counts.values, color='#4c72b0')
        plt.title('Динамика использования бота', fontsize=16)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/1_timeline_activity.png')
        print("   ✅ График активности сохранен.")

        # 2. HEATMAP
        df_time['hour'] = df_time['datetime'].dt.hour
        df_time['weekday'] = df_time['datetime'].dt.day_name()
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data = df_time.pivot_table(index='weekday', columns='hour', values='question', aggfunc='count', fill_value=0)
        heatmap_data = heatmap_data.reindex(days_order)
        
        plt.figure(figsize=(14, 6))
        sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='d')
        plt.title('Тепловая карта активности', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/2_activity_heatmap.png')
        print("   ✅ Тепловая карта сохранена.")

        # 3. LATENCY
        if 'latency' in df_time.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(df_time[df_time['latency'] < 60]['latency'], bins=30, kde=True, color='orange')
            plt.title('Скорость ответа (сек)', fontsize=16)
            plt.savefig(f'{OUTPUT_DIR}/3_latency_dist.png')
            print("   ✅ График задержек сохранен.")

    # 4. WORDCLOUD
    text = " ".join(df['question'].dropna().astype(str).tolist())
    stopwords = {'что', 'как', 'где', 'когда', 'можно', 'ли', 'на', 'по', 'за', 'или', 'для', 'не', 'я', 'а', 'в', 'у', 'с', 'это', 'подскажи', 'расскажи'}
    wordcloud = WordCloud(width=1600, height=800, background_color='white', stopwords=stopwords).generate(text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Облако слов', fontsize=16)
    plt.savefig(f'{OUTPUT_DIR}/4_wordcloud.png')
    print("   ✅ Облако слов сохранено.")

    # 5. TOP USERS
    if 'user_id' in df.columns:
        top_users = df['user_id'].value_counts().head(10)
        user_labels = [f"Student {uid}" for uid in top_users.index] 
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_users.values, y=user_labels, palette='viridis')
        plt.title('Топ-10 активных студентов (Без админов)', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/5_top_users.png')
        print("   ✅ Топ пользователей сохранен.")

    df.to_excel(f'{OUTPUT_DIR}/simple_report.xlsx', index=False)
    print(f"   ✅ Excel сохранен в {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_report()