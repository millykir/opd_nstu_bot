import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from datetime import datetime
import os

# --- НАСТРОЙКИ ---
LOG_FILE = 'chat_qa_log.txt'
OUTPUT_DIR = 'analytics_report'
plt.style.use('seaborn-v0_8-darkgrid')

def parse_log_file(filepath):
    data = []
    current_entry = {}
    
    patterns = {
        'user_id': re.compile(r'^UserID:\s*(.*)'),
        'username': re.compile(r'^Username:\s*(.*)'),
        'full_name': re.compile(r'^(?:Полное имя|Имя):\s*(.*)'),
        'timestamp': re.compile(r'^Время сообщения:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'),
        'latency': re.compile(r'^Задержка \(сек\):\s*([\d\.]+)'),
    }
    
    if not os.path.exists(filepath):
        print(f"Файл {filepath} не найден!")
        return pd.DataFrame()

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    buffer_q = []
    buffer_a = []
    state = 'META' 

    for line in lines:
        line = line.strip()
        if line.startswith('----------------'):
            if current_entry:
                current_entry['question'] = " ".join(buffer_q).strip()
                current_entry['answer'] = " ".join(buffer_a).strip()
                data.append(current_entry)
            current_entry = {}
            buffer_q = []
            buffer_a = []
            state = 'META'
            continue

        if line.startswith('Q:'):
            state = 'Q'
            clean_line = line[2:].strip()
            if clean_line: buffer_q.append(clean_line)
        elif line.startswith('A:'):
            state = 'A'
            clean_line = line[2:].strip()
            if clean_line: buffer_a.append(clean_line)
        else:
            if state == 'Q':
                buffer_q.append(line)
            elif state == 'A':
                buffer_a.append(line)
            else:
                for key, pattern in patterns.items():
                    match = pattern.match(line)
                    if match:
                        val = match.group(1).strip()
                        if key == 'latency':
                            try:
                                current_entry[key] = float(val)
                            except:
                                pass
                        elif key == 'timestamp':
                            try:
                                current_entry['datetime'] = datetime.strptime(val, '%Y-%m-%d %H:%M:%S')
                            except:
                                pass
                        else:
                            current_entry[key] = val

    if current_entry:
        current_entry['question'] = " ".join(buffer_q).strip()
        current_entry['answer'] = " ".join(buffer_a).strip()
        data.append(current_entry)

    return pd.DataFrame(data)

def generate_report():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print("1. Чтение и парсинг логов...")
    df = parse_log_file(LOG_FILE)
    if df.empty:
        print("Не удалось извлечь данные или файл пуст.")
        return

    print(f"   Всего записей найдено: {len(df)}")
    
    df_time = df.dropna(subset=['datetime']).copy()
    print(f"   Записей с датами (для графиков): {len(df_time)}")

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
    stopwords = {'что', 'как', 'где', 'когда', 'можно', 'ли', 'на', 'по', 'за', 'или', 'для', 'не', 'я', 'а', 'в', 'у', 'с', 'это'}
    wordcloud = WordCloud(width=1600, height=800, background_color='white', stopwords=stopwords).generate(text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Облако слов (темы вопросов)', fontsize=16)
    plt.savefig(f'{OUTPUT_DIR}/4_wordcloud_questions.png')
    print("   ✅ Облако слов сохранено.")

    # 5. TOP USERS
    if 'user_id' in df.columns:
        top_users = df['user_id'].value_counts().head(10)
        user_labels = []
        for uid in top_users.index:
            try:
                row = df[df['user_id'] == uid]
                name = row['full_name'].dropna().iloc[0] if 'full_name' in row and not row['full_name'].dropna().empty else "NoName"
            except:
                name = "Unknown"
            user_labels.append(f"{name} ({uid})")
            
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_users.values, y=user_labels, palette='viridis')
        plt.title('Топ-10 активных студентов', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/5_top_users.png')
        print("   ✅ Топ пользователей сохранен.")

    # EXCEL
    df.to_excel(f'{OUTPUT_DIR}/full_log_dump.xlsx', index=False)
    print(f"   ✅ Excel-отчет сохранен: {OUTPUT_DIR}/full_log_dump.xlsx")

if __name__ == "__main__":
    generate_report()
