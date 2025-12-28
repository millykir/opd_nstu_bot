import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime
import os
import warnings

# --- КОНФИГУРАЦИЯ ---
LOG_FILE = 'chat_qa_log.txt'
OUTPUT_DIR = 'analytics_pro_report'

warnings.simplefilter(action='ignore')
plt.style.use('seaborn-v0_8-whitegrid')

STOPWORDS = {
    'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой', 'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много', 'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 'между', 'привет', 'здравствуйте', 'спасибо', 'пожалуйста', 'бот', 'подскажи', 'скажи', 'расскажи'
}

def parse_log_file(filepath):
    print(f"DEBUG: Читаю файл {filepath}...")
    data = []
    current_entry = {}
    
    patterns = {
        'user_id': re.compile(r'^UserID:\s*(.*)'),
        'timestamp': re.compile(r'^Время сообщения:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'),
        'latency': re.compile(r'^Задержка \(сек\):\s*([\d\.]+)'),
    }
    
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    print(f"DEBUG: Прочитано строк: {len(lines)}")
    
    q_buf, a_buf = [], []
    state = 'META' 

    for i, line in enumerate(lines):
        line = line.strip()
        # Разделитель или конец файла
        if line.startswith('----------'):
            if current_entry and ('question' in current_entry or q_buf):
                # Сохраняем накопленное
                current_entry['question'] = " ".join(q_buf).strip()
                current_entry['answer'] = " ".join(a_buf).strip()
                # Если вопроса нет, ставим заглушку, чтобы не терять запись
                if not current_entry['question']: current_entry['question'] = "<Empty>"
                data.append(current_entry)
            
            # Сброс
            current_entry = {}
            q_buf, a_buf = [], []
            state = 'META'
            continue

        if line.startswith('Q:'):
            state = 'Q'
            clean_line = line[2:].strip()
            if clean_line: q_buf.append(clean_line)
        elif line.startswith('A:'):
            state = 'A'
            clean_line = line[2:].strip()
            if clean_line: a_buf.append(clean_line)
        else:
            if state == 'Q': q_buf.append(line)
            elif state == 'A': a_buf.append(line)
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

    # Последняя запись
    if current_entry and q_buf:
        current_entry['question'] = " ".join(q_buf).strip()
        current_entry['answer'] = " ".join(a_buf).strip()
        data.append(current_entry)

    print(f"DEBUG: Распознано записей (диалогов): {len(data)}")
    return pd.DataFrame(data)

def get_ngrams(text_series, n=2, top_k=10):
    ngrams_list = []
    # Работаем только с непустыми вопросами
    valid_texts = text_series.dropna().astype(str)
    for text in valid_texts:
        if text == "<Empty>": continue
        text = re.sub(r'[^\w\s]', '', text.lower())
        words = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
        if len(words) >= n:
            ngrams_list.extend(zip(*[words[i:] for i in range(n)]))
    return Counter(ngrams_list).most_common(top_k)

def generate_report():
    df = parse_log_file(LOG_FILE)
    if df.empty:
        print("❌ ОШИБКА: Не удалось извлечь данные. Проверьте формат файла.")
        return

    # Разделяем на данные с датами и без
    df_time = df.dropna(subset=['datetime'])
    print(f"DEBUG: Записей с датами (для графиков): {len(df_time)}")
    print(f"DEBUG: Записей всего (для текста): {len(df)}")

    # 1. RAG Success Rate (берем все данные)
    if 'answer' in df.columns:
        fail_phrases = ['к сожалению', 'не нашел', 'не смог найти', 'обратитесь к']
        df['is_failure'] = df['answer'].fillna('').str.lower().apply(lambda x: any(p in x for p in fail_phrases))
        failure_rate = df['is_failure'].mean() * 100
        success_rate = 100 - failure_rate
        
        plt.figure(figsize=(6, 6))
        plt.pie([success_rate, failure_rate], labels=['Успех', 'База не знает'], 
                autopct='%1.1f%%', colors=['#4CAF50', '#F44336'])
        plt.title('Эффективность базы знаний')
        plt.savefig(f'{OUTPUT_DIR}/1_quality.png')
        print(f"📊 Эффективность ответов: {success_rate:.1f}%")

    # 2. Активность по времени (только где есть даты)
    if not df_time.empty:
        df_time['hour'] = df_time['datetime'].dt.hour
        df_time['weekday'] = df_time['datetime'].dt.day_name()
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        hm_data = df_time.pivot_table(index='weekday', columns='hour', values='question', aggfunc='count', fill_value=0)
        # Сортировка дней
        existing_days = [d for d in days if d in hm_data.index]
        hm_data = hm_data.reindex(existing_days)
        
        plt.figure(figsize=(10, 5))
        sns.heatmap(hm_data, cmap='Blues', annot=False)
        plt.title('Тепловая карта активности')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/2_heatmap.png')
        print("📊 Тепловая карта построена.")

    # 3. NLP Анализ (берем все данные)
    if 'question' in df.columns:
        bigrams = get_ngrams(df['question'], n=2, top_k=10)
        if bigrams:
            plt.figure(figsize=(10, 6))
            phrases, counts = zip(*bigrams)
            phrases_str = [" ".join(p) for p in phrases]
            sns.barplot(x=list(counts), y=list(phrases_str), palette='viridis')
            plt.title('Топ популярных тем')
            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/3_topics.png')
            print("📊 Анализ тем завершен.")

    # 4. Сохранение в Excel
    df.to_excel(f'{OUTPUT_DIR}/report.xlsx', index=False)
    print(f"\n✅ ГОТОВО! Отчет сохранен в папку {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_report()
