import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import warnings
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# --- КОНФИГУРАЦИЯ (GENIUS MODE) ---
LOG_FILE = 'chat_qa_log.txt'
OUTPUT_DIR = 'analytics_video_studio'
FPS = 15  # Кадров в секунду (плавность)

# ЧЕРНЫЙ СПИСОК (Жесткий бан)
EXCLUDED_IDS = {'6753772275', '814358254', '1270577551'}

# Стиль "Матрица"
plt.style.use('dark_background')
warnings.simplefilter(action='ignore')

def parse_log_file(filepath):
    print(f"🔹 ЗАГРУЗКА ДАННЫХ ИЗ: {filepath}...")
    data = []
    current_entry = {}
    
    patterns = {
        'user_id': re.compile(r'^UserID:\s*(.*)'),
        'timestamp': re.compile(r'^Время сообщения:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'),
    }
    
    if not os.path.exists(filepath): return pd.DataFrame()

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    q_buf = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('----------'):
            if current_entry and 'timestamp' in current_entry:
                # Если это один из "лохов", мы даже не добавляем запись
                uid = current_entry.get('user_id', 'unknown')
                if str(uid) not in EXCLUDED_IDS:
                    data.append(current_entry)
            current_entry = {}
            continue

        # Нам нужны только ID и Время для видео
        for k, p in patterns.items():
            m = p.match(line)
            if m:
                val = m.group(1).strip()
                if k == 'timestamp':
                    try: val = datetime.strptime(val, '%Y-%m-%d %H:%M:%S')
                    except: continue
                current_entry[k] = val

    df = pd.DataFrame(data)
    print(f"🔹 ЧИСТЫХ ЗАПИСЕЙ (БЕЗ ТЕСТИРОВЩИКОВ): {len(df)}")
    return df

def generate_neural_pulse_video(df):
    """
    Генерирует видео 'Сердцебиение бота'. 
    Бегущий график активности за последние дни.
    """
    print("🎬 НАЧИНАЮ РЕНДЕР ВИДЕО 1: 'NEURAL PULSE' (ЭТО ЗАЙМЕТ ВРЕМЯ)...")
    
    # Агрегация по часам
    df = df.sort_values('timestamp')
    df['hour_bucket'] = df['timestamp'].dt.floor('h')
    counts = df.groupby('hour_bucket').size()
    
    # Создаем полный диапазон дат (заполняем нулями часы тишины)
    full_idx = pd.date_range(start=counts.index.min(), end=counts.index.max(), freq='h')
    counts = counts.reindex(full_idx, fill_value=0)
    
    # Настройки окна (показываем 48 часов за раз)
    window_size = 48 
    total_frames = len(counts) - window_size
    
    # Если данных мало, уменьшаем шаг
    step = max(1, total_frames // 200) # Ограничиваем видео ~200 кадрами для скорости
    
    fig, ax = plt.subplots(figsize=(10, 5))
    line, = ax.plot([], [], color='#00ff00', lw=2) # Хакерский зеленый
    
    # Эффект "сканера" (вертикальная линия)
    scanner = ax.axvline(x=0, color='red', alpha=0.5, linestyle='--')
    
    ax.set_ylim(0, counts.max() * 1.2)
    ax.set_facecolor('#050505')
    
    def init():
        line.set_data([], [])
        return line, scanner

    def update(frame_idx):
        start = frame_idx
        end = frame_idx + window_size
        
        # Данные для окна
        y_data = counts.iloc[start:end].values
        x_data = np.arange(len(y_data))
        
        line.set_data(x_data, y_data)
        
        # Сканер бегает
        scanner.set_xdata([len(y_data)-1]) # Всегда в конце
        
        # Дата в заголовке
        current_date = counts.index[end].strftime('%Y-%m-%d %H:%M')
        ax.set_title(f'SYSTEM MONITORING: {current_date}', color='white', fontsize=14, fontfamily='monospace')
        ax.set_xlim(0, window_size)
        
        return line, scanner

    # Создаем анимацию
    frames = range(0, total_frames, step)
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False)
    
    save_path = f'{OUTPUT_DIR}/1_neural_pulse.gif'
    ani.save(save_path, writer=PillowWriter(fps=FPS))
    print(f"✅ ВИДЕО 1 ГОТОВО: {save_path}")
    plt.close()

def generate_heatmap_evolution_video(df):
    """
    Генерирует видео 'Эволюция нагрузки'.
    Показывает, как меняется активность по дням недели неделя за неделей.
    """
    print("🎬 НАЧИНАЮ РЕНДЕР ВИДЕО 2: 'HEATMAP EVOLUTION'...")
    
    df['week'] = df['timestamp'].dt.to_period('W')
    weeks = df['week'].unique()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def update(week_val):
        ax.clear()
        # Фильтр данных за конкретную неделю (накапливающим итогом или скользящим)
        # Сделаем "Скользящее окно" - показываем текущую неделю + прошлую
        current_week_start = week_val.start_time
        subset = df[(df['timestamp'] >= current_week_start - timedelta(days=7)) & 
                    (df['timestamp'] <= current_week_start + timedelta(days=7))]
        
        if subset.empty: return
        
        subset['weekday'] = subset['timestamp'].dt.day_name()
        subset['hour'] = subset['timestamp'].dt.hour
        
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        matrix = subset.pivot_table(index='weekday', columns='hour', aggfunc='size', fill_value=0)
        matrix = matrix.reindex(days_order).fillna(0)
        
        # Чтобы сетка была полной 24 часа
        for h in range(24):
            if h not in matrix.columns: matrix[h] = 0
        matrix = matrix.sort_index(axis=1)
        
        sns.heatmap(matrix, cmap='magma', cbar=False, ax=ax, vmin=0, vmax=5) # vmax фикс для плавности
        
        ax.set_title(f'ACTIVITY SECTOR SCAN: Week of {week_val}', color='orange', fontsize=16)
        ax.set_xlabel('Hour (00-24)')
        ax.set_ylabel('')
        
    ani = FuncAnimation(fig, update, frames=weeks, repeat=True)
    
    save_path = f'{OUTPUT_DIR}/2_heatmap_evolution.gif'
    # Медленнее FPS, чтобы успеть рассмотреть неделю
    ani.save(save_path, writer=PillowWriter(fps=2)) 
    print(f"✅ ВИДЕО 2 ГОТОВО: {save_path}")
    plt.close()

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    df = parse_log_file(LOG_FILE)
    if df.empty:
        print("❌ НЕТ ДАННЫХ ДЛЯ ВИДЕО.")
        return

    generate_neural_pulse_video(df)
    generate_heatmap_evolution_video(df)
    
    print(f"\n🎥 СТУДИЯ ЗАВЕРШИЛА РАБОТУ. ФАЙЛЫ В ПАПКЕ: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
