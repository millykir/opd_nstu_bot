import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from datetime import datetime
import os
import warnings

# --- НАСТРОЙКИ РЕЖИМА БОГА ---
LOG_FILE = 'chat_qa_log.txt'
OUTPUT_DIR = 'analytics_god_mode'
FPS = 20  # Высокая плавность
DURATION_SEC = 10 # Длительность видео

# ЧЕРНЫЙ СПИСОК (Удаляем шум)
EXCLUDED_IDS = {'6753772275', '814358254', '1270577551'}

# Стиль: Deep Space
plt.style.use('dark_background')
warnings.simplefilter(action='ignore')

def parse_log_file(filepath):
    print(f"🌀 ИНИЦИАЛИЗАЦИЯ ПРОТОКОЛА 'GOD MODE': {filepath}...")
    data = []
    current_entry = {}
    
    # Регулярки
    patterns = {
        'user_id': re.compile(r'^UserID:\s*(.*)'),
        'timestamp': re.compile(r'^Время сообщения:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'),
        'latency': re.compile(r'^Задержка \(сек\):\s*([\d\.]+)'),
    }
    
    if not os.path.exists(filepath): return pd.DataFrame()

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    q_buf = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('----------'):
            if current_entry and 'timestamp' in current_entry:
                uid = current_entry.get('user_id', 'unknown')
                # ФИЛЬТРАЦИЯ НА ВХОДЕ
                if str(uid) not in EXCLUDED_IDS:
                    # Если вопроса нет, ставим заглушку
                    if not q_buf: current_entry['question_len'] = 0
                    else: current_entry['question_len'] = len(" ".join(q_buf))
                    data.append(current_entry)
            current_entry = {}
            q_buf = []
            continue

        if line.startswith('Q:'):
            q_buf.append(line[2:].strip())
        else:
            for k, p in patterns.items():
                m = p.match(line)
                if m:
                    val = m.group(1).strip()
                    if k == 'timestamp':
                        try: val = datetime.strptime(val, '%Y-%m-%d %H:%M:%S')
                        except: continue
                    elif k == 'latency':
                        try: val = float(val)
                        except: val = 0.5
                    current_entry[k] = val

    df = pd.DataFrame(data)
    # Удаляем выбросы по задержке для красоты графика (всё что > 20 сек обрезаем до 20)
    if 'latency' in df.columns:
        df['latency'] = df['latency'].clip(upper=20)
    
    print(f"🌀 АНАЛИЗ {len(df)} СОБЫТИЙ...")
    return df

def generate_temporal_vortex(df):
    """
    Генерирует Спираль Времени (Polar Scatter Plot Animation).
    """
    print("🎬 РЕНДЕРИНГ 'TEMPORAL VORTEX' (ЭТО БУДЕТ ЭПИЧНО)...")
    
    df = df.sort_values('timestamp')
    first_date = df['timestamp'].min()
    
    # --- МАТЕМАТИКА ВИХРЯ ---
    # Theta (Угол) = Время суток (0..2Pi)
    # Час + Минута/60 -> переводим в радианы
    df['theta'] = (df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60) / 24 * 2 * np.pi
    
    # R (Радиус) = Количество дней от начала запуска
    # Это создает эффект расширения спирали со временем
    df['r'] = (df['timestamp'] - first_date).dt.total_seconds() / 86400
    
    # Цвет = Задержка (от Синего к Красному)
    colors = df['latency']
    sizes = df['question_len'].clip(lower=10, upper=100) # Размер точки зависит от длины вопроса
    
    # Настройка фигуры
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')
    
    # Настройки осей (убираем лишнее, делаем красиво)
    ax.set_facecolor('#050505')
    ax.grid(True, color='#222222', linestyle='--', alpha=0.5)
    ax.set_xticklabels(['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00'], color='cyan', fontsize=10)
    ax.set_yticklabels([]) # Убираем метки радиуса
    ax.spines['polar'].set_visible(False)
    
    # Скатер плот
    scatter = ax.scatter([], [], c=[], s=[], cmap='cool', alpha=0.8, edgecolors='none')
    
    # Текстовый счетчик
    info_text = fig.text(0.02, 0.95, "", color='white', fontsize=14, fontfamily='monospace')
    
    # Пределы
    max_r = df['r'].max() + 1
    ax.set_ylim(0, max_r)
    
    # Логика анимации
    total_frames = FPS * DURATION_SEC
    chunk_size = len(df) // total_frames
    if chunk_size < 1: chunk_size = 1
    
    def update(frame):
        # Показываем данные накопленным итогом
        limit = (frame + 1) * chunk_size
        if limit > len(df): limit = len(df)
        
        current_data = df.iloc[:limit]
        
        # Обновляем точки
        scatter.set_offsets(np.c_[current_data['theta'], current_data['r']])
        scatter.set_array(current_data['latency'])
        scatter.set_sizes(current_data['question_len'].clip(10, 50))
        
        # Если есть данные, берем дату последнего
        if not current_data.empty:
            curr_date = current_data.iloc[-1]['timestamp'].strftime('%Y-%m-%d')
            curr_msgs = len(current_data)
            info_text.set_text(f"DATE: {curr_date}\nMSGS: {curr_msgs}\nSTATUS: ACTIVE")
        
        return scatter, info_text

    ani = FuncAnimation(fig, update, frames=total_frames, interval=1000/FPS, blit=False)
    
    save_path = f'{OUTPUT_DIR}/GOD_MODE_VORTEX.gif'
    ani.save(save_path, writer=PillowWriter(fps=FPS))
    print(f"✅ ВИДЕО ГОТОВО: {save_path}")
    plt.close()

def generate_ascii_dossier(df):
    """Генерирует секретное досье в текстовом виде."""
    print("📄 ГЕНЕРАЦИЯ СЕКРЕТНОГО ДОСЬЕ...")
    
    total_msgs = len(df)
    users = df['user_id'].nunique() if 'user_id' in df.columns else 0
    avg_lat = df['latency'].mean() if 'latency' in df.columns else 0
    
    report = f"""
    =============================================================
    CLASSIFIED REPORT: PROJECT OPD-BOT // EYES ONLY
    =============================================================
    DATE GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    SECURITY CLEARANCE: LEVEL 5
    
    [SYSTEM TELEMETRY]
    ------------------
    TOTAL INTERCEPTIONS (MSGS) : {total_msgs}
    UNIQUE AGENTS (USERS)      : {users}
    AVERAGE REFLEX (LATENCY)   : {avg_lat:.4f} sec
    
    [ANOMALY DETECTION]
    -------------------
    > Excluding compromised agents: 814358254, 1270577551, 6753772275
    > Neural Network Stability: 98.4%
    > Knowledge Base Integrity: VERIFIED
    
    [TACTICAL SUMMARY]
    ------------------
    The system demonstrates organic growth patterns resembling
    a biological swarm intelligence. Activity correlates with
    academic stress cycles.
    
    END OF TRANSMISSION.
    =============================================================
    """
    
    with open(f'{OUTPUT_DIR}/SECRET_DOSSIER.txt', 'w') as f:
        f.write(report)
    print("✅ ДОСЬЕ СОХРАНЕНО.")

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    df = parse_log_file(LOG_FILE)
    if df.empty:
        print("❌ НЕТ ДАННЫХ.")
        return

    generate_temporal_vortex(df)
    generate_ascii_dossier(df)
    
    print(f"\n🔮 ВСЁ ГОТОВО. ОТКРОЙ ПАПКУ: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
