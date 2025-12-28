import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime
import os
import warnings
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter

# --- CONFIGURATION (CYBERPUNK MODE) ---
LOG_FILE = 'chat_qa_log.txt'
OUTPUT_DIR = 'analytics_cyberpunk_report'

# ЧЕРНЫЙ СПИСОК (Михаил, Захар, Егор)
EXCLUDED_IDS = {'6753772275', '814358254', '1270577551'}

# СТИЛЬ: ТЕМНАЯ МАТЕРИЯ
plt.style.use('dark_background')
warnings.simplefilter(action='ignore')

def parse_log_file(filepath):
    print(f"ACCESSING MAINFRAME: {filepath}...")
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
    
    # ФИЛЬТРАЦИЯ "ЛОХОВ"
    if 'user_id' in df.columns:
        before = len(df)
        df = df[~df['user_id'].astype(str).isin(EXCLUDED_IDS)]
        print(f"SYSTEM PURGE: Удалено {before - len(df)} скомпрометированных записей.")
        
    return df

def generate_3d_report():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    df = parse_log_file(LOG_FILE)
    if df.empty: return

    df_time = df.dropna(subset=['datetime']).copy()
    print(f"PROCESSING {len(df_time)} DATA POINTS...")

    # Подготовка данных для 3D
    df_time['hour'] = df_time['datetime'].dt.hour
    df_time['weekday_num'] = df_time['datetime'].dt.weekday  # 0=Mon, 6=Sun
    df_time['msg_len'] = df_time['question'].str.len()
    
    # 1. 3D SCATTER ANIMATION (Вращающийся куб)
    # Оси: День недели (X), Час (Y), Длина сообщения (Z)
    print("RENDERING 3D HOLOGRAPHIC CUBE (GIF)... PLEASE WAIT...")
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Данные
    x = df_time['weekday_num'] + np.random.normal(0, 0.1, size=len(df_time)) # Джиттер
    y = df_time['hour'] + np.random.normal(0, 0.1, size=len(df_time))
    z = df_time['msg_len']
    colors = df_time['hour']
    
    scatter = ax.scatter(x, y, z, c=colors, cmap='plasma', s=40, alpha=0.8, edgecolors='w', linewidth=0.2)
    
    ax.set_xlabel('Day of Week (0-6)', color='cyan')
    ax.set_ylabel('Hour of Day (0-23)', color='magenta')
    ax.set_zlabel('Message Length (chars)', color='yellow')
    ax.set_title('DIGITAL BRAIN ACTIVITY MAPPING', color='white', fontsize=15)
    
    # Убираем серый фон осей
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)

    # Анимация вращения
    def update(frame):
        ax.view_init(elev=20, azim=frame)
        return scatter,

    # Создаем GIF (360 градусов, 1 кадр на 2 градуса = 180 кадров)
    try:
        ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50)
        ani.save(f'{OUTPUT_DIR}/1_3d_activity_spin.gif', writer=PillowWriter(fps=20))
        print("✅ 3D ANIMATION SAVED: 1_3d_activity_spin.gif")
    except Exception as e:
        print(f"⚠️ Animation skipped (needs ffmpeg/pillow): {e}")
        plt.savefig(f'{OUTPUT_DIR}/1_3d_static.png')

    plt.close()

    # 2. 3D SURFACE PLOT (Топография задержки)
    print("RENDERING LATENCY TOPOGRAPHY...")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Создаем сетку
    pivot = df_time.pivot_table(index='weekday_num', columns='hour', values='latency', aggfunc='mean').fillna(0)
    X, Y = np.meshgrid(pivot.columns, pivot.index)
    Z = pivot.values
    
    surf = ax.plot_surface(X, Y, Z, cmap='inferno', edgecolor='none', alpha=0.9)
    fig.colorbar(surf, shrink=0.5, aspect=5, label='Latency (sec)')
    
    ax.set_title('LATENCY MOUNTAINS (System Lag Topology)', fontsize=14, color='orange')
    ax.set_xlabel('Hour', color='white')
    ax.set_ylabel('Day', color='white')
    ax.set_zlabel('Lag', color='white')
    
    plt.savefig(f'{OUTPUT_DIR}/2_3d_latency_topo.png', dpi=300)
    print("✅ 3D TOPOGRAPHY SAVED.")

    # 3. NEON HEXBIN (Киберпанк активность)
    print("RENDERING NEON CLUSTERS...")
    plt.figure(figsize=(12, 8))
    
    # Рассчитываем lifetime
    user_stats = df_time.groupby('user_id').agg(
        msg_count=('question', 'count'),
        first_msg=('datetime', 'min'),
        last_msg=('datetime', 'max')
    )
    user_stats['lifetime'] = (user_stats['last_msg'] - user_stats['first_msg']).dt.total_seconds() / 86400
    
    hb = plt.hexbin(user_stats['lifetime'], user_stats['msg_count'], gridsize=20, cmap='spring', mincnt=1, bins='log')
    cb = plt.colorbar(hb, label='log10(User Density)')
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')
    
    plt.title('USER ENGAGEMENT CLUSTERS (NEON MODE)', fontsize=18, color='#00ff00')
    plt.xlabel('Days in System', fontsize=12, color='white')
    plt.ylabel('Total Messages', fontsize=12, color='white')
    plt.grid(True, alpha=0.2)
    
    plt.savefig(f'{OUTPUT_DIR}/3_neon_clusters.png', dpi=300)
    print("✅ NEON HEXBIN SAVED.")

    print(f"\n🚀 CYBERPUNK REPORT GENERATED: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_3d_report()
