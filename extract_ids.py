import re
import os

# --- Параметры ---
LOG_FILE_PATH = "chat_qa_log.txt"  # Имя вашего файла с логами
OUTPUT_FILE_PATH = "user_ids_for_broadcast.txt" # Файл, куда сохраним ID

def extract_unique_ids():
    """
    Извлекает все уникальные UserID из лог-файла и сохраняет их.
    """
    print(f"Анализ файла логов: {LOG_FILE_PATH}")
    
    if not os.path.exists(LOG_FILE_PATH):
        print(f"❌ ОШИБКА: Файл логов '{LOG_FILE_PATH}' не найден. Невозможно извлечь ID.")
        return

    try:
        with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"❌ ОШИБКА: Не удалось прочитать файл логов: {e}")
        return

    # Ищем все строки, начинающиеся с "UserID: " и забираем из них числа
    found_ids = re.findall(r"^UserID: (\d+)", content, re.MULTILINE)
    
    if not found_ids:
        print("⚠️ В файле логов не найдено ни одного UserID.")
        return

    # Используем set для автоматического удаления дубликатов, затем превращаем обратно в список
    unique_ids = sorted(list(set(found_ids)))
    
    print(f"✅ Найдено {len(found_ids)} записей ID. Уникальных ID: {len(unique_ids)}.")

    try:
        with open(OUTPUT_FILE_PATH, "w", encoding="utf-8") as f:
            for user_id in unique_ids:
                f.write(f"{user_id}\n")
        print(f"✅ Уникальные ID успешно сохранены в файл: {OUTPUT_FILE_PATH}")
    except Exception as e:
        print(f"❌ ОШИБКА: Не удалось сохранить ID в файл: {e}")

if __name__ == "__main__":
    extract_unique_ids()