#!/usr/bin/env python3
# coding: utf-8
"""
Скрипт №1 (v2): Автоматическое обогащение датасета с помощью Ollama.
Генерирует до 25 вариантов вопросов.
"""
import json
import requests
import time
from pathlib import Path
from tqdm import tqdm
import re

# --- КОНФИГУРАЦИЯ ---
PROJECT_ROOT = Path(__file__).parent
INPUT_DATA_PATH = PROJECT_ROOT / "data/opd_dataset.json" # Убедитесь, что здесь исходный файл с 1 вопросом
OUTPUT_DATA_PATH = PROJECT_ROOT / "data/opd_dataset_augmented.json"

OLLAMA_HOST = "http://localhost:11434"
# Используйте быструю модель для генерации! Llama3 или Gemma справятся отлично.
OLLAMA_MODEL_FOR_AUGMENTATION = "gpt-oss:20b" 

# --- ИЗМЕНЕНИЕ №1: Просим 25 вариантов ---
PROMPT_TEMPLATE = """
Твоя задача — перефразировать вопрос, создав 25 его вариантов.
Варианты должны быть в стиле реального студента: используй простой язык, синонимы, можешь добавить немного сленга, задавай вопросы под разными углами.

ВАЖНОЕ ПРАВИЛО: Твой ответ должен быть ТОЛЬКО валидным JSON-массивом из 25 строк. Никакого текста до или после.

[ПРИМЕР]
ОРИГИНАЛЬНЫЙ ВОПРОС:
"Можно ли не посещать занятия по дисциплине «Основы проектной деятельности»"

ТВОЙ ОТВЕТ (пример для 5, но ты должен сделать 25):
[
    "обязательно ли ходить на опд",
    "что будет если я не приду на опд",
    "можно не ходить на основы проектной деятельности",
    "прогулять опд можно?",
    "какие последствия за пропуск опд"
]
[/ПРИМЕР]

ОРИГИНАЛЬНЫЙ ВОПРОС:
"{question}"

ТВОЙ ОТВЕТ:
"""

def call_ollama_for_variations(question: str) -> list[str]:
    prompt = PROMPT_TEMPLATE.format(question=question)
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": OLLAMA_MODEL_FOR_AUGMENTATION,
        "prompt": prompt, "temperature": 0.7, "stream": False,
    }
    try:
        response = requests.post(url, json=payload, timeout=180) # Увеличим таймаут
        response.raise_for_status()
        raw_text = response.json().get("response", "").strip()
        
        match = re.search(r'\[.*\]', raw_text, re.DOTALL)
        if match:
            try:
                variations = json.loads(match.group(0))
                if isinstance(variations, list) and len(variations) > 0:
                    return [str(v) for v in variations]
            except json.JSONDecodeError:
                print(f"\nНе удалось распарсить JSON из ответа модели для вопроса: {question}")
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Ошибка подключения к Ollama: {e}")
    return []

def main():
    if not INPUT_DATA_PATH.exists():
        print(f"❌ ОШИБКА: Исходный файл данных не найден: {INPUT_DATA_PATH}"); return

    print(f"▶️  Загрузка исходных данных из {INPUT_DATA_PATH.name}...")
    with open(INPUT_DATA_PATH, 'r', encoding='utf-8') as f:
        # Убедитесь, что ваш исходный файл имеет формат {"question": "...", "answer": "..."}
        original_data = json.load(f)

    augmented_dataset = []
    print(f"🚀 Начинаем аугментацию {len(original_data)} записей. Это займет значительное время...")

    for item in tqdm(original_data, desc="Генерация вопросов"):
        # Убедитесь, что ваш исходный файл использует ключ "question"
        original_question = item.get("question")
        if not original_question: continue
            
        variations = call_ollama_for_variations(original_question)
        unique_questions = list(dict.fromkeys([original_question] + variations))
        
        new_item = {
            "id": item["id"], "questions": unique_questions, "answer": item["answer"],
            "topic": item["topic"], "source": item["source"]
        }
        augmented_dataset.append(new_item)
        time.sleep(0.5)

    print(f"\n✅ Аугментация завершена. Сгенерировано {sum(len(i['questions']) for i in augmented_dataset)} вопросов.")
    
    with open(OUTPUT_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(augmented_dataset, f, ensure_ascii=False, indent=2)
    print(f"🎉 Новый обогащенный датасет сохранен в: {OUTPUT_DATA_PATH}")

if __name__ == "__main__":
    main()