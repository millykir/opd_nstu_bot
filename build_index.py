#!/usr/bin/env python3
# coding: utf-8
"""
Создание FAISS-индекса (refactor). Этот скрипт поддерживает несколько форматов входного JSON:
 - список объектов вида {"id","questions": [..], "answer"}
 - список объектов по одной вариации: {"original_id","question_variant","answer"}
 - список объектов с одиночным полем "question" вместо "questions"

На выходе создаются:
 - knowledge_index.faiss
 - data_mapping.json (список записей в том же порядке, что и в индексе)
"""
import argparse
from pathlib import Path
import json
from collections import defaultdict
import numpy as np
from typing import List, Dict, Any

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm
except Exception as e:
    print("""ОШИБКА: Не найдены необходимые библиотеки.
Установите их командой: pip install sentence-transformers faiss-cpu numpy tqdm
""")
    raise


def load_and_normalize_data(data_path: Path) -> List[Dict[str, Any]]:
    """Загружает данные и приводит их к единому формату.
    Возвращает список словарей: {"id","questions" (list), "answer"}
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    # используем обычный dict вместо defaultdict(lambda: {...}) — типы понятнее для анализаторов
    grouped: Dict[str, Dict[str, Any]] = {}
    output: List[Dict[str, Any]] = []

    # Если входной файл уже содержит записи с questions -> оставить как есть
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and 'questions' in raw[0]:
        for item in raw:
            qlist = item.get('questions') or []
            # поддержка одиночного поля 'question'
            if 'question' in item and not qlist:
                qlist = [item['question']]
            qlist = [q.strip() for q in qlist if q and isinstance(q, str) and q.strip()]
            output.append({
                "id": item.get('id') or item.get('original_id'),
                "questions": qlist,
                "answer": item.get('answer', "")
            })
        return output

    # Иначе — возможно формат с множественными короткими записями
    for item in raw:
        # возможные поля
        oid = item.get('id') or item.get('original_id')
        q_single = item.get('question') or item.get('question_variant')
        ans = item.get('answer', "")
        if oid and q_single:
            # инициализируем структуру, если ещё нет
            if oid not in grouped:
                grouped[oid] = {"questions": [], "answer": ""}
            # убеждаемся, что questions — список
            if isinstance(q_single, str) and q_single.strip():
                grouped[oid]['questions'].append(q_single.strip())
            if not grouped[oid]['answer'] and ans:
                grouped[oid]['answer'] = ans
        else:
            # если встретили объект с полем questions
            if isinstance(item, dict) and 'questions' in item:
                qlist = item.get('questions') or []
                qlist = [q.strip() for q in qlist if q and isinstance(q, str) and q.strip()]
                output.append({
                    "id": item.get('id') or item.get('original_id'),
                    "questions": qlist,
                    "answer": item.get('answer', "")
                })

    # собрать сгруппированные
    for oid, v in grouped.items():
        output.append({"id": oid, "questions": v['questions'], "answer": v['answer']})

    return output


def build_index(data_path: Path, index_path: Path, map_path: Path, model_name: str, use_cuda: bool):
    data = load_and_normalize_data(data_path)
    print(f"✅ Подготовлено {len(data)} записей для индексирования.")

    # Разворачиваем список вопросов в passages и соответствующую карту
    passages: List[str] = []
    new_map: List[Dict[str, Any]] = []
    for item in data:
        oid = item.get('id') or f"item_{len(new_map)}"
        questions = item.get('questions') or []
        if not isinstance(questions, list):
            # защита на случай, если questions случайно строка
            questions = [questions] if isinstance(questions, str) else []
        if not questions:
            continue
        for q in questions:
            if not isinstance(q, str):
                continue
            text = q.strip()
            if not text:
                continue
            passages.append(text)
            new_map.append({
                "question_variant": text,
                "original_id": oid,
                "answer": item.get('answer', "")
            })

    if not passages:
        raise ValueError("Нет подходящих вопросов для индексирования.")

    device = 'cuda' if use_cuda else 'cpu'
    print(f"▶️ Загружаем модель эмбеддингов: {model_name} ({device})...")
    encoder = SentenceTransformer(model_name, device=device)

    print(f"▶️ Преобразуем {len(passages)} текстов в эмбеддинги...")
    # sentence-transformers может вернуть list[list[float]] или np.ndarray
    embeddings = encoder.encode(passages, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"Эмбеддинги имеют некорректную форму: {embeddings.shape}")
    print("✅ Эмбеддинги созданы.")

    # Используем IndexFlatIP + нормализованные векторы для косинус-похожести
    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dim)
    # faiss.IndexFlatIP.add принимает массив numpy с формой (n, dim)
    index.add(embeddings)  # type: ignore[call-arg]

    faiss.write_index(index, str(index_path))
    with open(map_path, 'w', encoding='utf-8') as f:
        json.dump(new_map, f, ensure_ascii=False, indent=2)

    print(f"✅ Индекс сохранён: {index_path}")
    print(f"✅ Карта данных сохранена: {map_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=Path, default=Path('data/opd_dataset_augmented.json'))
    parser.add_argument('--index_path', type=Path, default=Path('knowledge_index.faiss'))
    parser.add_argument('--map_path', type=Path, default=Path('data_mapping.json'))
    parser.add_argument('--model', type=str, default='BAAI/bge-m3')
    parser.add_argument('--use_cuda', action='store_true')
    args = parser.parse_args()

    if not args.data_path.exists():
        print(f"❌ Файл данных не найден: {args.data_path}")
        return

    build_index(args.data_path, args.index_path, args.map_path, args.model, args.use_cuda)


if __name__ == '__main__':
    main()
