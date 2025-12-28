#!/usr/bin/env python3
# coding: utf-8
"""
Telegram-бот интерфейс (v25 - FULL RESTORED + UX).
Полный функционал:
- RAG
- Безопасность (SQL/Code injection filter)
- Умная разбивка сообщений
- Обработка отложенных обновлений
- UX: Статус "Думаю..." и защита от спама
"""
import logging
import asyncio
import re
from datetime import datetime, timezone, timedelta
from typing import cast, Set
import functools

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, error, Message
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler

from rag_chatbot import RAGChatBot
from config import TELEGRAM_BOT_TOKEN, ADMIN_USER_IDS

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- UX: Глобальный список для блокировки спама ---
PROCESSING_USERS: Set[int] = set()

# Максимальная длина одного сообщения в Telegram
MAX_MESSAGE_LENGTH = 4096

def admin_only(func):
    """
    Декоратор, который ограничивает доступ к функции только для администраторов.
    """
    @functools.wraps(func)
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user_id = update.effective_user.id
        if user_id not in ADMIN_USER_IDS:
            logger.warning(f"Несанкционированный доступ к админ-команде от UserID: {user_id}")
            await update.message.reply_text("У вас нет прав для выполнения этой команды.")
            return
        logger.info(f"Администратор (UserID: {user_id}) вызвал команду: {func.__name__}")
        return await func(update, context, *args, **kwargs)
    return wrapped

# 2. Фильтр подозрительных сообщений (v2.0 - Усиленный)
def is_input_suspicious(text: str) -> bool:
    """
    Проверяет входящий текст на наличие явных признаков атаки.
    """
    if not text:
        return False

    text_lower = text.lower()

    # Паттерн 1: Признаки JSON или объектов кода.
    code_like_pattern = r'("[\w_]+"\s*:\s*({.*}|\[.*\]|".*"|true|false|[\d\.]+))'
    if re.search(code_like_pattern, text):
        return True

    # Паттерн 2: Явные попытки SQL-инъекций
    sql_injection_patterns = [
        "' or '1'='1", "union select", "drop table", "truncate table",
        "exec(", "xp_cmdshell", "information_schema"
    ]
    if any(p in text_lower for p in sql_injection_patterns):
        return True

    # Паттерн 3: Попытки указать пути к файлам или выполнить команды
    command_patterns = [
        "/etc/passwd", "ls -la", "process.env", "select * from",
        "require(", "import os", "subprocess.run"
    ]
    if any(p in text_lower for p in command_patterns):
        return True
    
    # Паттерн 4: Несбалансированные скобки (часто во вредоносном коде)
    if text.count('{') != text.count('}') or text.count('[') != text.count(']'):
        return True

    return False

# 3. Ограничитель запросов (Rate Limiter)
RATE_LIMIT_SECONDS = 10
RATE_LIMIT_REQUESTS = 5

def is_rate_limited(user_id: int, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """
    Проверяет, не превысил ли пользователь лимит запросов.
    """
    now = datetime.now(timezone.utc)
    if 'user_requests' not in context.bot_data:
        context.bot_data['user_requests'] = {}

    user_timestamps = context.bot_data['user_requests'].get(user_id, [])
    
    # Удаляем старые временные метки
    user_timestamps = [ts for ts in user_timestamps if now - ts < timedelta(seconds=RATE_LIMIT_SECONDS)]
    
    if len(user_timestamps) >= RATE_LIMIT_REQUESTS:
        logger.warning(f"Превышен лимит запросов для UserID: {user_id}. Блокировка.")
        return True 

    user_timestamps.append(now)
    context.bot_data['user_requests'][user_id] = user_timestamps
    return False 

async def send_smart_split_message(bot, chat_id: int, text: str, reply_to_message_id: int | None = None):
    """
    Отправляет текстовое сообщение, интеллектуально разбивая его на части.
    """
    if not text:
        logger.warning("Попытка отправить пустое сообщение в чат %s.", chat_id)
        return

    MAX_CHARS = 4096
    parts = []
    
    # Семантическая разбивка по идеям
    semantic_parts = re.split(r'(?m)(^\s*\*\*Идея \d+.*)', text)
    if len(semantic_parts) > 1:
        logger.info("Обнаружены семантические разделители. Применяется разбивка по идеям.")
        if not semantic_parts[0].strip():
            semantic_parts.pop(0)
        for i in range(0, len(semantic_parts), 2):
            if i + 1 < len(semantic_parts):
                parts.append((semantic_parts[i] + semantic_parts[i+1]).strip())
            else:
                parts.append(semantic_parts[i].strip())
    
    # Если не вышло, разбивка по абзацам
    if not parts or (len(parts) == 1 and len(parts[0]) > MAX_CHARS):
        logger.info("Семантическая разбивка не удалась. Применяется разбивка по абзацам.")
        parts = []
        paragraphs = text.split('\n')
        current_part = ''
        for p in paragraphs:
            if len(current_part) + len(p) + 1 < MAX_CHARS:
                current_part += p + '\n'
            else:
                if current_part:
                    parts.append(current_part.strip())
                current_part = p + '\n'
        if current_part:
            parts.append(current_part.strip())
            
    # Грубая разбивка, если всё остальное не помогло
    if not parts and len(text) > MAX_CHARS:
         logger.warning("Текст не удалось разделить умно, будет произведена грубая разбивка.")
         parts = [text[i:i + MAX_CHARS] for i in range(0, len(text), MAX_CHARS)]
    if not parts and len(text) <= MAX_CHARS:
        parts.append(text)
        
    if not parts:
        logger.error("КРИТИЧЕСКАЯ ОШИБКА: Не удалось разделить сообщение.")
        return

    logger.info(f"Сообщение будет отправлено в {len(parts)} частях.")
    for i, part in enumerate(parts):
        if not part: continue
        try:
            reply_id = reply_to_message_id if i == 0 else None
            await bot.send_message(chat_id=chat_id, text=part, reply_to_message_id=reply_id)
        except Exception as e:
            logger.exception("Ошибка при отправке части %s из %s в чат %s: %s", i + 1, len(parts), chat_id, e)

def escape_markdown_v2(text: str) -> str:
    """Экранирует специальные символы для Telegram MarkdownV2."""
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

MENU_TEXT = escape_markdown_v2("""
*Что я умею?*

Я — ваш личный консультант по дисциплине «Основы проектной деятельности» (ОПД). Моя задача — отвечать на ваши вопросы, используя официальную базу знаний.

Вы можете спросить меня о:
- Правилах посещения занятий и интенсивов
- Системе начисления баллов и получении зачета
- Сроках и правилах сдачи отчетов
- Конкурсе студенческих проектов
- И многом другом!

Также я могу помочь придумать идею для вашего проекта или подсказать ваше личное расписание интенсивов.

P.S. Могу даже имя команде придумать 😄

Нажмите на кнопку ниже, чтобы увидеть примеры вопросов.
""")

EXAMPLES_TEXT = escape_markdown_v2("""
*Примеры вопросов, которые вы можете задать:*

• Что такое ОПД?
• Сколько раз нужно ходить на интенсивы?
• Как получить зачет?
• Где моя следующая пара по ОПД? (потребуется ФИО)
• Где взять шаблон презентации?

*Примеры запросов для генерации идей:*

• Придумай идею для проекта, связанного с экологией в вузе
• Нужна креативная идея для социального проекта
• Придумай название нашей команды, мы очень любим пончики!

Просто напишите свой вопрос в чат.
""")

def build_menu_keyboard() -> InlineKeyboardMarkup:
    keyboard = [
        [InlineKeyboardButton("Что я умею?", callback_data='show_menu_info')],
        [InlineKeyboardButton("Примеры вопросов", callback_data='show_examples')],
    ]
    return InlineKeyboardMarkup(keyboard)

async def log_question_answer(question, answer, user, user_message_time, bot_response_time, path="chat_qa_log.txt"):
    def _write():
        try:
            q = (question or "").replace("\n", " ").strip()
            a = (answer or "").replace("\n", " ").strip()
            uid = getattr(user, 'id', '-')
            uname = f"@{user.username}" if getattr(user, 'username', None) else "-"
            full = getattr(user, 'full_name', None) or f"{getattr(user, 'first_name', '') or ''} {getattr(user, 'last_name', '') or ''}".strip()
            duration = (bot_response_time - user_message_time).total_seconds()
            user_time_str = user_message_time.strftime('%Y-%m-%d %H:%M:%S')
            bot_time_str = bot_response_time.strftime('%Y-%m-%d %H:%M:%S')

            with open(path, "a", encoding="utf-8") as f:
                f.write(
                    f"----------------------------------------\n"
                    f"UserID: {uid}\nUsername: {uname}\nПолное имя: {full}\n"
                    f"Время сообщения: {user_time_str}\n"
                    f"Время ответа: {bot_time_str}\n"
                    f"Задержка (сек): {duration:.2f}\n"
                    f"Q: {q}\nA: {a}\n"
                )
        except Exception as e:
            logger.exception("Ошибка при записи в лог Q/A: %s", e)
    await asyncio.to_thread(_write)

# --- Основная логика бота ---
print("Загрузка RAG-модели...")
rag_bot = RAGChatBot(debug=False)
print("RAG-модель успешно загружена.")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_name = (user.first_name if user and getattr(user, 'first_name', None) else "пользователь")
    welcome_message = (
        f"Здравствуйте, {user_name}!\n\n"
        "Я ваш консультант по дисциплине «Основы проектной деятельности» (ОПД).\n\n"
        "Задайте мне любой вопрос или напишите 'меню', чтобы узнать о моих возможностях."
    )
    if update.message and getattr(update.message, 'chat', None):
        try:
            await send_smart_split_message(context.bot, update.message.chat.id, welcome_message, reply_to_message_id=getattr(update.message, 'message_id', None))
            return
        except Exception:
            logger.debug("Fallback на reply_text для welcome_message")
            try:
                await update.message.reply_text(welcome_message)
                return
            except Exception:
                logger.exception("Не удалось отправить welcome_message.")
    if update.effective_chat and context.bot:
        await send_smart_split_message(context.bot, update.effective_chat.id, welcome_message)

async def show_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message and getattr(update.message, 'chat', None):
        try:
            await context.bot.send_message(chat_id=update.message.chat.id, text="Выберите опцию:", reply_markup=build_menu_keyboard())
            return
        except Exception:
            logger.debug("Fallback на reply_text для show_menu")
            try:
                await update.message.reply_text("Выберите опцию:", reply_markup=build_menu_keyboard())
                return
            except Exception:
                logger.exception("Не удалось отправить show_menu через reply_text.")
    if update.effective_chat and context.bot:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Выберите опцию:", reply_markup=build_menu_keyboard())

async def show_menu_and_log(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        logger.warning("show_menu_and_log не может быть выполнен без сообщения.")
        return
    message = update.message
    user_msg_time = (message.date if message.date else datetime.now(timezone.utc))
    question_text: str = cast(str, message.text) if message.text is not None else ""
    bot_resp_time = datetime.now(timezone.utc)

    await log_question_answer(
        question=question_text,
        answer="[Вызвано меню]",
        user=update.effective_user,
        user_message_time=user_msg_time,
        bot_response_time=bot_resp_time
    )
    await show_menu(update, context)

async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query: return
    try:
        await query.answer()
    except Exception as e:
        logger.exception("Не удалось выполнить query.answer(): %s", e)

    if not isinstance(query.message, Message):
        logger.warning("Сообщение с кнопкой недоступно, обработка callback'а прервана.")
        return
    chat_id = query.message.chat_id

    try:
        if query.data == 'show_menu_info':
            await context.bot.send_message(
                chat_id=chat_id,
                text=MENU_TEXT,
                parse_mode='MarkdownV2'
            )
        elif query.data == 'show_examples':
            await context.bot.send_message(
                chat_id=chat_id,
                text=EXAMPLES_TEXT,
                parse_mode='MarkdownV2'
            )
    except Exception as e:
        logger.exception(f"Не удалось отправить сообщение в ответ на callback '{query.data}' в чат {chat_id}: {e}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if not message or not message.text or not update.effective_user:
        logger.warning("handle_message вызван без текстового сообщения или пользователя.")
        return

    user_id = update.effective_user.id
    chat_id = message.chat.id
    user_question = message.text.strip()

    # 1. Проверка на флуд
    if is_rate_limited(user_id, context): return
    
    # 2. Проверка на подозрительный ввод
    if is_input_suspicious(user_question):
        logger.critical(f"!!! ОБНАРУЖЕНА ПОПЫТКА АТАКИ от UserID: {user_id}. Сообщение: '{user_question}'")
        joke_response = await asyncio.to_thread(rag_bot.generate_security_joke)
        await log_question_answer(question=user_question, answer=f"[ОТВЕТ НА АТАКУ]: {joke_response}", user=update.effective_user, user_message_time=message.date if message.date else datetime.now(timezone.utc), bot_response_time=datetime.now(timezone.utc))
        await message.reply_text(joke_response)
        return

    # --- UX: БЛОКИРОВКА СПАМА ---
    if user_id in PROCESSING_USERS:
        await message.reply_text("⏳ Не спеши! Я еще отвечаю на твой прошлый вопрос.")
        return
    
    # Добавляем юзера в список занятых и шлем "Думаю..."
    PROCESSING_USERS.add(user_id)
    try:
        status_msg = await message.reply_text("⏳ Думаю...")
    except:
        status_msg = None # Если вдруг не удалось отправить

    # Оборачиваем всю логику в try...finally, чтобы ГАРАНТИРОВАННО разблокировать юзера
    try:
        user_data = context.user_data if context.user_data is not None else {}
        user_msg_time = message.date if message.date else datetime.now(timezone.utc)
        
        logger.info(f"Получен вопрос от [user_id: {user_id}, chat_id: {chat_id}]: '{user_question}'")
        await context.bot.send_chat_action(chat_id=chat_id, action='typing')
        
        def process_question_in_background():
            try:
                if user_data.get('awaiting_fio'):
                    if user_question.lower() == 'стоп':
                        user_data.pop('awaiting_fio', None)
                        return "Хорошо, поиск по расписанию отменен. Чем еще могу помочь?"
                    else:
                        logger.info(f"Обработка сообщения как ФИО: '{user_question}'")
                        schedule_info_list = rag_bot.find_schedule_by_fio(user_question)
                        if schedule_info_list:
                            user_data.pop('awaiting_fio', None)
                            return rag_bot.format_schedule_response(schedule_info_list)
                        else:
                            return ("К сожалению, не удалось найти вас в списках.\n\n"
                                    "Пожалуйста, попробуйте ввести Фамилию, Имя и Отчество еще раз, проверив правильность написания.\n"
                                    "Если хотите отменить поиск, напишите \"стоп\".")
                else:
                    intent = rag_bot.classify_intent(user_question)
                    logger.info(f"Классифицированное намерение: '{intent}'")
                    
                    if intent == 'schedule_lookup':
                        if context.user_data is not None:
                             context.user_data['awaiting_fio'] = True
                        return "Пожалуйста, напишите ваши Фамилию, Имя и Отчество для поиска в расписании."
                    elif intent == 'creative_idea':
                        return rag_bot.answer_creatively(user_question)
                    elif intent == 'creative_team_name':
                        return rag_bot.answer_team_name_creatively(user_question)
                    elif intent == 'smalltalk':
                        return rag_bot.answer_smalltalk(user_question)
                    else: # 'rag_faq' or 'unclear'
                        return rag_bot.answer_by_rag(user_question)
            except Exception as e:
                logger.error(f"Ошибка при обработке вопроса в фоне: {e}", exc_info=True)
                return "Произошла внутренняя ошибка. Попробуйте задать вопрос иначе."

        bot_response = await asyncio.to_thread(process_question_in_background)

        # --- UX: УДАЛЯЕМ СТАТУС "ДУМАЮ" ПЕРЕД ОТВЕТОМ ---
        if status_msg:
            try:
                await status_msg.delete()
            except:
                pass

        bot_resp_time = datetime.now(timezone.utc)
        await log_question_answer(
            question=user_question, answer=bot_response, user=update.effective_user,
            user_message_time=user_msg_time, bot_response_time=bot_resp_time
        )
        
        await send_smart_split_message(
            bot=context.bot, chat_id=chat_id, text=bot_response, reply_to_message_id=message.message_id
        )
        logger.info(f"Отправлен ответ для [chat_id: {chat_id}]: '{(bot_response[:200].strip())}'")

    except Exception as e:
        logger.exception("Критическая ошибка в handle_message")
    finally:
        # --- UX: РАЗБЛОКИРОВКА ЮЗЕРА (ВСЕГДА) ---
        PROCESSING_USERS.discard(user_id)

async def _process_pending_updates(application):
    try:
        bot = application.bot
    except Exception as e:
        logger.warning("Не удалось получить объект бота: %s", e)
        return

    try:
        updates = await bot.get_updates(timeout=1)
    except Exception as e:
        logger.debug("Не удалось получить отложенные обновления: %s", e)
        return

    if not updates:
        logger.info("Нет отложенных обновлений для обработки.")
        return

    logger.info(f"Обнаружено {len(updates)} отложенных обновлений.")
    last_update_id = None
    seen_chats = set()

    apology_text = (
        "Привет! 👋 Прошу прощения за долгое молчание — я был на техобслуживании! 🤖\n\n"
        "Зато теперь я вернулся с *обновленной базой знаний* и новыми возможностями. "
        "Если ваш вопрос ниже всё ещё актуален, я постараюсь на него ответить."
    )

    for upd in updates:
        try:
            last_update_id = max(last_update_id or 0, upd.update_id)
            msg = upd.message
            if not msg or not getattr(msg, 'text', None): continue
            
            chat_id = msg.chat.id
            text = msg.text.strip()

            if chat_id not in seen_chats:
                try:
                    await bot.send_message(chat_id=chat_id, text=apology_text, parse_mode='Markdown')
                except: pass
                seen_chats.add(chat_id)

            response = None
            try:
                intent = rag_bot.classify_intent(text)
                if intent == 'schedule_lookup':
                    response = "Я вижу, вы спрашивали про расписание. Напишите ФИО."
                elif intent in ['creative_idea', 'creative_team_name']:
                    response = "Я вижу ваш запрос на креатив. Повторите его, я готов!"
                elif intent == 'smalltalk':
                    response = rag_bot.answer_smalltalk(text)
                else:
                    response = rag_bot.answer_by_rag(text)
            except Exception:
                response = "К сожалению, при попытке ответить произошла ошибка. Задайте вопрос снова."

            if response:
                await send_smart_split_message(bot, chat_id, response, reply_to_message_id=msg.message_id)
                
        except Exception: pass

    if last_update_id:
        try:
            await bot.get_updates(offset=last_update_id + 1)
            logger.info("Очередь очищена.")
        except Exception: pass

def main():
    if not TELEGRAM_BOT_TOKEN:
        print("❌ ОШИБКА: Токен Telegram не найден.")
        return
    print("Запуск Telegram-бота...")
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(_process_pending_updates).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("menu", show_menu_and_log))
    application.add_handler(CallbackQueryHandler(button_callback_handler))

    menu_triggers = ['меню', 'помощь', 'что ты умеешь', 'что ты можешь', 'команды']
    application.add_handler(MessageHandler(filters.Regex(r'(?i)^\s*(' + r'|'.join(menu_triggers) + r')\s*$'), show_menu_and_log))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_polling()
    print("Бот остановлен.")

if __name__ == '__main__':
    main()
