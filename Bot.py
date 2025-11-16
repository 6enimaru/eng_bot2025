import os
import asyncio
import glob
import chromadb
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# ==================== НАСТРОЙКИ ====================
TELEGRAM_BOT_TOKEN = "8587733755:AAHI-Y-yA-T8G01pC3AdSzNoPIH_GqTK0fc"
MODEL_PATH = r"C:\Users\black\OneDrive\Desktop\LLM_little_models\KviGPT-7b-Chat.i1-Q4_K_M.gguf"
# ===================================================

# Глобальные переменные для RAG системы
collection = None
embedding_model = None
llm = None

# ==================== ВАШ РАБОЧИЙ КОД ====================

def load_llm_model():
    """
    Загружает вашу локальную LLM модель
    """
    print("🔄 Загружаю локальную LLM...")
    
    try:
        # === ЗАГРУЗКА LLM МОДЕЛИ ===
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=4096,  # размер контекста
            n_threads=8,  # количество потоков
            verbose=False  # вывод информации о загрузке
        )
        
        print("✅ LLM модель успешно загружена!")
        return llm
        
    except Exception as e:
        print(f"❌ Ошибка загрузки LLM: {e}")
        print("💡 Убедитесь что:")
        print("   - Путь к модели правильный")
        print("   - Файл модели существует")
        print("   - Установлена библиотека: pip install llama-cpp-python")
        return None

def load_text_files(folder_path="C:/Users/black/OneDrive/Desktop/Grammar_RAG"):
    """
    Загружает все текстовые файлы из папки
    """
    print("📁 Загружаю текстовые файлы...")
    
    # Ищем все .txt файлы в текущей папке
    text_files = glob.glob(os.path.join(folder_path, "*.txt"))
    
    if not text_files:
        print("❌ Не найдено .txt файлов в текущей папке")
        return []
    
    chunks = []
    
    for file_path in text_files:
        filename = os.path.basename(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if content:  # Если файл не пустой
                chunk = {
                    "content": content,
                    "metadata": {
                        "source_file": filename,
                        "topic": filename.replace('.txt', ''),  # Название файла = тема
                        "size_chars": len(content),
                        "estimated_tokens": len(content) // 4
                    }
                }
                chunks.append(chunk)
                print(f"✅ Загружено: {filename} ({len(content)} символов)")
            
        except Exception as e:
            print(f"❌ Ошибка чтения {filename}: {e}")
    
    print(f"\n🎯 Всего загружено тем: {len(chunks)}")
    return chunks

def show_chunks_info(chunks):
    """
    Показывает информацию о загруженных темах
    """
    print("\n📊 ЗАГРУЖЕННЫЕ ТЕМЫ:")
    print("=" * 50)
    
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk["metadata"]
        content_preview = chunk["content"][:80] + "..." if len(chunk["content"]) > 80 else chunk["content"]
        
        print(f"{i}. {metadata['topic']}:")
        print(f"   📏 Размер: {metadata['size_chars']} символов")
        print(f"   🔤 Токенов: ~{metadata['estimated_tokens']}")
        print(f"   📝 Начало: {content_preview}")
        print()

def create_vector_db(chunks):
    """
    Создает векторную базу данных из чанков
    """
    print("🔄 Создаю векторную БД...")
    
    try:
        # Инициализируем модель для эмбеддингов
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Создаем ChromaDB клиент
        client = chromadb.PersistentClient(path="./grammar_db")
        
        # Создаем или получаем коллекцию
        collection = client.get_or_create_collection(name="grammar_topics")
        
        # Добавляем документы в базу
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            documents.append(chunk["content"])
            metadatas.append(chunk["metadata"])
            ids.append(f"chunk_{i}")
        
        # Создаем эмбеддинги и добавляем в базу
        embeddings = model.encode(documents).tolist()
        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print("✅ Векторная БД создана!")
        return collection, model
        
    except Exception as e:
        print(f"❌ Ошибка создания векторной БД: {e}")
        return None, None

def ask_question_with_llm(question):
    """
    Задает вопрос RAG системе с использованием вашей LLM
    """
    global collection, embedding_model, llm
    
    if collection is None:
        return "❌ Векторная БД не создана"
    
    if llm is None:
        return "❌ LLM не загружена"
    
    print(f"\n❓ Вопрос студента: {question}")
    print("🤔 Ищу материалы...")
    
    # Создаем эмбеддинг для вопроса
    question_embedding = embedding_model.encode([question]).tolist()
    
    # Ищем похожие чанки
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=3
    )
    
    # Показываем найденные материалы (в консоль)
    print("📚 Найденные материалы:")
    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"{i+1}. Тема: {metadata['topic']}")
        print(f"   Содержание: {doc[:100]}...")
    
    # === ПОДГОТОВКА ПРОМПТА ДЛЯ LLM ===
    context = ""
    for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
        context += f"ТЕМА: {metadata['topic']}\n{doc}\n\n"
    
    prompt = f"""Ты - учитель английского языка. Ответь на вопрос студента, используя ТОЛЬКО предоставленные материалы.

МАТЕРИАЛЫ ДЛЯ ОТВЕТА:
{context}

ВОПРОС СТУДЕНТА: {question}

ИНСТРУКЦИИ:
1. Ответь четко и понятно на русском или английском (в зависимости от вопроса)
2. Используй только информацию из предоставленных материалов
3. Если в материалах нет ответа - вежливо скажи об этом
4. Будь терпеливым и helpful учителем
5. Приводи примеры из материалов

ОТВЕТ УЧИТЕЛЯ:"""
    
    # === ВЫЗОВ ВАШЕЙ LLM МОДЕЛИ ===
    print("\n🧠 Генерирую ответ...")
    
    try:
        response = llm(
            prompt,
            max_tokens=500,  # Увеличим для более полных ответов
            temperature=0.3,  # Понизим для более точных ответов
            echo=False,
            stop=["Студент:", "Учитель:", "Вопрос:"]  # Стоп-слова для чистоты ответа
        )
        
        answer = response['choices'][0]['text'].strip()
        print(f"\n💡 ОТВЕТ УЧИТЕЛЯ: {answer}")
        return answer
        
    except Exception as e:
        error_msg = f"❌ Ошибка генерации ответа: {e}"
        print(error_msg)
        return error_msg

def initialize_rag_system():
    """
    Инициализирует всю RAG систему
    """
    global collection, embedding_model, llm
    
    print("🎯 ИНИЦИАЛИЗАЦИЯ RAG СИСТЕМЫ ДЛЯ TELEGRAM")
    print("=" * 50)
    
    # Загружаем LLM модель
    llm = load_llm_model()
    if llm is None:
        return False
    
    # Загружаем текстовые файлы с темами
    chunks = load_text_files()
    
    if not chunks:
        print("❌ Нечего загружать. Добавьте .txt файлы в папку.")
        return False
    
    # Показываем информацию о темах
    show_chunks_info(chunks)
    
    # Создаем векторную БД
    collection, embedding_model = create_vector_db(chunks)
    
    if collection and embedding_model:
        print(f"\n✅ RAG СИСТЕМА ГОТОВА! Загружено {len(chunks)} тем")
        return True
    
    return False

# ==================== TELEGRAM БОТ ====================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    user = update.effective_user
    welcome_text = f"""
👋 Привет, {user.first_name}!

Я - умный учитель английского! 📚

🎯 **Что я умею:**
• Объяснять грамматику английского
• Отвечать на вопросы по учебным материалам
• Помогать с пониманием времен и конструкций

💡 **Примеры вопросов:**
• "Объясни Present Perfect"
• "В чем разница между Past Simple и Present Perfect?"
• "Что такое условные предложения?"
• "Как использовать модальные глаголы?"

Просто задай вопрос - и я найду ответ в учебных материалах! 🎓
    """
    await update.message.reply_text(welcome_text)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /help"""
    help_text = """
📖 **Помощь по использованию бота:**

**Как задавать вопросы:**
Просто напишите вопрос на русском или английском о грамматике, временах, или любых темах английского языка.

**Примеры:**
• "Объясни Present Perfect"
• "Разница между Past Simple и Present Perfect"
• "Что такое reported speech?"
• "Как использовать артикли a/an/the?"

**Команды:**
/start - начать работу
/help - показать эту справку
/status - статус системы

Не стесняйтесь задавать вопросы! Я здесь, чтобы помочь! 😊
    """
    await update.message.reply_text(help_text)

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /status"""
    if llm and collection:
        status_text = "✅ Система полностью готова к работе!\n\n🤖 LLM модель: загружена\n🗄️ Векторная БД: создана\n📚 Материалы: доступны"
    else:
        status_text = "🔄 Система загружается или возникли ошибки"
    
    await update.message.reply_text(status_text)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик текстовых сообщений"""
    user_message = update.message.text.strip()
    
    if not user_message:
        await update.message.reply_text("📝 Пожалуйста, введите вопрос")
        return
    
    # Показываем, что бот печатает
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    # Получаем ответ от RAG системы
    try:
        # Запускаем в отдельном потоке, чтобы не блокировать Telegram
        response = await asyncio.get_event_loop().run_in_executor(
            None, ask_question_with_llm, user_message
        )
        
        # Отправляем ответ пользователю
        await update.message.reply_text(response)
        
    except Exception as e:
        error_msg = f"❌ Произошла ошибка при обработке вопроса: {str(e)}"
        print(error_msg)
        await update.message.reply_text(error_msg)

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик ошибок"""
    print(f"❌ Ошибка Telegram бота: {context.error}")

def main():
    """Основная функция запуска"""
    print("🚀 ЗАПУСК TELEGRAM RAG БОТА...")
    
    # Фикс для Windows
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Инициализируем RAG систему
    if not initialize_rag_system():
        print("❌ Не удалось инициализировать RAG систему. Бот не запущен.")
        return
    
    try:
        # Создаем Telegram бота
        app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        
        # Добавляем обработчики
        app.add_handler(CommandHandler("start", start_command))
        app.add_handler(CommandHandler("help", help_command))
        app.add_handler(CommandHandler("status", status_command))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        app.add_error_handler(error_handler)
        
        # Запускаем бота
        print("\n✅ TELEGRAM БОТ ЗАПУЩЕН!")
        print("📱 Напишите вашему боту в Telegram")
        print("⏹️  Для остановки нажмите Ctrl+C")
        print("\n" + "=" * 50)
        
        app.run_polling()
        
    except Exception as e:
        print(f"❌ Ошибка запуска Telegram бота: {e}")

if __name__ == "__main__":
    main()