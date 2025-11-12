
from llama_cpp import Llama
import sys

# ПОМЕНЯЙТЕ ПУТЬ НА СВОЙ:
model_path = r"c:\Users\black\OneDrive\Desktop\LLM_little_models\KviGPT-7b-Chat.i1-Q4_K_M.gguf"

try:
    # Инициализация модели
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,  # размер контекста
        n_threads=8,  # количество потоков
        verbose=True  # вывод информации о загрузке
    )
    
    print("✅ Модель успешно загружена!")
    
    # Простой запрос
    prompt = "You are an English teacher. We are learning Conditionals sentences. Rewrite the sentence, putting the verb in brackets in the correct tense, write only answer without explanation :  I wouldn’t eat there if I (be) you. It’s awful."
    
    response = llm(
        prompt,
        max_tokens=300,
        temperature=0.7,
        echo=False  # не выводить промпт в ответе
    )
    
    print("\n🤖 Ответ модели:")
    print(response['choices'][0]['text'])
    
except Exception as e:
    print(f"❌ Ошибка: {e}")