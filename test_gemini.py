import os
try:
    from google import genai
    # Пробуем инициализировать клиент с ключом из окружения или Streamlit Secrets
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents="Привет! Ответь одним словом: Работаешь?",
    )
    print("Статус Gemini API: УСПЕХ ✅")
    print("Ответ модели:", response.text)
except Exception as e:
    print("Статус Gemini API: ОШИБКА ❌")
    print("Детали ошибки:", e)
