from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import logging

app = Flask(__name__)
CORS(app)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MyAIAssistant:
    def __init__(self):
        self.model_loaded = False
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Загружаем модель (выполняется при первом запросе)"""
        try:
            logger.info("🔄 Загружаем AI модель...")
            
            # Вариант 1: Быстрая легкая модель для диалогов
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
            self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
            
            # Вариант 2: Или используем pipeline для простоты
            self.chat_pipeline = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",
                tokenizer=self.tokenizer,
                max_length=500
            )
            
            self.model_loaded = True
            logger.info("✅ Модель успешно загружена!")
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            raise

ai_assistant = MyAIAssistant()

# Системный промпт для определения личности AI
SYSTEM_PROMPT = """Ты - полезный AI ассистент по имени DeepHelper. Ты помогаешь пользователям с вопросами по программированию, математике и другим темам. Будь вежливым, полезным и точным в ответах."""

def generate_response(user_message):
    """Генерируем ответ с учетом контекста"""
    if not ai_assistant.model_loaded:
        ai_assistant.load_model()
    
    try:
        # Создаем промпт с контекстом
        prompt = f"{SYSTEM_PROMPT}\nПользователь: {user_message}\nАссистент:"
        
        # Генерируем ответ
        with torch.no_grad():
            inputs = ai_assistant.tokenizer.encode(prompt, return_tensors='pt')
            outputs = ai_assistant.model.generate(
                inputs,
                max_length=len(inputs[0]) + 200,  # Ограничиваем длину ответа
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=ai_assistant.tokenizer.eos_token_id,
                repetition_penalty=1.2
            )
            
        response = ai_assistant.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Извлекаем только ответ ассистента
        response = response.split("Ассистент:")[-1].strip()
        
        return response
        
    except Exception as e:
        logger.error(f"Ошибка генерации: {e}")
        return "Извините, произошла ошибка при обработке запроса."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'Пустое сообщение'}), 400
        
        logger.info(f"📨 Получено сообщение: {user_message}")
        
        # Генерируем ответ
        response = generate_response(user_message)
        
        logger.info(f"📤 Ответ сгенерирован: {response[:100]}...")
        
        return jsonify({
            'response': response,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Ошибка в /api/chat: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

@app.route('/api/status')
def status():
    return jsonify({
        'status': 'online',
        'model_loaded': ai_assistant.model_loaded,
        'name': 'DeepHelper AI'
    })

if __name__ == '__main__':
    logger.info("🚀 Запускаем AI сервер...")
    app.run(host='0.0.0.0', port=5000, debug=True)
