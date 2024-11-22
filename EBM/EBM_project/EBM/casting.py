import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import psycopg2

# Загрузка модели и токенизатора
model_path = "../best_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
with open(f"{model_path}/label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Подключение к базе данных
def get_db_connection():
    return psycopg2.connect(
        dbname="medical_data",
        user="postgres",
        password="01052004",
        host="localhost",
        port="5432"
    )


# Функция для предсказания диагноза
def predict(text, model, tokenizer, label_encoder, max_len=512, device=device):
    model.eval()
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).cpu().numpy()
    return label_encoder.inverse_transform(predicted_label)


# Функция для получения рекомендаций из базы данных
def find_recommendations_from_db(diagnosis_name):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT id, recommendations FROM diagnosis WHERE name = %s", (diagnosis_name,))
            row = cursor.fetchone()
            if row:
                diagnosis_id = row[0]
                recommendations = row[1]
                if isinstance(recommendations, str):
                    recommendations = recommendations.split(", ")
                return diagnosis_id, recommendations if recommendations else []
    finally:
        connection.close()
    return None, ["Рекомендации для данного диагноза отсутствуют."]


# Главная функция
def process_prediction(data):
    complaints = data.get("complaints", "")
    disease_history = data.get("disease_history", "")
    objective_status = data.get("objective_status", "")
    age = data.get("age", "")

    # Формируем текст для модели
    input_text = f"{complaints} | {disease_history} | {objective_status} | возраст: {age}"
    diagnosis_name = predict(input_text, model, tokenizer, label_encoder)[0]
    diagnosis_id, recommendations = find_recommendations_from_db(diagnosis_name)

    return {
        "diagnosis": diagnosis_name,
        "recommendations": recommendations,
        "diagnosis_id": diagnosis_id
    }