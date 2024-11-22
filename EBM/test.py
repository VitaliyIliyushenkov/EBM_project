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


# Получение или создание пациента
def get_or_create_patient(full_name, birth_date):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            # Проверка существования пациента
            cursor.execute("SELECT id FROM patients WHERE full_name = %s AND birth_date = %s", (full_name, birth_date))
            patient = cursor.fetchone()
            if patient:
                return patient[0]  # Возвращаем ID пациента

            # Если пациента нет, создаем новую запись
            cursor.execute(
                "INSERT INTO patients (full_name, birth_date) VALUES (%s, %s) RETURNING id",
                (full_name, birth_date)
            )
            connection.commit()
            return cursor.fetchone()[0]
    finally:
        connection.close()


# Сохранение визита в историю болезни с учетом возраста
def save_patient_visit(patient_id, complaints, disease_history, objective_status, age, diagnosis_id, notes):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO patient_visits (patient_id, complaints, disease_history, objective_status, age, diagnosis_id, notes)
                VALUES (%s, %s, %s, %s, %s, %s,  %s)
                """,
                (patient_id, complaints, disease_history, objective_status, age, diagnosis_id, notes)
            )
            connection.commit()
    finally:
        connection.close()


# Функция для получения рекомендаций из базы данных
def find_recommendations_from_db(diagnosis_name):
    connection = get_db_connection()
    recommendations = []
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT id, recommendations FROM diagnosis WHERE name = %s", (diagnosis_name,))
            row = cursor.fetchone()
            if row:
                diagnosis_id = row[0]
                recommendations = row[1]  # Получаем массив рекомендаций
                return diagnosis_id, recommendations
    finally:
        connection.close()
    return None, ["Рекомендации для данного диагноза отсутствуют."]


# Функция для предсказания диагноза
def predict(text, model, tokenizer, label_encoder, max_len, device):
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


# Основной процесс
def main():
    # Ввод информации о пациенте
    full_name = input("Введите полное имя пациента: ")
    birth_date = input("Введите дату рождения пациента (YYYY-MM-DD): ")
    patient_id = get_or_create_patient(full_name, birth_date)

    # Ввод данных для предсказания
    complaints = input("Введите жалобы пациента: ")
    disease_history = input("Введите историю болезни пациента: ")
    objective_status = input("Введите объективный статус пациента: ")
    age = int(input("Введите возраст пациента: "))  # Добавляем возраст


    # Предсказание диагноза
    text = f"{complaints}. {disease_history}. {objective_status}. возраст: {age}"  # Добавляем возраст в текст
    diagnosis_name = predict(text, model, tokenizer, label_encoder, 512, device)[0]
    diagnosis_id, diagnosis_recommendations = find_recommendations_from_db(diagnosis_name)

    # Вывод предсказания и рекомендаций
    print(f"Предсказанный диагноз: {diagnosis_name}")
    print("Рекомендации:")
    for rec in diagnosis_recommendations:
        print(f"- {rec}")

    # Одобрение диагноза врачом
    confirmation = input("Одобрить предсказанный диагноз? (да/нет): ").strip().lower()
    if confirmation == "да" and diagnosis_id:
        notes = input("Введите заметки врача: ").strip()
        save_patient_visit(patient_id, complaints, disease_history, objective_status, age, diagnosis_id, notes)  # Сохраняем возраст
        print("Диагноз и визит успешно сохранены в истории болезни.")
    else:
        print("Диагноз не сохранен.")


if __name__ == "__main__":
    main()
