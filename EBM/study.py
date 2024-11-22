import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import RandomOverSampler
import pymorphy2
import pickle
import psycopg2

# Лемматизация текста
morph = pymorphy2.MorphAnalyzer()

def lemmatize_text(text):
    words = text.split()
    return " ".join([morph.parse(word)[0].normal_form for word in words])

# Подключение к базе данных
def load_data_from_db():
    connection = psycopg2.connect(
        dbname="medical_data",
        user="postgres",
        password="01052004",
        host="localhost",
        port="5432"
    )
    texts, labels = [], []
    with connection.cursor() as cursor:
        cursor.execute("SELECT complaints, disease_history, objective_status, age, name FROM diagnosis;")
        rows = cursor.fetchall()
        for complaints, disease_history, objective_status, age, diagnosis in rows:
            # Лемматизация текстов
            complaints = lemmatize_text(complaints)
            disease_history = lemmatize_text(disease_history)
            objective_status = lemmatize_text(objective_status)

            # Преобразование возраста в категорию
            age_category = categorize_age(age)

            # Формирование текста для модели
            text = f"{complaints} | {disease_history} | {objective_status} | возрастная категория: {age_category}"
            texts.append(text)
            labels.append(diagnosis)
    connection.close()
    return texts, labels

# Кастомный Dataset
# Разбивает текст на три компонента (жалобы, анамнез, объективный статус) пригодный для Bert
class MedicalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, label_encoder):
        self.texts = texts
        self.labels = label_encoder.transform(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_parts = self.texts[idx].split(" | ")
        complaints, disease_history, objective_status = text_parts
        encoded_input = self.tokenizer(
            complaints,
            disease_history,
            objective_status,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded_input["input_ids"].squeeze(0),
            "attention_mask": encoded_input["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }

# Гиперпараметры
max_len = 512
batch_size = 8
epochs = 10
learning_rate = 2e-5

# Загрузка начальных данных
texts, labels = load_data_from_db()
current_texts, current_labels = texts, labels

label_encoder = LabelEncoder()
label_encoder.fit(labels)

# Балансировка классов
# Это увеличивает количество менее представленных классов, чтобы модель могла их лучше различать
ros = RandomOverSampler(random_state=42)
texts_balanced, labels_balanced = ros.fit_resample(np.array(texts).reshape(-1, 1), labels)
texts_balanced = texts_balanced.flatten()

# Разделение на обучающую и валидационную выборки
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts_balanced, labels_balanced, test_size=0.2, random_state=42
)

tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
train_dataset = MedicalDataset(train_texts, train_labels, tokenizer, max_len, label_encoder)
val_dataset = MedicalDataset(val_texts, val_labels, tokenizer, max_len, label_encoder)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Модель и оптимизатор
num_labels = len(label_encoder.classes_)
model = AutoModelForSequenceClassification.from_pretrained("DeepPavlov/rubert-base-cased", num_labels=num_labels)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Функция для преобразования возраста в категорию
def categorize_age(age):
    if age < 18:
        return "младше 18"
    elif 18 <= age <= 35:
        return "18-35"
    elif 36 <= age <= 60:
        return "36-60"
    else:
        return "старше 60"

# Оценка модели
# Считает предсказания, сравнивает их с истинными метками и выводит метрики классификации: Precision, Recall, F1-Score.
def evaluate_model(model, loader, device):
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    report = classification_report(true_labels, preds, target_names=label_encoder.classes_, digits=4, labels=np.arange(len(label_encoder.classes_)))
    print(report)
    return f1_score(true_labels, preds, average="weighted")

# Обучение с ранней остановкой
def train_model_with_early_stopping(model, train_loader, val_loader, optimizer, device, epochs, patience=3):
    best_val_f1 = 0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}: Training loss: {avg_loss:.4f}")

        val_f1 = evaluate_model(model, val_loader, device)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            model.save_pretrained("./best_model")
            tokenizer.save_pretrained("./best_model")
            with open("./best_model/label_encoder.pkl", "wb") as file:
                pickle.dump(label_encoder, file)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping!")
                break

    print("Training complete. Best validation F1:", best_val_f1)
    print("Модель и LabelEncoder сохранены!")

# Запуск обучения
train_model_with_early_stopping(model, train_loader, val_loader, optimizer, device, epochs)

