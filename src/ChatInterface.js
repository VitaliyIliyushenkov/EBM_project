import React, { useState, useEffect } from "react";
import axios from "axios";

function ChatInterface({ selectedPatient }) {
    const [messages, setMessages] = useState([]);
    const [visits, setVisits] = useState([]);
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [formData, setFormData] = useState({
        complaints: "",
        disease_history: "",
        objective_status: "",
        full_name: "",
        birth_date: "",
        age: ""
    });
    const [prediction, setPrediction] = useState(null);
    const [notes, setNotes] = useState("");
    const [isLoading, setIsLoading] = useState(false);

    useEffect(() => {
        if (selectedPatient) {
            setFormData({
                ...formData,
                full_name: selectedPatient.full_name,
                birth_date: selectedPatient.birth_date,
                age: selectedPatient.age_category || formData.age
            });
            axios.get(`http://127.0.0.1:8000/api/patients/${selectedPatient.id}/visits/`)
                .then(response => setVisits(response.data))
                .catch(error => {
                    console.error(error);
                    alert("Ошибка при загрузке визитов.");
                });
        }
    }, [selectedPatient]);

    const handleFormChange = (e) => {
        const { name, value } = e.target;
        setFormData({ ...formData, [name]: value });
    };

    const handleOpenModal = () => {
        setIsModalOpen(true);
    };

    const handleCloseModal = () => {
        setIsModalOpen(false);
        setPrediction(null); // Сбрасываем предсказание при закрытии
    };

    const handlePredict = () => {
        setIsLoading(true);
        axios.post("http://127.0.0.1:8000/api/predict/", formData)
            .then(response => {
                setPrediction(response.data);
                setIsLoading(false);
            })
            .catch(error => {
                console.error(error);
                alert("Ошибка при предсказании диагноза.");
                setIsLoading(false);
            });
    };

    const getAgeCategory = (age) => {
        if (age < 18) return "младше 18";
        if (age >= 18 && age <= 35) return "18-35";
        if (age >= 36 && age <= 60) return "36-60";
        return "старше 60";
    };

    const handleSaveVisit = () => {
        const dataToSave = {
            ...formData,
            age: getAgeCategory(formData.age),
            diagnosis: prediction?.diagnosis,
            recommendations: prediction?.recommendations,
            notes
        };

        axios.post("http://127.0.0.1:8000/api/save_visit/", dataToSave)
            .then(() => {
                alert("Визит успешно сохранен.");

                axios.get(`http://127.0.0.1:8000/api/patients/${selectedPatient.id}/visits/`)
                    .then(response => {
                        setVisits(response.data);
                    })
                    .catch(error => {
                        console.error(error);
                        alert("Ошибка при загрузке визитов.");
                    });

                setIsModalOpen(false);
                setNotes("");
                setPrediction(null);
            })
            .catch(error => {
                console.error(error);
                alert("Ошибка при сохранении визита.");
            });
    };

    return (
        <div style={{ flex: 1, padding: "10px" }}>
            <h3>Чат с пациентом</h3>
            <div style={{ height: "70vh", overflowY: "auto", border: "1px solid #ccc", padding: "10px" }}>
                {messages.map((msg, index) => (
                    <div key={index} style={{ textAlign: msg.user === "doctor" ? "right" : "left" }}>
                        <p>{msg.text}</p>
                    </div>
                ))}
                {visits.length > 0 && (
                    <div>
                        <h4>Прошлые визиты:</h4>
                        {visits.map((visit, index) => (
                            <div key={index}>
                                <p><strong>Дата визита:</strong> {visit.visit_date}</p>
                                <p><strong>Жалобы:</strong> {visit.complaints}</p>
                                <p><strong>История болезни:</strong> {visit.disease_history}</p>
                                <p><strong>Объективный статус:</strong> {visit.objective_status}</p>
                                <p><strong>Диагноз:</strong> {visit.diagnosis || 'Не указан'}</p>
                                <p><strong>Рекомендации:</strong> {visit.recommendations || 'Не указаны'}</p>
                                <p><strong>Заметки врача:</strong> {visit.notes || 'Нет заметок'}</p>
                            </div>
                        ))}
                    </div>
                )}
            </div>
            <button onClick={handleOpenModal}>Добавить запись</button>
            {isModalOpen && (
                <div className="modal">
                    <div className="modal-content">
                        <h4>Добавить новый визит</h4>
                        <form>
                            <label>Жалобы:</label>
                            <textarea name="complaints" value={formData.complaints} onChange={handleFormChange} />
                            <label>История болезни:</label>
                            <textarea name="disease_history" value={formData.disease_history} onChange={handleFormChange} />
                            <label>Объективный статус:</label>
                            <textarea name="objective_status" value={formData.objective_status} onChange={handleFormChange} />
                            <label>Возраст:</label>
                            <input type="number" name="age" value={formData.age} onChange={handleFormChange} />
                        </form>
                        <button onClick={handlePredict} disabled={isLoading}>
                            {isLoading ? "Загрузка..." : "Предсказать диагноз"}
                        </button>

                        {prediction && (
                            <div>
                                <h4>Результаты предсказания</h4>
                                <p><strong>Диагноз:</strong> {prediction.diagnosis}</p>
                                <p><strong>Рекомендации:</strong> {prediction.recommendations?.join(", ") || "Нет рекомендаций"}</p>
                                <label>Заметки врача:</label>
                                <textarea value={notes} onChange={(e) => setNotes(e.target.value)} />
                                <button onClick={handleSaveVisit}>Сохранить визит</button>
                            </div>
                        )}
                        <button onClick={handleCloseModal}>Закрыть</button>
                    </div>
                </div>
            )}
        </div>
    );
}

export default ChatInterface;
