import React, { useState, useEffect } from "react";
import axios from "axios";

function PatientsList({ onSelectPatient }) {
    const [patients, setPatients] = useState([]);
    const [isFormOpen, setIsFormOpen] = useState(false); // состояние для открытия/закрытия формы
    const [newPatient, setNewPatient] = useState({
        full_name: "",
        birth_date: "",
        age_category: "",
    });

    // Загрузка списка пациентов
    useEffect(() => {
        axios.get("http://127.0.0.1:8000/api/patients/")
            .then(response => setPatients(response.data))
            .catch(error => console.error(error));
    }, []);

    // Обработчик изменения полей формы
    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setNewPatient({
            ...newPatient,
            [name]: value,
        });
    };

    // Обработчик отправки формы
    const handleSubmit = (e) => {
        e.preventDefault();
        axios.post("http://127.0.0.1:8000/api/patients/", newPatient)
            .then((response) => {
                alert("Пациент успешно добавлен");
                setPatients([...patients, response.data]); // Добавляем нового пациента в список
                setIsFormOpen(false); // Закрываем форму
                setNewPatient({ full_name: "", birth_date: "", age_category: "" }); // очищаем форму
            })
            .catch((error) => {
                console.error(error);
                alert("Ошибка при добавлении пациента");
            });
    };

    return (
        <div style={{ width: "20%", borderRight: "1px solid #ccc", padding: "10px" }}>
            <h3>Список пациентов</h3>
            <ul>
                {patients.map(patient => (
                    <li key={patient.id} onClick={() => onSelectPatient(patient)}>
                        {patient.full_name}
                    </li>
                ))}
            </ul>
            <button onClick={() => setIsFormOpen(true)}>Добавить пациента</button>

            {/* Форма для добавления пациента */}
            {isFormOpen && (
                <div style={{ marginTop: "20px", borderTop: "1px solid #ccc", paddingTop: "10px" }}>
                    <h4>Добавить нового пациента</h4>
                    <form onSubmit={handleSubmit}>
                        <div>
                            <label>ФИО:</label>
                            <input
                                type="text"
                                name="full_name"
                                value={newPatient.full_name}
                                onChange={handleInputChange}
                                required
                            />
                        </div>
                        <div>
                            <label>Дата рождения:</label>
                            <input
                                type="date"
                                name="birth_date"
                                value={newPatient.birth_date}
                                onChange={handleInputChange}
                                required
                            />
                        </div>
                        <div>
                            <label>Возрастная категория:</label>
                            <select
                                name="age_category"
                                value={newPatient.age_category}
                                onChange={handleInputChange}
                                required
                            >
                                <option value="">Выберите категорию</option>
                                <option value="младше 18">младше 18</option>
                                <option value="18-35">18-35</option>
                                <option value="36-60">36-60</option>
                                <option value="старше 60">старше 60</option>
                            </select>
                        </div>
                        <button type="submit">Сохранить</button>
                        <button type="button" onClick={() => setIsFormOpen(false)}>
                            Отмена
                        </button>
                    </form>
                </div>
            )}
        </div>
    );
}

export default PatientsList;
