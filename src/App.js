import React, { useState } from "react";
import PatientsList from "./PatientsList";
import ChatInterface from "./ChatInterface";

function App() {
    const [selectedPatient, setSelectedPatient] = useState(null);

    return (
        <div style={{ display: "flex" }}>
            <PatientsList onSelectPatient={setSelectedPatient} />
            {selectedPatient ? (
                <ChatInterface selectedPatient={selectedPatient} />
            ) : (
                <div style={{ flex: 1, textAlign: "center", padding: "20px" }}>
                    Выберите пациента, чтобы начать чат.
                </div>
            )}
        </div>
    );
}

export default App;
