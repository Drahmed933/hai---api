
from flask import Flask, request, jsonify
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

# Dummy model setup (replace with real trained model)
dummy_model = MultiOutputClassifier(RandomForestClassifier())
dummy_model.fit([[65, 10, 1, 1, 0, 0, 1, 55.0, 13.0, 1, 0, 0, 0, 0.75, 1]], [[1, 0, 0, 0, 0, 0]])

# Labels and advice
labels = ["VAE", "CLABSI", "CAUTI", "SSI", "C. diff", "HAP"]
advice_dict = {
    "VAE": "Check ventilator settings and assess readiness for extubation.",
    "CLABSI": "Evaluate central line necessity and inspect site for signs of infection.",
    "CAUTI": "Review catheter need and ensure aseptic technique in care.",
    "SSI": "Inspect surgical wound for signs of infection and review perioperative protocols.",
    "C. diff": "Assess for recent antibiotic use. Consider stool test and isolation if needed.",
    "HAP": "Monitor respiratory status. Consider chest imaging and early antibiotics."
}

# Set up Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        input_data = [
            data["age"],
            data["icu_days"],
            int(data["ventilator"]),
            int(data["central_line"]),
            int(data["urinary_catheter"]),
            int(data["surgery"]),
            int(data["antibiotics"]),
            data["crp"],
            data["wbc"],
            int(data["fever"]),
            int(data["diarrhea"]),
            int(data["sputum"]),
            int(data["wound"]),
            data["risk_score"],
            int(data["gender"])
        ]

        preds = dummy_model.predict([input_data])[0]
        results = {}
        advice = {}

        for i, label in enumerate(labels):
            risk = "High Risk" if preds[i] == 1 else "Low Risk"
            results[label] = risk
            if preds[i] == 1:
                advice[label] = advice_dict[label]

        return jsonify({"predictions": results, "advice": advice})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    from pyngrok import ngrok, conf
import os

# Kill any running processes
!pkill streamlit

# Run Streamlit app
os.system('streamlit run hai_dashboard_app.py &')

# Start the tunnel
public_url = ngrok.connect(8501)
print("Your Streamlit dashboard is ready at:", public_url)
