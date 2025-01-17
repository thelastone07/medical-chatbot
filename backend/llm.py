from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import json
from preds import *

app = Flask(__name__)
CORS(app)

API_NAME = 'GEMINI_API_KEY'
    
with open('secret.json','r') as f:
    data = json.load(f)
gemini_key = data.get(API_NAME)
genai.configure(api_key = gemini_key)

model = genai.GenerativeModel('models/gemini-1.5-flash-001')

current_disease = ""

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    try:
        response = model.generate_content(user_message)
        return jsonify({"response" : response.text})
    except Exception as e:
        return jsonify({"error":str(e)}), 500

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json
    symptoms = data['symptoms']
    
    disease = get_disease(symptoms)
    print(disease)
    current_disease = disease
    try :
        response = model.generate_content(f"Given {symptoms} and {disease}, rephrase the statements and inform user that he or she might have the {disease}")
        return jsonify({"response" : response.text})
    except Exception as e:
        return jsonify({"error":str(e)}), 500

@app.route("/api/explain", methods=["GET"])
def explain():
    disease = request.args.get('disease', None)
    if not disease:
        return jsonify("Kindly provide a disease"), 500
    desc = get_description(disease)
    prec = get_precautions(disease)
    try :
        response = model.generate_content(f"Given {disease}, {desc} and {prec}. Just rephrase the statements. Donot add anything else. End the statement with the advise that consult doctor if condition persists")
        return jsonify({"response" : response.text})
    except Exception as e:
        return jsonify({"error":str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
