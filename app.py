from flask import Flask, request, jsonify, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained scaler and model
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Modern, Colorful HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sales AI Predictor</title>
    <style>
        :root {
            --primary: #6366f1;
            --secondary: #a855f7;
            --bg: #0f172a;
            --card-bg: #1e293b;
            --text: #f8fafc;
        }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            display: flex; 
            justify-content: center; 
            align-items: center; 
            height: 100vh; 
            background: linear-gradient(135deg, var(--bg) 0%, #1e1b4b 100%);
            color: var(--text);
        }
        .container { 
            background: var(--card-bg); 
            padding: 2.5rem; 
            border-radius: 1.5rem; 
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 10px 10px -5px rgba(0, 0, 0, 0.04); 
            max-width: 450px; 
            width: 100%;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        h2 { 
            margin-top: 0; 
            text-align: center; 
            background: linear-gradient(to right, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 1.8rem;
        }
        p.subtitle { text-align: center; color: #94a3b8; margin-bottom: 2rem; font-size: 0.9rem; }
        label { display: block; margin-bottom: 0.5rem; color: #cbd5e1; font-weight: 500; }
        input { 
            width: 100%; 
            padding: 0.75rem; 
            margin-bottom: 1.5rem; 
            background: #0f172a;
            border: 1px solid #334155; 
            border-radius: 0.75rem; 
            color: white;
            font-size: 1rem;
            box-sizing: border-box;
            transition: border-color 0.2s;
        }
        input:focus { outline: none; border-color: var(--primary); }
        button { 
            width: 100%; 
            padding: 0.75rem; 
            background: linear-gradient(to right, var(--primary), var(--secondary)); 
            color: white; 
            border: none; 
            border-radius: 0.75rem; 
            font-weight: 600;
            font-size: 1rem;
            cursor: pointer; 
            transition: transform 0.2s, opacity 0.2s;
        }
        button:hover { transform: translateY(-2px); opacity: 0.9; }
        .result-box { 
            margin-top: 2rem; 
            padding: 1rem; 
            background: rgba(99, 102, 241, 0.1); 
            border-radius: 0.75rem; 
            border: 1px dashed var(--primary);
            text-align: center;
        }
        .prediction-value { 
            font-size: 1.5rem; 
            font-weight: bold; 
            color: #fbbf24;
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Sales Predictor AI</h2>
        <p class="subtitle">Enter your advertising budget to estimate sales</p>
        
        <form action="/predict_web" method="post">
            <label>TV Advertising ($)</label>
            <input type="number" step="0.01" name="TV" placeholder="0.00" required>
            
            <label>Radio Advertising ($)</label>
            <input type="number" step="0.01" name="Radio" placeholder="0.00" required>
            
            <label>Newspaper Advertising ($)</label>
            <input type="number" step="0.01" name="Newspaper" placeholder="0.00" required>
            
            <button type="submit">Analyze & Predict</button>
        </form>

        {% if prediction %}
        <div class="result-box">
            <span style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase;">Estimated Revenue</span>
            <span class="prediction-value">${{ prediction }}k</span>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict_web', methods=['POST'])
def predict_web():
    tv = float(request.form.get('TV'))
    radio = float(request.form.get('Radio'))
    news = float(request.form.get('Newspaper'))
    
    features = np.array([[tv, radio, news]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    
    return render_template_string(HTML_TEMPLATE, prediction=f"{float(prediction[0]):,.2f}")

@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    features = np.array([[data['TV'], data['Radio'], data['Newspaper']]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    return jsonify({'sales_prediction': round(float(prediction[0]), 2)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)