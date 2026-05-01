# -*- coding: utf-8 -*-
import pickle
import numpy as np
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates'),
    static_folder=os.path.join(os.path.dirname(__file__), '..', 'static'),
)
CORS(app)

# ── Load model ──────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), '..', 'models')

with open(os.path.join(BASE, 'best_model.pkl'), 'rb') as f:
    model = pickle.load(f)
with open(os.path.join(BASE, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)
with open(os.path.join(BASE, 'label_encoders.pkl'), 'rb') as f:
    label_encoders = pickle.load(f)
with open(os.path.join(BASE, 'model_meta.pkl'), 'rb') as f:
    meta = pickle.load(f)

FEATURES = meta['features']
CAT_COLS = ['district', 'direction', 'furniture_std', 'legal_std', 'loai_bds']

print(f"[OK] Model loaded: {meta['best_model_name']}  R²={meta['metrics']['R2']:.4f}")

# ── Helper ───────────────────────────────────────────────────
def safe_encode(col, val):
    le = label_encoders[col]
    s  = str(val)
    return int(le.transform([s])[0]) if s in le.classes_ else 0

def build_feature_row(data):
    row = []
    for feat in FEATURES:
        val = data.get(feat, 0)
        if feat in CAT_COLS:
            val = safe_encode(feat, val)
        row.append(float(val))
    return row

# ── Routes ───────────────────────────────────────────────────
@app.route('/')
def index():
    districts = sorted(label_encoders['district'].classes_.tolist())
    directions = sorted(label_encoders['direction'].classes_.tolist())
    furniture  = sorted(label_encoders['furniture_std'].classes_.tolist())
    legal      = sorted(label_encoders['legal_std'].classes_.tolist())
    return render_template('index.html',
                           districts=districts,
                           directions=directions,
                           furniture=furniture,
                           legal=legal)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Validate
        for field in ['area_m2', 'bedrooms_num', 'district', 'loai_bds']:
            if not data.get(field):
                return jsonify({'success': False, 'error': f'Thiếu trường: {field}'}), 400

        # Defaults
        data.setdefault('direction',     'Không rõ')
        data.setdefault('furniture_std', 'Không rõ')
        data.setdefault('legal_std',     'Sổ đỏ/Sổ hồng')
        data.setdefault('floors_num',    0)
        data.setdefault('frontage_m',    0)
        data.setdefault('road_width_m',  0)

        row   = build_feature_row(data)
        X     = np.array([row])
        price = float(model.predict(X)[0])
        price = max(0.5, price)          # giá tối thiểu 0.5 tỷ

        return jsonify({
            'success':        True,
            'price_billion':  round(price, 3),
            'price_million':  round(price * 1000, 1),
            'price_display':  f'{price:.2f}',
            'price_per_m2':   round(price * 1000 / float(data['area_m2']), 1),
            'model_used':     meta['best_model_name'],
            'r2_score':       round(meta['metrics']['R2'], 4),
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/districts')
def districts():
    return jsonify({'districts': sorted(label_encoders['district'].classes_.tolist())})

@app.route('/model-info')
def model_info():
    return jsonify({
        'best_model':  meta['best_model_name'],
        'metrics':     meta['metrics'],
        'all_results': meta['all_results'],
    })

if __name__ == '__main__':
    print("[*] Server chay tai: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
