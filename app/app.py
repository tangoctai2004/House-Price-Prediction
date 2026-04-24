from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
# import pickle
# import numpy as np

app = Flask(__name__)
CORS(app)

# ==========================================
# TÀI LƯU Ý: Đây là khung sườn Backend.
# Khi Thái train xong model, em sẽ load model ở đây.
# ==========================================

# TODO: Mở comment khi có file model thực tế
# Load model AI đã train
# try:
#     with open('../models/best_model.pkl', 'rb') as f:
#         model = pickle.load(f)
# except FileNotFoundError:
#     model = None
#     print("Chưa có model. Vui lòng bảo Thái train model trước!")

@app.route('/')
def home():
    """Route mặc định render giao diện trang chủ"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API nhận data từ Frontend và trả về giá dự đoán"""
    try:
        # Lấy dữ liệu user nhập từ giao diện (JSON)
        data = request.get_json()
        
        area = float(data.get('area', 0))
        bedrooms = int(data.get('bedrooms', 0))
        district = data.get('district', '')
        
        # TODO: Cần transform data này giống y hệt cách Đức làm trong file Tiền xử lý
        # input_features = np.array([[area, bedrooms, ...]])
        
        # # Chạy dự đoán
        # if model:
        #     prediction = model.predict(input_features)[0]
        # else:
        #     prediction = 5200000000 # Mock data khi chưa có model
            
        # Tạm thời trả về số ảo (Mock Data) để Đông test Frontend
        mock_prediction = area * 100000000 # Cứ 1m2 = 100 triệu
        
        return jsonify({
            'success': True,
            'predicted_price_vnd': mock_prediction,
            'message': 'Dự đoán thành công (đang dùng số liệu ảo)'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    # Chạy server ở port 5000
    app.run(debug=True, host='0.0.0.0', port=5000)
