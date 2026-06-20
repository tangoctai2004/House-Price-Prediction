import os
import base64
import json
import time
import urllib.parse
import urllib.request
import sys
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_cors import CORS
import pickle
import pandas as pd
from secrets import token_hex
from werkzeug.security import check_password_hash, generate_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY') or token_hex(32)

cors_origins = [origin.strip() for origin in os.environ.get('CORS_ORIGINS', '').split(',') if origin.strip()]
if cors_origins:
    CORS(app, resources={r"/api/*": {"origins": cors_origins}})

# Google OAuth: set env vars, or create app/oauth_config.py from oauth_config.example.py.
try:
    from oauth_config import GOOGLE_CLIENT_ID as GOOGLE_CLIENT_ID_LOCAL
    from oauth_config import GOOGLE_CLIENT_SECRET as GOOGLE_CLIENT_SECRET_LOCAL
except ModuleNotFoundError:
    try:
        from app.oauth_config import GOOGLE_CLIENT_ID as GOOGLE_CLIENT_ID_LOCAL
        from app.oauth_config import GOOGLE_CLIENT_SECRET as GOOGLE_CLIENT_SECRET_LOCAL
    except ModuleNotFoundError:
        GOOGLE_CLIENT_ID_LOCAL = ''
        GOOGLE_CLIENT_SECRET_LOCAL = ''

GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID', '').strip() or GOOGLE_CLIENT_ID_LOCAL.strip()
GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET', '').strip() or GOOGLE_CLIENT_SECRET_LOCAL.strip()

GOOGLE_CONFIGURED = bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)

# Simple in-memory user store (demo)
USERS = {
    'admin@prophet.vn': {
        'password_hash': generate_password_hash('123456', method='pbkdf2:sha256'),
        'name': 'Admin',
        'phone': '0912 345 678',
        'gender': 'Nam',
        'dob': '2004-01-15',
    },
}

PREDICTION_HISTORY = {}
SAVED_PROPERTIES = {}

def password_matches(user, password):
    password_hash = user.get('password_hash')
    if password_hash:
        return check_password_hash(password_hash, password)

    legacy_password = user.get('password')
    return legacy_password is not None and legacy_password == password

def _decode_jwt_payload(token):
    try:
        payload = token.split('.')[1]
        payload += '=' * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload).decode('utf-8'))
    except Exception:
        return None

def verify_google_credential(credential):
    if not GOOGLE_CLIENT_ID:
        raise ValueError('Google Client ID chưa được cấu hình.')

    if google_id_token is not None and google_requests is not None:
        try:
            return google_id_token.verify_oauth2_token(
                credential,
                google_requests.Request(),
                GOOGLE_CLIENT_ID
            )
        except Exception:
            pass

    payload = _decode_jwt_payload(credential)
    if not payload:
        raise ValueError('Google credential không hợp lệ.')
    if payload.get('aud') != GOOGLE_CLIENT_ID:
        raise ValueError('Google Client ID không khớp.')
    if int(payload.get('exp', 0)) < int(time.time()):
        raise ValueError('Google credential đã hết hạn.')
    return payload

def exchange_google_code(code, redirect_uri):
    data = urllib.parse.urlencode({
        'code': code,
        'client_id': GOOGLE_CLIENT_ID,
        'client_secret': GOOGLE_CLIENT_SECRET,
        'redirect_uri': redirect_uri,
        'grant_type': 'authorization_code',
    }).encode('utf-8')
    req = urllib.request.Request(
        'https://oauth2.googleapis.com/token',
        data=data,
        headers={'Content-Type': 'application/x-www-form-urlencoded'},
        method='POST'
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode('utf-8'))

def user_from_google_id_token(id_token_value):
    payload = _decode_jwt_payload(id_token_value)
    if not payload:
        raise ValueError('Google token invalid.')
    if payload.get('aud') != GOOGLE_CLIENT_ID:
        raise ValueError('Google Client ID mismatch.')
    if int(payload.get('exp', 0)) < int(time.time()):
        raise ValueError('Google token expired.')
    return payload


# ==========================================
# Khởi động app: Load model 1 lần duy nhất (Singleton Pattern)
# ==========================================
models = {}

def load_model(name):
    # Lấy đường dẫn an toàn cho dù chạy từ root hay thư mục app/
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', f'{name}.pkl')
    if not os.path.exists(model_path): # Fallback
        model_path = f'models/{name}.pkl'
    try:
        with open(model_path, 'rb') as f:
            models[name] = pickle.load(f)
        print(f"[OK] Da load thanh cong {name}")
    except FileNotFoundError:
        print(f"[ERROR] Chua tim thay model {name}. Vui long chay file run_training.py truoc!")
        models[name] = None

# Load mô hình duy nhất đã được gộp bằng Pipeline
load_model('best_model_pipeline')

transformer_bundle = {}

def load_transformer_model():
    """Load optional Transformer experiment model without breaking the main web app."""
    transformer_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'transformer')
    output_dir = os.path.join(transformer_dir, 'outputs')
    checkpoint_path = os.path.join(output_dir, 'transformer_model.pt')
    preprocessing_path = os.path.join(output_dir, 'preprocessing.pkl')

    if not (os.path.exists(checkpoint_path) and os.path.exists(preprocessing_path)):
        print("[WARNING] Chua tim thay Transformer outputs, bo qua model Transformer")
        return

    try:
        if transformer_dir not in sys.path:
            sys.path.insert(0, transformer_dir)
        import torch
        from transformer_model import HousePriceTransformer

        torch.set_num_threads(1)

        with open(preprocessing_path, 'rb') as f:
            preprocessing = pickle.load(f)

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model = HousePriceTransformer(**checkpoint['model_params'])
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        transformer_bundle.update({
            'torch': torch,
            'model': model,
            'preprocessing': preprocessing,
        })
        print("[OK] Da load thanh cong Transformer model")
    except Exception as e:
        transformer_bundle.clear()
        print("[WARNING] Khong load duoc Transformer model:", e)

def predict_with_transformer(input_data):
    if not transformer_bundle:
        return None

    torch = transformer_bundle['torch']
    model = transformer_bundle['model']
    preprocessing = transformer_bundle['preprocessing']

    numeric_features = preprocessing['numeric_features']
    categorical_features = preprocessing['categorical_features']

    numeric = preprocessing['scaler'].transform(input_data[numeric_features])

    cat_ids = []
    row = input_data.iloc[0]
    for col in categorical_features:
        mapping = preprocessing['category_mappings'][col]
        cat_ids.append(mapping.get(str(row[col]), 0))

    with torch.no_grad():
        output = model(
            torch.tensor(numeric, dtype=torch.float32),
            torch.tensor([cat_ids], dtype=torch.long),
        )
    return float(output.item())

load_transformer_model()

# Load metadata (MAE, Features importance, etc.)
model_metadata = {}
try:
    meta_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'model_meta.pkl')
    if not os.path.exists(meta_path): meta_path = 'models/model_meta.pkl'
    with open(meta_path, 'rb') as f:
        model_metadata = pickle.load(f)
    print("[OK] Da load thanh cong model_meta")
except:
    model_metadata = {'metrics': {'MAE': 0.85}, 'feature_names': []}
    print("[WARNING] Khong tim thay model_meta.pkl, dung mac dinh")

# Load data cho gợi ý căn nhà tương tự
dataframes = {}
try:
    csv_chung_cu = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed', 'cleaned_chung_cu.csv')
    csv_nha_dat = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed', 'cleaned_nha_dat.csv')
    if not os.path.exists(csv_chung_cu): csv_chung_cu = 'data/processed/cleaned_chung_cu.csv'
    if not os.path.exists(csv_nha_dat): csv_nha_dat = 'data/processed/cleaned_nha_dat.csv'
    
    dataframes['chung_cu'] = pd.read_csv(csv_chung_cu)
    dataframes['nha_dat'] = pd.read_csv(csv_nha_dat)
    print("[OK] Da load du lieu CSV de lam goi y")
except Exception as e:
    print("[ERROR] Khong load duoc du lieu CSV:", e)

@app.route('/')
def home():
    """Route mặc định render giao diện trang chủ"""
    return render_template('index.html', user=session.get('user'))

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json(silent=True) or {}
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    user = USERS.get(email)
    if user and password_matches(user, password):
        session['user'] = {'email': email, 'name': user['name'], 'avatar': user.get('avatar', '')}
        return jsonify({'success': True, 'name': user['name']})
    return jsonify({'success': False, 'error': 'Email hoặc mật khẩu không đúng.'}), 401

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json(silent=True) or {}
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    name = data.get('name', 'Người dùng')
    if not email or not password:
        return jsonify({'success': False, 'error': 'Vui lòng nhập đầy đủ thông tin.'}), 400
    if email in USERS:
        return jsonify({'success': False, 'error': 'Email này đã được đăng ký.'}), 400
    USERS[email] = {'password_hash': generate_password_hash(password, method='pbkdf2:sha256'), 'name': name}
    session['user'] = {'email': email, 'name': name}
    return jsonify({'success': True, 'name': name})

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/')

# ── Google OAuth Routes ────────────────────────────────────────────
@app.route('/auth/google')
def google_login():
    if not GOOGLE_CONFIGURED:
        return jsonify({'error': 'Google OAuth is not configured. Add GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET.'}), 503
    state = token_hex(16)
    session['google_oauth_state'] = state
    redirect_uri = url_for('google_callback', _external=True)
    params = urllib.parse.urlencode({
        'response_type': 'code',
        'client_id': GOOGLE_CLIENT_ID,
        'redirect_uri': redirect_uri,
        'scope': 'openid email profile',
        'state': state,
        'access_type': 'online',
        'prompt': 'select_account',
    })
    return redirect(f'https://accounts.google.com/o/oauth2/v2/auth?{params}')

@app.route('/auth/google/callback')
def google_callback():
    if not GOOGLE_CONFIGURED:
        return redirect('/?auth_error=google_unavailable')
    if request.args.get('state') != session.pop('google_oauth_state', None):
        return redirect('/?auth_error=google_state')
    code = request.args.get('code')
    if not code:
        return redirect('/?auth_error=google_no_code')
    try:
        token = exchange_google_code(code, url_for('google_callback', _external=True))
        user_info = user_from_google_id_token(token.get('id_token', ''))
        email = user_info.get('email', '').strip().lower()
        name = user_info.get('name') or email.split('@')[0]
        picture = user_info.get('picture', '')
        if email not in USERS:
            USERS[email] = {'password_hash': None, 'name': name, 'picture': picture, 'provider': 'google'}
        session['user'] = {'email': email, 'name': name, 'picture': picture, 'provider': 'google'}
        return redirect('/')
    except Exception:
        return redirect('/?auth_error=google_failed')

@app.route('/auth/google/token', methods=['POST'])
def google_token_login():
    return jsonify({'success': False, 'error': 'Google token login is disabled. Use /auth/google.'}), 410
    data = request.get_json(silent=True) or {}
    credential = data.get('credential', '')
    try:
        user_info = verify_google_credential(credential)
        email = user_info.get('email', '').strip().lower()
        if not email:
            return jsonify({'success': False, 'error': 'Google không trả về email.'}), 400

        name = user_info.get('name') or email.split('@')[0]
        picture = user_info.get('picture', '')
        if email not in USERS:
            USERS[email] = {'password_hash': None, 'name': name, 'picture': picture, 'provider': 'google'}
        session['user'] = {'email': email, 'name': name, 'picture': picture, 'provider': 'google'}
        return jsonify({'success': True, 'name': name})
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/me')
def me():
    user = session.get('user')
    if user:
        return jsonify({'logged_in': True, 'name': user['name']})
    return jsonify({'logged_in': False})

@app.route('/api/auth/status')
def auth_status():
    return jsonify({
        'google_configured': bool(GOOGLE_CONFIGURED)
    })

@app.route('/analytics')
def analytics():
    """Trang phân tích dữ liệu bất động sản từ dữ liệu processed"""
    import json as _json

    cc = dataframes.get('chung_cu', pd.DataFrame())
    nd = dataframes.get('nha_dat', pd.DataFrame())

    def safe_dict(series):
        return {str(k): round(float(v), 2) for k, v in series.items() if pd.notna(k) and str(k) != 'nan'}

    stats = {
        'total_cc': len(cc),
        'total_nd': len(nd),
        'total': len(cc) + len(nd),
        # Price stats
        'cc_price_mean': round(float(cc['price_billion'].mean()), 2) if len(cc) else 0,
        'cc_price_median': round(float(cc['price_billion'].median()), 2) if len(cc) else 0,
        'cc_price_min': round(float(cc['price_billion'].min()), 2) if len(cc) else 0,
        'cc_price_max': round(float(cc['price_billion'].max()), 2) if len(cc) else 0,
        'nd_price_mean': round(float(nd['price_billion'].mean()), 2) if len(nd) else 0,
        'nd_price_median': round(float(nd['price_billion'].median()), 2) if len(nd) else 0,
        # District counts
        'cc_dist_count': safe_dict(cc['district'].value_counts().head(10)),
        'nd_dist_count': safe_dict(nd['district'].value_counts().head(10)),
        # Avg price by district
        'cc_price_by_dist': safe_dict(cc.groupby('district')['price_billion'].mean().sort_values(ascending=False).head(10)),
        'nd_price_by_dist': safe_dict(nd.groupby('district')['price_billion'].mean().sort_values(ascending=False).head(10)),
        # Legal breakdown
        'cc_legal': safe_dict(cc['legal_std'].value_counts()),
        'nd_legal': safe_dict(nd['legal_std'].value_counts()),
        # Furniture
        'cc_furniture': safe_dict(cc['furniture_std'].value_counts()),
        'nd_furniture': safe_dict(nd['furniture_std'].value_counts()),
        # Bedroom dist
        'cc_bedroom': safe_dict(cc['bedrooms_num'].value_counts().sort_index()),
        'nd_bedroom': safe_dict(nd['bedrooms_num'].value_counts().sort_index().head(6)),
        # Area ranges
        'cc_area_mean': round(float(cc['area_m2'].mean()), 1) if len(cc) else 0,
        'nd_area_mean': round(float(nd['area_m2'].mean()), 1) if len(nd) else 0,
    }

    return render_template('analytics.html',
                           stats=stats,
                           stats_json=_json.dumps(stats, ensure_ascii=False),
                           user=session.get('user'))

@app.route('/about')
def about():
    return render_template('about.html', user=session.get('user'))

@app.route('/news')
def news():
    return render_template('news.html', user=session.get('user'))

@app.route('/search')
def search():
    return render_template('search.html', user=session.get('user'))

@app.route('/api/search')
def api_search():
    """Search properties from CSV data with filters"""
    import math
    q        = request.args.get('q', '').strip().lower()
    ptype    = request.args.get('type', 'all')       # all / chung_cu / nha_dat
    sort     = request.args.get('sort', 'default')
    district  = request.args.get('district', '').strip()
    direction = request.args.get('direction', '').strip()
    per_page  = 12

    if ptype not in {'all', 'chung_cu', 'nha_dat'}:
        return jsonify({'success': False, 'error': 'Loại bất động sản không hợp lệ.'}), 400

    if sort not in {'default', 'price_asc', 'price_desc'}:
        return jsonify({'success': False, 'error': 'Kiểu sắp xếp không hợp lệ.'}), 400

    try:
        price_min = float(request.args.get('price_min', 0))
        price_max = float(request.args.get('price_max', 9999))
        page      = max(1, int(request.args.get('page', 1)))
    except (TypeError, ValueError):
        return jsonify({'success': False, 'error': 'Tham số tìm kiếm không hợp lệ.'}), 400

    if price_min < 0 or price_max < 0 or price_min > price_max:
        return jsonify({'success': False, 'error': 'Khoảng giá không hợp lệ.'}), 400

    results = []
    for dtype in (['chung_cu', 'nha_dat'] if ptype == 'all' else [ptype]):
        df = dataframes.get(dtype, pd.DataFrame())
        if df.empty:
            continue
        mask = pd.Series([True] * len(df), index=df.index)
        if q:
            text_mask = df['district'].astype(str).str.lower().str.contains(q, na=False)
            if 'title' in df.columns:
                text_mask |= df['title'].astype(str).str.lower().str.contains(q, na=False)
            mask &= text_mask
        if district:
            mask &= df['district'].astype(str).str.lower() == district.lower()
        if direction:
            mask &= df['direction'].astype(str).str.lower() == direction.lower()
        mask &= (df['price_billion'] >= price_min) & (df['price_billion'] <= price_max)
        sub = df[mask].copy()
        sub['_type'] = dtype
        sub['_id'] = sub.index
        results.append(sub)

    if results:
        combined = pd.concat(results, ignore_index=True)
    else:
        combined = pd.DataFrame()

    if not combined.empty and sort in {'price_asc', 'price_desc'}:
        combined = combined.sort_values('price_billion', ascending=(sort == 'price_asc'))

    total = len(combined)
    start = (page - 1) * per_page
    page_df = combined.iloc[start:start + per_page]

    def row_to_dict(row):
        import json as _json
        imgs = []
        if 'image_urls' in row.index and pd.notna(row.get('image_urls')):
            raw = str(row['image_urls']).strip()
            try:
                parsed = _json.loads(raw)
                imgs = [u for u in parsed if isinstance(u, str) and u.startswith('http')]
            except Exception:
                # fallback: try comma split
                imgs = [u.strip().strip('"') for u in raw.split(',') if 'http' in u]
        thumb = imgs[0] if imgs else ''
        return {
            'id': int(row.get('_id', 0)),
            'type': row.get('_type', ''),
            'title': str(row.get('title', f"BĐS tại {row.get('district','')}")),
            'district': str(row.get('district', '')),
            'price': round(float(row.get('price_billion', 0)), 2),
            'area': float(row.get('area_m2', 0)),
            'bedrooms': int(row.get('bedrooms_num', 0)) if pd.notna(row.get('bedrooms_num')) else 0,
            'image': thumb,
            'legal': str(row.get('legal_std', '')),
            'furniture': str(row.get('furniture_std', '')),
        }

    items = [row_to_dict(row) for _, row in page_df.iterrows()]
    return jsonify({
        'total': total,
        'page': page,
        'pages': math.ceil(total / per_page) if total else 0,
        'items': items
    })

@app.route('/predict', methods=['POST'])
def predict():
    """API nhận data từ Frontend và trả về giá dự đoán"""
    try:
        # Lấy dữ liệu user nhập từ giao diện (JSON)
        data = request.get_json(silent=True) or {}
        
        # Nhận diện loại bất động sản cần dự đoán (mặc định chung_cu nếu Frontend không gửi)
        property_type = data.get('property_type', 'chung_cu')
        
        # --- VALIDATION ĐẦU VÀO (cho phép bỏ trống, gán giá trị mặc định) ---
        area = float(data.get('area', '') or 60)  # Mặc định 60m² nếu bỏ trống
        if area < 0 or area > 50000:
            return jsonify({'success': False, 'error': f'Diện tích {area}m2 không hợp lệ!'}), 400
            
        bedrooms = int(data.get('bedrooms', '') or 2)  # Mặc định 2 phòng ngủ
        if bedrooms < 0 or bedrooms > 100:
            return jsonify({'success': False, 'error': f'Số phòng ngủ {bedrooms} không hợp lệ!'}), 400

        if property_type == 'nha_dat':
            floors = int(data.get('floors', '') or 1)
            if floors < 0 or floors > 200:
                return jsonify({'success': False, 'error': f'Số tầng {floors} không hợp lệ!'}), 400
            
            frontage = float(data.get('frontage', '') or 0)
            if frontage < 0 or frontage > 1000:
                return jsonify({'success': False, 'error': f'Mặt tiền {frontage}m không hợp lệ!'}), 400
                
            road_width = float(data.get('road_width', '') or 0)
            if road_width < 0 or road_width > 1000:
                return jsonify({'success': False, 'error': f'Đường vào {road_width}m không hợp lệ!'}), 400
        # ------------------------------------------------
        
        # 1. Chỉ cần tạo thẳng 1 DataFrame 1 dòng từ input của Frontend
        # Chú ý: Các key của dictionary phải khớp Y HỆT tên cột lúc Thái train
        input_dict = {
            'area_m2': area,
            'bedrooms_num': bedrooms,
            'district': data.get('district', 'Khác'),
            'direction': data.get('direction', 'Không rõ'),
            'furniture_std': data.get('furniture', 'Không rõ'),
            'legal_std': data.get('legal', 'Không rõ'),
            'floors_num': 0,
            'frontage_m': 0.0,
            'road_width_m': 0.0,
            'loai_bds': property_type
        }
        
        if property_type == 'nha_dat':
            input_dict['floors_num'] = floors
            input_dict['frontage_m'] = frontage
            input_dict['road_width_m'] = road_width
        elif property_type != 'chung_cu':
            return jsonify({'success': False, 'error': 'Loại bất động sản không hợp lệ (chỉ nhận chung_cu hoặc nha_dat)'}), 400

        # Chuyển Dictionary thành DataFrame (chỉ có 1 dòng dữ liệu)
        input_data = pd.DataFrame([input_dict])
        
        # Đảm bảo các cột categorical là kiểu chuỗi giống lúc train
        for c in ['district', 'direction', 'furniture_std', 'legal_std', 'loai_bds']:
            input_data[c] = input_data[c].astype(str)
        
        # 2. Đưa thẳng DataFrame vào model dự đoán
        model = models.get('best_model_pipeline')
        prediction_billion = 0
        if model:
            prediction_billion = float(model.predict(input_data)[0])
            prediction_vnd = float(prediction_billion * 1_000_000_000)
        else:
            prediction_billion = float(data.get('area', 0)) * 0.1
            prediction_vnd = float(prediction_billion * 1_000_000_000)

        # Tạo DataFrame riêng cho Transformer có thêm tinh_thanh
        input_data_transformer = input_data.copy()
        input_data_transformer['tinh_thanh'] = str(data.get('province', 'Khác'))

        transformer_prediction_billion = predict_with_transformer(input_data_transformer)
        transformer_result = None
        if transformer_prediction_billion is not None:
            transformer_result = {
                'price_billion': round(transformer_prediction_billion, 2),
                'predicted_price_vnd': float(transformer_prediction_billion * 1_000_000_000),
                'difference_billion': round(transformer_prediction_billion - prediction_billion, 2),
            }
            
        # 3. Tính toán Feature Contributions (XAI) & Confidence Interval
        mae = model_metadata.get('metrics', {}).get('MAE', 0.85)
        fi = model_metadata.get('feature_importance', {})
        total_importance = model_metadata.get('total_importance', 1.0)  # Tổng importance từ training
        
        # Mapping tên tiếng Việt chuyên nghiệp
        human_names = {
            'area_m2': 'Diện tích',
            'bedrooms_num': 'Phòng ngủ',
            'district': 'Vị trí',
            'legal_std': 'Pháp lý',
            'furniture_std': 'Nội thất',
            'direction': 'Hướng nhà',
            'floors_num': 'Số tầng',
            'frontage_m': 'Mặt tiền',
            'road_width_m': 'Đường rộng'
        }

        # Trích xuất đóng góp thực tế từ mô hình (Top 4 yếu tố)
        # Công thức: impact = prediction × (feature_importance / total_importance)
        # → Mỗi feature đóng góp tỷ lệ % thực sự vào giá dự đoán
        contributions = []
        # Các cột số
        for feat in ['area_m2', 'bedrooms_num', 'floors_num', 'frontage_m', 'road_width_m']:
            val = float(data.get(feat, 0)) if feat in data else 0
            if val > 0:
                imp = float(fi.get(feat, 0))
                if imp > 0:
                    ratio = imp / total_importance if total_importance > 0 else 0
                    contributions.append({
                        "feature": human_names.get(feat, feat),
                        "impact": round(prediction_billion * ratio, 2),
                        "unit": "tỷ"
                    })
        
        # Các cột phân loại (khớp với lựa chọn của người dùng)
        for feat_group, user_val in [
            ('district', data.get('district')),
            ('legal_std', input_dict['legal_std']),
            ('furniture_std', input_dict['furniture_std']),
            ('direction', input_dict['direction'])
        ]:
            if not user_val: continue
            key = f"{feat_group}_{user_val}"
            imp = float(fi.get(key, 0))
            if imp > 0:
                ratio = imp / total_importance if total_importance > 0 else 0
                contributions.append({
                    "feature": f"{human_names.get(feat_group)} ({user_val})",
                    "impact": round(prediction_billion * ratio, 2),
                    "unit": "tỷ"
                })
        
        # Sắp xếp lấy 4 yếu tố ảnh hưởng mạnh nhất
        contributions = sorted(contributions, key=lambda x: x['impact'], reverse=True)[:4]
        if not contributions: # Fallback nếu không có fi
            contributions = [{"feature": "Diện tích", "impact": round(prediction_billion*0.4, 2), "unit": "tỷ"}]

        # 4. Tìm các căn nhà tương tự và Tính trung bình khu vực
        similar_properties = []
        district_avg_m2 = 0
        df_target = dataframes.get(property_type)
        if df_target is not None and not df_target.empty and prediction_billion > 0:
            user_dist = data.get('district', '')
            
            # Tính trung bình khu vực
            same_dist_all = df_target[df_target['district'] == user_dist]
            if not same_dist_all.empty:
                valid_area = same_dist_all[same_dist_all['area_m2'] > 0]
                if not valid_area.empty:
                    dist_prices = valid_area['price_billion'] * 1000 / valid_area['area_m2']
                    district_avg_m2 = dist_prices.mean()

            # Lấy 10-15 căn tương tự
            min_p = prediction_billion * 0.8
            max_p = prediction_billion * 1.2
            filtered = df_target[(df_target['price_billion'] >= min_p) & (df_target['price_billion'] <= max_p)]
            if not filtered.empty:
                same_dist = filtered[filtered['district'] == user_dist]
                if len(same_dist) >= 12:
                    sampled = same_dist.sample(12)
                else:
                    others = filtered[filtered['district'] != user_dist]
                    needed = min(12 - len(same_dist), len(others))
                    if needed > 0:
                        sampled = pd.concat([same_dist, others.sample(needed)])
                    else:
                        sampled = same_dist
                
                import ast
                for idx, row in sampled.iterrows():
                    img_url = ""
                    raw_imgs = str(row.get('image_urls', '[]'))
                    if raw_imgs.startswith('['):
                        try:
                            imgs = ast.literal_eval(raw_imgs)
                            if imgs and isinstance(imgs, list): 
                                nice_imgs = [img for img in imgs if '1275x717' in img or 'crop' not in img]
                                img_url = nice_imgs[0] if nice_imgs else imgs[0]
                        except: pass

                    prop = {
                        "id": int(idx),
                        "property_type": property_type,
                        "price_billion": round(float(row['price_billion']), 2),
                        "area_m2": float(row['area_m2']),
                        "district": str(row['district']),
                        "title": str(row.get('title', f"BĐS tại {row['district']}"))[:80],
                        "image": img_url
                    }
                    similar_properties.append(prop)

        # Tính đơn giá/m2
        price_per_m2 = (prediction_billion * 1000) / float(area) if float(area) > 0 else 0

        # 5. Save to history
        user = session.get('user')
        if user:
            from datetime import datetime
            PREDICTION_HISTORY.setdefault(user['email'], []).insert(0, {
                'timestamp': datetime.now().strftime('%d/%m/%Y %H:%M'),
                'property_type': property_type,
                'district': data.get('district', 'Khác'),
                'area': float(area),
                'price_billion': round(float(prediction_billion), 2),
            })

        return jsonify({
            'success': True,
            'predicted_price_vnd': prediction_vnd,
            'price_billion': round(prediction_billion, 2),
            'price_low': round(max(0, prediction_billion - mae), 2),
            'price_high': round(prediction_billion + mae, 2),
            'price_per_m2': round(price_per_m2, 1),
            'district_avg_m2': round(district_avg_m2, 1),
            'mae': round(mae, 2),
            'transformer_prediction': transformer_result,
            'contributions': contributions,
            'similar_properties': similar_properties,
            'message': 'Dự đoán thành công'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

# ── Profile, History, Saved ─────────────────────────────────────────

@app.route('/profile')
def profile():
    user = session.get('user')
    if not user:
        return redirect('/')
    stored = USERS.get(user['email'], {})
    profile_data = {
        **user,
        'phone': stored.get('phone', ''),
        'gender': stored.get('gender', ''),
        'dob': stored.get('dob', ''),
        'avatar': stored.get('avatar', ''),
    }
    return render_template('profile.html', user=user, profile=profile_data)

@app.route('/api/upload-avatar', methods=['POST'])
def upload_avatar():
    user = session.get('user')
    if not user:
        return jsonify({'success': False, 'error': 'Chưa đăng nhập.'}), 401
    file = request.files.get('avatar')
    if not file or not file.filename:
        return jsonify({'success': False, 'error': 'Chưa chọn file.'}), 400
    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in ('jpg', 'jpeg', 'png', 'webp'):
        return jsonify({'success': False, 'error': 'Chỉ hỗ trợ JPG, PNG, WEBP.'}), 400
    import hashlib
    filename = hashlib.md5(user['email'].encode()).hexdigest() + '.' + ext
    save_path = os.path.join(app.static_folder, 'uploads', filename)
    file.save(save_path)
    avatar_url = f'/static/uploads/{filename}'
    USERS[user['email']]['avatar'] = avatar_url
    session['user'] = {**user, 'avatar': avatar_url}
    return jsonify({'success': True, 'avatar': avatar_url})

@app.route('/api/update-profile', methods=['POST'])
def update_profile():
    user = session.get('user')
    if not user:
        return jsonify({'success': False, 'error': 'Chưa đăng nhập.'}), 401
    data = request.get_json(silent=True) or {}
    email = user['email']
    new_name = data.get('name', '').strip()
    old_password = data.get('old_password', '')
    new_password = data.get('new_password', '')

    if new_name and new_name != user['name']:
        USERS[email]['name'] = new_name
        session['user'] = {**user, 'name': new_name}

    for field in ('phone', 'gender', 'dob'):
        val = data.get(field)
        if val is not None:
            USERS[email][field] = val.strip() if isinstance(val, str) else val

    if new_password:
        if len(new_password) < 6:
            return jsonify({'success': False, 'error': 'Mật khẩu mới tối thiểu 6 ký tự.'}), 400
        stored = USERS.get(email)
        if stored and stored.get('password_hash') and not check_password_hash(stored['password_hash'], old_password):
            return jsonify({'success': False, 'error': 'Mật khẩu cũ không đúng.'}), 400
        USERS[email]['password_hash'] = generate_password_hash(new_password, method='pbkdf2:sha256')

    return jsonify({'success': True, 'name': session['user']['name']})

@app.route('/history')
def history():
    user = session.get('user')
    if not user:
        return redirect('/')
    items = PREDICTION_HISTORY.get(user['email'], [])
    return render_template('history.html', user=user, items=items)

@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    user = session.get('user')
    if not user:
        return jsonify({'success': False}), 401
    PREDICTION_HISTORY.pop(user['email'], None)
    return jsonify({'success': True})

@app.route('/saved')
def saved():
    user = session.get('user')
    if not user:
        return redirect('/')
    saved_keys = SAVED_PROPERTIES.get(user['email'], [])
    props = []
    for key in saved_keys:
        ptype, pid = key.split('/')
        df = dataframes.get(ptype)
        if df is not None and int(pid) in df.index:
            row = df.loc[int(pid)]
            import json as _json
            imgs = []
            try:
                imgs = _json.loads(str(row.get('image_urls', '[]')))
                imgs = [u for u in imgs if isinstance(u, str) and u.startswith('http')]
            except Exception:
                pass
            props.append({
                'id': int(pid),
                'type': ptype,
                'title': str(row.get('title', f"BĐS tại {row.get('district', '')}"))[:80],
                'district': str(row.get('district', '')),
                'price': round(float(row.get('price_billion', 0)), 2),
                'area': float(row.get('area_m2', 0)),
                'bedrooms': int(row.get('bedrooms_num', 0)) if pd.notna(row.get('bedrooms_num')) else 0,
                'image': imgs[0] if imgs else '',
            })
    return render_template('saved.html', user=user, properties=props)

@app.route('/api/save-property', methods=['POST'])
def save_property():
    user = session.get('user')
    if not user:
        return jsonify({'success': False, 'error': 'Chưa đăng nhập.'}), 401
    data = request.get_json(silent=True) or {}
    ptype = data.get('type', '')
    pid = data.get('id', '')
    if ptype not in ('chung_cu', 'nha_dat'):
        return jsonify({'success': False, 'error': 'Loại BĐS không hợp lệ.'}), 400
    key = f"{ptype}/{pid}"
    email = user['email']
    saved = SAVED_PROPERTIES.setdefault(email, [])
    if key not in saved:
        saved.insert(0, key)
    return jsonify({'success': True, 'saved': True})

@app.route('/api/unsave-property', methods=['POST'])
def unsave_property():
    user = session.get('user')
    if not user:
        return jsonify({'success': False, 'error': 'Chưa đăng nhập.'}), 401
    data = request.get_json(silent=True) or {}
    key = f"{data.get('type', '')}/{data.get('id', '')}"
    email = user['email']
    saved = SAVED_PROPERTIES.get(email, [])
    if key in saved:
        saved.remove(key)
    return jsonify({'success': True, 'saved': False})

@app.route('/api/is-saved')
def is_saved():
    user = session.get('user')
    if not user:
        return jsonify({'saved': False})
    key = f"{request.args.get('type', '')}/{request.args.get('id', '')}"
    saved = SAVED_PROPERTIES.get(user['email'], [])
    return jsonify({'saved': key in saved})

@app.route('/property/<property_type>/<int:prop_id>')
def property_detail(property_type, prop_id):
    """Trang chi tiết của một bất động sản"""
    try:
        df = dataframes.get(property_type)
        if df is None or prop_id not in df.index:
            return "Không tìm thấy bất động sản!", 404
            
        prop_data = df.loc[prop_id].to_dict()
        prop_data['id'] = prop_id
        prop_data['type'] = property_type
        
        # Parse image_urls
        fallback_image = url_for('static', filename='hero-bg.png')
        try:
            import json
            images = json.loads(prop_data.get('image_urls', '[]'))
            images = [img for img in images if isinstance(img, str) and img.startswith(('http://', 'https://'))]
            if isinstance(images, list) and len(images) > 0:
                prop_data['image'] = images[0]
                prop_data['all_images'] = images
            else:
                prop_data['image'] = fallback_image
                prop_data['all_images'] = [prop_data['image']]
        except:
            prop_data['image'] = fallback_image
            prop_data['all_images'] = [prop_data['image']]
            
        # Add title and description safe fallback
        if 'title' not in prop_data or pd.isna(prop_data['title']):
            prop_data['title'] = f"Bất động sản tại {prop_data.get('district', 'Hà Nội')}"
            
        if 'description' not in prop_data or pd.isna(prop_data['description']):
            prop_data['description'] = "Đang cập nhật mô tả cho bất động sản này."
        
        # Generate fake phone per property id
        import random as _rand
        _rand.seed(prop_id * 37 + 7)
        phone_full = f"0{_rand.choice([9,8,7])}{_rand.randint(10,99)} {_rand.randint(100,999)} {_rand.randint(100,999)}"
        phone_masked = phone_full[:-3] + '***'
        prop_data['phone_full'] = phone_full
        prop_data['phone_masked'] = phone_masked

        # Filter to only high-res images (1275x717)
        all_imgs = prop_data.get('all_images', [])
        hires = [u for u in all_imgs if '1275x717' in u]
        prop_data['all_images'] = hires if hires else all_imgs[:8]

        logged_in = 'user' in session
        return render_template('property_detail.html', prop=prop_data, logged_in=logged_in, user=session.get('user'))
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    # Chạy server ở port 5000
    debug = os.environ.get('FLASK_DEBUG', '').lower() in {'1', 'true', 'yes'}
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=debug, host=host, port=port, use_reloader=False)
