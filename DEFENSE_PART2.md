# TÀI LIỆU BẢO VỆ — PHẦN 2: DEMO, BÁO CÁO, Q&A
# HỆ THỐNG DỰ ĐOÁN GIÁ BẤT ĐỘNG SẢN (ProphetEstate)

---

## 4. KỊCH BẢN DEMO

### 4.1 Chuẩn bị môi trường

```bash
# Bước 1: Cài dependencies
pip install -r requirements.txt

# Bước 2 (tùy chọn): Chạy lại training nếu cần
python run_training.py

# Bước 3: Khởi động server
cd app
python app.py
# → Mở trình duyệt: http://127.0.0.1:5000
```

### 4.2 Kịch bản demo 5 phút

**Bước 1 — Giới thiệu trang chủ (30s)**
- Mở trang chủ, giới thiệu giao diện ProphetEstate
- Chỉ ra: "11.8K+ BĐS phân tích", "43 Quận/Huyện", "AI XGBoost"

**Bước 2 — Demo dự đoán Chung Cư (90s)**
- Chọn tab "Chung Cư"
- Nhập: Diện tích=65m², Phòng ngủ=2, Quận=Cầu Giấy, Hướng=Đông-Nam, Nội thất=Đầy đủ, Pháp lý=Sổ đỏ
- Bấm "Dự Đoán Giá Trị" → Modal hiện kết quả
- Chỉ ra: Giá dự đoán, khoảng ước tính (±MAE), đơn giá/m²
- **Highlight XAI**: "Diện tích đóng góp +1.38 tỷ, Vị trí Cầu Giấy đóng góp +0.8 tỷ"
- Cuộn xuống: BĐS tương tự (có ảnh, giá, link chi tiết)

**Bước 3 — Demo dự đoán Nhà Đất (60s)**
- Chuyển tab "Nhà Đất" → form hiện thêm: Số tầng, Mặt tiền, Đường vào
- Nhập: 80m², 4PN, Thanh Xuân, 3 tầng, Mặt tiền 5m, Đường vào 6m
- So sánh kết quả: Badge "Cao hơn/Thấp hơn khu vực X%"

**Bước 4 — Demo Analytics (60s)**
- Vào trang Phân tích → KPI cards, biểu đồ phân bố theo quận
- Giá TB theo quận (horizontal bar chart)
- Tỷ lệ pháp lý, nội thất (doughnut chart)
- Bảng xếp hạng quận đắt nhất

**Bước 5 — Demo Search + Auth (60s)**
- Tìm kiếm BĐS: filter theo loại, quận, khoảng giá, sắp xếp
- Đăng ký tài khoản → lịch sử dự đoán, BĐS đã lưu
- Click vào BĐS → trang chi tiết (gallery ảnh, thông tin, liên hệ)

### 4.3 Test Cases hay nhất

| # | Test Case | Mục đích show |
|---|---|---|
| 1 | CC 30m² Ba Đình vs CC 200m² Gia Lâm | Diện tích + Vị trí ảnh hưởng mạnh |
| 2 | Nhà đất mặt tiền 8m vs 3m (cùng quận) | Mặt tiền = feature quan trọng nhà đất |
| 3 | Cùng căn hộ, thay đổi Pháp lý: Sổ đỏ vs HĐMB | Pháp lý ảnh hưởng giá |
| 4 | Nhập quận "Khác" (unseen category) | handle_unknown='ignore' hoạt động |
| 5 | Diện tích cực lớn (5000m²) hoặc âm | Validation đầu vào chặn lỗi |

---

## 5. DÀN Ý BÁO CÁO IN QUYỂN

### MỤC LỤC ĐỀ XUẤT

```
CHƯƠNG 1: MỞ ĐẦU
  1.1 Lý do chọn đề tài
      → Thị trường BĐS VN phức tạp, thiếu công cụ định giá khách quan
  1.2 Mục tiêu đề tài
      → Xây dựng hệ thống AI dự đoán giá BĐS dựa trên dữ liệu thực
  1.3 Phạm vi và giới hạn
      → Chung cư + nhà đất tại HN, HCM, ĐN; dữ liệu từ batdongsan.com.vn
  1.4 Phương pháp nghiên cứu
      → Thu thập dữ liệu (Web Scraping) → EDA → ML → Triển khai Web

CHƯƠNG 2: CƠ SỞ LÝ THUYẾT
  2.1 Tổng quan Machine Learning
      → Supervised Learning, Regression, Train/Test Split
  2.2 Các thuật toán sử dụng
      → Linear Regression, Decision Tree, Random Forest, XGBoost
  2.3 Tiền xử lý dữ liệu
      → StandardScaler, OneHotEncoder, Pipeline, ColumnTransformer
  2.4 Đánh giá mô hình
      → RMSE, MAE, R², MAPE, Cross-Validation, Overfitting Analysis
  2.5 Explainable AI (XAI)
      → Feature Importance, SHAP values (lý thuyết)

CHƯƠNG 3: PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG
  3.1 Phân tích yêu cầu
      → Use Case: Dự đoán giá, Tìm kiếm BĐS, Phân tích thị trường
  3.2 Thiết kế kiến trúc
      → Sơ đồ 3 tầng: Data Layer → ML Layer → Web Layer
  3.3 Thiết kế cơ sở dữ liệu
      → Schema CSV: price_billion, area_m2, district, direction...
  3.4 Thiết kế giao diện
      → Wireframe trang chủ, form dự đoán, modal kết quả

CHƯƠNG 4: TRIỂN KHAI
  4.1 Thu thập dữ liệu (Web Crawling)
      → crawler.py, curl_cffi, BeautifulSoup, chống anti-bot
  4.2 Tiền xử lý và EDA
      → Lọc outlier, xử lý missing values, phân tích phân phối
  4.3 Huấn luyện mô hình
      → Pipeline, 4 thuật toán, hyperparameter tuning
  4.4 Xây dựng ứng dụng Web
      → Flask, Jinja2, Chart.js, REST API

CHƯƠNG 5: ĐÁNH GIÁ KẾT QUẢ
  5.1 Kết quả huấn luyện
      → Bảng so sánh R², RMSE, MAE, MAPE của 4 model
  5.2 Phân tích Overfitting
      → R² Train vs Test, Overfit Gap
  5.3 Cross-Validation
      → CV R² mean ± std cho từng model
  5.4 Feature Importance
      → Top features: area_m2, district, floors_num
  5.5 Đánh giá hệ thống Web
      → Thời gian phản hồi, UX, responsive design

CHƯƠNG 6: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN
  6.1 Kết luận
  6.2 Hạn chế
      → In-memory storage, chưa có DB, dữ liệu tĩnh
  6.3 Hướng phát triển
      → Database thực, cập nhật dữ liệu tự động, Deep Learning
```

---

## 6. BỘ CÂU HỎI PHÒNG THỦ Q&A (15 CÂU)

### CÂU 1: Hệ thống này giải quyết bài toán gì trong Machine Learning?
**Trả lời**: Bài toán **Supervised Regression** — hồi quy có giám sát. Input là vector đặc trưng (diện tích, quận, hướng...), output là giá trị liên tục (giá nhà tính bằng tỷ VNĐ). Khác với Classification (phân loại) vì output không phải nhãn rời rạc mà là số thực.

### CÂU 2: Tại sao chọn XGBoost mà không dùng Deep Learning?
**Trả lời**: Với dữ liệu dạng bảng (tabular data) có ~12.000 mẫu và ~10 features, XGBoost thường **vượt trội hơn Deep Learning**. Lý do: (1) Deep Learning cần lượng data lớn hơn nhiều (>100K), (2) XGBoost xử lý tốt mixed features (số + categorical), (3) XGBoost có built-in regularization chống overfit, (4) Thời gian train nhanh hơn nhiều. Nhiều nghiên cứu (Grinsztajn et al., 2022) đã chứng minh tree-based models vẫn là SOTA cho tabular data.

### CÂU 3: StandardScaler hoạt động thế nào? Tại sao cần chuẩn hóa?
**Trả lời**: StandardScaler biến đổi z = (x - μ) / σ, đưa mỗi feature về mean=0, std=1. Cần chuẩn hóa vì: nếu area_m2 có range 10-1000 nhưng bedrooms có range 0-10, thì các thuật toán dựa trên khoảng cách (Linear Regression) sẽ bị bias bởi feature có scale lớn. **Lưu ý**: XGBoost thực ra không cần scale (vì nó dựa trên thứ tự, không phải khoảng cách), nhưng ta dùng chung Pipeline cho cả 4 model nên vẫn giữ.

### CÂU 4: OneHotEncoder là gì? handle_unknown='ignore' nghĩa là gì?
**Trả lời**: OneHotEncoder chuyển categorical thành binary vector. VD: district có 43 giá trị → tạo 43 cột, mỗi cột = 0 hoặc 1. `handle_unknown='ignore'` nghĩa là: nếu lúc predict user nhập quận chưa có trong training data → tất cả 43 cột = 0 (không crash). Nếu không có tham số này, hệ thống sẽ raise error khi gặp category mới.

### CÂU 5: R² Score nghĩa là gì? R²=0.85 có tốt không?
**Trả lời**: R² = 1 - (SS_res / SS_tot). Nó đo tỷ lệ variance của biến mục tiêu được giải thích bởi model. R²=0.85 nghĩa là model giải thích được 85% sự biến thiên của giá nhà, 15% còn lại là noise/factors chưa capture. Với bài toán giá BĐS (có nhiều yếu tố chủ quan như tâm lý, đầu cơ), R²=0.85 là **tốt**.

### CÂU 6: Cross-Validation 5-Fold là gì? Tại sao cần?
**Trả lời**: Chia data thành 5 phần bằng nhau, lần lượt dùng 1 phần làm test, 4 phần làm train → train 5 lần → lấy trung bình R². Cần vì: (1) Đánh giá model trên TOÀN BỘ dữ liệu (không chỉ 1 lần split), (2) Đo độ ổn định: nếu CV std nhỏ → model ổn định, nếu std lớn → model nhạy cảm với data split.

### CÂU 7: Overfitting là gì? Làm sao phát hiện trong code?
**Trả lời**: Overfitting = model "học thuộc" training data nhưng dự đoán kém trên data mới. Phát hiện bằng cách so sánh R²_train vs R²_test. Trong code: `overfit_gap = r2_train - r2_test`. Nếu gap > 0.15 → cảnh báo overfit. XGBoost chống overfit bằng: `max_depth=7` (giới hạn độ sâu cây), `subsample=0.8` (mỗi cây chỉ dùng 80% data), `learning_rate=0.05` (bước học nhỏ), `colsample_bytree=0.8` (mỗi cây chỉ dùng 80% features).

### CÂU 8: Tại sao dùng Pipeline thay vì tự gọi StandardScaler rồi model.fit()?
**Trả lời**: Pipeline giải quyết 2 vấn đề nghiêm trọng: (1) **Data leakage**: Nếu fit StandardScaler trên toàn bộ data rồi mới split → test set đã "thấy" thông tin từ train set (mean, std). Pipeline đảm bảo scaler chỉ fit trên train. (2) **Consistency**: Khi predict dữ liệu mới, Pipeline tự động apply đúng transformation đã learn lúc train. Nếu tự code, rất dễ quên bước scale hoặc scale sai.

### CÂU 9: Giải thích Feature Importance trong XGBoost?
**Trả lời**: XGBoost tính Feature Importance dựa trên **gain** — tổng improvement (giảm loss) mà mỗi feature mang lại khi được dùng để split node trong tất cả 300 cây. Feature có gain cao = ảnh hưởng mạnh đến dự đoán. Trong code, ta trích xuất `base_model.feature_importances_` rồi chia cho `total_importance` để ra tỷ lệ %. Sau đó nhân với `predicted_price` để ra "impact" bằng tiền.

### CÂU 10: Dữ liệu crawl được có vấn đề gì? Làm sao xử lý?
**Trả lời**: Vấn đề chính: (1) **Missing values**: nhiều tin thiếu hướng, pháp lý → điền "Không rõ", (2) **Outliers**: giá 0.01 tỷ hoặc 500 tỷ → lọc bỏ (chỉ giữ 1-200 tỷ), (3) **Duplicate**: cùng BĐS đăng nhiều lần → dedup bằng listing_id, (4) **Inconsistent format**: "2 tỷ 5", "2.500.000.000" → chuẩn hóa về float (billion), (5) **Bias**: batdongsan.com.vn thiên về HN/HCM, ít tỉnh lẻ → model predict tốt hơn ở HN/HCM.

### CÂU 11: curl_cffi khác gì requests thường? Tại sao không dùng Selenium?
**Trả lời**: `requests` thường bị batdongsan.com.vn phát hiện và chặn (403) vì TLS fingerprint của Python khác Chrome. `curl_cffi` giả lập TLS fingerprint của Chrome (`impersonate="chrome"`) → server nghĩ là trình duyệt thật. Không dùng Selenium vì: (1) Chậm hơn 10-50x (phải render full DOM), (2) Tốn RAM (mỗi tab ~100MB), (3) batdongsan.com.vn là server-rendered HTML → không cần JavaScript execution.

### CÂU 12: Hệ thống lưu trữ user data bằng gì? Điểm yếu?
**Trả lời**: Hiện tại lưu **in-memory** bằng Python dict (`USERS = {}`, `PREDICTION_HISTORY = {}`, `SAVED_PROPERTIES = {}`). **Điểm yếu nghiêm trọng**: (1) Restart server = mất hết data, (2) Không scale được (nhiều worker = nhiều bản copy khác nhau), (3) RAM tăng dần khi nhiều user. **Giải pháp**: Chuyển sang SQLite/PostgreSQL (đã có schema_design.md trong docs).

### CÂU 13: Nếu scale lên 1000 users đồng thời, chỗ nào sập trước?
**Trả lời**: (1) **API /predict**: Mỗi request phải `pd.DataFrame()` + `model.predict()` → CPU-bound, Flask single-threaded sẽ bottleneck. Giải pháp: dùng Gunicorn multi-worker + cache kết quả. (2) **CSV loading**: 2 file CSV ~30MB load vào RAM → 1000 users truy cập search/analytics sẽ tốn RAM. Giải pháp: chuyển sang database có indexing. (3) **In-memory dict**: 1000 users × lịch sử → RAM explosion. (4) **Không có rate limiting**: có thể bị DDoS.

### CÂU 14: Đoạn code nào em tự viết, đoạn nào AI viết? Làm sao chứng minh?
**Trả lời** (gợi ý): "Toàn bộ pipeline ML (chọn thuật toán, tuning hyperparameters, đánh giá overfitting) được xây dựng dựa trên kiến thức môn học. Em hiểu rõ từng bước: tại sao chọn XGBoost, tại sao max_depth=7 chứ không phải 20, tại sao cần cross-validation. Phần UI/UX có sự hỗ trợ của AI để tăng tốc, nhưng em nắm vững logic backend: cách Pipeline hoạt động, cách tính XAI, cách tìm BĐS tương tự."

### CÂU 15: Nếu có thêm 1 tuần, em sẽ cải thiện gì?
**Trả lời**: (1) **Database**: Thay in-memory bằng PostgreSQL/SQLite (đã thiết kế schema), (2) **Feature Engineering**: Thêm feature giá/m² theo quận (market benchmark), khoảng cách đến trung tâm, (3) **Hyperparameter Tuning**: Dùng GridSearchCV/Optuna tìm tham số tối ưu thay vì manual, (4) **SHAP Values**: Thay thế feature importance đơn giản bằng SHAP để giải thích chính xác hơn per-prediction, (5) **Auto-update data**: Cron job chạy crawler hàng tuần để cập nhật dữ liệu mới, (6) **A/B Testing**: So sánh XGBoost vs LightGBM vs CatBoost.
