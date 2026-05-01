# 🏠 DỰ ÁN: Hệ Thống Web Dự Đoán Giá Bất Động Sản Việt Nam

**Môn học:** Phát triển Hệ thống Thông minh — Học viện Bưu chính Viễn thông (PTIT)  
**Đề số:** 2 — Dự đoán Giá (Regression)  
**Deadline:** Chiều thứ 5, ngày 07/05/2026

---

## 1. TỔNG QUAN ĐỀ TÀI

### Mục tiêu
Xây dựng một hệ thống web cho phép người dùng nhập thông số căn nhà (diện tích, số phòng ngủ, quận/huyện, hướng nhà...) và nhận về **giá dự đoán bằng VNĐ** từ mô hình Machine Learning.

### Điểm nổi bật (so với các nhóm khác)
- **Tự crawl dữ liệu thực** từ batdongsan.com.vn (không dùng dataset có sẵn)
- So sánh **4 thuật toán ML** và chọn ra mô hình tốt nhất
- Giao diện web **hiển thị chi tiết từng căn nhà** như trang BĐS thật

### Quy trình tổng thể (End-to-End Pipeline)

```
[1. CRAWL DATA]  →  [2. CLEAN DATA]  →  [3. TRAIN AI]  →  [4. WEB APP]  →  [5. BÁO CÁO]
   (Tài)              (Đức)              (Thái)           (Tài+Đông)       (Mai Anh)
```

---

## 2. NHÓM THỰC HIỆN

| Thành viên | Vai trò | Phụ trách chính |
|---|---|---|
| **Tài** | Leader / Tích hợp | Crawl data, Flask API, tích hợp hệ thống, review |
| **Thái** | ML Engineer | Train models, tuning, export model .pkl |
| **Đức** | Data Engineer | EDA, tiền xử lý, vẽ biểu đồ phân tích |
| **Đông** | Frontend Dev | HTML/CSS/JS giao diện web |
| **Mai Anh** | Tài liệu & QA | Viết báo cáo, nghiên cứu lý thuyết, testing |

---

## 3. GIẢI THÍCH CHI TIẾT TỪNG BƯỚC

### BƯỚC 1: CRAWL DỮ LIỆU (Tài phụ trách)

**Mục đích:** Thu thập dữ liệu thực về giá BĐS tại Việt Nam từ batdongsan.com.vn.

**Cách hoạt động:**
- Dùng thư viện `curl_cffi` (Python) giả lập trình duyệt Chrome để truy cập trang web
- Tự động lật qua từng trang danh sách, vào từng tin đăng, rút thông tin ra
- Mỗi căn nhà cào xong → ghi ngay 1 dòng vào file CSV

**Dữ liệu cào được (25 cột):**

| Nhóm | Các trường | Dùng cho |
|---|---|---|
| **Cốt lõi** | Giá, Diện tích, Số PN, Số WC | Train AI |
| **Vị trí** | Địa chỉ, Quận/Huyện | Train AI |
| **Kỹ thuật** | Hướng nhà, Hướng BC, Nội thất, Pháp lý, Số tầng | Train AI |
| **Hiển thị** | Tiêu đề, Mô tả, Ảnh, Tên dự án, Loại BĐS, Mã tin | Hiện trên Web |

**Mục tiêu:** 2000 - 3000 bản ghi  
**File đầu ra:** `data/raw/raw_data.csv`

---

### BƯỚC 2: LÀM SẠCH & PHÂN TÍCH DỮ LIỆU (Đức phụ trách)

**Mục đích:** Biến dữ liệu thô (text lộn xộn) thành bảng số sạch sẽ để máy tính đọc được.

**Công việc cụ thể:**
1. **Xử lý giá:** Chuyển "5.2 tỷ" → `5200000000`, "800 triệu" → `800000000`
2. **Xử lý diện tích:** Chuyển "74 m²" → `74.0`
3. **Xử lý missing values:** Tin nào thiếu giá/diện tích → loại bỏ hoặc bổ sung
4. **Loại bỏ outliers:** Giá bất thường (0 đồng, 999 tỷ...) → loại
5. **Encoding:** "Quận 1" → số 1, "Đông Nam" → số 5 (One-Hot Encoding)
6. **EDA (Exploratory Data Analysis):** Vẽ biểu đồ phân tích:
   - Heatmap tương quan giữa các đặc trưng
   - Histogram phân phối giá
   - Boxplot giá theo quận

**File đầu ra:**
- `data/processed/processed_data.csv`
- `notebooks/01_eda.ipynb`
- `notebooks/02_preprocessing.ipynb`

---

### BƯỚC 3: HUẤN LUYỆN MÔ HÌNH AI (Thái phụ trách)

**Mục đích:** Dạy máy tính cách định giá nhà bằng cách cho nó "học" từ dữ liệu đã có.

**Quy trình:**
1. Chia dữ liệu: 80% để dạy (train), 20% để thi (test)
2. Lần lượt train 4 thuật toán:

| # | Thuật toán | Đặc điểm |
|---|---|---|
| 1 | **Linear Regression** | Baseline đơn giản nhất, vẽ đường thẳng |
| 2 | **Decision Tree** | Chia dữ liệu theo cây quyết định |
| 3 | **Random Forest** | Tập hợp nhiều cây → chính xác hơn |
| 4 | **XGBoost** | Thuật toán mạnh nhất hiện nay cho bài toán bảng |

3. So sánh bằng các chỉ số:
   - **RMSE** (Root Mean Squared Error): Sai số trung bình
   - **MAE** (Mean Absolute Error): Sai số tuyệt đối trung bình
   - **R² Score**: Mô hình giải thích được bao nhiêu % biến động giá (càng gần 1 càng tốt)

4. Chọn model tốt nhất → xuất file `.pkl` (khối não AI đóng gói)

**File đầu ra:**
- `models/best_model.pkl` — Não AI
- `models/scaler.pkl` — Bộ chuẩn hóa dữ liệu
- `notebooks/03_training.ipynb`
- `notebooks/04_evaluation.ipynb`

---

### BƯỚC 4: XÂY DỰNG WEB (Tài + Đông phụ trách)

**Mục đích:** Tạo giao diện web để người dùng sử dụng AI dự đoán giá.

**Kiến trúc:**
```
Người dùng (Chrome)
    ↓ Nhập: 80m², Quận 1, 3 PN
    ↓
[Frontend - HTML/CSS/JS]  ←→  [Backend - Flask API]  ←→  [Model .pkl]
    ↓                              ↓
Hiển thị: 8.5 Tỷ VNĐ         Load não AI, tính toán
```

**Phân công:**

| Người | Làm gì |
|---|---|
| **Đông** | Code HTML/CSS giao diện (form nhập, trang kết quả, biểu đồ Chart.js) |
| **Tài** | Code Flask API (`/predict`), load model, xử lý request/response, tích hợp |

**Các trang web:**
- Trang chủ: Form nhập thông số + hiển thị kết quả dự đoán
- Dashboard: Biểu đồ phân tích dữ liệu (giá theo quận, phân phối giá...)
- About: Giới thiệu nhóm và đề tài

---

### BƯỚC 5: BÁO CÁO (Mai Anh phụ trách chính, Tài review)

**Cấu trúc báo cáo 7 chương:**

| Chương | Nội dung | Ai viết |
|---|---|---|
| 1 | Tổng quan: Đặt vấn đề, mục tiêu, phân công | Mai Anh |
| 2 | Cơ sở lý thuyết: ML, Regression, 4 thuật toán | Mai Anh |
| 3 | Thu thập & Tiền xử lý dữ liệu (kèm biểu đồ EDA) | Đức → Mai Anh tổng hợp |
| 4 | Xây dựng & Đánh giá mô hình (bảng so sánh) | Thái → Mai Anh tổng hợp |
| 5 | Phát triển hệ thống Web | Đông + Tài → Mai Anh tổng hợp |
| 6 | Kết quả & Demo (screenshot) | Mai Anh |
| 7 | Kết luận & Hướng phát triển | Tài + Mai Anh |

---

## 4. CẤU TRÚC THƯ MỤC DỰ ÁN

```
house-price-prediction/
├── requirements.txt            # Thư viện Python cần cài
├── PROJECT_PLAN.md             # File này
├── data/
│   ├── crawl/crawler.py        # Code cào dữ liệu (Tài)
│   ├── raw/raw_data.csv        # Dữ liệu thô
│   └── processed/              # Dữ liệu đã làm sạch
├── notebooks/                  # Jupyter Notebooks (Đức & Thái)
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_training.ipynb
│   └── 04_evaluation.ipynb
├── models/                     # Não AI đã train
│   └── best_model.pkl
├── app/app.py                  # Flask Backend (Tài)
├── templates/                  # HTML (Đông)
├── static/css/ js/ images/     # CSS/JS (Đông)
└── docs/report.pdf             # Báo cáo (Mai Anh)
```

---

## 5. TIMELINE CHI TIẾT (14 NGÀY)

### TUẦN 1: DATA + ML (23/4 → 29/4)

| Ngày | Tài (Leader) | Thái (ML) | Đức (Data) | Đông (FE) | Mai Anh (Docs) |
|---|---|---|---|---|---|
| **T5 23/4** | Setup GitHub, họp kick-off, bắt đầu crawl | Setup môi trường Python | Clone repo, cài Jupyter | Phác thảo wireframe UI | Tạo Google Docs, viết mục lục |
| **T6 24/4** | Crawl data (~500 tin) | Review data crawl | Test chất lượng data | Code layout HTML/CSS | Viết Chương 1 |
| **T7 25/4** | ✅ Crawl xong 2000+ tin | Review dataset | Bắt đầu EDA | Code form nhập liệu | Viết Chương 2 (2.1, 2.2) |
| **CN 26/4** | Setup Flask API cơ bản | Train LR + DT | ✅ Tiền xử lý xong | Code JS gọi API | Viết Chương 2 (thuật toán) |
| **T2 27/4** | Review code team | ✅ Train RF + XGBoost | Vẽ biểu đồ Feature Importance | Thêm Chart.js | Viết Chương 3 |
| **T3 28/4** | ✅ Tích hợp model vào web | ✅ Export model .pkl | Viết phần data cho báo cáo | Polish UI | Viết Chương 4 |
| **T4 29/4** | ✅ Web chạy E2E, họp demo | Fix bugs model | Hỗ trợ testing | Responsive design | Tổng hợp feedback |

### TUẦN 2: TÍCH HỢP + BÁO CÁO (30/4 → 6/5)

| Ngày | Tài | Thái | Đức | Đông | Mai Anh |
|---|---|---|---|---|---|
| **T5 30/4** | Fix bugs web | Thêm comments notebook | Export biểu đồ PNG | Dashboard page | Viết Chương 5 |
| **T6 1/5** | Chụp screenshot gửi MA | Viết nội dung ML cho báo cáo | Viết nội dung data cho báo cáo | Polish cuối | ✅ Bản nháp 7 chương |
| **T7 2/5** | Sửa báo cáo + format | Viết README.md | Kiểm tra biểu đồ | — | Format báo cáo đẹp |
| **CN 3/5** | Testing toàn diện | Test 10 trường hợp | Test 10 trường hợp | Test 10 trường hợp | Ghi kết quả test |
| **T2 4/5** | ✅ Báo cáo final PDF | Clean code | — | — | ✅ Export PDF |
| **T3 5/5** | Chuẩn bị Q&A | Ôn phần ML | Ôn phần Data | Ôn phần Web | Ôn phần lý thuyết |
| **T4 6/5** | ✅ Tổng duyệt cả nhóm | Tổng duyệt | Tổng duyệt | Tổng duyệt | Tổng duyệt |

### 🎯 THỨ 5, 7/5: BÁO CÁO
- **Sáng:** Cả nhóm review lần cuối
- **Chiều:** Báo cáo trước giảng viên

---

## 6. CÁC MỐC DEADLINE CỨNG

| Mốc | Ngày | Chịu trách nhiệm |
|---|---|---|
| CSV 2000+ records | 25/4 (T7) | **Tài** |
| Dữ liệu sạch | 26/4 (CN) | **Đức** |
| 4 models so sánh xong | 27/4 (T2) | **Thái** |
| Model .pkl export | 28/4 (T3) | **Thái** |
| Web chạy end-to-end | 28/4 (T3) | **Tài** |
| Báo cáo bản nháp | 1/5 (T6) | **Mai Anh** |
| Báo cáo final PDF | 4/5 (T2) | **Tài + Mai Anh** |
| Tổng duyệt | 6/5 (T4) | **Cả nhóm** |

---

## 7. TECH STACK

| Layer | Công nghệ |
|---|---|
| Crawl Data | Python, curl_cffi, BeautifulSoup |
| ML / AI | scikit-learn, XGBoost, pandas, numpy |
| Visualization | matplotlib, seaborn, Chart.js |
| Backend | Flask (Python) |
| Frontend | HTML, CSS, JavaScript |
| Notebook | Jupyter / Google Colab |
| Version Control | Git + GitHub |

---

## 8. CÂU HỎI GIẢNG VIÊN CÓ THỂ HỎI

| Câu hỏi | Ai trả lời |
|---|---|
| Tại sao chọn đề tài giá nhà? | Tài |
| Dữ liệu lấy từ đâu, crawl thế nào? | Tài |
| Tiền xử lý dữ liệu ra sao? Missing values? | Đức |
| Tại sao XGBoost tốt hơn Linear Regression? | Thái |
| R² Score, RMSE nghĩa là gì? | Mai Anh |
| Feature nào quan trọng nhất? | Thái hoặc Đức |
| Hạn chế của hệ thống? Hướng phát triển? | Tài |
| Web dùng công nghệ gì? | Đông |

---

## 9. QUY TẮC LÀM VIỆC NHÓM

1. **Commit code lên GitHub mỗi ngày** — Tài check hàng ngày
2. **Ai bị stuck phải báo group chat NGAY** — không tự ôm
3. **Họp nhanh 15 phút mỗi tối** — check tiến độ
4. **Deadline nội bộ sớm hơn 2 ngày** — có buffer sửa lỗi
5. **Mỗi người phải hiểu phần mình** — sẵn sàng trả lời GV
