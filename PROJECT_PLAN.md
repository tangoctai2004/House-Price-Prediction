# 🏠 DỰ ÁN: Hệ Thống Web Dự Đoán Giá Bất Động Sản Việt Nam

**Môn học:** Phát triển Hệ thống Thông minh — Học viện Bưu chính Viễn thông (PTIT)  
**Đề số:** 2 — Dự đoán Giá (Regression)  
**Deadline:** Chiều thứ 5, ngày 07/05/2026

---

## 1. TỔNG QUAN ĐỀ TÀI

### Mục tiêu
Xây dựng một hệ thống web cho phép người dùng nhập thông số căn nhà (diện tích, số phòng ngủ, tỉnh/thành, quận/huyện, hướng nhà...) và nhận về **giá dự đoán bằng VNĐ** từ mô hình Machine Learning.

### Điểm nổi bật (so với các nhóm khác)
- **Tự crawl dữ liệu thực** từ batdongsan.com.vn (11.831 bản ghi, không dùng dataset có sẵn).
- **Hệ thống AI Pipeline hiện đại:** Tích hợp bộ chuẩn hóa (StandardScaler) và mã hóa (OneHotEncoder) vào cùng 1 Pipeline duy nhất, giúp web hoạt động cực kỳ mượt mà và không bao giờ bị lỗi khi gặp dữ liệu mới.
- So sánh **4 thuật toán ML** và chọn ra mô hình tốt nhất (XGBoost).
- Giao diện web **hiển thị chi tiết từng căn nhà**, bộ lọc tìm kiếm nâng cao nhiều lớp, UI/UX hiện đại chuẩn doanh nghiệp.
- **Tích hợp Cơ sở dữ liệu thật (Database)** quản lý người dùng và lịch sử thay vì chỉ lưu tạm thời.

### Quy trình tổng thể (End-to-End Pipeline)

```text
[1. CRAWL] → [2. CLEAN & EDA] → [3. TRAIN AI & PIPELINE] → [4. DATABASE & WEB APP] → [5. ĐÁNH GIÁ & BÁO CÁO]
   (Tài)          (Tài)                 (Thái)                 (Đông + Đức)                 (Đức + Mai Anh)
```

---

## 2. NHÓM THỰC HIỆN

| Thành viên | Vai trò | Phụ trách chính |
|---|---|---|
| **Tài** | Leader / Full-stack | Crawl, Clean, EDA, API, Quản lý tiến độ chung |
| **Thái** | ML Engineer | Xây dựng AI Pipeline, Train 4 thuật toán, Export model |
| **Đức** | Backend & Data Analyst | **Tích hợp Database (SQLite/SQLAlchemy)**, Kiểm thử mô hình |
| **Đông** | Frontend Dev | Giao diện Web (HTML/CSS/JS), Tìm kiếm nâng cao, Lọc dữ liệu |
| **Mai Anh** | Documentation | Tổng hợp báo cáo, Nghiên cứu lý thuyết, QA |

---

## 3. GIẢI THÍCH CHI TIẾT TỪNG BƯỚC

### BƯỚC 1: CRAWL DỮ LIỆU ✅ (Tài — Hoàn thành)

**Mục đích:** Thu thập dữ liệu thực về giá BĐS tại Hà Nội, TP.HCM, Đà Nẵng từ batdongsan.com.vn.

**Cách hoạt động:**
- Dùng thư viện `curl_cffi` (Python) giả lập trình duyệt Chrome để vượt tường lửa.
- Mỗi căn nhà cào xong → ghi ngay 1 dòng vào file CSV.
- Đã thu thập **11.831 bản ghi** (5.452 Chung cư + 6.379 Nhà đất).

---

### BƯỚC 2: LÀM SẠCH & PHÂN TÍCH DỮ LIỆU (EDA) ✅ (Tài — Hoàn thành)

> Chi tiết kỹ thuật xem file `DATA_CLEANING_GUIDE.md`

**Quyết định thiết kế:** Dữ liệu có 2 loại BĐS (Chung cư và Nhà riêng). Chúng ta gom chung vào 1 tập dữ liệu và thêm cột `loai_bds` làm một feature đặc trưng để mô hình tự học sự khác biệt thay vì phải chia đôi model.

**Các xử lý đã thực hiện:**
1. Lọc outlier (giữ các nhà 1-200 tỷ, 10-1000m2).
2. Xử lý Missing Values theo Median của từng Quận/Huyện.
3. Chuẩn hóa text cho Nội thất, Pháp lý, Hướng nhà.

---

### BƯỚC 3: HUẤN LUYỆN MÔ HÌNH AI & PIPELINE ✅ (Thái — Hoàn thành)

**Mục đích:** Train 1 hệ thống AI đồng nhất cho mọi loại BĐS bằng `sklearn.Pipeline`.

**Quy trình Pipeline v2:**
1. Đọc file CSV tổng hợp đã làm sạch.
2. Dùng `ColumnTransformer`:
   - `StandardScaler` cho biến số (diện tích, phòng ngủ, số tầng...).
   - `OneHotEncoder(handle_unknown='ignore')` cho biến phân loại (quận, hướng, nội thất...).
3. Train 4 thuật toán: **Linear Regression, Decision Tree, Random Forest, XGBoost**.
4. So sánh bằng **RMSE, MAE, R² Score** → Chọn **XGBoost**.
5. Gói toàn bộ Preprocessor và XGBoost vào **1 Pipeline duy nhất** và xuất ra file.

**File đầu ra:**
- `models/best_model_pipeline.pkl` (Chỉ cần 1 file duy nhất này để chạy Web).
- `notebooks/03_training.ipynb` và `notebooks/04_evaluation.ipynb`.

---

### BƯỚC 4: XÂY DỰNG WEB & DATABASE 🔄 (Đông + Đức — Đang thực hiện)

**Kiến trúc Hệ thống:**
```text
Người dùng (Trình duyệt)
    ↓ Chọn Tỉnh/Thành → Quận/Huyện + Nhập thông số
    ↓
[Frontend (Đông)]  ←→  [Backend Flask + Database (Đức + Tài)]
                                    ↓
                       Đưa dữ liệu vào best_model_pipeline.pkl
                                    ↓
                          Lưu Lịch sử vào Database
                                    ↓
                       Trả về: Giá dự đoán (tỷ VNĐ)
```

**Phân công chi tiết:**
- **Đông (FE):** Làm giao diện đẹp mắt, hiệu ứng Glassmorphism, bộ lọc nâng cao (chọn Tỉnh tự động nhảy ra Quận), tích hợp Chart.js cho Analytics.
- **Đức (BE/DB):** **(MỚI)** Chuyển đổi bộ nhớ In-Memory sang Cơ sở dữ liệu thật (SQLite). Tạo các bảng `Users`, `PredictionHistory`, `SavedProperties` bằng SQLAlchemy.

---

### BƯỚC 5: BÁO CÁO 🔄 (Mai Anh — Đang thực hiện)

**Cấu trúc báo cáo 7 chương:**
| Chương | Nội dung | Ai viết |
|---|---|---|
| 1 | Tổng quan: Đặt vấn đề, mục tiêu, phân công | Mai Anh |
| 2 | Cơ sở lý thuyết: ML, Regression, 4 thuật toán | Mai Anh |
| 3 | Thu thập & Tiền xử lý dữ liệu (EDA) | Mai Anh tổng hợp |
| 4 | Xây dựng & Đánh giá mô hình AI | Mai Anh tổng hợp |
| 5 | Phát triển Web & Cơ sở dữ liệu | Đức + Đông + Tài |
| 6 | Kết quả & Demo (screenshot) | Mai Anh |
| 7 | Kết luận & Hướng phát triển | Tài + Mai Anh |

---

## 4. CẤU TRÚC THƯ MỤC DỰ ÁN MỚI NHẤT

```text
house-price-prediction/
├── PROJECT_PLAN.md             # Kế hoạch tổng thể (File này)
├── DATA_CLEANING_GUIDE.md      # Hướng dẫn làm sạch data
├── requirements.txt            # Thư viện Python cần cài
├── data/                       # Chứa dữ liệu raw và processed
├── notebooks/                  
│   ├── 01_eda.ipynb            # Phân tích EDA
│   ├── 02_preprocessing.ipynb  # Code tiền xử lý
│   ├── 03_training.ipynb       # Code train AI bằng Pipeline (Mới)
│   └── 04_evaluation.ipynb     # Đánh giá, vẽ biểu đồ sai số (Mới)
├── models/                     
│   ├── best_model_pipeline.pkl # "Não bộ" AI duy nhất đang dùng
│   ├── model_meta.pkl          # Thông số config model
│   └── *.png                   # Biểu đồ đánh giá
├── app/app.py                  # Flask Backend
├── app/templates/              # Các trang HTML (Giao diện)
└── app/static/                 # CSS/JS và Hình ảnh
```

---

## 5. TIMELINE CHI TIẾT CẬP NHẬT (14 NGÀY)

### TUẦN 1: DATA + ML (23/4 → 29/4) ✅ (Đã xong)
- Hoàn thành Crawl data 11.831 căn (5.452 Chung cư + 6.379 Nhà đất).
- Tiền xử lý và tạo biểu đồ EDA.
- Thiết kế UI HTML/CSS bản Alpha.
- Hoàn thành bản nháp 4 thuật toán Machine Learning.

### TUẦN 2: TỐI ƯU HÓA, DATABASE + BÁO CÁO (30/4 → 6/5) 🔄 (Hiện tại)

| Ngày | Tài (Leader) | Thái (ML) | Đức (DB & Test) | Đông (FE) | Mai Anh (Docs) |
|---|---|---|---|---|---|
| **T5 30/4** | Cấu hình Git, review code | Refactor: Đổi sang Pipeline v2 | Nghiên cứu Flask-SQLAlchemy | Thêm bộ lọc Tỉnh/Quận | Viết lý thuyết ML |
| **T6 1/5** | Hỗ trợ fix UI bugs | Update NB 03 & 04 khớp với Pipeline | **Setup DB, viết code ORM** | CSS hoàn thiện các nút | Tổng hợp Chương 3+4 |
| **T7 2/5** | Review tính năng Search | Viết feature importance | **Chuyển Login/Reg sang DB** | Tinh chỉnh Analytics | Viết Chương 5 |
| **CN 3/5** | Testing End-to-End | Test file .pkl | **Lưu Prediction History vô DB**| Responsive Mobile | Hoàn thiện bản nháp |
| **T2 4/5** | ✅ Chốt chức năng | Hỗ trợ viết báo cáo | Sửa bug DB nếu có | Sửa bug UI | ✅ Sinh file PDF cuối |
| **T3 5/5** | Chuẩn bị Q&A | Ôn kiến thức Pipeline | Ôn phần DB & Data | Ôn kiến thức Web | Đọc lại toàn bộ |
| **T4 6/5** | ✅ Tổng duyệt cả nhóm | Tổng duyệt | Tổng duyệt | Tổng duyệt | Tổng duyệt |

### 🎯 THỨ 5, 7/5: BÁO CÁO (DEADLINE)
- **Sáng:** Cả nhóm review lần cuối trước giờ G.
- **Chiều:** Báo cáo trước giảng viên (Chuẩn bị demo Web mượt mà nhất).

---

## 6. TECH STACK

| Layer | Công nghệ |
|---|---|
| Crawl Data | Python, curl_cffi, BeautifulSoup |
| ML / AI | scikit-learn (`Pipeline`, `ColumnTransformer`), XGBoost |
| Backend & DB | Flask, **SQLite / SQLAlchemy** |
| Frontend | HTML, CSS (Glassmorphism), Vanilla JS |
| Version Control | Git + GitHub |

---

## 7. CÂU HỎI GIẢNG VIÊN CÓ THỂ HỎI (CẬP NHẬT)

1. **Tại sao không dùng 2 model riêng cho Chung cư / Nhà đất nữa?** → (Thái/Tài): Vì dữ liệu đã lớn hơn (12K bản ghi), việc gom chung và dùng `loai_bds` làm Feature + sử dụng XGBoost giúp mô hình tự học tương quan tốt hơn, quản lý 1 file `pipeline.pkl` gọn nhẹ và ít rủi ro hơn trên Web.
2. **Dữ liệu lưu ở đâu? Web có dùng Database không?** → (Đức): Hệ thống dùng SQLite kết hợp SQLAlchemy để lưu trữ tài khoản người dùng và lịch sử dự đoán vĩnh viễn, khắc phục nhược điểm mất dữ liệu của bộ nhớ RAM (In-Memory).
3. **Tại sao dùng OneHotEncoder trong Pipeline thay vì LabelEncoder?** → (Thái): Vì LabelEncoder sinh ra số nguyên có tính phân bậc (0 < 1 < 2), làm mô hình hiểu sai về Quận/Huyện. OneHotEncoder sinh biến giả chuẩn xác hơn cho các biến danh mục không có thứ tự.
4. **Lọc theo tỉnh và quận hoạt động thế nào?** → (Đông): Dùng JavaScript bắt sự kiện thay đổi của Select Box Tỉnh, từ đó linh động đổi các option trong Select Box Quận tương ứng mà không cần tải lại trang.

---

## 8. QUY TẮC LÀM VIỆC NHÓM
1. **Commit code lên GitHub mỗi ngày** — Chống mất source.
2. **Ai bị stuck phải báo group chat NGAY** — Tránh "nước đến chân mới nhảy".
3. **Mỗi người phải hiểu phần mình** — Sẵn sàng trả lời GV.
