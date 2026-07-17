<div align="center">

# 🏠 ProphetEstate — Dự Đoán Giá Bất Động Sản Việt Nam

**Ứng dụng AI tiên mạng đến định giá bất động sản chính xác nhất — dựa trên gần 12.000 giao dịch thực tế tại Việt Nam**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-017CEE?style=for-the-badge)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## 📋 Mục Lục

- [Giới Thiệu](#-giới-thiệu)
- [Tính Năng Chính](#-tính-năng-chính)
- [Kiến Trúc Hệ Thống](#-kiến-trúc-hệ-thống)
- [Cấu Trúc Thư Mục](#-cấu-trúc-thư-mục)
- [Yêu Cầu Hệ Thống](#-yêu-cầu-hệ-thống)
- [Hướng Dẫn Cài Đặt](#-hướng-dẫn-cài-đặt)
  - [Trên macOS](#-trên-macos)
  - [Trên Windows](#-trên-windows)
- [Hướng Dẫn Sử Dụng](#-hướng-dẫn-sử-dụng)
- [Mô Hình AI](#-mô-hình-ai)
- [Dữ Liệu](#-dữ-liệu)
- [API Endpoints](#-api-endpoints)
- [Thành Viên Nhóm](#-thành-viên-nhóm)

---

## 🎯 Giới Thiệu

**ProphetEstate** là một hệ thống dự đoán giá bất động sản tại Việt Nam sử dụng các mô hình Machine Learning và Deep Learning. Hệ thống được xây dựng dựa trên **gần 12.000 tin rao bán thực tế** được thu thập từ các sàn giao dịch bất động sản lớn nhất Việt Nam, bao gồm dữ liệu từ **Hà Nội**, **TP. Hồ Chí Minh** và **Đà Nẵng**.

### Điểm nổi bật

- 🤖 **Dual AI Engine**: Kết hợp XGBoost (ML truyền thống) và Transformer (Deep Learning) để đưa ra dự đoán chính xác nhất
- 📊 **Explainable AI (XAI)**: Giải thích minh bạch từng yếu tố ảnh hưởng đến giá dự đoán
- 🔍 **Tìm kiếm & Lọc thông minh**: Tìm kiếm BĐS theo nhiều tiêu chí với dữ liệu thực tế
- 📈 **Phân tích thị trường**: Dashboard trực quan hóa xu hướng thị trường BĐS
- 🏡 **Gợi ý tương tự**: Tự động gợi ý các bất động sản tương tự dựa trên kết quả dự đoán

---

## ✨ Tính Năng Chính

| Tính năng | Mô tả |
|-----------|-------|
| **🤖 Định Giá AI** | Dự đoán giá Chung cư & Nhà đất bằng mô hình Transformer Deep Learning |
| **🔍 Tìm Kiếm BĐS** | Tìm kiếm bất động sản theo vị trí, giá, diện tích, số phòng ngủ với dữ liệu thực |
| **📊 Phân Tích Thị Trường** | Dashboard biểu đồ tương tác phân tích xu hướng giá theo quận/huyện |
| **🧠 Mô Hình AI** | Trang so sánh chi tiết hiệu năng giữa XGBoost và Transformer |
| **📰 Tin Tức BĐS** | Cập nhật tin tức, phân tích thị trường bất động sản mới nhất |
| **👤 Tài Khoản** | Đăng ký, đăng nhập (Email + Google OAuth), lưu lịch sử dự đoán |
| **💾 Lưu BĐS Yêu Thích** | Lưu các bất động sản quan tâm để xem lại sau |
| **📱 Responsive Design** | Giao diện tương thích tốt trên mọi thiết bị |

---

## 🏗 Kiến Trúc Hệ Thống

```
┌──────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Jinja2 + CSS + JS)              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │
│  │ Định Giá │ │ Tìm Kiếm │ │ Phân Tích│ │ Mô Hình  │ │Tin Tức │ │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └───┬────┘ │
│       │            │            │            │            │      │
├───────┴────────────┴────────────┴────────────┴────────────┴──────┤
│                      BACKEND (Flask Python)                      │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                    API Layer (REST)                       │    │
│  │   /predict  /search  /analytics  /auth  /property/:id    │    │
│  └────────────────────────┬─────────────────────────────────┘    │
│                           │                                      │
│  ┌────────────────────────┴─────────────────────────────────┐    │
│  │                   AI / ML Engine                          │    │
│  │  ┌─────────────────┐    ┌──────────────────────────┐     │    │
│  │  │ XGBoost Pipeline │    │ Transformer Deep Learning │     │    │
│  │  │ (scikit-learn)   │    │ (PyTorch)                 │     │    │
│  │  │ • CC: R²=0.6734  │    │ • CC: R²=0.6451           │     │    │
│  │  │ • NĐ: R²=0.7587  │    │ • NĐ: R²=0.7355           │     │    │
│  │  └─────────────────┘    └──────────────────────────┘     │    │
│  └──────────────────────────────────────────────────────────┘    │
│                           │                                      │
│  ┌────────────────────────┴─────────────────────────────────┐    │
│  │                      Data Layer                           │    │
│  │   cleaned_chung_cu.csv (5,452 mẫu)                       │    │
│  │   cleaned_nha_dat.csv  (6,379 mẫu)                       │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📁 Cấu Trúc Thư Mục

```
House-Price-Prediction/
│
├── app/                            # 🌐 Ứng dụng Web
│   ├── app.py                      #    Server chính Flask (backend + API)
│   ├── static/                     #    Tài nguyên tĩnh
│   │   ├── style.css               #       CSS chính
│   │   ├── style-override.css      #       CSS ghi đè & responsive
│   │   ├── script.js               #       JavaScript frontend
│   │   ├── logo.png                #       Logo ProphetEstate
│   │   ├── hero-bg.png             #       Ảnh nền trang chủ
│   │   └── images/                 #       Ảnh minh họa khác
│   └── templates/                  #    Giao diện HTML (Jinja2)
│       ├── index.html              #       Trang chủ + Form dự đoán
│       ├── search.html             #       Trang tìm kiếm BĐS
│       ├── analytics.html          #       Trang phân tích thị trường
│       ├── models.html             #       Trang so sánh mô hình AI
│       ├── news.html               #       Trang tin tức
│       ├── property_detail.html    #       Chi tiết bất động sản
│       ├── profile.html            #       Hồ sơ người dùng
│       ├── history.html            #       Lịch sử dự đoán
│       ├── saved.html              #       BĐS đã lưu
│       ├── about.html              #       Giới thiệu
│       ├── _header.html            #       Header dùng chung
│       ├── _footer.html            #       Footer dùng chung
│       └── _account_tabs.html      #       Tab tài khoản
│
├── data/                           # 📊 Dữ liệu
│   ├── crawl/                      #    Script thu thập dữ liệu
│   │   └── crawler.py              #       Crawler từ batdongsan.com.vn
│   ├── raw/                        #    Dữ liệu thô (chưa xử lý)
│   └── processed/                  #    Dữ liệu đã làm sạch
│       ├── cleaned_chung_cu.csv    #       5,452 mẫu chung cư
│       ├── cleaned_nha_dat.csv     #       6,379 mẫu nhà đất
│       ├── cleaned_full_web.csv    #       Toàn bộ dữ liệu gộp
│       └── eda_*.png               #       Biểu đồ EDA
│
├── models/                         # 🤖 Mô hình AI đã huấn luyện
│   ├── ml_traditional/             #    Mô hình ML truyền thống
│   │   ├── national/               #       Mô hình toàn quốc
│   │   │   ├── model_cc_with_city.pkl      # XGBoost Chung cư
│   │   │   ├── model_nd_with_city.pkl      # XGBoost Nhà đất
│   │   │   └── comparison_report.md        # Báo cáo so sánh
│   │   └── hanoi_only/             #       Mô hình riêng Hà Nội
│   └── transformer/                #    Mô hình Transformer
│       ├── national/               #       Transformer toàn quốc
│       │   ├── transformer_model_cc.pt     # Model Chung cư
│       │   ├── transformer_model_nd.pt     # Model Nhà đất
│       │   ├── preprocessing_cc.pkl        # Preprocessing CC
│       │   └── preprocessing_nd.pkl        # Preprocessing NĐ
│       └── hanoi_only/             #       Transformer riêng Hà Nội
│
├── notebooks/                      # 📓 Jupyter Notebooks
│   ├── 01_eda.ipynb                #    Khám phá & phân tích dữ liệu (EDA)
│   ├── 02_preprocessing.ipynb      #    Tiền xử lý & làm sạch dữ liệu
│   ├── 03_train_ml_national.ipynb  #    Huấn luyện ML toàn quốc
│   ├── 04_train_ml_hanoi.ipynb     #    Huấn luyện ML riêng Hà Nội
│   ├── 05_train_transformer_national.ipynb  # Huấn luyện Transformer toàn quốc
│   └── 06_train_transformer_hanoi.ipynb     # Huấn luyện Transformer Hà Nội
│
├── docs/                           # 📄 Tài liệu & Báo cáo
│   ├── BÁO CÁO/                   #    Báo cáo đồ án
│   └── database/                   #    Tài liệu cơ sở dữ liệu
│
├── transformer_model.py            # 🧠 Định nghĩa kiến trúc Transformer
├── run_training.py                 # 🚀 Script huấn luyện mô hình
├── predict_transformer.py          # 🔮 Script dự đoán bằng Transformer
├── requirements.txt                # 📦 Danh sách thư viện Python
├── .gitignore                      # 🚫 File ignore cho Git
└── README.md                       # 📖 File hướng dẫn (file này)
```

---

## 💻 Yêu Cầu Hệ Thống

### Phần mềm bắt buộc

| Phần mềm | Phiên bản tối thiểu | Ghi chú |
|-----------|---------------------|---------|
| **Python** | 3.10 trở lên | Khuyến nghị Python 3.11 hoặc 3.12 |
| **pip** | 21.0 trở lên | Đi kèm khi cài Python |
| **Git** | 2.30 trở lên | Để clone repository |

### Phần cứng khuyến nghị

| Thành phần | Tối thiểu | Khuyến nghị |
|------------|-----------|-------------|
| **RAM** | 4 GB | 8 GB trở lên |
| **Ổ cứng trống** | 2 GB | 5 GB |
| **CPU** | Bất kỳ | Multi-core |
| **GPU** | Không cần | NVIDIA CUDA (tùy chọn, tăng tốc Transformer) |

> **📝 Lưu ý:** GPU không bắt buộc. PyTorch sẽ tự động sử dụng CPU nếu không có GPU NVIDIA. Ứng dụng vẫn hoạt động bình thường.

---

## 🚀 Hướng Dẫn Cài Đặt

### 🍎 Trên macOS

#### Bước 1: Cài đặt Homebrew (nếu chưa có)

Mở **Terminal** (tìm trong Spotlight bằng `Cmd + Space`, gõ "Terminal"):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### Bước 2: Cài đặt Python và Git

```bash
# Cài Python (nếu chưa có)
brew install python@3.12

# Kiểm tra phiên bản Python
python3 --version
# Kết quả mong đợi: Python 3.12.x

# Cài Git (thường đã có sẵn trên macOS)
brew install git

# Kiểm tra phiên bản Git
git --version
```

#### Bước 3: Clone repository

```bash
# Di chuyển đến thư mục bạn muốn lưu project
cd ~/Desktop

# Clone repository từ GitHub
git clone https://github.com/tangoctai2004/House-Price-Prediction.git

# Di chuyển vào thư mục project
cd House-Price-Prediction
```

#### Bước 4: Tạo môi trường ảo (Virtual Environment) — *Khuyến nghị*

```bash
# Tạo môi trường ảo tên "venv"
python3 -m venv venv

# Kích hoạt môi trường ảo
source venv/bin/activate

# Sau khi kích hoạt, terminal sẽ hiển thị (venv) ở đầu dòng
# (venv) user@MacBook House-Price-Prediction %
```

#### Bước 5: Cài đặt thư viện

```bash
# Cập nhật pip lên phiên bản mới nhất
pip install --upgrade pip

# Cài đặt tất cả thư viện cần thiết
pip install -r requirements.txt
```

> ⏱️ **Thời gian cài đặt:** Khoảng 3-10 phút tùy tốc độ mạng. Thư viện `torch` (PyTorch) có dung lượng lớn (~800MB).

#### Bước 6: Chạy ứng dụng

```bash
python3 app/app.py
```

Kết quả mong đợi trên Terminal:
```
[OK] Da load thanh cong model_cc_with_city.pkl
[OK] Da load thanh cong model_nd_with_city.pkl
[OK] Da load thanh cong 4 Transformer models
[OK] Da load du lieu CSV de lam goi y
 * Serving Flask app 'app'
 * Running on http://localhost:5003
```

#### Bước 7: Mở trình duyệt

Truy cập: **http://localhost:5003**

#### Tắt ứng dụng

- Nhấn `Ctrl + C` trong Terminal để dừng server
- Gõ `deactivate` để thoát môi trường ảo

---

### 🪟 Trên Windows

#### Bước 1: Cài đặt Python

1. Truy cập **https://www.python.org/downloads/**
2. Tải phiên bản **Python 3.12** (hoặc mới hơn)
3. Chạy file `.exe` đã tải
4. ⚠️ **QUAN TRỌNG:** Tick chọn ✅ **"Add python.exe to PATH"** ở màn hình đầu tiên
5. Nhấn **"Install Now"**

Sau khi cài xong, mở **Command Prompt** (nhấn `Win + R`, gõ `cmd`, nhấn Enter) và kiểm tra:

```cmd
python --version
```
Kết quả mong đợi: `Python 3.12.x`

```cmd
pip --version
```
Kết quả mong đợi: `pip 24.x.x from ...`

#### Bước 2: Cài đặt Git

1. Truy cập **https://git-scm.com/download/win**
2. Tải và cài đặt Git (giữ nguyên cấu hình mặc định, nhấn Next cho đến khi hoàn tất)
3. Kiểm tra:

```cmd
git --version
```
Kết quả mong đợi: `git version 2.x.x.windows.x`

#### Bước 3: Clone repository

Mở **Command Prompt** hoặc **PowerShell**:

```cmd
:: Di chuyển đến Desktop (hoặc thư mục bất kỳ bạn muốn)
cd %USERPROFILE%\Desktop

:: Clone repository từ GitHub
git clone https://github.com/tangoctai2004/House-Price-Prediction.git

:: Di chuyển vào thư mục project
cd House-Price-Prediction
```

#### Bước 4: Tạo môi trường ảo (Virtual Environment) — *Khuyến nghị*

```cmd
:: Tạo môi trường ảo tên "venv"
python -m venv venv

:: Kích hoạt môi trường ảo
venv\Scripts\activate

:: Sau khi kích hoạt, Command Prompt sẽ hiển thị (venv) ở đầu dòng
:: (venv) C:\Users\...\House-Price-Prediction>
```

> **📝 Lưu ý PowerShell:** Nếu gặp lỗi khi kích hoạt venv trên PowerShell, chạy lệnh sau trước:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Sau đó kích hoạt lại:
> ```powershell
> venv\Scripts\Activate.ps1
> ```

#### Bước 5: Cài đặt thư viện

```cmd
:: Cập nhật pip
python -m pip install --upgrade pip

:: Cài đặt tất cả thư viện
pip install -r requirements.txt
```

> ⏱️ **Thời gian cài đặt:** Khoảng 5-15 phút. Thư viện `torch` (PyTorch) có dung lượng lớn (~2.5GB trên Windows do kèm CUDA binaries).
>
> **💡 Mẹo tiết kiệm dung lượng:** Nếu máy không có GPU NVIDIA, bạn có thể cài phiên bản PyTorch CPU nhỏ gọn hơn:
> ```cmd
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

#### Bước 6: Xử lý lỗi AirPlay / Port bị chiếm (nếu có)

Nếu gặp lỗi `Port 5003 is in use`, đổi port bằng cách:

```cmd
:: Chạy ứng dụng trên port khác (ví dụ 8080)
set PORT=8080
python app/app.py
```

#### Bước 7: Chạy ứng dụng

```cmd
python app/app.py
```

Kết quả mong đợi:
```
[OK] Da load thanh cong model_cc_with_city.pkl
[OK] Da load thanh cong model_nd_with_city.pkl
[OK] Da load thanh cong 4 Transformer models
[OK] Da load du lieu CSV de lam goi y
 * Serving Flask app 'app'
 * Running on http://localhost:5003
```

#### Bước 8: Mở trình duyệt

Truy cập: **http://localhost:5003**

#### Tắt ứng dụng

- Nhấn `Ctrl + C` trong Command Prompt để dừng server
- Gõ `deactivate` để thoát môi trường ảo

---

### 🔧 Xử Lý Sự Cố Thường Gặp

<details>
<summary><b>❌ Lỗi "ModuleNotFoundError: No module named 'flask'"</b></summary>

**Nguyên nhân:** Chưa cài đặt thư viện hoặc chưa kích hoạt môi trường ảo.

**Cách khắc phục:**
```bash
# Kiểm tra đã kích hoạt venv chưa (phải thấy (venv) ở đầu dòng)
# macOS:
source venv/bin/activate

# Windows:
venv\Scripts\activate

# Cài lại thư viện
pip install -r requirements.txt
```
</details>

<details>
<summary><b>❌ Lỗi "Address already in use" / "Port is in use"</b></summary>

**Nguyên nhân:** Port đang bị ứng dụng khác chiếm (thường là AirPlay trên macOS).

**Cách khắc phục:**
```bash
# macOS: Tắt AirPlay Receiver
# System Settings → General → AirDrop & Handoff → AirPlay Receiver → Tắt

# Hoặc dùng port khác:
# macOS:
PORT=8080 python3 app/app.py

# Windows:
set PORT=8080
python app/app.py
```
</details>

<details>
<summary><b>❌ Lỗi "python: command not found" (macOS)</b></summary>

**Nguyên nhân:** macOS dùng `python3` thay vì `python`.

**Cách khắc phục:** Thay `python` bằng `python3` trong tất cả các lệnh:
```bash
python3 app/app.py
```
</details>

<details>
<summary><b>❌ Lỗi "pip install" chạy quá chậm</b></summary>

**Cách khắc phục:** Sử dụng mirror gần Việt Nam:
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
</details>

---

## 📖 Hướng Dẫn Sử Dụng

### 1. Định Giá AI

1. Truy cập trang chủ **http://localhost:5003**
2. Chọn loại bất động sản: **Chung Cư** hoặc **Nhà Đất**
3. Nhập thông tin:
   - Diện tích (m²)
   - Số phòng ngủ
   - Tỉnh/Thành phố & Quận/Huyện
   - Hướng nhà
   - Nội thất & Pháp lý
   - *(Nhà đất)* Số tầng, Mặt tiền, Đường vào
4. Nhấn **"Dự Đoán Giá Trị"**
5. Xem kết quả: giá dự đoán, phân tích mức độ ảnh hưởng, và các bất động sản tương tự

### 2. Tìm Kiếm BĐS

- Truy cập tab **"Tìm kiếm"** trên thanh điều hướng
- Lọc theo: vị trí, khoảng giá, diện tích, số phòng ngủ
- Nhấn vào từng bất động sản để xem chi tiết

### 3. Phân Tích Thị Trường

- Truy cập tab **"Phân tích"**
- Xem biểu đồ phân phối giá, so sánh giữa các quận/huyện
- Phân tích xu hướng theo loại hình BĐS

### 4. Tài Khoản Demo

Sử dụng tài khoản có sẵn để trải nghiệm đầy đủ tính năng:

| Trường | Giá trị |
|--------|---------|
| **Email** | `admin@prophet.vn` |
| **Mật khẩu** | `123456` |

---

## 🤖 Mô Hình AI

### Tổng quan mô hình

Hệ thống sử dụng **2 loại mô hình**, mỗi loại được huấn luyện riêng biệt cho **Chung cư** và **Nhà đất**:

#### 1. XGBoost Pipeline (ML Truyền thống)

| Chỉ số | Chung cư | Nhà đất |
|--------|----------|---------|
| **R² (Test)** | 0.6734 | 0.7587 |
| **MAE** | 1.67 tỷ | 2.99 tỷ |
| **RMSE** | 3.14 tỷ | 5.90 tỷ |
| **CV R² (5-Fold)** | 0.697 ± 0.027 | 0.751 ± 0.008 |

#### 2. Transformer Deep Learning (PyTorch)

| Chỉ số | Chung cư | Nhà đất |
|--------|----------|---------|
| **R² (Test)** | 0.6451 | 0.7355 |
| **MAE** | 1.78 tỷ | 3.27 tỷ |
| **RMSE** | 3.16 tỷ | 6.08 tỷ |
| **Overfit Gap** | 6.7% | 9.4% |

### Đặc trưng đầu vào (Features)

| Feature | Mô tả | Loại |
|---------|--------|------|
| `area_m2` | Diện tích (m²) | Số |
| `bedrooms_num` | Số phòng ngủ | Số |
| `floors_num` | Số tầng *(chỉ nhà đất)* | Số |
| `frontage_m` | Mặt tiền (m) *(chỉ nhà đất)* | Số |
| `road_width_m` | Đường vào (m) *(chỉ nhà đất)* | Số |
| `city` | Tỉnh/Thành phố | Phân loại |
| `district` | Quận/Huyện | Phân loại |
| `direction` | Hướng nhà | Phân loại |
| `furniture_std` | Tình trạng nội thất | Phân loại |
| `legal_std` | Tình trạng pháp lý | Phân loại |

### Notebooks huấn luyện

| Notebook | Nội dung |
|----------|----------|
| `01_eda.ipynb` | Khám phá dữ liệu (EDA): phân phối giá, tương quan, outliers |
| `02_preprocessing.ipynb` | Làm sạch, chuẩn hóa, xử lý missing values |
| `03_train_ml_national.ipynb` | Huấn luyện XGBoost trên toàn bộ dữ liệu |
| `04_train_ml_hanoi.ipynb` | Huấn luyện XGBoost riêng cho Hà Nội |
| `05_train_transformer_national.ipynb` | Huấn luyện Transformer toàn quốc |
| `06_train_transformer_hanoi.ipynb` | Huấn luyện Transformer riêng Hà Nội |

---

## 📊 Dữ Liệu

### Nguồn dữ liệu

Dữ liệu được thu thập từ các sàn giao dịch bất động sản trực tuyến lớn nhất Việt Nam thông qua web scraping (`data/crawl/crawler.py`).

### Thống kê dữ liệu

| Loại BĐS | Số mẫu | Thành phố | Số Quận/Huyện |
|-----------|--------|-----------|---------------|
| **Chung cư** | 5,452 | Hà Nội, TP.HCM, Đà Nẵng | 43 |
| **Nhà đất** | 6,379 | Hà Nội, TP.HCM | 36 |
| **Tổng cộng** | **11,831** | **3 thành phố** | — |

### Tiền xử lý dữ liệu

- Loại bỏ các bản ghi trùng lặp và thiếu dữ liệu quan trọng
- Chuẩn hóa tên quận/huyện, hướng nhà, nội thất, pháp lý
- Xử lý outliers bằng phương pháp IQR
- Mã hóa biến phân loại bằng One-Hot Encoding (XGBoost) và Label Encoding (Transformer)

---

## 🔌 API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `GET` | `/` | Trang chủ + Form dự đoán |
| `POST` | `/predict` | API dự đoán giá BĐS |
| `GET` | `/search` | Trang tìm kiếm |
| `GET` | `/api/search` | API tìm kiếm BĐS (JSON) |
| `GET` | `/analytics` | Trang phân tích thị trường |
| `GET` | `/api/analytics/data` | API dữ liệu phân tích (JSON) |
| `GET` | `/models` | Trang so sánh mô hình |
| `GET` | `/news` | Trang tin tức |
| `GET` | `/property/<type>/<id>` | Chi tiết bất động sản |
| `POST` | `/login` | Đăng nhập |
| `POST` | `/register` | Đăng ký tài khoản |
| `GET` | `/logout` | Đăng xuất |
| `GET` | `/profile` | Trang hồ sơ |
| `GET` | `/history` | Lịch sử dự đoán |
| `POST` | `/api/save-property` | Lưu BĐS yêu thích |

### Ví dụ gọi API dự đoán

```bash
curl -X POST http://localhost:5003/predict \
  -H "Content-Type: application/json" \
  -d '{
    "property_type": "chung_cu",
    "area": "70",
    "bedrooms": "2",
    "province": "TP. Hồ Chí Minh",
    "district": "Quận 7",
    "direction": "Đông - Nam",
    "furniture": "Đầy đủ",
    "legal": "Sổ đỏ/Sổ hồng"
  }'
```

---

## 🛠 Công Nghệ Sử Dụng

| Lớp | Công nghệ |
|-----|-----------|
| **Backend** | Python 3.12, Flask 3.0 |
| **Frontend** | HTML5, CSS3, JavaScript (Vanilla) |
| **Template Engine** | Jinja2 |
| **ML Framework** | scikit-learn, XGBoost |
| **DL Framework** | PyTorch (Transformer) |
| **Data Processing** | pandas, NumPy, SciPy |
| **Visualization** | Matplotlib, Seaborn |
| **Web Scraping** | curl_cffi, BeautifulSoup4 |
| **Authentication** | Flask Session, Google OAuth 2.0 |

---

## 👥 Thành Viên Nhóm

<div align="center">

| Thành viên | Vai trò |
|------------|---------|
| **Tạ Ngọc Tài** | Team Lead / Full-stack Developer |
| **Trịnh Quang Thái** | ML Engineer / Data Scientist |
| **Nguyễn Đức Đồng** | Frontend Developer |
| **Nguyễn Hoàng Đức** | Data Analyst / Web Scraping |
| **Bùi Lê Mai Anh** | UI/UX Designer / Tester |

</div>

---

## 📄 License

Project được phát triển phục vụ mục đích học tập và nghiên cứu.

---

<div align="center">

**Made with ❤️ by ProphetEstate Team**

*Đồ án Khoa học Dữ liệu — 2025*

</div>
