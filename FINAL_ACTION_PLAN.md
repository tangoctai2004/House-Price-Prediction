# 📋 KẾ HOẠCH KHẮC PHỤC TOÀN DIỆN — ProphetEstate

**Ngày lập:** 03/05/2026  
**Deadline:** Chiều 07/05/2026 (còn **4 ngày**)  
**Mục tiêu:** Nâng cấp bài tập lớn từ mức Khá (6.6/10) lên Giỏi (8.5+/10)

---

## 📊 Kết Quả Rà Soát Toàn Bộ Source Code

Sau khi kiểm tra xuyên suốt **15 file**, tôi phát hiện **12 vấn đề** chia thành 3 mức độ:

| Mức độ | Số lượng | Ý nghĩa |
|:---:|:---:|---|
| 🔴 Nghiêm trọng | 2 | Mất điểm nặng nếu giảng viên hỏi |
| 🟡 Cần sửa | 5 | Gây mâu thuẫn hoặc lỗi nhỏ |
| 🟢 Nên cải thiện | 5 | Polish, nâng tầm chuyên nghiệp |

---

## 🔴 NHÓM A — NGHIÊM TRỌNG (Ưu tiên 1 — Làm trước Thứ 2 4/5)

### A1. Thiếu Giải Thích Dự Đoán (Explainability)
- **Ai làm:** **Thái (ML)** viết logic Backend → **Đông (FE)** hiển thị.
- **Backend:** Thêm logic tính toán Feature Contribution vào `/predict`.
- **Frontend:** Vẽ biểu đồ thanh ngang nhỏ hiển thị 5 yếu tố ảnh hưởng nhất.

### A2. Thiếu Khoảng Tin Cậy (Confidence Interval)
- **Ai làm:** **Thái (ML)** thêm MAE vào response → **Đông (FE)** hiển thị.
- **Frontend:** Hiển thị `📊 Khoảng ước tính: 2.8 — 3.7 Tỷ (±Sai số trung bình)`.

---

## 🟡 NHÓM B — CẦN SỬA (Ưu tiên 2 — Làm trước Thứ 3 5/5)

### B1. Footer Ghi "Sắp Ra Mắt" Cho TP.HCM & Đà Nẵng
- **Vấn đề:** Mâu thuẫn vì thực tế web đã có dữ liệu 3 thành phố.
- **Sửa:** Xóa chữ `(Sắp ra mắt)` trong `_footer.html`.

### B2. Trang Analytics Chỉ Nói "Hà Nội"
- **Sửa:** Đổi text trong `analytics.html` thành "Việt Nam" hoặc "Hà Nội, TP.HCM & Đà Nẵng".

### B3. Trường "Hướng Ban Công" Là Ghost Field
- **Vấn đề:** Form có nhập nhưng AI không dùng.
- **Sửa:** Xóa trường này khỏi `index.html`.

### B4. Default Bias Trong Form Dự Đoán
- **Vấn đề:** Mặc định "Sổ đỏ", "Đầy đủ nội thất" làm giá luôn bị cao.
- **Sửa:** Đổi mặc định thành "-- Chọn --" hoặc "Không rõ".

### B5. Code Chuyển Trang Trùng Lặp
- **Sửa:** Xóa đoạn JS chuyển trang thừa trong các file template (đã có trong `script.js`).

---

## 🟢 NHÓM C — NÊN CẢI THIỆN (Ưu tiên 3)

1. **Validation trực quan:** Báo lỗi nếu để trống diện tích.
2. **Analytics nhà đất:** Thêm biểu đồ pháp lý/nội thất cho Nhà đất (hiện chỉ có Chung cư).
3. **Thống nhất hướng nhà:** Sửa dấu gạch ngang `"Đông - Nam"` vs `"Đông Nam"`.
4. **requirements.txt:** Thêm `Werkzeug>=3.0`.
5. **Data loading:** Sửa lỗi load model 2 lần khi khởi động Flask.

---

## 📅 LỊCH TRÌNH PHÂN CÔNG

| Thành viên | Việc cần làm |
|---|---|
| **Thái (ML)** | Tính Feature Contribution, MAE, **Cập nhật Notebook 03/04**. |
| **Đông (FE)** | Hiển thị biểu đồ XAI, Khoảng tin cậy, Sửa UI/UX, Xóa field thừa. |
| **Tài (Leader)** | Review code, Sửa lỗi text, Xóa code thừa, Quản lý Git. |
| **Đức (DB)** | Tích hợp SQLite để lưu lịch sử và tài khoản. |

---

## 💡 Ghi chú cho Thái (ML) về Notebooks:
Thái **bắt buộc** phải cập nhật lại Notebook `03_training.ipynb` và `04_evaluation.ipynb` để:
1. Đồng bộ kết quả sai số (MAE) với con số hiển thị trên Web.
2. Lưu lại các biểu đồ Feature Importance (yếu tố quan trọng) để đưa vào báo cáo, khớp với phần giải thích trên giao diện web.
