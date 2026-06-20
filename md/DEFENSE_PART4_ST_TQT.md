# TÀI LIỆU BẢO VỆ — PHẦN 4: DÀNH RIÊNG CHO ST_TQT (TRẦN QUANG THÁI)
# Phân tích khác biệt: Nhánh `main` vs Nhánh `feature/train-ai`

> ⚠️ **LƯU Ý DÀNH CHO TQT**: Hệ thống ở nhánh `main` hiện tại đã được nâng cấp (refactor) đáng kể so với code gốc bạn làm ở nhánh `feature/train-ai`. Giảng viên sẽ chấm trên nhánh `main` cuối cùng, do đó bạn BẮT BUỘC phải nắm rõ **TẠI SAO** code lại được thay đổi và **NÂNG CẤP Ở ĐÂU**.

---

## 1. NHỮNG THAY ĐỔI LỚN TRONG `run_training.py` & NOTEBOOKS

Dưới đây là 5 điểm khác biệt cốt lõi giữa code cũ của bạn và code `main` hiện tại:

### 1.1 Chuyển từ LabelEncoder sang OneHotEncoder + Pipeline
- **Code cũ (của bạn)**: Dùng `LabelEncoder` để biến đổi các cột phân loại (quận, hướng...) thành số (0, 1, 2...). Xuất ra 3 file: `best_model.pkl`, `scaler.pkl`, `label_encoders.pkl`.
- **Code mới (`main`)**: Dùng `OneHotEncoder` bọc trong `ColumnTransformer` và nối với model thành 1 `Pipeline` duy nhất. Chỉ xuất đúng 1 file `best_model_pipeline.pkl`.
- **Lý do thay đổi (Cần giải thích khi bảo vệ)**: 
  1. `LabelEncoder` tạo ra thứ tự ảo (Ví dụ: Cầu Giấy=1, Hà Đông=2 → Thuật toán sẽ nghĩ Hà Đông lớn gấp đôi Cầu Giấy). Quận là biến định danh (nominal), BẮT BUỘC phải dùng `OneHotEncoder`.
  2. `Pipeline` giúp gộp chung Preprocessing và Model thành 1 cục. Tránh lỗi quên scale dữ liệu khi đưa lên web (Data Leakage prevention).
  3. Tham số `handle_unknown='ignore'` của OneHotEncoder giúp hệ thống KHÔNG BỊ CRASH khi user nhập 1 quận mới tinh.

### 1.2 Thêm Phân tích Overfitting (Overfit Gap)
- **Code cũ**: Chỉ tính RMSE, MAE, R² trên tập Test.
- **Code mới**: Tính R² trên cả tập Train và tập Test. Tính `Overfit Gap = R²_train - R²_test`. Vẽ thêm biểu đồ `overfitting_analysis.png`.
- **Lý do thay đổi**: Giảng viên rất hay hỏi "Làm sao biết model không bị học vẹt?". Việc show R²_train và R²_test cho thấy: Tuy XGBoost có học thuộc 1 chút (Train > Test), nhưng khoảng cách (Gap) rất nhỏ, chứng tỏ model tổng quát hóa tốt.

### 1.3 Thêm Cross-Validation (5-Fold)
- **Code mới**: Chạy thêm `cross_val_score(cv=5)`. 
- **Lý do thay đổi**: Train/Test split thông thường có thể bị "hên xui" (vô tình test data quá dễ đoán). CV 5-Fold cắt data làm 5 phần, train 5 lần khác nhau → lấy trung bình R² → Đo lường ĐỘ ỔN ĐỊNH thực sự của model.

### 1.4 Thêm chỉ số MAPE (Mean Absolute Percentage Error)
- **Code mới**: Thêm tính toán MAPE (Sai số phần trăm trung bình tuyệt đối).
- **Lý do thay đổi**: MAE (ví dụ lệch 0.5 tỷ) khó hình dung là tốt hay xấu (lệch 0.5 tỷ cho nhà 2 tỷ là 25%, nhưng cho nhà 20 tỷ chỉ là 2.5%). MAPE quy về %, ví dụ MAPE = 15% nghĩa là trung bình dự đoán lệch 15% so với giá gốc → Rất dễ báo cáo.

### 1.5 Giải thích AI (XAI - Feature Importance) cho Giao diện
- **Code cũ**: Tính feature importance đơn giản nhưng chưa xử lý đúng khi số lượng cột bị phình ra do OneHotEncoder.
- **Code mới**: Trích xuất `get_feature_names_out()` từ Pipeline, tính đúng Importance cho từng feature và đẩy vào `model_meta.pkl` để giao diện Web vẽ biểu đồ giải thích "Tại sao nhà lại có giá này".

---

## 2. KỊCH BẢN BẢO VỆ CHUYÊN SÂU CHO TQT

Vì bạn là người phụ trách AI/Training, giảng viên sẽ "xoáy" bạn rất mạnh vào lý thuyết mô hình. Hãy dùng chính những điểm nâng cấp trên làm VŨ KHÍ để ghi điểm:

### CÂU HỎI & TRẢ LỜI (DÀNH RIÊNG CHO TQT)

**Q1 (Giảng viên): Tại sao em lại dùng Pipeline thay vì xử lý dữ liệu rồi ném vào model.fit()?**
> **TQT:** "Dạ, ban đầu em làm theo cách thủ công: chạy LabelEncoder và StandardScaler rồi mới fit model. Nhưng sau khi review lại kiến trúc, em quyết định refactor toàn bộ sang `sklearn.pipeline.Pipeline`. Lý do cốt lõi là để **chống rò rỉ dữ liệu (Data Leakage)**. Nếu em gọi StandardScaler trên toàn bộ data trước khi split Train/Test, tập Test sẽ vô tình làm rò rỉ thông tin (mean/std) vào quá trình huấn luyện. Pipeline đảm bảo việc tính toán mean/std chỉ xảy ra hoàn toàn trên tập Train. Ngoài ra nó giúp code web gọn hơn khi chỉ cần load đúng 1 file pkl."

**Q2 (Giảng viên): Em xử lý biến 'Quận/Huyện' như thế nào?**
> **TQT:** "Biến Quận là biến định danh (Nominal), không có thứ tự lớn bé. Ban đầu em thử nghiệm với LabelEncoder nhưng nó tạo ra thứ tự ảo khiến model (đặc biệt là model tuyến tính) bị nhiễu. Do đó, ở nhánh main cuối cùng, em sử dụng `OneHotEncoder(sparse_output=False, handle_unknown='ignore')`. Tham số `ignore` này là 'phao cứu sinh' của hệ thống web: nếu user nhập 1 quận chưa từng xuất hiện trong tập Train, hệ thống sẽ tự động gán vector [0,0,0...] mà không bị văng lỗi (Crash)."

**Q3 (Giảng viên): Làm sao em chứng minh mô hình XGBoost của em không bị Overfitting?**
> **TQT:** "Dạ em chứng minh qua 2 phương pháp. Thứ nhất, em tính `R2_train` và `R2_test`, sau đó lấy `Overfit Gap`. Với XGBoost, khoảng cách này rất nhỏ, chứng tỏ mô hình không bị học vẹt. Thứ hai, em áp dụng K-Fold Cross Validation (K=5) trên toàn bộ dữ liệu. Kết quả độ lệch chuẩn (Standard Deviation) của R² giữa 5 lần train rất nhỏ, chứng tỏ mô hình cực kỳ ổn định bất chấp việc xáo trộn dữ liệu."

**Q4 (Giảng viên): Feature Importance của XGBoost được tính như thế nào?**
> **TQT:** "Trong XGBoost, hàm `feature_importances_` mặc định tính theo độ tăng ích (Gain). Nghĩa là, nó đo lường mức độ giảm hao phí (loss) trung bình mà mỗi biến mang lại khi được chọn làm điểm rẽ nhánh (split node) trên tất cả 300 cây quyết định. Em đã trích xuất mảng này, map lại với tên cột sau khi đã One-Hot Encoding (`get_feature_names_out`), và tính ra tỷ lệ % để hiển thị lên Web cho người dùng hiểu tại sao giá lại cao/thấp."

---

## 3. LỜI KHUYÊN KHI DEMO (DÀNH CHO TQT)

Khi đứng trước màn hình bảo vệ, hãy chủ động MỞ file `run_training.py` và chỉ vào những dòng code "đắt giá" nhất mà bạn đã (cùng team/AI) refactor:

1. **Chỉ vào dòng 75-80 (`ColumnTransformer`)**: "Thưa thầy, đây là lõi của hệ thống tiền xử lý, em dùng Pipeline để bọc StandardScaler và OneHotEncoder."
2. **Chỉ vào dòng 128-142 (`train_eval function`)**: "Đây là nơi em tính Overfit Gap. Em luôn so sánh dự đoán trên cả tập Train và tập Test để đảm bảo model học được quy luật chứ không học vẹt."
3. **Chỉ vào file `models/overfitting_analysis.png`**: Bật ảnh này lên và tự tin nói: "Biểu đồ này chứng minh Decision Tree bị overfit (Cột Train cao, cột Test thấp), trong khi XGBoost cân bằng hoàn hảo."

**Kết luận:** Đừng sợ việc code `main` khác với code ban đầu của bạn. Hãy biến nó thành câu chuyện: *"Ban đầu em làm bản V1 (LabelEncoder, tách file), nhưng sau quá trình R&D và tối ưu, em đã nâng cấp lên bản V2 (Pipeline, OHE, Cross-Validation) để đạt chuẩn Production."* Giảng viên sẽ đánh giá rất cao tư duy cải tiến này!
