# 🎓 CẨM NANG DEMO & VẤN ĐÁP MỔ XẺ SOURCE CODE BÀI TẬP LỚN
**Môn học:** Phát triển Hệ thống Thông minh (Intelligent Systems Development)
**Dự án:** ProphetEstate - AI Dự đoán Giá Bất động sản
**Hình thức:** Nộp Báo cáo Lý thuyết + Chạy Demo thực tế + GV mổ xẻ hỏi sâu vào Hệ thống

---

## PHẦN 1: ĐÁNH GIÁ TỔNG QUAN (REVIEW)

### 1. Dưới góc độ Kỹ sư Machine Learning (MLE)
*Một dự án AI thực tế không chỉ nằm ở file Notebook, mà ở khả năng đưa model vào production. Hệ thống của bạn làm rất tốt điều này.*

**Ưu điểm vượt trội (Điểm cộng lớn):**
- **Data Pipeline bài bản:** Việc tách rõ quá trình từ Crawl dữ liệu -> Tiền xử lý (Xử lý Outlier, điền khuyết Missing Values) -> Huấn luyện model là chuẩn quy trình CRISP-DM thực tế.
- **Sử dụng `sklearn.Pipeline`:** Đây là minh chứng rõ nhất của sự chuyên nghiệp. Khác với sinh viên thường gọi `fit_transform` riêng lẻ, việc bạn gói `ColumnTransformer` (OneHotEncoder + StandardScaler) cùng `XGBRegressor` vào 1 file `.pkl` duy nhất giúp backend không bị rò rỉ dữ liệu (Data Leakage) và code web rất gọn gàng.
- **Explainable AI (XAI):** Rất ít dự án sinh viên làm được phần này. Việc trích xuất `feature_importance` để vẽ biểu đồ và hiển thị mức độ ảnh hưởng của từng đặc trưng giúp AI không còn là một "hộp đen" (Black box).
- **Chỉ số Đánh giá Thực tế:** Tính toán MAE (Mean Absolute Error) để tạo ra "Khoảng tin cậy" (Confidence Interval) $\pm 2.48$ tỷ là một cách tiếp cận cực kỳ khoa học, giúp người dùng cuối hiểu được sai số của hệ thống.

**Điểm hạn chế (Trade-off chấp nhận được):**
- Dữ liệu Crawl từ Web đôi khi phụ thuộc vào người bán (nhập sai giá, ảo giá), dẫn đến model có R² ~ 0.78 (không quá cao nhưng thực tế với ngành BĐS).
- Việc xử lý Missing Values bằng Median là an toàn nhưng có thể dùng các phương pháp nâng cao hơn (ví dụ dùng KNN Imputer) để chính xác hơn.

### 2. Dưới góc độ Giảng viên đánh giá
*Chấm điểm dựa trên tính trọn vẹn, logic nghiệp vụ và trải nghiệm người dùng (UX).*

**Điểm mạnh:**
- **Giải quyết bài toán thực tế:** Bất động sản là bài toán kinh điển nhưng khó. Nhóm không chỉ dự đoán 1 loại, mà xử lý cả 2 luồng (Chung cư & Nhà đất) với bộ features khác biệt (Nhà đất có Mặt tiền, Đường rộng). Thể hiện tư duy phân tích nghiệp vụ tốt.
- **UI/UX xuất sắc:** Khác biệt hoàn toàn với các đồ án chỉ có giao diện Gradio/Streamlit sơ sài. Giao diện Web (SaaS) mượt mà, Pop-up Modal, Validation báo lỗi đỏ, hiệu ứng loading, tất cả tạo cảm giác đây là một "Sản phẩm thương mại" (Commercial Product) chứ không chỉ là bài tập.
- **Feature "Bán hàng" rất đỉnh:** Việc đưa ra "Giá Trung Bình Khu Vực" và gợi ý "10-15 Căn nhà tương tự" chứng tỏ nhóm hiểu rằng người dùng không chỉ cần 1 con số, họ cần **ngữ cảnh** để ra quyết định đầu tư.

**Đánh giá tổng thể:** Dự án hoàn toàn xứng đáng đạt mức **9.0 - 9.5/10** (Thậm chí 10 nếu kỹ năng trình bày bảo vệ trôi chảy).

---

## PHẦN 2: HƯỚNG PHÁT TRIỂN THÊM (FUTURE WORK)
*(Dùng để trả lời câu hỏi: "Nếu có thêm thời gian, em sẽ phát triển hệ thống này như thế nào?")*

1. **Về mặt Dữ liệu (Data):**
   - **Bổ sung dữ liệu không gian (Geospatial Data):** Tích hợp tọa độ (Kinh độ/Vĩ độ) để tính khoảng cách đến trung tâm, bệnh viện, trường học. (Giá BĐS phụ thuộc rất lớn vào khoảng cách).
   - **Xây dựng Data Warehouse/Database:** Thay vì dùng file `.csv` tĩnh, sẽ chuyển sang dùng CSDL (như SQL Server/PostgreSQL) và tự động chạy Cron-job thu thập dữ liệu hàng tuần để cập nhật model.
2. **Về mặt Mô hình (Model):**
   - **Tối ưu hóa Hyperparameters:** Sử dụng `GridSearchCV` hoặc `Optuna` để dò tìm bộ tham số tốt nhất cho XGBoost thay vì setup thủ công.
   - **XAI nâng cao:** Sử dụng thư viện **SHAP** (SHapley Additive exPlanations) thay vì chỉ dùng Feature Importance mặc định để giải thích chính xác từng dự đoán một cách cục bộ (Local explanation).
3. **Về mặt Hệ thống (System):**
   - **Lưu trữ lịch sử:** Cho phép người dùng đăng nhập, lưu lại các dự đoán, biến nó thành công cụ theo dõi biến động tài sản.
   - **Deploy thực tế:** Đưa ứng dụng lên AWS, Heroku hoặc Render để người dùng có thể truy cập qua Internet.

---

## PHẦN 3: KỊCH BẢN DEMO "ĂN ĐIỂM" & BỘ CÂU HỎI MỔ XẺ SOURCE CODE

Vì hình thức thi là Show bài tập + GV hỏi xoáy vào dự án, điểm số phụ thuộc 50% vào việc Demo có trơn tru không và 50% vào việc hiểu code đến đâu. 

### 🌟 Bước 1: Kịch bản Demo thực tế (Dẫn dắt GV)
Đừng đợi GV hỏi mới bật web lên. Hãy chủ động dẫn dắt luồng suy nghĩ của GV:
1. **Mở đầu bằng Analytics:** "Thưa thầy, hệ thống của nhóm em không chỉ là một form nhập liệu dự đoán. Đầu tiên em xin phép show trang **Phân tích dữ liệu**. Đây là kết quả EDA trên 11.831 bản ghi thật (5.452 Chung cư + 6.379 Nhà đất) nhóm cào từ Batdongsan.com.vn, được chia rõ ràng thành 2 luồng: Chung cư và Nhà đất." *(Chỉ cho GV thấy các biểu đồ pháp lý, nội thất, xếp hạng giá).*
2. **Demo Dự đoán có logic:** Nhập 1 ví dụ cụ thể, ví dụ: *Chung cư Cầu Giấy, 2 PN, 75m2, Nội thất đầy đủ*. Bấm dự đoán.
3. **Khoe ngay tính năng XAI (Explainable AI):** Khi pop-up hiện lên, **ngay lập tức** chỉ vào phần *Phân tích mức độ ảnh hưởng*. Nói luôn: *"Hệ thống của em không chỉ nhả ra 1 con số vô hồn. Tụi em dùng thuật toán để trích xuất độ quan trọng (Feature Importance) từ mô hình XGBoost, cho người dùng biết tại sao lại ra giá đó (ví dụ do yếu tố Vị trí Cầu Giấy và Diện tích đẩy giá lên)."*
4. **Khoe Khoảng Tin Cậy (Confidence Interval):** *"Và tụi em biết AI không thể đúng 100%, nên em có đưa vào Khoảng Sai Số $\pm 2.48$ tỷ dựa trên chỉ số MAE lúc train model để cảnh báo rủi ro cho người dùng."*

### 🔪 Bước 2: Chuẩn bị cho màn "Mổ xẻ Source Code"
Giảng viên sẽ yêu cầu mở code ra (VS Code hoặc Notebook) và hỏi thẳng vào luồng dữ liệu. Dưới đây là các câu "tủ":

**Q1: "Mở code chỗ train model ra. Sao dùng XGBoost? Và chỗ nào lưu model lại?"**
- **Mở ngay:** File `notebooks/03_training.ipynb` (Section 7 & 9) HOẶC `run_training.py`.
- **Trả lời:** Em dùng XGBoost vì dữ liệu BĐS có tính phi tuyến tính cao và nhiều biến phân loại (Categorical). Em lưu model bằng `pickle` thành file `best_model_pipeline.pkl`. 

**Q2: "Mở app.py lên. Web lấy model lúc nào? Dùng model như thế nào?"**
- **Mở ngay:** File `app.py` dòng 120-140 (Chỗ `load_model()`).
- **Trả lời:** Em load model **1 lần duy nhất** khi Flask khởi động (Singleton Pattern) để tiết kiệm RAM. Khi người dùng submit, em bắt API `/predict`, gói input thành DataFrame và gọi thẳng `pipeline.predict(input)`.
- **Điểm ăn tiền:** Nhấn mạnh chữ **Pipeline**. *"Thưa thầy, nhờ em lưu nguyên Pipeline (gồm cả StandardScaler và OneHotEncoder) nên ở app.py em không cần viết code scale tay hay encode tay nữa, tránh bị lỗi Data Leakage ạ."*

**Q3: "Cái biểu đồ độ quan trọng (XAI) trên web vẽ bằng cái gì? Dữ liệu lấy ở đâu hay là fake?"**
- **Mở ngay:** File `run_training.py` (Dòng 115) HOẶC `03_training.ipynb` (Section 9).
- **Trả lời:** Dạ không fake ạ. Ngay sau khi train xong XGBoost, em trích xuất `base_model.feature_importances_`. Sau đó em map với các tên biến sau khi đã OneHot, rồi lưu toàn bộ vào 1 file tên là `model_meta.pkl`. Ở `app.py` em đọc file meta này, kết hợp với Input người dùng để chọn ra 4 yếu tố tác động mạnh nhất đưa lên biểu đồ.

**Q4: "Dữ liệu ban đầu toàn chữ với lỗi, em làm sạch kiểu gì?"**
- **Mở ngay:** `02_preprocessing.ipynb` (Bước 1 & 3 & 4).
- **Trả lời:** Đầu tiên em quy chuẩn giá (triệu/m2 sang Tỷ). Sau đó những trường bị trống như Phòng ngủ em fill bằng giá trị **Median (Trung vị) của chính Quận đó** (chứ không phải trung vị của toàn tập). Những nhà diện tích vô lý (<15m2) hoặc phòng ngủ vô lý (>25) em loại bỏ hoàn toàn bằng logic nghiệp vụ BĐS.

**Q5: "Có sợ bị Overfitting không?"**
- **Mở ngay:** `04_evaluation.ipynb` (Bước 2 & 3 - Có thể show hình).
- **Trả lời:** Em chia tập Train/Test theo tỷ lệ 80:20 (random_state=42). Kết quả đánh giá trên tập Test hoàn toàn độc lập với tập Train. R-squared trên tập Test đạt 78%, RMSE/MAE tương đối ổn định nên mô hình hoàn toàn có sức mạnh tổng quát hóa (Generalization) ạ.

### 💡 Lời Khuyên Cuối Cùng
Hãy nhớ nguyên tắc: **Tự tin, Không cãi, và Dẫn chứng bằng Code**. 
Nếu GV chê mô hình độ chính xác chưa cao, hãy mỉm cười đồng ý: *"Dạ vâng, dữ liệu BĐS cào từ web rất nhiều tin ảo. Điểm R2 78% là ngưỡng thực tế. Thay vì cố push nó lên 99% bằng cách Overfitting, tụi em tập trung vào việc **Giải thích AI (XAI)** và cung cấp **Khoảng tin cậy** để ứng dụng có tính thực tiễn cao nhất ạ."*
