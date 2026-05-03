# Thiết kế Cơ sở dữ liệu cho Dự án

## Tổng quan
Sử dụng SQLite + SQLAlchemy. SQLite lưu dữ liệu thành một file duy nhất (ví dụ `app.db`), không cần cài đặt server phức tạp. SQLAlchemy là ORM (Object-Relational Mapping), giúp thao tác với database bằng code Python thay vì viết câu lệnh SQL thuần.

## Các bảng (Tables)

### 1. Bảng `User` (Người dùng)
Lưu trữ thông tin người dùng đăng ký, đăng nhập.
- `id` (Integer, Primary Key)
- `email` (String, Unique, Not Null)
- `password_hash` (String, Nullable - vì có thể login bằng Google)
- `name` (String, Not Null)
- `phone` (String, Nullable)
- `gender` (String, Nullable)
- `dob` (Date, Nullable)
- `avatar` (String, Nullable)
- `provider` (String, Default 'local' - để biết login bằng web hay google)
- `created_at` (DateTime, Default = Now)

### 2. Bảng `PredictionHistory` (Lịch sử dự đoán)
Lưu lại các lần người dùng đã thực hiện dự đoán.
- `id` (Integer, Primary Key)
- `user_id` (Integer, Foreign Key -> User.id)
- `property_type` (String) - chung_cu hoặc nha_dat
- `district` (String)
- `area` (Float)
- `bedrooms` (Integer)
- `price_vnd` (Float)
- `price_billion` (Float)
- `input_data` (JSON) - Lưu toàn bộ tham số khác (hướng, pháp lý...)
- `created_at` (DateTime, Default = Now)

### 3. Bảng `SavedProperty` (Bất động sản đã lưu)
Lưu các căn nhà (từ CSV) mà người dùng bấm yêu thích/lưu lại.
- `id` (Integer, Primary Key)
- `user_id` (Integer, Foreign Key -> User.id)
- `property_type` (String) - chung_cu hoặc nha_dat
- `property_id` (Integer) - ID tương ứng trong file CSV
- `created_at` (DateTime, Default = Now)

## Kế hoạch triển khai (Implementation Plan)
1. Cài đặt thư viện `Flask-SQLAlchemy`
2. Tạo file `app/models.py` chứa định nghĩa các bảng
3. Cập nhật `app/app.py` để kết nối SQLite
4. Thay thế các biến bộ nhớ (`USERS`, `PREDICTION_HISTORY`, `SAVED_PROPERTIES`) bằng các câu truy vấn Database.
