# Dự đoán giá nhà bằng Transformer

Thư mục này là phần thử nghiệm Transformer chạy riêng, chưa thay đổi web Flask cũ.

## Ý tưởng

Luồng xử lý:

```text
Dữ liệu nhà đất cũ
  -> dọn dữ liệu
  -> chia train / validation / test
  -> cột số được chuẩn hóa
  -> cột chữ được đổi thành mã và embedding
  -> Transformer học quan hệ giữa các thông tin
  -> dự đoán giá nhà
  -> tính MAE, RMSE, R2, MAPE
```

Giải thích ngắn:

- Cột số như diện tích, phòng ngủ, số tầng được đổi bằng công thức `(x - mean) / std`.
- Cột chữ như quận, hướng, pháp lý được đổi thành mã số rồi qua embedding.
- Train dùng để mô hình học.
- Validation dùng để chọn phiên bản mô hình tốt nhất.
- Test dùng cuối cùng để chấm điểm chính thức.

## Cách chạy

Cài thư viện:

```bash
pip install -r requirements.txt
```

Train Transformer:

```bash
python transformer/train_transformer.py
```

Dự đoán thử một căn nhà:

```bash
python transformer/predict_transformer.py --area-m2 80 --bedrooms-num 3 --district "Cầu Giấy" --floors-num 4 --frontage-m 5 --road-width-m 6 --loai-bds nha_dat
```

## File kết quả

Sau khi train, kết quả nằm trong `transformer/outputs/`:

```text
transformer_model.pt              model Transformer tốt nhất
preprocessing.pkl                 scaler và bảng mã category
transformer_meta.pkl/json         số liệu MAE, RMSE, R2, MAPE
training_loss.png                 biểu đồ train/validation MSE
prediction_scatter.png            giá thật vs giá dự đoán
error_distribution.png            phân bố sai số
comparison_with_xgboost.csv       bảng so sánh với model cũ
transformer_vs_xgboost.png        biểu đồ so sánh với model cũ
```

## Câu giải thích khi bảo vệ

Nhóm em xây dựng thêm mô hình Transformer cho dữ liệu bảng. Các cột số được chuẩn hóa về cùng thang đo, còn các cột chữ như quận, hướng và pháp lý được mã hóa thành embedding. Sau đó Transformer học mối quan hệ giữa các thông tin của căn nhà để dự đoán giá. Mô hình được đánh giá bằng MAE, RMSE, R2 và MAPE, rồi so sánh trực tiếp với XGBoost cũ.
