# PHẦN CỦA TÀI — CRAWLER + NOTEBOOK 01 (EDA) + 02 (PREPROCESSING)

## BẠN LÀM GÌ TRONG PROJECT NÀY?

Bạn phụ trách **đầu vào của toàn bộ hệ thống** — không có phần bạn thì không có gì để train, không có gì để predict. Cụ thể:

1. **Crawler** (`data/crawl/crawler.py`) — Cào 14.015 bản ghi thô từ batdongsan.com.vn
2. **02_preprocessing.ipynb** — Làm sạch → 11.831 bản ghi sạch
3. **01_eda.ipynb** — Phân tích khám phá, 5 biểu đồ dẫn đến quyết định kiến trúc

---

## CRAWLER — CÁC CÂU GV SẼ HỎI

### "Em cào dữ liệu bằng cách nào?"

> Em tự viết crawler bằng Python, cào trực tiếp từ batdongsan.com.vn — sàn giao dịch BĐS lớn nhất Việt Nam. Mỗi tin đăng em lấy 25 cột thông tin: giá, diện tích, phòng ngủ, quận, hướng nhà, nội thất, pháp lý, ảnh, mô tả...

### "Trang web có chặn không? Xử lý thế nào?"

> **Có.** batdongsan.com.vn dùng Cloudflare Bot Management — nó chặn mọi thư viện HTTP thông thường như `requests`, `urllib3` vì phát hiện TLS fingerprint không phải trình duyệt thật.
>
> Em dùng **`curl_cffi`** với tham số `impersonate="chrome"` — thư viện này giả lập hoàn toàn TLS fingerprint của Chrome (cipher suites, TLS extensions, HTTP/2 frame settings). Cloudflare không phân biệt được với người dùng thật.
>
> Ngoài ra em dùng **random delay 1.5-3 giây** giữa mỗi request để mô phỏng hành vi người thật. Nếu bị trả HTTP 429 (Too Many Requests) hoặc 403, crawler tự **chờ 20 giây + đổi session mới** rồi retry tối đa 3 lần.

### "Dữ liệu có bị trùng không? Xử lý sao?"

> Em chống trùng **2 lớp**:
> - **Lớp 1**: Set `seen_urls` — mỗi URL cào rồi thì bỏ qua
> - **Lớp 2**: Set `seen_ids` — mỗi listing_id (mã tin) đã lưu thì bỏ qua, dù URL khác
>
> Lý do cần 2 lớp: cùng 1 căn nhà có thể xuất hiện ở nhiều trang khác nhau với URL khác nhau, nhưng listing_id giống nhau.

### "Nếu crawler bị ngắt giữa chừng thì sao?"

> Em thiết kế **Resume mode**: khi khởi động, crawler đọc file CSV cũ, nạp lại tất cả listing_id và URL đã cào vào 2 set. Rồi tính `cần thêm = 15.000 - đã có` và chỉ cào phần thiếu. Không bao giờ mất dữ liệu cũ.
>
> Ngoài ra, mỗi bản ghi được **ghi vào CSV ngay lập tức** (không đợi hết vòng lặp). Nên dù bị ngắt đột ngột, dữ liệu đã cào vẫn còn.

### "Cào từ bao nhiêu danh mục? Tại sao?"

> **27 danh mục** — gồm 4 loại BĐS (chung cư, nhà riêng, biệt thự, đất nền) × nhiều tỉnh (Hà Nội, TP.HCM, Đà Nẵng, Bình Dương, Hải Phòng, Đồng Nai, Khánh Hòa, Bắc Ninh, Long An, Quảng Ninh). Mỗi danh mục cào tối đa 250 trang.
>
> Tại sao nhiều danh mục? Vì mỗi danh mục chỉ có ~50-100 trang kết quả. Nếu chỉ cào 1 danh mục sẽ không đủ 15.000 mẫu. Cần đa dạng về loại BĐS và địa lý để model học tổng quát.

---

## PREPROCESSING (02) — CÁC CÂU GV SẼ HỎI

### "Dữ liệu thô có vấn đề gì? Xử lý ra sao?"

> Dữ liệu thô có **4 vấn đề chính**:
>
> **1. Lỗi đảo cột (304 dòng):** Scraper đôi khi parse nhầm — giá tổng (ví dụ "3,5 tỷ") bị ghi vào cột `price_per_m2`, còn giá/m² ("52 triệu/m²") bị ghi vào cột `price`. Em phát hiện bằng cách kiểm tra: nếu cột `price` chứa chuỗi "triệu/m" → chắc chắn bị đảo → swap lại 2 cột.
>
> **Code:** `mask = df['price'].str.contains('triệu/m', na=False)` rồi swap values.
>
> **2. Tin không có giá (1.275 dòng):** Nhiều tin ghi "Giá thỏa thuận" — không có giá cụ thể → loại bỏ vì model cần target số.
>
> **3. Trùng lặp:** `drop_duplicates(subset='listing_id')` — loại 24 dòng trùng.
>
> **4. Dữ liệu text cần parse thành số:** Giá dạng "3,5 tỷ", "850 triệu", "52 triệu/m²" → cần hàm `parse_price()` chuyển tất cả về đơn vị tỷ VNĐ.

### "Hàm parse_price hoạt động thế nào?"

> Hàm xử lý 4 trường hợp:
> - `"3,5 tỷ"` → extract số 3.5 → trả 3.5
> - `"850 triệu"` → extract 850 → chia 1000 → trả 0.85
> - `"52 triệu/m²"` + area 60m² → 52 × 60 / 1000 → trả 3.12 (tỷ)
> - `"Thỏa thuận"` → trả None → bị loại

### "Phân loại Chung cư / Nhà đất bằng cách nào?"

> Dùng **regex** trên cột `property_type` (lấy từ breadcrumb trang web):
> - Chứa "chung cư" → Chung cư
> - Chứa "nhà riêng" hoặc "biệt thự" → Nhà đất
> - Không match → loại bỏ (chỉ 9 dòng, < 0.1%)

### "Quận có ít dữ liệu thì xử lý sao?"

> Quận có **< 30 bản ghi** thì gom vào nhóm "Khác". Lý do: nếu để nguyên, OHE (One-Hot Encoding) tạo ra 1 cột riêng cho quận đó, nhưng chỉ có vài mẫu → model không đủ dữ liệu để học, gây overfitting hoặc noise.
>
> Các quận bị gom: Hoàn Kiếm, Đan Phượng, Thạch Thất, Thường Tín, Thanh Oai (CC) + thêm vài quận nữa (NĐ).

### "Fill NaN phòng ngủ bằng median — tại sao không dùng mean?"

> **Median** vì phân phối phòng ngủ không đều — đa số căn hộ CC có 2-3 phòng ngủ, nhưng 1 số penthouse có 5-6. Mean bị kéo lệch bởi outlier, median ổn định hơn.
>
> Ngoài ra em fill theo **nhóm quận** (`groupby('district')`) trước, rồi mới dùng global median cho những quận toàn NaN. Lý do: căn hộ ở Ba Đình thường có 3PN, ở Gia Lâm thường 2PN — fill theo quận chính xác hơn fill 1 số cho tất cả.

### "Lọc outlier bằng ngưỡng cố định — tại sao không dùng IQR?"

> Em dùng ngưỡng dựa trên **domain knowledge**:
> - CC: area 20-300m², price 0.3-50 tỷ, bedrooms 1-6
> - NĐ: area 15-1000m², price 0.5-200 tỷ, bedrooms 1-25, road_width ≤ 50m
>
> Tại sao không IQR? Vì phân phối giá BĐS cực kỳ lệch phải (skewness ~3-4). IQR × 1.5 sẽ loại quá nhiều dữ liệu hợp lệ ở đuôi phải (những căn biệt thự đắt nhưng có thật). Ngưỡng cố định cho phép giữ lại các giá trị hợp lệ ở range cao.

### "Chuẩn hóa nội thất/pháp lý — tại sao cần?"

> Dữ liệu thô rất bẩn: nội thất có thể ghi "Full nội thất", "Đầy đủ", "NT cao cấp", "Cơ bản CĐT"... → em chuẩn hóa về **4 nhóm**: Đầy đủ / Cơ bản / Không nội thất / Không rõ.
>
> Tương tự pháp lý: "Sổ đỏ", "Sổ hồng", "Có sổ", "Sẵn sổ" → tất cả gộp thành **"Sổ đỏ/Sổ hồng"**. Giúp OHE tạo ít cột hơn và model học hiệu quả hơn.

---

## EDA (01) — CÁC CÂU GV SẼ HỎI

### "EDA phát hiện được gì quan trọng?"

> **3 phát hiện dẫn đến quyết định kiến trúc:**
>
> 1. **Boxplot**: Phân phối giá CC và NĐ hoàn toàn khác (CC median ~7.2 tỷ, NĐ ~12.95 tỷ)
> 2. **Scatter plot**: 2 cluster rõ ràng trên mặt phẳng (area, price) — không overlap
> 3. **Heatmap**: Bộ features quan trọng khác nhau — NĐ có floors (r=0.386) và frontage (r=0.202), CC không có
>
> → Quyết định: CC và NĐ cần được phân biệt. Em tạo cột `loai_bds` để model biết đang xử lý loại nào.

### "Skewness 2.75 và 3.98 — nghĩa là gì?"

> Skewness đo **độ lệch** của phân phối. Skewness = 0 là đối xứng (chuẩn). Skewness > 1 là lệch phải (đuôi dài bên phải).
>
> CC=2.75, NĐ=3.98 → phân phối giá lệch phải nặng: đa số căn hộ giá 3-8 tỷ, nhưng 1 số penthouse/biệt thự có giá rất cao (50-100+ tỷ) tạo đuôi dài.
>
> Hệ quả: Linear Regression bị ảnh hưởng nhiều (giả định phân phối chuẩn), XGBoost xử lý tốt hơn nhờ tree-based splitting không phụ thuộc phân phối.

### "Heatmap tương quan nói lên điều gì?"

> - CC: `area_m2` có tương quan cao nhất với giá (r=0.667). Phòng ngủ cũng liên quan (r=0.437).
> - NĐ: `area_m2` vẫn quan trọng nhất (r=0.520), nhưng thêm `floors_num` (r=0.386) và `frontage_m` (r=0.202) — 2 feature này chỉ có ý nghĩa với nhà đất, không tồn tại trong chung cư.
>
> → Bằng chứng từ dữ liệu: CC và NĐ cần bộ features khác nhau.

---

## TÓM TẮT — PHẢI NHỚ

| Bạn cần nhớ | Số |
|-------------|-----|
| Bản ghi thô | **14.015** (9.329 từ batch đầu + thêm từ mở rộng) |
| Lỗi đảo cột | **304 dòng** (output notebook ghi 304) |
| Tin không có giá | **1.275 dòng** loại bỏ |
| CC sau tiền xử lý | **4.494** (notebook) → **5.452** (sau merge thêm data) |
| NĐ sau tiền xử lý | **3.381** (notebook) → **6.379** (sau merge thêm data) |
| Quận gom vào "Khác" | < 30 mẫu |
| Nội thất | 4 nhóm: Đầy đủ / Cơ bản / Không nội thất / Không rõ |
| Pháp lý | 4 nhóm: Sổ đỏ/Sổ hồng / HĐMB / Đang chờ sổ / Khác |
| Skewness CC / NĐ | **2.75 / 3.98** |
| Tương quan area-price CC | **r = 0.667** |

> **Mẹo:** Khi GV hỏi, luôn bắt đầu bằng **vấn đề** rồi mới nói **giải pháp**. Ví dụ: "Dữ liệu có 304 dòng bị đảo cột vì scraper parse nhầm — em phát hiện bằng cách kiểm tra chuỗi 'triệu/m' trong cột price". Đừng chỉ nói "em dùng mask swap" — GV muốn thấy bạn hiểu **tại sao** chứ không phải **cú pháp**.
