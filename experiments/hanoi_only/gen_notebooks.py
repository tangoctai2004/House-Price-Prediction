"""Generate Hanoi-only notebooks from original notebooks."""
import json, copy, sys, os
try:
    sys.stdout.reconfigure(encoding='utf-8')
except: pass
sys.path.insert(0, os.path.dirname(__file__))

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
NB_DIR = os.path.join(BASE, 'notebooks')
OUT_DIR = os.path.dirname(__file__)

def read_nb(name):
    with open(os.path.join(NB_DIR, name), 'r', encoding='utf-8') as f:
        return json.load(f)

def save_nb(nb, name):
    path = os.path.join(OUT_DIR, name)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"  Saved: {path}")

def clear_outputs(nb):
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
    return nb

def replace_in_source(cells, old, new):
    for cell in cells:
        cell['source'] = [line.replace(old, new) for line in cell['source']]

def make_cell(ctype, source_lines):
    c = {"cell_type": ctype, "metadata": {}, "source": source_lines}
    if ctype == "code":
        c["outputs"] = []
        c["execution_count"] = None
    return c

# ============ 03_training_hanoi.ipynb ============
print("Generating 03_training_hanoi.ipynb...")
nb3 = read_nb('03_training.ipynb')
nb3 = clear_outputs(nb3)

# Replace title
nb3['cells'][0] = make_cell("markdown", [
    "# \U0001f3e0 BƯỚC 3: HUẤN LUYỆN MÔ HÌNH AI — HÀ NỘI\n",
    "**Dự án:** Dự đoán Giá Bất Động Sản Hà Nội  \n",
    "**Mô tả:** Train 4 thuật toán ML trên dữ liệu **chỉ Hà Nội**, sử dụng `sklearn.Pipeline`\n",
    "\n",
    "> **Phiên bản:** Hanoi-only experiment  \n",
    "> **Dữ liệu:** Lọc district thuộc Hà Nội từ dataset tổng"
])

# Insert Hanoi districts + filter after load data cell (cell index ~3 in original = section 3)
hanoi_filter_cell = make_cell("code", [
    "# Danh sách quận/huyện Hà Nội\n",
    "HANOI_DISTRICTS = [\n",
    "    'Ba Đình', 'Hoàn Kiếm', 'Tây Hồ', 'Long Biên', 'Cầu Giấy',\n",
    "    'Đống Đa', 'Hai Bà Trưng', 'Hoàng Mai', 'Thanh Xuân',\n",
    "    'Hà Đông', 'Bắc Từ Liêm', 'Nam Từ Liêm',\n",
    "    'Sơn Tây', 'Ba Vì', 'Chương Mỹ', 'Đan Phượng', 'Đông Anh',\n",
    "    'Gia Lâm', 'Hoài Đức', 'Mê Linh', 'Mỹ Đức', 'Phú Xuyên',\n",
    "    'Phúc Thọ', 'Quốc Oai', 'Sóc Sơn', 'Thạch Thất', 'Thanh Oai',\n",
    "    'Thanh Trì', 'Thường Tín', 'Ứng Hòa',\n",
    "]\n",
    "\n",
    "# Lọc chỉ giữ dữ liệu Hà Nội\n",
    "df_all['district'] = df_all['district'].astype(str).str.strip()\n",
    "df_all = df_all[df_all['district'].isin(HANOI_DISTRICTS)].copy()\n",
    "print(f'\\n📍 Dataset Hà Nội: {df_all.shape[0]} bản ghi')\n",
    "print(f'Phân bổ loại BĐS:')\n",
    "print(df_all['loai_bds'].value_counts())\n",
    "print(f'\\nCác quận/huyện có dữ liệu:')\n",
    "print(', '.join(df_all['district'].value_counts().index.tolist()))"
])

hanoi_filter_md = make_cell("markdown", [
    "### Lọc dữ liệu chỉ Hà Nội\n",
    "Lọc các dòng có `district` thuộc danh sách quận/huyện Hà Nội."
])

# Find the outlier filter cell (contains 'Sau khi lọc outlier') and insert after it
insert_idx = None
for i, cell in enumerate(nb3['cells']):
    src = ''.join(cell['source'])
    if 'Sau khi lọc outlier' in src:
        insert_idx = i + 1
        break

if insert_idx:
    nb3['cells'].insert(insert_idx, hanoi_filter_md)
    nb3['cells'].insert(insert_idx + 1, hanoi_filter_cell)

# Update save paths: ../models -> ./models_hanoi
replace_in_source(nb3['cells'], "'../models/best_model_pipeline.pkl'", "'./models_hanoi/best_model_pipeline.pkl'")
replace_in_source(nb3['cells'], "'../models/model_meta.pkl'", "'./models_hanoi/model_meta.pkl'")
replace_in_source(nb3['cells'], "'../models/model_comparison.png'", "'./models_hanoi/model_comparison.png'")
replace_in_source(nb3['cells'], "'../models/feature_importance.png'", "'./models_hanoi/feature_importance.png'")
replace_in_source(nb3['cells'], "os.makedirs('../models'", "os.makedirs('./models_hanoi'")
replace_in_source(nb3['cells'], "Giá BĐS Việt Nam", "Giá BĐS Hà Nội")

# Update URLs to local paths
replace_in_source(nb3['cells'],
    "URL_CC = 'https://raw.githubusercontent.com/tangoctai2004/House-Price-Prediction/refs/heads/main/data/processed/cleaned_chung_cu.csv'",
    "URL_CC = '../../data/processed/cleaned_chung_cu.csv'")
replace_in_source(nb3['cells'],
    "URL_ND = 'https://raw.githubusercontent.com/tangoctai2004/House-Price-Prediction/refs/heads/main/data/processed/cleaned_nha_dat.csv'",
    "URL_ND = '../../data/processed/cleaned_nha_dat.csv'")

# Update test cases to Hanoi-specific
for cell in nb3['cells']:
    src = ''.join(cell['source'])
    if 'Quận 1' in src and 'predict_price' in src:
        cell['source'] = [
            "def predict_price(area_m2, bedrooms, district_name, furniture, legal, loai='chung_cu',\n",
            "                  floors=0, frontage=0, road_width=0, direction='Không rõ'):\n",
            "    \"\"\"\n",
            "    Dự đoán giá BĐS theo tỷ VNĐ.\n",
            "    Sử dụng Pipeline nên không cần encode/scale thủ công.\n",
            "    \"\"\"\n",
            "    sample = pd.DataFrame([{\n",
            "        'area_m2': area_m2, 'bedrooms_num': bedrooms,\n",
            "        'district': district_name, 'direction': direction,\n",
            "        'furniture_std': furniture, 'legal_std': legal,\n",
            "        'floors_num': floors, 'frontage_m': frontage,\n",
            "        'road_width_m': road_width, 'loai_bds': loai\n",
            "    }])\n",
            "    return best_pipeline.predict(sample)[0]\n",
            "\n",
            "test_cases = [\n",
            "    ('Chung cư 75m², 2PN, Cầu Giấy, Đầy đủ, Sổ đỏ',\n",
            "     dict(area_m2=75, bedrooms=2, district_name='Cầu Giấy',\n",
            "          furniture='Đầy đủ', legal='Sổ đỏ/Sổ hồng', loai='chung_cu')),\n",
            "    ('Chung cư 50m², 2PN, Hà Đông, Cơ bản, HĐMB',\n",
            "     dict(area_m2=50, bedrooms=2, district_name='Hà Đông',\n",
            "          furniture='Cơ bản', legal='HĐMB', loai='chung_cu')),\n",
            "    ('Nhà đất 45m², 4PN, Thanh Xuân, Đầy đủ, Sổ đỏ, 5 tầng',\n",
            "     dict(area_m2=45, bedrooms=4, district_name='Thanh Xuân',\n",
            "          furniture='Đầy đủ', legal='Sổ đỏ/Sổ hồng', loai='nha_dat',\n",
            "          floors=5, frontage=4.2, road_width=4.0)),\n",
            "    ('Nhà đất 60m², 3PN, Ba Đình, Đầy đủ, Sổ đỏ, 4 tầng',\n",
            "     dict(area_m2=60, bedrooms=3, district_name='Ba Đình',\n",
            "          furniture='Đầy đủ', legal='Sổ đỏ/Sổ hồng', loai='nha_dat',\n",
            "          floors=4, frontage=4.0, road_width=5.0)),\n",
            "]\n",
            "\n",
            "print('🔮 THỬ NGHIỆM DỰ ĐOÁN GIÁ (Hà Nội):')\n",
            "print('='*60)\n",
            "for desc, params in test_cases:\n",
            "    price = predict_price(**params)\n",
            "    print(f'  📍 {desc}')\n",
            "    print(f'     → Giá dự đoán: {price:.2f} tỷ VNĐ (~{price*1000:.0f} triệu)\\n')"
        ]

save_nb(nb3, '03_training_hanoi.ipynb')

# ============ 04_evaluation_hanoi.ipynb ============
print("Generating 04_evaluation_hanoi.ipynb...")
nb4 = read_nb('04_evaluation.ipynb')
nb4 = clear_outputs(nb4)

# Replace title
nb4['cells'][0] = make_cell("markdown", [
    "# 📊 BƯỚC 4: ĐÁNH GIÁ MÔ HÌNH AI — HÀ NỘI\n",
    "**Mô tả:** Đánh giá chi tiết model Hà Nội-only  \n",
    "**Yêu cầu:** Chạy `03_training_hanoi.ipynb` trước để có file model\n",
    "\n",
    "> **Phiên bản:** Hanoi-only experiment"
])

# Update paths
replace_in_source(nb4['cells'], "'../models/best_model_pipeline.pkl'", "'./models_hanoi/best_model_pipeline.pkl'")
replace_in_source(nb4['cells'], "'../models/model_meta.pkl'", "'./models_hanoi/model_meta.pkl'")
replace_in_source(nb4['cells'], "'../models/prediction_analysis.png'", "'./models_hanoi/prediction_analysis.png'")
replace_in_source(nb4['cells'], "'../models/error_distribution.png'", "'./models_hanoi/error_distribution.png'")
replace_in_source(nb4['cells'], "'../models/xai_feature_importance.png'", "'./models_hanoi/xai_feature_importance.png'")

# Update URLs to local
replace_in_source(nb4['cells'],
    "URL_CC = 'https://raw.githubusercontent.com/tangoctai2004/House-Price-Prediction/refs/heads/main/data/processed/cleaned_chung_cu.csv'",
    "URL_CC = '../../data/processed/cleaned_chung_cu.csv'")
replace_in_source(nb4['cells'],
    "URL_ND = 'https://raw.githubusercontent.com/tangoctai2004/House-Price-Prediction/refs/heads/main/data/processed/cleaned_nha_dat.csv'",
    "URL_ND = '../../data/processed/cleaned_nha_dat.csv'")

# Add Hanoi filter in rebuild test set section
for i, cell in enumerate(nb4['cells']):
    src = ''.join(cell['source'])
    if 'Rebuild test set' in src and cell['cell_type'] == 'markdown':
        # The next code cell rebuilds test set - add Hanoi filter
        for j in range(i+1, len(nb4['cells'])):
            if nb4['cells'][j]['cell_type'] == 'code':
                old_src = nb4['cells'][j]['source']
                # Add Hanoi filter lines after the district astype line
                new_src = []
                for line in old_src:
                    new_src.append(line)
                    if "df_all[c] = df_all[c].astype(str)" in line:
                        new_src.append("\n")
                        new_src.append("# Lọc chỉ Hà Nội\n")
                        new_src.append("HANOI_DISTRICTS = [\n")
                        new_src.append("    'Ba Đình','Hoàn Kiếm','Tây Hồ','Long Biên','Cầu Giấy',\n")
                        new_src.append("    'Đống Đa','Hai Bà Trưng','Hoàng Mai','Thanh Xuân',\n")
                        new_src.append("    'Hà Đông','Bắc Từ Liêm','Nam Từ Liêm',\n")
                        new_src.append("    'Đông Anh','Gia Lâm','Hoài Đức','Thanh Trì',\n")
                        new_src.append("]\n")
                        new_src.append("df_all = df_all[df_all['district'].isin(HANOI_DISTRICTS)].copy()\n")
                nb4['cells'][j]['source'] = new_src
                break
        break

# Update conclusion text
replace_in_source(nb4['cells'], "Hà Nội, TP.HCM và Đà Nẵng", "Hà Nội")
replace_in_source(nb4['cells'], "gần 8.000", "khoảng 3.600")

save_nb(nb4, '04_evaluation_hanoi.ipynb')

print("\nDone! Created:")
print("  - 03_training_hanoi.ipynb")
print("  - 04_evaluation_hanoi.ipynb")
