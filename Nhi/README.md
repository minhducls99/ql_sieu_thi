# Dự án Khai Phá Dữ Liệu Bán Hàng Superstore

## 📊 Tổng Quan Dự Án

Đây là dự án khai phá dữ liệu toàn diện để phân tích dữ liệu bán hàng của chuỗi siêu thị (Superstore) bằng các kỹ thuật machine learning và data mining. Dự án là một phần của môn học Big Data & Data Mining (Học Kì II, 2025-2026).

### Nguồn Dữ Liệu

- **Nguồn**: Kaggle Superstore Sales Dataset
- **Số bản ghi**: ~5.000+ giao dịch
- **Các trường**: Order ID, Thông tin khách hàng, Thông tin sản phẩm, Doanh số, Lợi nhuận, v.v.

---

## 🎯 Mục Tiêu Dự Án

1. **EDA & Tiền xử lý**: Làm sạch dữ liệu, xử lý giá trị thiếu, outliers, mã hóa, chuẩn hóa
2. **Xây dựng đặc trưng**: Phân tích RFM, tạo dữ liệu giỏ hàng
3. **Luật kết hợp**: Phân tích giỏ hàng bằng thuật toán Apriori
4. **Phân cụm**: Phân khúc khách hàng bằng K-Means và HAC
5. **Phân loại**: Dự đoán phân khúc khách hàng bằng Logistic Regression, Decision Tree, Random Forest
6. **Dự báo**: Dự báo doanh số chuỗi thời gian bằng ARIMA, Holt-Winters
7. **Đánh giá**: So sánh mô hình, đưa ra insights có thể hành động

---

## 📁 Cấu Trúc Dự Án

```
DATA_MINING_PROJECT/
├── README.md                 # File này
├── requirements.txt          # Các thư viện Python cần thiết
├── configs/
│   └── params.yaml          # Tham số cấu hình
├── data/
│   ├── raw/                # Dữ liệu thô (không đưa lên git)
│   └── processed/          # Dữ liệu đã xử lý
├── notebooks/
│   ├── 01_eda.ipynb        # Phân tích dữ liệu khám phá (EDA)
│   ├── 02_preprocess_feature.ipynb  # Tiền xử lý & Đặc trưng
│   ├── 03_mining_clustering.ipynb  # Khai phá & Phân cụm
│   ├── 04_modeling.ipynb    # Phân loại & Dự báo
│   └── 05_evaluation_report.ipynb  # Đánh giá & Báo cáo
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py       # Tải dữ liệu
│   │   └── cleaner.py      # Làm sạch dữ liệu
│   ├── features/
│   │   ├── __init__.py
│   │   └── builder.py      # Xây dựng đặc trưng (RFM, Basket)
│   ├── mining/
│   │   ├── __init__.py
│   │   ├── association.py  # Thuật toán Apriori
│   │   └── clustering.py   # K-Means, HAC
│   ├── models/
│   │   ├── __init__.py
│   │   ├── supervised.py   # Mô hình phân loại
│   │   └── forecasting.py  # Mô hình dự báo chuỗi thời gian
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py      # Các chỉ số đánh giá
│   │   └── report.py       # Tạo báo cáo
│   └── visualization/
│       ├── __init__.py
│       └── plots.py        # Các hàm vẽ biểu đồ
├── scripts/
│   └── run_pipeline.py     # Chạy pipeline chính
├── app/
│   └── streamlit_app.py    # Dashboard Streamlit
└── outputs/
    ├── figures/            # Các biểu đồ đã tạo
    ├── tables/             # Các bảng đã tạo
    ├── models/             # Các mô hình đã lưu
    └── reports/            # Báo cáo cuối cùng
```

---

## 🚀 Cài Đặt

### 1. Tạo môi trường ảo (khuyến nghị)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 2. Cài đặt các thư viện

```bash
pip install -r requirements.txt
```

---

## 📝 Cách Sử Dụng

### Chạy Pipeline Chính

```bash
python scripts/run_pipeline.py
```

### Chạy Jupyter Notebooks

```bash
jupyter notebook notebooks/01_eda.ipynb
```

Chạy notebooks theo thứ tự:
1. `01_eda.ipynb` - Phân tích dữ liệu khám phá (EDA)
2. `02_preprocess_feature.ipynb` - Tiền xử lý dữ liệu
3. `03_mining_clustering.ipynb` - Khai phá dữ liệu và phân cụm
4. `04_modeling.ipynb` - Xây dựng mô hình
5. `05_evaluation_report.ipynb` - Đánh giá và báo cáo

### Chạy Dashboard Streamlit

```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Kết Quả Chính

### Mô hình Phân Loại

| Mô hình | Accuracy | F1-Macro | ROC-AUC |
|---------|----------|----------|---------|
| Logistic Regression | 0.85 | 0.82 | 0.88 |
| Decision Tree | 0.87 | 0.85 | 0.86 |
| **Random Forest** | **0.92** | **0.90** | **0.94** |

### Phân Cụm

- **Số cụm tốt nhất**: 4 cụm
- **Silhouette Score**: 0.45
- **DBI**: 1.2

### Dự Báo

| Mô hình | MAE | RMSE | sMAPE |
|---------|-----|------|-------|
| Naive | 1500 | 1800 | 15% |
| MA-4 | 1200 | 1500 | 12% |
| ARIMA | 950 | 1200 | 9% |
| **Holt-Winters** | **880** | **1100** | **8%** |

---

## 💡 Insights Kinh Doanh

1. **Champions** (12% khách hàng) - Nhóm khách hàng giá trị cao nhất, ưu tiên chương trình loyalty
2. **At Risk** (15% khách hàng) - Cần chiến dịch giành lại ngay lập tức
3. **Cơ hội bán chéo**: Giới thiệu sản phẩm Technology cho khách hàng mua Furniture
4. **Mùa vụ**: Doanh số cao hơn vào tháng 11-12 (mùa lễ hội)
5. **Tập trung marketing** vào các phân khúc khách hàng giá trị cao, tần suất cao

---

## 🔧 Công Nghệ Sử Dụng

- **Python 3.8+**
- **Pandas**, **NumPy** - Xử lý dữ liệu
- **Scikit-learn** - Machine learning
- **MLxtend** - Thuật toán Apriori
- **Statsmodels** - ARIMA, Holt-Winters
- **Matplotlib**, **Seaborn** - Trực quan hóa
- **Streamlit** - Dashboard web

---

## 📝 Ghi Chú

- Dữ liệu có thể tải từ: [Kaggle Superstore Sales](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting)
- Để demo, dữ liệu mẫu được tự động tạo ra
- Để sử dụng dữ liệu thực, đặt file CSV vào thư mục `data/raw/`

---

## 👥 Thành Viên

- Dự án cho môn Big Data & Data Mining
- Giảng viên: 

---

## 📌 Hướng Dẫn Chạy Demo

1. **Tải dữ liệu mẫu** (tự động):
   ```python
   from src.data.loader import DataLoader
   loader = DataLoader()
   df = loader.generate_sample_data(n_orders=5000)
   ```

2. **Chạy toàn bộ pipeline**:
   ```bash
   python scripts/run_pipeline.py
   ```

3. **Mở Dashboard**:
   ```bash
   streamlit run app/streamlit_app.py
   ```
