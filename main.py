import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', None)

PATH_1 = r"C:\Users\ADMIN\PycharmProjects\PythonProject\VNM_Finan.csv"
PATH_2 = r"C:\Users\ADMIN\PycharmProjects\PythonProject\VNM_price.csv"
NEW_PATH_1 = r"C:\Users\ADMIN\PycharmProjects\PythonProject\VNM_finance_reverse.csv"


def clean_data():
    df1_raw = pd.read_csv(PATH_1, skiprows=6, thousands=',')
    df2 = pd.read_csv(PATH_2, skiprows=5, thousands=',')

    df1 = pd.read_csv(NEW_PATH_1, skiprows=1, thousands=',')
    df1 = df1.iloc[1:, 0:62]
    df1.columns = [col.strip() for col in df1.columns]

    drop_cols = [
        'Chỉ tiêu Báo cáo kết quả kinh doanh (Tỷ đồng)', 'Chỉ tiêu Bảng cân đối (Tỷ đồng)',
        'Chỉ số định giá', 'Chỉ số hiệu quả hoạt động', 'Chỉ số hiệu suất hoạt động',
        'Chỉ số cơ cấu nguồn vốn', 'Chỉ số khả năng thanh toán', 'Thông tin doanh nghiệp'
    ]
    df1.drop(columns=[c for c in drop_cols if c in df1.columns], inplace=True)

    df1 = df1.map(lambda x: str(x).replace('%', '') if isinstance(x, str) else x)

    df1.set_index('CHỈ TIÊU', inplace=True)
    df1 = df1.apply(pd.to_numeric, errors='coerce')
    df1.dropna(how='all', axis=0, inplace=True)

    df2.columns = [col.strip() for col in df2.columns]
    df2.set_index('NGÀY', inplace=True)
    return df1, df2


def ex1(df1, df2):
    # Dictionary dịch sang tiếng Việt
    rename_dict = {
        'mean': 'Trung bình',
        'std': 'Độ lệch chuẩn',
        'min': 'Thấp nhất',
        '25%': 'Tứ phân vị 1 (Q1)',
        '50%': 'Trung vị (Q2)',
        '75%': 'Tứ phân vị 3 (Q3)',
        'max': 'Cao nhất',
        'variance': 'Phương sai'
    }

    # Xử lý thống kê cho Tài chính
    stats1 = df1.describe().T
    stats1['variance'] = df1.var(numeric_only=True)
    if 'count' in stats1.columns:
        stats1 = stats1.drop(columns='count')
    stats1 = stats1.rename(columns=rename_dict)

    # Xử lý thống kê cho Cổ phiếu
    stats2 = df2.describe().T
    stats2['variance'] = df2.var(numeric_only=True)
    if 'count' in stats2.columns:
        stats2 = stats2.drop(columns='count')
    stats2 = stats2.rename(columns=rename_dict)

    print("--- THỐNG KÊ TÀI CHÍNH ---")
    print(stats1)
    print("\n--- THỐNG KÊ CỔ PHIẾU ---")
    print(stats2)

    # Xuất file CSV
    stats1.to_csv('VNM_Thong_Ke_Tai_Chinh.csv', encoding='utf-8-sig')
    stats2.to_csv('VNM_Thong_Ke_Co_Phieu.csv', encoding='utf-8-sig')

    df2_idx = pd.to_datetime(df2.index, dayfirst=True, errors='coerce')
    df_plot = df2.copy()
    df_plot.index = df2_idx
    df_plot = df_plot[df_plot.index.notnull()].sort_index()

    plt.figure(figsize=(20, 10))
    plt.plot(df_plot.index, df_plot['GIÁ ĐÓNG CỬA'], color='blue', linewidth=2, label='Giá đóng cửa')
    plt.title('Biến động giá cổ phiếu VNM', fontsize=20, color='red', fontweight='bold')
    plt.xlabel('Thời gian', fontsize=15)
    plt.ylabel('Giá (VNĐ)', fontsize=15)
    plt.grid()
    plt.show()


def preprocessing(df1, df2):
    df2.index = pd.to_datetime(df2.index, dayfirst=True, errors='coerce')
    df2 = df2[df2.index.notnull()].sort_index()

    df1.index = df1.index.astype(str)
    df1 = df1[df1.index != 'nan']
    df1 = df1[df1.index.str.contains('/', na=False)]

    def get_end(q_str):
        q_str = str(q_str).strip()
        q, year = q_str.split('/')
        month = {'Q1': 3, 'Q2': 6, 'Q3': 9, 'Q4': 12}[q.upper()]
        return pd.Timestamp(year=int(year), month=month, day=1) + pd.offsets.MonthEnd(0)

    price_each_quarter = []

    for q_label in df1.index:
        try:
            chot_quy = get_end(q_label)
            bat_dau = chot_quy - pd.Timedelta(days=14)
            ket_thuc = chot_quy + pd.Timedelta(days=14)

            mask = (df2.index >= bat_dau) & (df2.index <= ket_thuc)
            data = df2.loc[mask, 'GIÁ ĐÓNG CỬA']

            if not data.empty:
                price_each_quarter.append({
                    'CHỈ TIÊU': q_label,
                    'Giá đóng cửa': data.mean(),
                    'Timeline': chot_quy
                })
        except:
            continue

    if not price_each_quarter:
        return None

    df_price_q = pd.DataFrame(price_each_quarter).set_index('CHỈ TIÊU')
    df_price_q = df_price_q.sort_values('Timeline')

    df_price_q['Thay đổi giá'] = df_price_q['Giá đóng cửa'].diff()
    df_price_q['% thay đổi'] = df_price_q['Giá đóng cửa'].pct_change() * 100

    cols_tai_chinh = [
        'Biên lợi nhuận gộp', 'Biên lợi nhuận ròng', 'P/E', 'EPS (VNĐ/CP)',
        'Tăng trưởng EPS', 'ROE LTM', 'Nợ phải trả / Vốn chủ sở hữu',
        'Khả năng thanh toán tổng quát', 'Vòng quay tài sản (vòng)', 'Giá trị sổ sách (VNĐ/CP)'
    ]

    available_cols = [c for c in cols_tai_chinh if c in df1.columns]
    df1_filtered = df1[available_cols]

    df_final = df1_filtered.merge(df_price_q.drop(columns=['Timeline']),
                                  left_index=True, right_index=True, how='inner')

    df_final.to_csv('VNM_Du_Lieu_Tong_Hop.csv', encoding='utf-8-sig')
    return df_final


def ex4(df_final):
    print("\n--- 5 DÒNG ĐẦU TIÊN CỦA BỘ DỮ LIỆU TỔNG HỢP ---")
    print(df_final.head())

    # Tính toán tương quan Pearson
    correlation_matrix = df_final.corr()
    price_correlation = correlation_matrix['Giá đóng cửa'].sort_values(ascending=False)

    print("\n--- ĐỘ TƯƠNG QUAN PEARSON VỚI GIÁ ĐÓNG CỬA ---")
    print(price_correlation)

    # Chỉ vẽ biểu đồ cột để so sánh mức độ ảnh hưởng của các yếu tố
    plt.figure(figsize=(12, 6))
    factors_only = price_correlation.drop(['Giá đóng cửa', 'Thay đổi giá', '% thay đổi'], errors='ignore')
    factors_only.plot(kind='bar', color='skyblue')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title('Mức độ ảnh hưởng của các chỉ số tài chính đến Giá VNM (Hệ số Pearson)', fontsize=14)
    plt.ylabel('Hệ số tương quan')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    top_factor = factors_only.abs().idxmax()
    top_value = factors_only[top_factor]
    print(f"\n=> KẾT LUẬN: Chỉ số có ảnh hưởng mạnh nhất đến giá là: {top_factor} ({top_value:.2f})")

    return price_correlation


if __name__ == '__main__':
    df1, df2 = clean_data()
    ex1(df1, df2)
    df_final = preprocessing(df1, df2)
    if df_final is not None:
        ex4(df_final)