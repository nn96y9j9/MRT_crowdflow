# 讀取檔案
def dataset():
    column = ["Date", "時段", "Station", "出站", "CrowdFlow"]
    df = pd.DataFrame(columns=column)
    for i in range(17, 24):
        dataset = pd.read_excel(f"./MRTmining/20{i}捷運人流_日期時間合併.xlsx", engine="openpyxl")
        df = pd.concat([df, dataset], ignore_index=True)
    return df


# 基本處理


def process(df):
    try:
        df["Date"] = pd.to_datetime(df["Date"])
    except ValueError as e:
        print(f"Error reading{e}")
    df = df.drop("出站", axis=1)
    df["Weekday"] = df["Date"].dt.weekday
    # weekday = ['Mon', 'Tue', 'Wed', 'Thr', 'Fri', 'Sat', 'Sun']
    # df['Weekday'] = df['Weekday'].map(lambda x: weekday[x])
    df["時段"] = df["時段"].astype(int)
    df["CrowdFlow"] = df["CrowdFlow"].astype(int)
    df["Station"] = df["Station"].astype(str)
    print(type(df["Weekday"]))
    return df


# 車站路線


def station(df):
    station_to_routes = {
        "動物園": "BR",
        "木柵": "BR",
        "萬芳社區": "BR",
        "萬芳醫院": "BR",
        "辛亥": "BR",
        "麟光": "BR",
        "六張犁": "BR",
        "科技大樓": "BR",
        "大安": ["BR", "R"],
        "忠孝復興": ["BR", "BL"],
        "南京復興": ["BR", "G"],
        "中山國中": "BR",
        "松山機場": "BR",
        "大直": "BR",
        "劍南路": "BR",
        "西湖": "BR",
        "港墘": "BR",
        "文德": "BR",
        "內湖": "BR",
        "大湖公園": "BR",
        "葫洲": "BR",
        "東湖": "BR",
        "南港軟體園區": "BR",
        "南港展覽館": ["BR", "BL"],
        "象山": "R",
        "台北101/世貿": "R",
        "世貿": "R",
        "信義安和": "R",
        "大安森林公園": "R",
        "東門": ["R", "O"],
        "中正紀念堂": ["R", "G"],
        "台大醫院": "R",
        "台北車站": ["R", "BL"],
        "中山": ["R", "G"],
        "雙連": "R",
        "民權西路": ["R", "O"],
        "圓山": "R",
        "劍潭": "R",
        "士林": "R",
        "芝山": "R",
        "明德": "R",
        "石牌": "R",
        "唭哩岸": "R",
        "奇岩": "R",
        "北投": "R",
        "新北投": "R",
        "復興崗": "R",
        "忠義": "R",
        "關渡": "R",
        "竹圍": "R",
        "紅樹林": "R",
        "淡水": "R",
        "頂埔": "BL",
        "永寧": "BL",
        "土城": "BL",
        "海山": "BL",
        "亞東醫院": "BL",
        "府中": "BL",
        "BL板橋": "BL",
        "新埔": "BL",
        "江子翠": "BL",
        "龍山寺": "BL",
        "西門": ["BL", "G"],
        "善導寺": "BL",
        "忠孝新生": ["BL", "O"],
        "忠孝敦化": "BL",
        "國父紀念館": "BL",
        "市政府": "BL",
        "永春": "BL",
        "後山埤": "BL",
        "昆陽": "BL",
        "南勢角": "O",
        "景安": "O",
        "永安市場": "O",
        "頂溪": "O",
        "古亭": ["O", "G"],
        "松江南京": ["O", "G"],
        "行天宮": "O",
        "中山國小": "O",
        "大橋頭站": "O",
        "台北橋": "O",
        "菜寮": "O",
        "三重": "O",
        "先嗇宮": "O",
        "頭前庄": "O",
        "新莊": "O",
        "輔大": "O",
        "丹鳳": "O",
        "迴龍": "O",
        "三重國小": "O",
        "三和國中": "O",
        "徐匯中學": "O",
        "三民高中": "O",
        "蘆洲": "O",
        "新店": "G",
        "新店區公所": "G",
        "七張": "G",
        "小碧潭": "G",
        "大坪林": "G",
        "景美": "G",
        "萬隆": "G",
        "公館": "G",
        "台電大樓": "G",
        "小南門": "G",
        "北門": "G",
        "台北小巨蛋": "G",
        "南京三民": "G",
        "松山": "G",
        "南港": "BL",
    }
    df["Routes"] = df["Station"].map(station_to_routes)
    return df


# 假日


def hoilday(df):
    hoilday = pd.read_excel("./MRTmining/holiday_Calendar.xlsx")
    hoilday["Date"] = pd.to_datetime(
        hoilday["年"].astype(str)
        + "-"
        + hoilday["月"].astype(str)
        + "-"
        + hoilday["日"].astype(str)
    )
    hoildays = hoilday["Date"]
    df["Hoilday"] = df["Date"].isin(hoildays)
    df[df["Hoilday"] == False]
    return df


# 尖離峰


def peak(df):
    df["Peak"] = False
    filter_condition = (df["Weekday"].isin(["Mon", "Tue", "Wed", "Thr", "Fri"])) & (
        df["時段"].isin([7, 8, 17, 18, 19])
    )
    df.loc[filter_condition, "Peak"] = True
    return df


# 車站代碼轉換


def station_code():
    map_station = {}
    with open("./MRTmining/station.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        map_station = {rows[0]: rows[1] for rows in reader}
    return map_station


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import openpyxl

    # 預處理
    df = dataset()
    df = process(df)
    df = station(df)
    df = hoilday(df)
    df = peak(df)
    print(df.head())
    print("-" * 50)
    print(df.info())
    print("-" * 50)
    non_date_cols = df.select_dtypes(exclude=["datetime64"])
    non_date_describe = non_date_cols.describe().astype(int)
    print(non_date_describe)
    print("-" * 50)
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.rc("font", family="Microsoft JhengHei")
    import seaborn as sns

    # 做相關矩陣
    corr_data = pd.DataFrame(df[["時段", "Station", "CrowdFlow", "Weekday", "Hoilday"]])
    import csv

    map_station = station_code()
    corr_data["Station"] = (
        corr_data["Station"].replace("BL板橋", "板橋").replace("Y板橋", "板橋")
    )
    corr_data["Station"] = corr_data["Station"].map(map_station).astype(int)
    method = ["pearson", "spearman", "kendall"]
    for i in method:
        correlation_matrix = corr_data.corr(method=i)
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title(f"{i} Correlation Heatmap")
        plt.show()
        # 輸出結果
        print(f"{i} Correlation Matrix")
        print(correlation_matrix)
        print("-" * 50)

    # 隨機抽樣查看人流分布狀況
    df_sample = df.sample(n=1000, random_state=42)
    sns.displot(df_sample["CrowdFlow"], bins=50, kde=True, aspect=2, height=6)
    plt.title("CrowdFlow Distribution")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # 查看是否有統計上的顯著
    from scipy.stats import kendalltau

    columns = ["時段", "Station", "Weekday", "Hoilday"]
    for i in columns:
        tau, p_value = kendalltau(df_sample[i], df_sample["CrowdFlow"])
        print(f"{i}與人流的相關程度")
        print(f"Kendall Tau: {tau}")
        print(f"P-value: {p_value}")
        print("-" * 50)

    # 時段與人流的圖表(以平均數畫圖)
    data_time = df[["時段", "CrowdFlow"]].groupby("時段").mean().reset_index()
    data_time = data_time.sort_values("時段")
    print(data_time["CrowdFlow"].describe().astype(int))
    x = data_time["時段"].astype(str)
    y = data_time["CrowdFlow"]

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker="o")
    plt.xlabel("時段")
    plt.ylabel("人流量")
    plt.xticks(x)

    plt.show()
