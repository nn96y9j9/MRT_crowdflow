import csv
import os
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
import openpyxl

# 操作 browser 的 API

from selenium import webdriver
from selenium.webdriver.chrome.service import Service

# ChromeDriver 的下載管理工具

from webdriver_manager.chrome import ChromeDriverManager

# 處理逾時例外的工具

from selenium.common.exceptions import TimeoutException

from selenium.common.exceptions import ElementNotInteractableException

# 面對動態網頁，等待某個元素出現的工具，通常與 exptected_conditions 搭配

from selenium.webdriver.support.ui import WebDriverWait

# 搭配 WebDriverWait 使用，對元素狀態的一種期待條件，若條件發生，則等待結束，往下一行執行

from selenium.webdriver.support import expected_conditions as EC

# 期待元素出現要透過什麼方式指定，通常與 EC、WebDriverWait 一起使用

from selenium.webdriver.common.by import By

# 強制等待 (執行期間休息一下)

from time import sleep

# with open(os.path.join("科技執法總清單20230611.csv"), "r", newline="", encoding="utf-8") as file:

# 啟動瀏覽器工具的選項

my_options = webdriver.ChromeOptions()

# my_options.add_argument("--headless") #不開啟實體瀏覽器背景執行

my_options.add_argument("--start-maximized")  # 最大化視窗
my_options.add_argument("--incognito")  # 開啟無痕模式
my_options.add_argument("--disable-popup-blocking")  # 禁用彈出攔截
my_options.add_argument("--disable-notifications")  # 取消 chrome 推播通知
my_options.add_argument("--lang=zh-TW")  # 設定為正體中文
my_options.add_argument(
    "--user-agent=Mozilla/5.0 (Macintosh Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36"
)

# 開啟網頁

chrome_path = "C:\Program Files\Google\Chrome\Application\Chrome.exe"
options = webdriver.ChromeOptions()
options.binary_location = chrome_path
driver = webdriver.Chrome(options=options)


# 要讀取的目錄路徑
directory_path = "./register_first"  # 將這裡替換為您的目錄路徑

# 列出目錄中所有的檔案
file_list = os.listdir(directory_path)
data_list = []
for i in file_list:
    print(i.split(".")[0])
    data_list.append(i.split(".")[0])

# 抓經緯度的正規表達式
all = r"@(-?\d+\.\d+),(-?\d+\.\d+),"

for index in data_list:
    df = pd.read_csv(f"./register_first/{index}.csv", header=0)
    print(df)
    for i in range(len(df["商 業 所 在 地"])):
        address = f"{df['商 業 所 在 地'][i]}"
        url = "https://www.google.com/maps/place/{}".format(address)
        driver.get(url)
        sleep(6)
        title = driver.title
        if title != "Google 地圖":
            current_url = driver.current_url
            match = re.search(all, current_url)
            if match:
                match_Longitude = match.group(2)
                match_Latitude = match.group(1)
                df.loc[i, "經度"] = match_Longitude
                df.loc[i, "緯度"] = match_Latitude
                print("第{}筆座標轉換成功,{},{}".format(i, match_Longitude, match_Latitude))
            else:
                print("第{}筆無法獲得座標資訊".format(i))
        else:
            print("{}座標轉換失敗,索引值為{}".format(address, i))
    df.drop(["Unnamed: 5", "output_table_0.xlsx", "序號"], axis=1, inplace=True)
    df.to_csv(f"./position/{index}.csv", index=False, encoding="utf_8_sig")
driver.quit()
