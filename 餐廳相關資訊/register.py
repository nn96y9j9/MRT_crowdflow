import pandas as pd
import numpy as np
from bs4 import BeautifulSoup as bs
import requests
import openpyxl
import os
import re
def filename():
    directory_path = "./"
    file_list = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".xlsx"):
                file_list.append(os.path.join(root, file))

    return file_list

# 使用正規表達式抓取檔案路徑中的數字部分
def extract_numbers(file_path):
    pattern = r"\d+"
    match = re.search(pattern, file_path)
    if match:
        return match.group()

def label():
    for i in file_list:
        file = i.split("/")[1]
        print(file)
        df = pd.read_excel(file,engine='openpyxl',header=1)
        for j in range(len(df['統一編號'])):
            try:
                url = f'https://alltwcompany.com/nd-B-{str(df.loc[j,"統一編號"])[:8]}-{df.loc[j,"商 業 名 稱"]}.html'
                print(url)
                req = requests.get(url)
                print(req.status_code)
                soup = bs(req.text,"lxml")
                register = soup.select_one("div.txt ul li")
                df.loc[j, "營登"] = register.text
                print(f'{df.loc[j,"商 業 名 稱"]}:\n{df.loc[j, "營登"]}')
            except AttributeError:
                print(f'{df.loc[j,"商 業 名 稱"]}登記失敗')
        df.to_csv(f"./register_first/{str(i).split('.')[0]}.csv", encoding="utf-8-sig", index=False)
        print(f'{str(i).split(".")[0]}.csv完成')

if __name__ == "__main__":
    file_list = filename()
    for i in file_list:
        data_list = extract_numbers(i)
        df = pd.read_excel(f"{data_list}.xlsx",engine='openpyxl',header=1)
        for j in range(len(df['統一編號'])):
            try:
                url = f'https://alltwcompany.com/nd-B-{str(df.loc[j,"統一編號"])[:8]}-{df.loc[j,"商 業 名 稱"]}.html'
                print(df.loc[j,"商 業 名 稱"])
                print('*'*100)
                req = requests.get(url)
                soup = bs(req.text,"lxml")
                register = soup.select_one("div.txt ul li")
                df.loc[j, "營登"] = register.text
            except AttributeError:
                print('-'*60)
                print(f'{df.loc[j,"商 業 名 稱"]}登陸失敗')
                print('-'*60)
        df.to_csv(f'./register_first/{data_list}.csv', encoding="utf-8-sig", index=False)
        print(f'{data_list}.csv完成')