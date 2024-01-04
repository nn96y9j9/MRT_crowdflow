import datetime
from time import sleep

import pandas as pd
import requests as req
from bs4 import BeautifulSoup as bs


def get_current_time():
    return datetime.datetime.now().strftime("%Y%m%d%H%M")


def request_page(url):
    try:
        response = req.get(url)
        return response
    except req.RequestException as e:
        print(e)


def convert_to_dataframe(response):
    soup = bs(response, "lxml")
    df = pd.read_html(str(soup))[0]
    return df


if __name__ == "__main__":
    url_high_volume = (
        "http://127.0.0.1:5000/"  # Replace this with the URL you want to request
    )
    url_BL = "http://127.0.0.1:5000/BL.html"
    url_BR = "http://127.0.0.1:5000/BR.html"
    url_list = [url_high_volume, url_BL, url_BR]
    dataset_high_volume = pd.DataFrame()
    dataset_BR = pd.DataFrame()
    dataset_BL = pd.DataFrame()
    count = 0
    num = 0
    while num <= 10:
        for index, content in enumerate(url_list):
            print(index, content)
            html = request_page(content).text
            df = convert_to_dataframe(html)
            print(df.info())
            if index == 0:
                dataset_high_volume = pd.concat(
                    [dataset_high_volume, df], ignore_index=True
                )
                print(dataset_high_volume.head())
            elif index == 1:
                dataset_BL = pd.concat([dataset_BL, df], ignore_index=True)
                print(dataset_BL.head())
            else:
                dataset_BR = pd.concat([dataset_BR, df], ignore_index=True)
                print(dataset_BR.head())
            print("-" * 100)
            df = pd.DataFrame()
        interval_minutes = 5
        sleep(interval_minutes * 60)
        count += 1
        if count == 12:
            dataset_high_volume.to_csv(f"./dataset/高運量{get_current_time()}.csv")
            dataset_BL.to_csv(f"./dataset/板南線{get_current_time()}.csv")
            dataset_BR.to_csv(f"./dataset/文湖線{get_current_time()}.csv")
            count = 0
            num += 1
            print("-" * 100)
            print(f"{get_current_time()}完成了")
            dataset_high_volume = pd.DataFrame()
            dataset_BL = pd.DataFrame()
            dataset_BR = pd.DataFrame()
