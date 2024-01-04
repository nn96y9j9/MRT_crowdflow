import json

import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
from flask import Flask, render_template
from information import USERNAME, PASSWORD

username = USERNAME
password = PASSWORD

app = Flask(__name__)


@app.route("/")
def calculate():
    url = "https://api.metro.taipei/metroapiex/CarWeight.asmx"

    payload = f'<?xml version="1.0" encoding="utf-8"?>\r\n<soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" \r\nxmlns:xsd="http://www.w3.org/2001/XMLSchema" \r\nxmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">\r\n    <soap:Body>\r\n        <getCarWeightByInfoEx xmlns="http://tempuri.org/">\r\n            <userName>{username}</userName>\r\n            <passWord>{password}</passWord>\r\n        </getCarWeightByInfoEx>\r\n    </soap:Body>\r\n</soap:Envelope>'
    headers = {
        "Content-Type": "text/xml; charset=utf-8",
        "Cookie": "__cf_bm=YHtEuzavrsic3qMwJy91xoocTdMasDiVtoRZpF2TQpI-1690594644-0-Aegw2NsOv8/ZAgSY/V315CeYCvrl99QSZZVzyWHCXotOpcgU86p3qrPKuuH21RIo8sOEqb0tBlsc5Qp1Cn5qGSQ=; TS01232bc6=0110b39fae0ea5231672c55a032ff65c6638e0f7196d6af40a3c4726b6ed408b1172c390e6e8190d826fa79e1a1e65e08d57ac167a",
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    soup = bs(response.text, "lxml")
    json_data = json.loads(soup.select_one("p").get_text())
    df = pd.DataFrame(json_data)
    print(df)
    table_html = df.to_html()

    return render_template("index.html", table=table_html)


@app.route("/BR.html")
def BR():
    url = "https://api.metro.taipei/metroapi/CarWeightBR.asmx"

    payload = f'<?xml version="1.0" encoding="utf-8"?>\r\n<soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\r\nxmlns:xsd="http://www.w3.org/2001/XMLSchema"\r\nxmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">\r\n    <soap:Body>\r\n        <getCarWeightBRInfo xmlns="http://tempuri.org/">\r\n            <userName>{username}</userName>\r\n            <passWord>{password}</passWord>\r\n        </getCarWeightBRInfo>\r\n    </soap:Body>\r\n</soap:Envelope>'
    headers = {
        "Content-Type": "text/xml;charset = utf-8",
        "Cookie": "__cf_bm=nxaHOM654t6He79kWRarYoGPsBXBTEND3wUdylbqiio-1690681872-0-AXDcxrv1R/4K6eY3hy1redpZdYs7KthogeG2ff3WFbomIrlfzY252q86Kxyo8OLDncp8XUIpX1LYxU28900wbfo=; TS01232bc6=0110b39fae9849df83eef367b97082422d4aa102c8a0557fee2aa0c6f394d33ec7097fbb3b38aecd40f8caff839fcbc01efcf30fe2",
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    soup = bs(response.text, "lxml")
    json_data = json.loads(soup.select_one("getcarweightbrinforesult").get_text())
    df = pd.DataFrame(json_data)
    table_html = df.to_html()
    return render_template("index.html", table=table_html)


@app.route("/BL.html")
def BL():
    url = "https://api.metro.taipei/metroapi/CarWeight.asmx"

    payload = f'<?xml version="1.0" encoding="utf-8"?>\r\n<soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\r\nxmlns:xsd="http://www.w3.org/2001/XMLSchema"\r\nxmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">\r\n    <soap:Body>\r\n        <getCarWeightByInfo xmlns="http://tempuri.org/">\r\n            <userName>{username}</userName>\r\n            <passWord>{password}</passWord>\r\n        </getCarWeightByInfo>\r\n    </soap:Body>\r\n</soap:Envelope>'
    headers = {
        "Content-Type": "text/xml;charset = utf-8",
        "Cookie": "__cf_bm=m69u1Rxo2Tne0MpQnOniGai57Wa1vgO.ljWf0.Is2_Q-1690683023-0-AR+eSnFITVVXsQNlSc0e2BRrnrnN4LUfGevdDYlC4oauYeyhWoGEV98m3pbaUG3iCwmSNNhKeoMaioup/8PwcIw=; TS01232bc6=0110b39faee3824331ebfa8d57eeca2190a2f8eebd83fd25c8121e5b29f7dbb88cc9ccddb76ea28dcf09623309400c3304a8209601",
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    soup = bs(response.text, "lxml")
    data_list = soup.get_text()
    json_data = json.loads(data_list)
    json_data = json.loads(json_data)
    df = pd.DataFrame(json_data)
    table_html = df.to_html()
    return render_template("index.html", table=table_html)


if __name__ == "__main__":
    app.run(debug=True)
