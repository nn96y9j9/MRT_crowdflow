import requests
from information import USERNAME, PASSWORD

username = USERNAME
password = PASSWORD

url = "https://api.metro.taipei/metroapi/CarWeight.asmx"

payload = f'<?xml version="1.0" encoding="utf-8"?>\r\n<soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">\r\n    <soap:Body>\r\n        <getCarWeightByInfo xmlns="http://tempuri.org/">\r\n            <userName>{username}</userName>\r\n            <passWord>{PASSWORD}</passWord>\r\n        </getCarWeightByInfo>\r\n    </soap:Body>\r\n</soap:Envelope>'
headers = {
    "Content-Type": "text/xml; charset=utf 8",
    "Cookie": "__cf_bm=WkGpcG.y0wyaA7aKzYMSZgLWYe9YhML.R0mMw842.fg-1691398571-0-AU/it9cNXDSF4EjuSjhKDuoQwEmIxLz5x+0EBYA/nNJ+IaMDDjcZyKvVd/O2Mt1u8+Q5a6HyIlCRh8O/Ag9TqdY=; TS01232bc6=0110b39fae90b20a582ec88c91a6f7c59ab955c1a70249a3dff37835d8cd45ff96da2b298fa55c94969ae3a59d76ca75dfccec25c6",
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
