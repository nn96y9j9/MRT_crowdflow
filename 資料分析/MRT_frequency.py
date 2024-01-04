import requests
import pandas as pd
from pandas import json_normalize
import json

url = "https://www.metro.taipei/OpenData.aspx?SN=6230C0D75360BFF5"

payload = {}
headers = {
    'Cookie': 'ASP.NET_SessionId=aishsk3uquioo5iphxqeugw3; TS0131301b=01ca5d79a1dcceb1bb3440ba2ec600b6083fe61cca32a4be4596b2674a8b915ade687d1c9690e56d6aee68671e8ba27b59b1515d56eb6e8a48f01db800ae1446897b7b1945'
}

response = requests.request("GET", url, headers=headers, data=payload)
json_ = json.loads(response.content.decode('utf-8-sig'))

df = pd.DataFrame.from_dict(json_)
df.to_csv('MRT_frequency.csv', index=False)
