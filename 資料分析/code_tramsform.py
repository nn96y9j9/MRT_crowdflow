import csv

with open("./dataset/station.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    map_station = {rows[0]: rows[1] for rows in reader}
print(map_station)
