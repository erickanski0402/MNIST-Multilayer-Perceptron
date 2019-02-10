import csv

rawData = [];

with open('C:/Users/erick/Desktop/train.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')
    line_count = 0
    for row in csv_reader:
        rawData.append(row);

print(rawData)
