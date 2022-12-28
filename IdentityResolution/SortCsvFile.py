# assign dataset
import csv
from difflib import SequenceMatcher

import pandas as pd

csvData = pd.read_csv("csv_example_output.csv", delimiter=",")

# displaying unsorted data frame
# print("\nBefore sorting:")
# print(csvData)

# sort data frame
csvData.sort_values(["Cluster ID"],
                    axis=0,
                    ascending=[True],
                    inplace=True)



# print("\nAfter sorting:")

# print(csvData)
# with open("FinalData.csv", 'w') as csvfile:
#     # creating a csv writer object
#     csvwriter = csv.writer(csvfile)
#     # writing the fields
#     csvwriter.writerow(csvData)
#
#     # writing the data rows
#     csvwriter.writerows(csvData)
Csv = csv.reader(open("csv_example_output.csv", 'r', encoding='utf-8-sig'))

Name_to_search= 'aditi'

List_of_similar_records = []


for row in Csv:
    if len(row) == 0:
        continue
    else:
        s = SequenceMatcher(None,Name_to_search,row[3])
        if s.ratio() >=.50:
            List_of_similar_records.append(row)

List_of_similar_records = sorted(List_of_similar_records)

for row in List_of_similar_records:
    print(row)

Func = open("GFG-1.html", "w")

# Adding input data to the HTML file
Func.write("<html>\n<head>\n<title> \nOutput Data in an HTML file \
           </title>\n</head> <body>{{List_of_similar_records}}</h1>\
           \n<h2>A <u>CS</u> Portal for Everyone</h2> \n</body></html>")
Func.write(List_of_similar_records)
Func.write("</h1>\
           \n<h2>A <u>CS</u> Portal for Everyone</h2> \n</body></html>")

# Saving the data into the HTML file
Func.close()





