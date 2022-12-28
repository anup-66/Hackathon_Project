import os
import csv
import re
import logging
import optparse

from multiprocessing import freeze_support, active_children
import dedupe
from gitdb.util import read
from unidecode import unidecode

def preProcess(column):

    column = unidecode(column)
    column = re.sub('  +', ' ', column)
    column = re.sub('\n', ' ', column)
    # column = re.sub('ï..', ' ', column)
    column = column.strip().strip('"').strip("'").lower().strip()

    if not column:
        column = None
    return column

def readData(filename):
#
    data_d = {}
    with open(filename,encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        # reader = read.csv(f, fileEncoding="UTF-8-BOM")
        for row in reader:

            clean_row = [(k, preProcess(v)) for (k, v) in row.items()]
            print(row)
            row_id = int(row['Id'])
            data_d[row_id] = dict(clean_row)

    return data_d


if __name__ == '__main__':

    
    input_file = 'Hackathon01.csv'
    output_file = 'csv_example_output.csv'
    settings_file = 'csv_example_learned_settings'
    training_file = 'csv_example_training.json'

    print('importing data ...')
    data_d = readData(input_file)

    if os.path.exists(settings_file):
        print('reading from', settings_file)
        with open(settings_file, 'rb') as f:
            deduper = dedupe.StaticDedupe(f,num_cores=0)
    else:
        fields = [
            {'field': 'Name', 'type': 'String'},
            {'field': 'Address', 'type': 'String'},
            {'field': 'Zip', 'type': 'Exact', 'has missing': True},
            {'field': 'Phone', 'type': 'Exact', 'has missing': True},
        ]
        deduper = dedupe.Dedupe(fields,num_cores=0)

        if os.path.exists(training_file):
            print('reading labeled examples from ', training_file)
            with open(training_file, 'rb') as f:
                deduper.prepare_training(data_d, f)
        else:
            deduper.prepare_training(data_d)

        print('starting active labeling...')

        dedupe.console_label(deduper)

        deduper.train()

        with open(training_file, 'w') as tf:
            deduper.write_training(tf)

        with open(settings_file, 'wb') as sf:
            deduper.write_settings(sf)

        print('clustering...')

        clustered_dupes = deduper.partition(data_d,0.5)
        print(clustered_dupes)
        print('# duplicate sets', len(clustered_dupes))

        cluster_membership = {}
        for cluster_id, (records, scores) in enumerate(clustered_dupes):
            print("cluster_Id",cluster_id,(records,scores))
            for record_id, score in zip(records, scores):
                cluster_membership[record_id] = {
                    "Cluster ID": cluster_id,
                    "confidence_score": score
                }

        with open(output_file, 'w',encoding='utf-8-sig') as f_output, open(input_file,encoding='utf-8-sig') as f_input:

            reader = csv.DictReader(f_input)
            fieldnames = ['Cluster ID', 'confidence_score'] + reader.fieldnames

            writer = csv.DictWriter(f_output, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                row_id = int(row['Id'])
                row.update(cluster_membership[row_id])
                writer.writerow(row)
