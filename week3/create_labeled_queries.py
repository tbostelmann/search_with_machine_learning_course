import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv
import re

# Useful if you want to perform stemming.
import nltk
stemmer = nltk.stem.PorterStemmer()

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
output_file_name = r'/workspace/datasets/fasttext/labeled_queries.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")

args = parser.parse_args()
output_file_name = args.output

if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
queries_df = pd.read_csv(queries_file_name)[['category', 'query']]
queries_df = queries_df[queries_df['category'].isin(categories)]

# IMPLEMENT ME: Convert queries to lowercase, and optionally implement other normalization, like stemming.
queries_df['query'] = queries_df['query'].str.lower()
regex_pat = re.compile('[^a-zA-Z0-9]+')
queries_df['query'] = queries_df['query'].str.replace(regex_pat, ' ', regex=True)
regex_pat = re.compile(' +')
queries_df['query'] = queries_df['query'].str.replace(regex_pat, ' ', regex=True)
queries_df['query'] = queries_df['query'].apply(lambda x: stemmer.stem(x))


def pick_column(x):
    if x['count'] < min_queries:
        return x['parent']
    else:
        return x['category']


# IMPLEMENT ME: Roll up categories to ancestors to satisfy the minimum number of queries per category.
while True:
    cat_count_df = queries_df.groupby(['category'])['category'].count()
    cat_count_df = cat_count_df.reset_index(name='count')
    if len(cat_count_df[cat_count_df['count'] < min_queries].index) == 0:
        break
    else:
        print('rolling up min_queries to parents...')

    queries_df = pd.merge(queries_df, cat_count_df, on='category', how='outer')
    queries_df = pd.merge(queries_df, parents_df, on='category', how='left')
    queries_df['count'] = pd.to_numeric(queries_df['count'])
    queries_df['category'] = queries_df.apply(lambda row: pick_column(row), axis=1)
    queries_df.drop('count', axis=1, inplace=True)
    queries_df.drop('parent', axis=1, inplace=True)

queries_df = queries_df[queries_df['category'].notnull()]
queries_df = queries_df[queries_df['category'] != 'cat00000']
queries_df = queries_df[queries_df['query'].str.replace(' ', '') != '']

# Create labels in fastText format.
queries_df['label'] = '__label__' + queries_df['category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
queries_df = queries_df[queries_df['category'].isin(categories)]
queries_df['output'] = queries_df['label'] + ' ' + queries_df['query']
queries_df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)
