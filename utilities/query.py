# A simple client for querying driven by user input on the command line.  Has hooks for the various
# weeks (e.g. query understanding).  See the main section at the bottom of the file
from opensearchpy import OpenSearch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import argparse
import json
import os
from getpass import getpass
from urllib.parse import urljoin
import pandas as pd
import re
import sys
import logging
import fasttext
import nltk
from sentence_transformers import SentenceTransformer


logging.basicConfig(filename="query.log", format='%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

stemmer = nltk.stem.PorterStemmer()
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# expects clicks and impressions to be in the row
def create_prior_queries_from_group(
        click_group):  # total impressions isn't currently used, but it mayb worthwhile at some point
    click_prior_query = ""
    # Create a string that looks like:  "query": "1065813^100 OR 8371111^89", where the left side is the doc id and the right side is the weight.  In our case, the number of clicks a document received in the training set
    if click_group is not None:
        for item in click_group.itertuples():
            try:
                click_prior_query += "%s^%.3f  " % (item.doc_id, item.clicks / item.num_impressions)

            except KeyError as ke:
                pass  # nothing to do in this case, it just means we can't find priors for this doc
    return click_prior_query


# expects clicks from the raw click logs, so value_counts() are being passed in
def create_prior_queries(doc_ids, doc_id_weights,
                         query_times_seen):  # total impressions isn't currently used, but it mayb worthwhile at some point
    click_prior_query = ""
    # Create a string that looks like:  "query": "1065813^100 OR 8371111^89", where the left side is the doc id and the right side is the weight.  In our case, the number of clicks a document received in the training set
    click_prior_map = ""  # looks like: '1065813':100, '8371111':809
    if doc_ids is not None and doc_id_weights is not None:
        for idx, doc in enumerate(doc_ids):
            try:
                wgt = doc_id_weights[doc]  # This should be the number of clicks or whatever
                click_prior_query += "%s^%.3f  " % (doc, wgt / query_times_seen)
            except KeyError as ke:
                pass  # nothing to do in this case, it just means wloade can't find priors for this doc
    return click_prior_query


def create_vector_query(encoded_query, nearest_neighbor_range):
    query_obj = {
        "size": nearest_neighbor_range,
        "query": {
            "bool": {
                "filter": {
                    "bool": {
                        "must": [
                            {
                                "term": {
                                    "onlineAvailability": "false"
                                }
                            }
                        ]
                    }
                },
                "must": [
                    {
                        "knn": {
                            "embedding": {
                                "vector": encoded_query[0].tolist(),
                                "k": nearest_neighbor_range
                            }
                        }
                    }
                ]
            }
        }
    }
    return query_obj


# Hardcoded query here.  Better to use search templates or other query config.
def create_query(user_query, category_predictions, click_prior_query, filters, sort="_score", sortDir="desc", size=10, source=None, useSynonyms=False, use_filter=True):
    if useSynonyms:
        match_name = 'name.synonyms'
    else:
        match_name = 'name'

    query_obj = {
        'size': size,
        "sort": [
            {sort: {"order": sortDir}}
        ],
        "query": {
            "function_score": {
                "query": {
                    "bool": {
                        "must": [
                        ],
                        "should": [  #
                            {
                                "match": {
                                    match_name: {
                                        "query": user_query,
                                        "fuzziness": "1",
                                        "prefix_length": 2,
                                        # short words are often acronyms or usually not misspelled, so don't edit
                                        "boost": 0.01
                                    }
                                }
                            },
                            {
                                "match_phrase": {  # near exact phrase match
                                    "name.hyphens": {
                                        "query": user_query,
                                        "slop": 1,
                                        "boost": 50
                                    }
                                }
                            },
                            {
                                "multi_match": {
                                    "query": user_query,
                                    "type": "phrase",
                                    "slop": "6",
                                    "minimum_should_match": "2<75%",
                                    "fields": [f"{match_name}^10", "name.hyphens^10", "shortDescription^5",
                                               "longDescription^5", "department^0.5", "sku", "manufacturer", "features",
                                               "categoryPath"]
                                }
                            },
                            {
                                "terms": {
                                    # Lots of SKUs in the query logs, boost by it, split on whitespace so we get a list
                                    "sku": user_query.split(),
                                    "boost": 50.0
                                }
                            },
                            {  # lots of products have hyphens in them or other weird casing things like iPad
                                "match": {
                                    "name.hyphens": {
                                        "query": user_query,
                                        "operator": "OR",
                                        "minimum_should_match": "2<75%"
                                    }
                                }
                            }
                        ],
                        "minimum_should_match": 1,
                        "filter": filters  #
                    }
                },
                "boost_mode": "multiply",  # how _score and functions are combined
                "score_mode": "sum",  # how functions are combined
                "functions": [
                    {
                        "filter": {
                            "exists": {
                                "field": "salesRankShortTerm"
                            }
                        },
                        "gauss": {
                            "salesRankShortTerm": {
                                "origin": "1.0",
                                "scale": "100"
                            }
                        }
                    },
                    {
                        "filter": {
                            "exists": {
                                "field": "salesRankMediumTerm"
                            }
                        },
                        "gauss": {
                            "salesRankMediumTerm": {
                                "origin": "1.0",
                                "scale": "1000"
                            }
                        }
                    },
                    {
                        "filter": {
                            "exists": {
                                "field": "salesRankLongTerm"
                            }
                        },
                        "gauss": {
                            "salesRankLongTerm": {
                                "origin": "1.0",
                                "scale": "1000"
                            }
                        }
                    },
                    {
                        "script_score": {
                            "script": "0.0001"
                        }
                    }
                ]

            }
        }
    }

    logger.info(f"category_predictions: {category_predictions} - use_filter: {use_filter}")
    logger.info(f"bool: {bool(category_predictions and use_filter)}")
    if category_predictions and use_filter:
        query_obj["query"]["function_score"]["query"]["bool"]["must"] = [
            {
                "terms": {
                    "categoryPathIds": category_predictions
                }
            }
        ]

    if click_prior_query is not None and click_prior_query != "":
        query_obj["query"]["function_score"]["query"]["bool"]["should"].append({
            "query_string": {
                # This may feel like cheating, but it's really not, esp. in ecommerce where you have all this prior data,  You just can't let the test clicks leak in, which is why we split on date
                "query": click_prior_query,
                "fields": ["_id"]
            }
        })
    if user_query == "*" or user_query == "#":
        # replace the bool
        try:
            query_obj["query"] = {"match_all": {}}
        except:
            print("Couldn't replace query for *")
    if source is not None:  # otherwise use the default and retrieve all source
        query_obj["_source"] = source
    return query_obj


def search(client, user_query, classifier_model,
           index="bbuy_products", sort="_score", sortDir="desc",
           useSynonyms=False, category_filter=True, is_vector_search=False):
    #### W3: classify the query
    #### W3: create filters and boosts
    # Note: you may also want to modify the `create_query` method above
    category_query = re.sub('[^a-zA-Z0-9]+', ' ', query)
    category_query = re.sub(' +', ' ', category_query)
    category_query = stemmer.stem(category_query)
    category_predictions = classifier_model.predict(category_query, k=3, threshold=0.1)
    logger.info(f'category_predictions: {category_predictions}')
    total_prediction = 0
    categories = []
    for (i, p) in enumerate(category_predictions[0]):
        if total_prediction < 0.6:
            total_prediction += category_predictions[1][i]
            categories.append(str(category_predictions[0][i]).replace('__label__', ''))
    logger.info(f'categories: {categories}')

    if is_vector_search:
        encoded_query = sentence_model.encode([user_query])
        requested_results = 10
        nearest_neighbor_range = requested_results
        while True:
            query_obj = create_vector_query(encoded_query, nearest_neighbor_range)
            logger.info(json.dumps(query_obj))
            response = client.search(query_obj, index=index)
            if response and response['hits']['hits'] and len(response['hits']['hits']) >= requested_results:
                break
            else:
                nearest_neighbor_range = nearest_neighbor_range * 2
        print_results(response)
    else:
        query_obj = create_query(user_query, categories, click_prior_query=None, filters=None, sort=sort, sortDir=sortDir, source=["name", "shortDescription", "categoryPathIds"], useSynonyms=useSynonyms, use_filter=category_filter)
        logger.info(json.dumps(query_obj))
        response = client.search(query_obj, index=index)
        print_results(response)


def print_results(response, abbreviated=True):
    if response and response['hits']['hits'] and len(response['hits']['hits']) > 0:
        hits = response['hits']['hits']
        for (i, hit) in enumerate(hits, start=1):
            if abbreviated:
                print(f'\'pos\': {i}, \'score\': {hit["_score"]}], \'name\': {hit["_source"]["name"]}, \'onlineAvailability\': {hit["_source"]["onlineAvailability"]}, \'categoryPathIds\': {hit["_source"]["categoryPathIds"]}')
            else:
                print(json.dumps(hit, indent=2))


if __name__ == "__main__":
    host = 'localhost'
    port = 9200
    auth = ('admin', 'admin')  # For testing only. Don't store credentials in code.
    parser = argparse.ArgumentParser(description='Build LTR.')
    general = parser.add_argument_group("general")
    general.add_argument("-i", '--index', default="bbuy_products",
                         help='The name of the main index to search')
    general.add_argument("-s", '--host', default="localhost",
                         help='The OpenSearch host name')
    general.add_argument("-p", '--port', type=int, default=9200,
                         help='The OpenSearch port')
    general.add_argument('--user',
                         help='The OpenSearch admin.  If this is set, the program will prompt for password too. If not set, use default of admin/admin')
    general.add_argument('--synonyms', action='store_true')
    general.add_argument('--no-synonyms', dest='synonyms', action='store_false')
    general.add_argument('--model-file', dest='model_file', default="/workspace/datasets/fasttext/model_project3.bin")
    general.add_argument('-v', '--vector', dest='vector', action='store_true', default=False, help='run vector query')
    general.set_defaults(synonyms=False)

    args = parser.parse_args()

    if len(vars(args)) == 0:
        parser.print_usage()
        exit()

    use_synonyms = args.synonyms
    host = args.host
    port = args.port
    if args.user:
        password = getpass()
        auth = (args.user, password)

    base_url = "https://{}:{}/".format(host, port)
    opensearch = OpenSearch(
        hosts=[{'host': host, 'port': port}],
        http_compress=True,  # enables gzip compression for request bodies
        http_auth=auth,
        # client_cert = client_cert_path,
        # client_key = client_key_path,
        use_ssl=True,
        verify_certs=False,  # set to true if you have certs
        ssl_assert_hostname=False,
        ssl_show_warn=False,

    )
    index_name = args.index
    model_file = args.model_file
    is_vector_search = args.vector

    model = fasttext.load_model(model_file)

    query_prompt = "\nEnter your query (type 'Exit' to exit or hit ctrl-c):"
    print(query_prompt)
    toggle_filter_re = r'cf=([True|False])+'
    toggle_vector_search_re = r'v=([True|False])'
    for line in sys.stdin:
        cf = True
        cf_match = re.match(toggle_filter_re, line)
        if cf_match:
            cf = bool(cf_match.groups()[0])
            line.replace(f'cf={str(cf)}', '')
        v_match = re.match(toggle_vector_search_re, line)
        v = True
        if v_match:
            v = bool(v_match.groups()[0])
            line.replace(f'v={str(v)}')
        query = line.rstrip()
        logger.info(f'query: {query}')
        if query == "Exit":
            break
        search(client=opensearch, user_query=query, classifier_model=model, index=index_name, useSynonyms=use_synonyms, category_filter=cf, is_vector_search=v)

        print(query_prompt)
