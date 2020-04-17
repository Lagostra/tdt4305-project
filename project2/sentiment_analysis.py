from base64 import b64decode
import re
import argparse

from pyspark import SparkContext, SparkConf
conf = SparkConf()
sc = SparkContext(conf=conf)

parser = argparse.ArgumentParser(description='Analyze sentiment of reviews to find top businesses.')
parser.add_argument('path', type=str)
parser.add_argument('--normalize', '-n', dest='normalize', action='store_true', help='Normalize review scores by review length')
parser.add_argument('-k', type=int, default=10, dest='k', help='The number of businesses to be returned')
parser.add_argument('--reduce', '-r', dest='reduce', action='store_true', help='Reduce the dataset to allow for faster test runs')
parser.add_argument('--fraction', '-f', dest='fraction', type=int, default=0.1, help='Fraction of dataset to be used. Only applicable when --reduce is specified.')
parser.add_argument('--afinn', '-a', dest='afinn', type=str, default='../data/AFINN-111.txt', help='Path to the AFINN file for word polarities.')
parser.add_argument('--stopwords', '-s', dest='stopwords', type=str, default='../data/stopwords.txt', help='Path to the file with english stopwords.')

args = parser.parse_args()

path = args.path
k = args.k
NORMALIZE_SCORES = args.normalize

REDUCE_DATASET = args.reduce
DATASET_FRACTION = args.fraction
afinn_path = args.afinn
stopwords_path = args.stopwords


reviews = sc.textFile(path) \
    .zipWithIndex() \
    .filter(lambda x: x[1] > 0) \
    .map(lambda x: x[0].replace('"', '').split('\t')) \
    .map(lambda row: tuple(b64decode(row[i]).decode('utf8') if i == 3 else row[i] for i in range(len(row))))
# "review_id","user_id","business_id","review_text","review_date"

afinn = sc.textFile(afinn_path) \
    .map(lambda x: x.split('\t')) \
    .map(lambda row: (row[0], int(row[1])))

stopwords = sc.textFile(stopwords_path)

if REDUCE_DATASET:
    count = reviews.count()
    
    reviews = reviews.zipWithIndex() \
        .filter(lambda row: row[1] < count * DATASET_FRACTION) \
        .map(lambda row: row[0])


def tokenize(review):
    return re.sub('[^a-zA-Z ]+', '', review).lower().split()

tokenized = reviews.map(lambda row: ((row[0], row[1], row[2]), row[3])) \
    .flatMapValues(tokenize)

stopwords_removed = tokenized \
    .map(lambda row: (row[1], row[0])) \
    .subtractByKey(stopwords.map(lambda row: (row, row))) \
    .map(lambda row: (row[1], row[0]))


if NORMALIZE_SCORES:
    word_polarity = stopwords_removed \
        .map(lambda row: (row[1], row[0])) \
        .leftOuterJoin(afinn) \
        .map(lambda row: row[1]) \
        .map(lambda row: (row[0], 0 if row[1] is None else row[1]))
else:
    word_polarity = stopwords_removed \
        .map(lambda row: (row[1], row[0])) \
        .join(afinn) \
        .map(lambda row: row[1]) \
        .map(lambda row: (row[0], 0 if row[1] is None else row[1]))


if NORMALIZE_SCORES:
    review_polarity = word_polarity.map(lambda row: (row[0], (row[1], 1))) \
        .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])) \
        .map(lambda row: (row[0], row[1][0] / row[1][1]))
else:
    review_polarity = word_polarity \
        .reduceByKey(lambda x, y: (x + y))


business_ranking = review_polarity.map(lambda row: (row[0][2], row[1])) \
        .reduceByKey(lambda x, y: x + y) \
        .sortBy(lambda row: row[1], ascending=False) \
        .zipWithIndex().filter(lambda row: row[1] < k).map(lambda row: row[0])

business_ranking.map(lambda row: row[0]) \
        .coalesce(1) \
        .saveAsTextFile('ranking_output')

