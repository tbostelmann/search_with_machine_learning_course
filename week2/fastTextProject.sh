SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd $SCRIPT_DIR/.. && pwd)"
DATA_DIR="/workspace/datasets/fasttext"
TRAINING_DATA_FULL="$DATA_DIR/pruned_labeled_products.txt"
echo "SCRIPT_DIR=$SCRIPT_DIR"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "DATA_DIR=$DATA_DIR"

set -e

# Create labeled products
if [ ! -f "$TRAINING_DATA_FULL" ]; then
  python $PROJECT_ROOT/week2/createContentTrainingData.py --min_products 500 --output $TRAINING_DATA_FULL
fi

# Randomly shuffle in a deterministic manner
shuf $TRAINING_DATA_FULL --random-source=<(seq 99999) > $DATA_DIR/shuffled_labeled_products.txt

# Normalize label data
cat $DATA_DIR/shuffled_labeled_products.txt | \
  sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" | sed "s/[^[:alnum:]_]/ /g" | \
  tr -s ' ' > $DATA_DIR/normalized_labeled_products.txt

# Split labeled data into training and test.
head -n 10000 $DATA_DIR/normalized_labeled_products.txt > \
  $DATA_DIR/normalized_labeled_products.train
tail -n 10000 $DATA_DIR/normalized_labeled_products.txt > \
  $DATA_DIR/normalized_labeled_products.test

cat $DATA_DIR/normalized_labeled_products.train | wc -l
cat $DATA_DIR/normalized_labeled_products.test | wc -l

# Set word ngrams to 2 to learn from bigrams
~/fastText-0.9.2/fasttext supervised -input $DATA_DIR/normalized_labeled_products.train \
  -output $DATA_DIR/model_project2 -lr 1.0 -epoch 25 -wordNgrams 2
~/fastText-0.9.2/fasttext test $DATA_DIR/model_project2.bin $DATA_DIR/normalized_labeled_products.test

cut -d' ' -f2- $DATA_DIR/shuffled_labeled_products.txt > $DATA_DIR/titles.txt

cat /workspace/datasets/fasttext/titles.txt | \
 sed -e "s/\([.\!?,'/()]\)/ \1 /g" | \
  tr "[:upper:]" "[:lower:]" | \
   sed "s/[^[:alnum:]]/ /g" | \
    tr -s ' ' > $DATA_DIR/normalized_titles.txt

cat /workspace/datasets/fasttext/normalized_titles.txt | tr " " "\n" | grep "...." | sort | uniq -c | sort -nr | \
 head -1000 | grep -oE '[^ ]+$' > /workspace/datasets/fasttext/top_words.txt

~/fastText-0.9.2/fasttext skipgram -epoch 25 -minCount 20 -input $DATA_DIR/normalized_titles.txt -output $DATA_DIR/title_model

python ./week2/createSynonyms.py


