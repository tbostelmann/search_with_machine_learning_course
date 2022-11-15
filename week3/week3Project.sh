set -e

labeled_queries_file_name='/workspace/datasets/fasttext/labeled_queries.txt'
shuffled_labeled_queries_file_name='/workspace/datasets/fasttext/shuffled_labeled_queries'

#cut -d' ' -f1 /workspace/datasets/fasttext/labeled_queries.txt | sort | uniq | wc

shuf /workspace/datasets/fasttext/labeled_queries.txt --random-source=<(seq 99999999) > \
  "$shuffled_labeled_queries_file_name.txt"

# Split labeled data into training and test.
head -n 200000 "$shuffled_labeled_queries_file_name.txt" > \
  "$shuffled_labeled_queries_file_name.train"
tail -n 50000 "$shuffled_labeled_queries_file_name.txt" > \
  "$shuffled_labeled_queries_file_name.test"

wc "$shuffled_labeled_queries_file_name.train"
wc "$shuffled_labeled_queries_file_name.test"

~/fastText-0.9.2/fasttext supervised -input "$shuffled_labeled_queries_file_name.train" \
  -output '/workspace/datasets/fasttext/model_project3' -lr 0.45 -epoch 25
# Note: As we increase the number of epochs p1 suffers but p3/p5 improve
# using min_queries 1000, epoch 50 and train/test size of 50000/10000:
#N       9973
#P@1     0.429
#R@1     0.429
#N       9973
#P@3     0.203
#R@3     0.61
#N       9973
#P@5     0.135
#R@5     0.677
# Note: Using min_queries 1000, epoch of 25 and train/test size of 10000/10000:
#N       7258
#P@1     0.364
#R@1     0.364
#N       7258
#P@3     0.172
#R@3     0.517
#N       7258
#P@5     0.12
#R@5     0.599
# Using min_queries 1000, epoch 25 and train/test size of 50000/10000:
#N       9973
#P@1     0.423
#R@1     0.423
#N       9973
#P@3     0.203
#R@3     0.61
#N       9973
#P@5     0.137
#R@5     0.684
# Note: Using min_queries 1000, epoch of 25 and train/test size of 100000/10000:
#N       10000
#P@1     0.439
#R@1     0.439
#N       10000
#P@3     0.212
#R@3     0.636
#N       10000
#P@5     0.142
#R@5     0.712
# Note: Using min_queries 1000, epoch of 25 and train/test size of 200000/50000:
#N       50000
#P@1     0.482
#R@1     0.482
#N       50000
#P@3     0.227
#R@3     0.68
#N       50000
#P@5     0.15
#R@5     0.751
# Note: Using min_queries 10000, epoch of 25 and train/test size of 200000/50000:
#N       50000
#P@1     0.554
#R@1     0.554
#N       50000
#P@3     0.258
#R@3     0.775
#N       50000
#P@5     0.168
#R@5     0.842
#~/fastText-0.9.2/fasttext supervised -input "$shuffled_labeled_queries_file_name.train" \
#  -output '/workspace/datasets/fasttext/model_project3' -lr 0.5 -epoch 50
~/fastText-0.9.2/fasttext test '/workspace/datasets/fasttext/model_project3.bin' \
  "$shuffled_labeled_queries_file_name.test" >> /workspace/datasets/fasttext/model_project3.test.1
tail -3 /workspace/datasets/fasttext/model_project3.test.1
~/fastText-0.9.2/fasttext test '/workspace/datasets/fasttext/model_project3.bin' \
  "$shuffled_labeled_queries_file_name.test" 3 >> /workspace/datasets/fasttext/model_project3.test.3
tail -3 /workspace/datasets/fasttext/model_project3.test.3
~/fastText-0.9.2/fasttext test '/workspace/datasets/fasttext/model_project3.bin' \
  "$shuffled_labeled_queries_file_name.test" 5 >> /workspace/datasets/fasttext/model_project3.test.5
tail -3 /workspace/datasets/fasttext/model_project3.test.5
