import argparse
from pathlib import Path
import fasttext


# Directory for product data
directory = r'/workspace/datasets/fasttext/'

parser = argparse.ArgumentParser(description='Generate synonyms')
general = parser.add_argument_group("general")
general.add_argument("--model", default=f'{directory}/title_model.bin', help="path to skipgram model")
general.add_argument("--words", default=f'{directory}/top_words.txt', help="path to line-separated words file")
general.add_argument("--output", default=f'{directory}/synonyms.csv', help="comma separated output of words/synonyms")
general.add_argument("--threshold", default="0.75", help="nearest neighbor similarity threshold")

args = parser.parse_args()
model_file = args.model
words_file = Path(args.words)
output_file = Path(args.output)
nn_threshold = float(args.threshold)

if __name__ == '__main__':
    print(f'using threshold: {str(nn_threshold)}')
    print(f'loading words from: {words_file}')
    with open(words_file, 'r') as f:
        word_list = f.readlines()
    print(f'number words: {len(word_list)}')

    print(f'loading model: {model_file}')
    model = fasttext.load_model(model_file)

    print(f'outputting csv list to: {output_file}')
    with open(output_file, 'w') as output:
        for w in word_list:
            word = w.strip()
            nn_words = model.get_nearest_neighbors(word)
            csv_items = [word]
            for nn_word in nn_words:
                if nn_word[0] > nn_threshold:
                    csv_items.append(nn_word[1].strip())
            output.write(','.join(csv_items) + '\n')

