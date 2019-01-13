import sys
import numpy as np
import scipy.stats
from tqdm import tqdm


def main(embedding_file_name, verification_file_name):
    embeddings = dict()  # String -> numpy array

    print(f'Loading embeddings from {embedding_file_name} ...')
    with open(embedding_file_name) as lines:
        split = lines.readline().split()
        dictionary_size = int(split[0])

        for line in tqdm(lines, total=dictionary_size, unit="words"):
            split = line.split()
            word = split[0]
            embedding = np.array([float(val)
                                  for val in split[1:]], dtype=float)
            embeddings[word] = embedding

    actual_values = []
    predicted_values = []
    oov_skip_count = 0
    oov_words = set()

    with open(verification_file_name) as lines:
        lines.readline()  # Abandon header

        for line in lines:
            split = line.split(',')
            word1, word2, actual_similarity = line.split(',')

            if word1 not in embeddings:
                oov_words.add(word1)
                oov_skip_count += 1
                continue

            if word2 not in embeddings:
                oov_words.add(word2)
                oov_skip_count += 1
                continue

            predicted_similarity = np.dot(embeddings[word1], embeddings[word2])/(
                np.linalg.norm(embeddings[word1])*np.linalg.norm(embeddings[word2]))
            actual_values.append(float(actual_similarity))
            predicted_values.append(predicted_similarity)

    spearman_rho, _ = scipy.stats.spearmanr(actual_values, predicted_values)
    pearson_r, _ = scipy.stats.pearsonr(actual_values, predicted_values)

    print(f'Spearman\'s rho: {spearman_rho}')
    print(f'Pearson\'s r: {pearson_r}')
    print(f'Skipped {oov_skip_count} verification pair(s) due to OOV.')
    print(f'Missing words: {oov_words}')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage: python wordsim_evaluate.py embedding_file_name.txt verification_file_name.csv')
        sys.exit(1)

    embedding_file_name, verification_file_name = sys.argv[1], sys.argv[2]
    main(embedding_file_name, verification_file_name)

