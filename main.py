import string
import numpy as np

dok_num = int(input())

dok = []

for i in range(0, dok_num):
    document = input().translate(str.maketrans('', '', string.punctuation)).strip().lower()
    dok.append(document)

words = input()
words_list = words.split(" ")

k = int(input())

unique_words = sorted(set(word for sentence in dok for word in sentence.split())) 
result_matrix = []
for sentence in dok:
    words = set(sentence.split())
    row = [1 if word in words else 0 for word in unique_words]
    result_matrix.append(row)

key_words_list = [1 if word in unique_words and word in words_list else 0 for word in unique_words]


C = np.array(result_matrix).T
U, s, Vt = np.linalg.svd(C, full_matrices=False)

sr = np.copy(s)
sr[k:] = 0
    
Sr = np.diag(sr)
Ck = Sr.dot(Vt)

sk = np.take(s, range(k), axis=0)
Sk = np.diag(sk)

Vk = np.take(Vt, range(k), axis=0)
Ck = Sk.dot(Vk)

q = np.array(key_words_list).T
Sk_1 = np.linalg.inv(Sk)
UkT = np.take(U.T, range(k), axis=0)

reduced_query = Sk_1.dot(UkT).dot(q)


def compute_norm(vector):
    return np.sqrt(np.sum(vector**2))

def cosine_similarity(vec1, vec2):
    norm1 = compute_norm(vec1)
    norm2 = compute_norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))



column_norms = []
column_similarities = []

for doc_vec in Ck.T:
    column_norm = compute_norm(doc_vec)
    column_norms.append(round(column_norm, 4))
    
    similarity = cosine_similarity(reduced_query, doc_vec)
    column_similarities.append(round(similarity, 2))  

print(column_similarities)
