from typing import Set, Dict, Tuple, Optional, List
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# dataset here : https://fasttext.cc/docs/en/crawl-vectors.html

def load_embedding(fname: str, limit: Optional[int] = None) -> Tuple[List[str], np.ndarray]:
    """
    Open a .vec files `fname` and returns a list of words and a numpy matrix
    limit is an optional integer to limit the vocabulary size
    (and take only the first embeddings)
    """
    with open(fname, "r") as f:
        count, dim = map(int, next(f).strip("\n").split())
        limit = count if limit is None else limit
        words = []
        output = np.zeros((limit, dim))
        for line in f:
            word, vec = line.strip("\n").split(" ", 1)
            vec = np.asarray(np.fromstring(vec, sep=" ", dtype="float"))
            output[len(words)] = vec
            words.append(word)
            if len(words) == limit:
                break
    return words, output[: len(words)]


# stockage des mots et de la matrice dans des variables
words = load_embedding("wiki.fr.vec", 200)[0]
matrix = load_embedding("wiki.fr.vec", 200)[1]

# affichage des mots
#print(words)
#print(matrix)
#print(matrix.shape)


# application de la reduction de dimension et verification de la dimension obtenue
reduced_matrix = TSNE(init="random", learning_rate="auto").fit_transform(matrix)
print(reduced_matrix.shape) # renvoie (200,2)

# vecteurs correspondant aux deux colonnes de la matrice
X = [row[0] for row in reduced_matrix]
Y = [row[1] for row in reduced_matrix]


# creation et affichage du nuage de points
plt.scatter(X, Y)
plt.title("Representation des 200 premiers mots")
# annotation du nuage de points avec les mots représentés
for i in range(len(words)):
    plt.annotate(words[i], (X[i], Y[i]))
plt.show()
@bbb
