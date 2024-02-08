import numpy as np
import os
import scipy.io

file_path = os.path.dirname(__file__)
datafile_ = scipy.io.loadmat(f'{file_path}/../data/binaryalphadigs.mat')
    
def lire_alpha_digits(characters_to_read, datafile=datafile_, indices=None):
    def character_index(c):
        if c.isnumeric():
            return int(c)
        return ord(c) - ord('A') + 10

    if indices is None:
        indices = [character_index(c) for c in characters_to_read]
    
    X = np.array([datafile['dat'][indices[0]][i].flatten() for i in range(len(datafile['dat'][indices[0]]))])
    for index in indices[1:]:
        X_i = np.array([datafile['dat'][index][i].flatten() for i in range(len(datafile['dat'][index]))])
        X = np.concatenate([X, X_i])
    return X