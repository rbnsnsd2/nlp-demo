from os import path
import pickle

from utils import dctConstr

root = path.dirname(path.abspath(__file__))
rel_path = path.join(root, 'models/')

with open(rel_path+'literary_monsters.pkl', 'rb') as f:
    clf_monster = pickle.load(f)
with open(rel_path+'text_vectorizer.pkl', 'rb') as f:
    textVectorizer = pickle.load(f)

