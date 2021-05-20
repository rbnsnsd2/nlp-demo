import re
import numpy as np
from collections import Counter, defaultdict

def split_by_paragraphs(data:str) -> []:
    processed = data.lower()
    while '\n\n\n' in processed:
        processed = processed.replace('\n\n\n','\n\n')
    out = processed.split('\n\n')
    return [o.replace("\n", " ") for o in out]


class dctConstr():

    def __init__(self, **kwargs):
        self.tfidf_state = False
        self.UNKN = '\u22b9'
        self.specialchar = {
            'UNKN': u'\u22b9'
        }

        self.terms = self.specialchar
        self.counts = Counter(list(self.specialchar.values()))
        kwargs = {**kwargs}
        self.num_terms = len(self.terms)

        if kwargs.get("stop_words"):
            self.stop_words = [term for term in kwargs["stop_words"]]
        else:
            self.stop_words = []

        if kwargs.get("ignore_case"):
            self.case = kwargs["ignore_case"]
        else:
            self.case = False

        if kwargs.get("char_level"):
            self.char_level = kwargs["char_level"]
        else:
            self.char_level = False

    def _sub_punctuation(self, document):
        doc = re.sub(r"[\!\_\.\n\'\:\;,\?]+", " ", document)
        return doc

    def _segmenter(self, document):
        if self.case is True:
            document = document.lower()
        doc = self._sub_punctuation(document)
        seg = re.split(r"\s+", doc)
        seg = [i for i in seg if i]
        if self.stop_words:
            seg = self._remove_stop_words(seg)
        return seg

    def _remove_stop_words(self, terms):
        """remove terms from list if in stop_words
        """
        _keepers = [t for t in terms if t not in self.stop_words]
        return _keepers

    def _term_inator(self, _counter):
        """re-index terms-idx
        """
        _terms = {k: i for i, k in enumerate(_counter.keys(), start=0)}
        _indices = {k: i for i, k in _terms.items()}
        self.terms = _terms
        self.indices = _indices
        self.num_terms = len(_terms)

    def trimmer(self, **kwargs):
        """top refers to the most frequent percentage of terms
        bottom to the least frequent percentage of terms
        min is the lowest number of occurances to keep
        """
        kwargs = {**kwargs}
        _len = len(self.counts)
        
        if kwargs.get("max_num"):
            self.counts = Counter(
                el for el in self.counts.elements() if self.counts[el] <= kwargs["max_num"])
        if kwargs.get("min_num"):
            self.counts = Counter(
                el for el in self.counts.elements() if self.counts[el] >= kwargs["min_num"])

        print(f"before trim number of terms: {_len}")
        print(f"after trim: {len(self.counts)}")

        self._term_inator(self.counts)

    def constructor(self, document):
        """the constructor can be used iteratively
        """
        if self.char_level is False:
            _terms = self._segmenter(document)
        else:
            _terms = document

        self.counts.update(_terms)
        self._term_inator(self.counts)

    def terms_to_idx(self, terms):
        cat_idx = [self.terms.get(term, 0) for term in terms]
        return cat_idx

    def _terms_to_bow(self, terms):
        _idx = [self.terms.get(term, 0) for term in terms]
        _count = Counter(_idx)
        _bow = sorted([
            (idx, ct) for idx, ct in zip(_count.keys(), _count.values())
            ])
        return _bow

    def bow_to_vec(self, bow):
        vec = np.zeros(self.num_terms + 1)
        for idx, _ct in bow:
            vec[idx] = _ct
        return vec

    def to_idx(self, document):
        _terms = self._segmenter(document)
        return self.terms_to_idx(_terms)

    def vec_to_terms(self, vec):
        _strings = [self.indices.get(item, 0) for item in vec]
        if self.char_level is True:
            _doc = "".join(_strings)
        else:
            _doc = " ".join(_strings)
        return _doc

    def to_count_vec(self, document):
        _terms = self._segmenter(document)
        _count = Counter(_terms)
        _count_vec = [_count[i] for i in self.terms]
        return _count_vec

    def __call__(self, document):
        if self.char_level is True:
            char_idx = [self.terms.get(term, 0) for term in document]
            return char_idx
        else:
            _terms = self._segmenter(document)
            bow = self._terms_to_bow(_terms)
            return bow