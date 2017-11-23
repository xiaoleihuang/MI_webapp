from gensim.models import LdaModel
from gensim.corpora import Dictionary
from nltk.tokenize import word_tokenize

def clean_doc(doc, stopwords=False, lemma=False):
    """Clean and tokenize document
    :param doc: one string document
    :param stopwords: whether to use stopwords
    """
    new_doc = word_tokenize(doc.lower())
    if stopwords:
        global stop_filter
        new_doc = [word for word in new_doc if word not in stop_filter]
    if lemma:
        global snowball_stemmer
        new_doc = [snowball_stemmer.stem(word) for word in new_doc]
    return new_doc

class LDA_Infer():
    def __init__(self, modelpath, dictpath, select_topics={1:'family',2:'health',3:'alcohol',4:'drive'}):
        # load lda model
        self.ldamodel = LdaModel.load(modelpath)
        self.select_topics = select_topics
        self.infer_dict = Dictionary.load(dictpath)

    def infer(self, input_doc):
        new_doc = clean_doc(input_doc)
        new_doc = self.infer_dict.doc2bow(new_doc)
        topic_dist = self.ldamodel[new_doc]

        selected_dists = dict.fromkeys(list(self.select_topics.values()), 0.0)
        for pairs in topic_dist:
            if pairs[0] in self.select_topics:
                selected_dists[self.select_topics[pairs[0]]] = pairs[1]
        return selected_dists

class LIWC_Infer():
    pass