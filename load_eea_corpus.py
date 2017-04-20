#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/chartbeat-labs/textacy
# make sure you have downloaded the language model
# $ python -m spacy.en.download all

from __future__ import print_function
from pyLDAvis import show
import eea_corpus
import textacy
import vis

corpus = eea_corpus.EEACorpus().load_or_create_corpus('data.csv')

print('Corpus: ', corpus)

# These are the metadata columns
# [(0, u'expires'),
#  (1, u'description'),
#  (2, u'issued'),
#  (3, u'modified'),
#  (4, u'Regions/Places/Cities/Seas...'),
#  (5, u'Countries'),
#  (6, u'WorkflowState'),
#  (7, u'topics'),
#  (8, u'url'),
#  (9, u'Content types'),
#  (10, u'Time coverage'),
#  (11, u'format'),
#  (12, u'organisation'),
#  (13, u'language')]


def published_match_func(doc):
    return doc.metadata[6] == 'published'

# find published docs
for doc in corpus.get(published_match_func, limit=3):
    triples = textacy.extract.subject_verb_object_triples(doc)
    print('Published doc: ', doc, list(triples))


def url_match_func(url):
    return lambda doc: doc.metadata[8] == url

# find doc with specific url
url = 'http://www.eea.europa.eu/publications/C23I92-826-5409-5'
for doc in corpus.get(url_match_func(url), limit=3):
    print('specific url:', doc)

# get terms list
for doc in corpus.get(url_match_func(url), limit=3):
    tlist = doc.to_terms_list(ngrams=1, named_entities=True, as_strings=True)
    terms = list(tlist)
    print(terms)

# Represent corpus as a document-term matrix, with flexible weighting and
# filtering:
docs = (
    doc.to_terms_list(ngrams=1, named_entities=True, as_strings=True)
    for doc in corpus
)
doc_term_matrix, id2term = textacy.vsm.doc_term_matrix(docs, weighting='tfidf',
                                                       normalize=True,
                                                       smooth_idf=True,
                                                       min_df=2, max_df=0.95,
                                                       max_n_terms=100000)

print('DTM: ', repr(doc_term_matrix))


# Train and interpret a topic model:

# model = textacy.tm.TopicModel('nmf', n_topics=10)
model = textacy.tm.TopicModel('lda', n_topics=10)
model.fit(doc_term_matrix)

# Transform the corpus and interpret our model:
doc_topic_matrix = model.transform(doc_term_matrix)

print('DocTopicMatrix shape', doc_topic_matrix.shape)

print('Discovered topics:')
for topic_idx, top_terms in model.top_topic_terms(id2term, top_n=10):
    print('topic', topic_idx, ':', '   '.join(top_terms))

# Show top 2 doc within first 2 topics
for (topic_idx, top_docs) in model.top_topic_docs(doc_topic_matrix,
                                                  topics=[0, 1],
                                                  top_n=2):
    print(topic_idx)
    for j in top_docs:
        print(corpus[j].metadata[7])

prep_data = vis.prepare(model.model, doc_term_matrix, id2term)
show(prep_data)

# model.save('model.saved')
# model.termite_plot(doc_term_matrix, id2term, topics=-1, n_terms=2,
#                    sort_terms_by="seriation")
