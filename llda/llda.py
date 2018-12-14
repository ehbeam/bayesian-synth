#!/usr/bin/env python

# Labeled latent Dirichlet allocation

# Adapted by Ellie Beam from code available under the MIT License
# (c) 2010-2011 Nakatani Shuyo / Cybozu Labs Inc.

import vocabulary
import functools, os, sys, string, random, numpy, warnings
import pandas as pd

class LLDA:
    def __init__(self, K, alpha, beta, docs, ids, labelset):
        self.alpha = alpha
        self.beta = beta
        self.docs = docs
        self.ids = ids
        self.labelset = labelset

    def term_to_id(self, term):
        if term not in self.vocas_id:
            voca_id = len(self.vocas)
            self.vocas_id[term] = voca_id
            self.vocas.append(term)
        else:
            voca_id = self.vocas_id[term]
        return voca_id

    def complement_label(self, label, labelmap=None):
        if not labelmap: labelmap = self.labelmap
        if not label: return numpy.ones(len(self.labelmap))
        vec = numpy.zeros(len(self.labelmap))
        vec[0] = 1.0
        for x in label: vec[self.labelmap[x]] = 1.0
        return vec

    def set_corpus(self, labelset, corpus, labels):
        labelset.insert(0, "common")
        self.labelmap = dict(zip(labelset, range(len(labelset))))
        self.K = len(self.labelmap)

        self.vocas = []
        self.vocas_id = dict()
        self.labels = numpy.array([self.complement_label(label) for label in labels])
        self.docs = [[self.term_to_id(term) for term in doc] for doc in corpus]

        M = len(corpus)
        V = len(self.vocas)

        self.z_m_n = []
        self.n_m_z = numpy.zeros((M, self.K), dtype=int)
        self.n_z_t = numpy.zeros((self.K, V), dtype=int)
        self.n_z = numpy.zeros(self.K, dtype=int)

        for m, doc, label in zip(range(M), self.docs, self.labels):
            N_m = len(doc)
            z_n = [numpy.random.multinomial(1, label / label.sum()).argmax() for x in range(N_m)]
            self.z_m_n.append(z_n)
            for t, z in zip(doc, z_n):
                self.n_m_z[m, z] += 1
                self.n_z_t[z, t] += 1
                self.n_z[z] += 1

    def inference(self):
        V = len(self.vocas)
        for m, doc, label in zip(range(len(self.docs)), self.docs, self.labels):
            for n in range(len(doc)):
                t = doc[n]
                z = self.z_m_n[m][n]
                self.n_m_z[m, z] -= 1
                self.n_z_t[z, t] -= 1
                self.n_z[z] -= 1

                denom_a = self.n_m_z[m].sum() + self.K * self.alpha
                denom_b = self.n_z_t.sum(axis=1) + V * self.beta
                p_z = label * (self.n_z_t[:, t] + self.beta) / denom_b * (self.n_m_z[m] + self.alpha) / denom_a
                new_z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()

                self.z_m_n[m][n] = new_z
                self.n_m_z[m, new_z] += 1
                self.n_z_t[new_z, t] += 1
                self.n_z[new_z] += 1

    def phi(self):
        V = len(self.vocas)
        return (self.n_z_t + self.beta) / (self.n_z[:, numpy.newaxis] + V * self.beta)

    def theta(self):
        """document-topic distribution"""
        n_alpha = self.n_m_z + self.labels * self.alpha
        return n_alpha / n_alpha.sum(axis=1)[:, numpy.newaxis]

    def worddist(self):
        """topic-word distribution"""
        return self.n_z_t / self.n_z[:, numpy.newaxis]

    def perplexity(self, docs=None):
        if docs == None: docs = self.docs
        phi = self.phi()
        thetas = self.theta()

        log_per = N = 0
        for doc, theta in zip(docs, thetas):
            for w in doc:
                log_per -= numpy.log(numpy.inner(phi[:,w], theta))
            N += len(doc)
        return numpy.exp(log_per / N)

def load_corpus(filename):
    corpus = []
    labels = []
    labelmap = dict()
    f = open(filename, 'r')
    for line in f:
        mt = re.match(r'\[(.+?)\](.+)', line)
        if mt:
            label = mt.group(1).split(',')
            for x in label: labelmap[x] = 1
            line = mt.group(2)
        else:
            label = None
        doc = re.findall(r'\w+(?:\'\w+)?',line.lower())
        if len(doc)>0:
            corpus.append(doc)
            labels.append(label)
    f.close()
    return labelmap.keys(), corpus, labels

def docs2tops(llda, docs=None, ids=None):
    """per-document topic distribution"""
    if docs == None: docs = llda.docs
    if ids == None: ids = llda.ids
    phi = llda.worddist()
    log_per = 0
    N = 0
    Kalpha = llda.K * llda.alpha
    prob = pd.DataFrame(index=range(len(ids)), columns=range(41))
    for m, doc in enumerate(docs):
        theta = llda.n_m_z[m] / (len(llda.docs[m]) + Kalpha) # list of probabilities for each topic
        prob.loc[m] = {i:val for i,val in enumerate(theta)}
    return prob

def output_word_topic_dist(llda):
    phi = llda.phi()
    for k, label in enumerate(llda.labelset):
        print("\n-- Label {}: {}".format(k, label))
        for w in numpy.argsort(-phi[k])[:20]:
            print("{}: {}".format(llda.vocas[w], phi[k,w]))

def llda_learning(llda, name, iteration, voca, anat, op, op_i, tol=0.01):
    pre_perp = llda.perplexity()
    print ("Initial perplexity: %f" % (pre_perp))
    for i in range(iteration):
        llda.inference()
        perp = llda.perplexity()
        print ("-- %d Perplexity: %f" % (i + 1, perp))
        if pre_perp != None:
            if abs(pre_perp - perp) < tol:
                pre_perp = None
            else:
                pre_perp = perp
        else:
            break
    output_word_topic_dist(llda)
    print("\nFitting on validation set...")
    perp = llda.perplexity(docs=llda.docs)
    print ("\nPerplexity: %f\n" % (perp))
    op = pd.read_csv("../llda_perplexity.csv", index_col=None, header=0)
    op.loc[(op["X"] == name) & (op["INDEX"] == op_i), "PERPLEXITY"] = perp
    op.to_csv(path_or_buf="../llda_perplexity.csv", sep=",", index=False)

def run_llda(name, op_i, stopwords=False, seed=42):

    splits = {"train": [], "validate": [], "test": []}
    for split in splits.keys():
        splits[split] = [int(id.strip()) for id in open("../../data/pmids_{}.txt".format(split)).readlines()]

    op = pd.read_csv("../llda_perplexity.csv", index_col=None, header=0)
    alpha = float(op.loc[(op["X"] == name) & (op["INDEX"] == op_i), "ALPHA"])
    beta = float(op.loc[(op["X"] == name) & (op["INDEX"] == op_i), "BETA"])
    iteration = int(op.loc[(op["X"] == name) & (op["INDEX"] == op_i), "ITERATION"])
    n_text = int(op.loc[(op["X"] == name) & (op["INDEX"] == op_i), "N_TEXT"])

    random.seed(seed)
    numpy.random.seed(seed)

    scores = pd.read_csv("../../data/rdoc_scores.csv", index_col=0, header=0)

    rdoc = pd.read_csv("../../data/rdoc_constructs.csv", index_col=None, header=0)
    vocab = sorted(list(set(rdoc["TOKEN"])))

    if "fulltexts" in name:
        dtm = pd.read_csv("../../data/dtm.csv.gz", compression="gzip", index_col=0, header=0)
    if "abstracts" in name:
        dtm = pd.read_csv("../../data/atm.csv", index_col=0, header=0)

    train_labels = []
    train_corpus = []
    for id in splits["train"][:n_text]:
        train_labels.append([lab for lab in scores if scores.loc[id,lab] > 0])
        train_corpus.append([word for word in vocab if dtm.loc[id,word] > 0])
    train_labelset = list(set(reduce(list.__add__, train_labels)))
    K = len(train_labelset)

    voca = vocabulary.Vocabulary(stopwords)
    train_docs = [voca.doc_to_ids(doc) for doc in train_corpus]

    llda = LLDA(K, alpha, beta, train_docs, splits["train"][:n_text], train_labelset)
    llda.set_corpus(train_labelset, train_corpus, train_labels)
    print("M={}, V={}, K={}, alpha={}, beta={}".format(len(train_corpus), len(llda.vocas), K, alpha, beta))

    anat = pd.read_csv("../../data/coordinates.csv", index_col=0, header=0)

    llda_learning(llda, name, iteration, voca, anat, op, op_i)

