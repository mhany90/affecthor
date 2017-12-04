#!/usr/bin/python3

import pandas as pd
import numpy as np
import argparse
from datetime import datetime

def get_embeddings(efile_name):
    df = pd.read_csv(efile_name, sep = '\t', header=None, skiprows=1, quoting=3)
    embeddings = df.iloc[:,:df.shape[1]-1].astype("float64")
    vocab = df.iloc[:,df.shape[1]-1].astype("str")
    return embeddings, vocab

def compare(em1, v1, em2, v2):
    if (em1.shape[1] > em2.shape[1]):
        return em1, v1, em2, v2
    elif (em2.shape[1] > em1.shape[1]):
        return em2, v2, em1, v1
    elif (em1.shape[1] == em2.shape[1]):
        if (em1.shape[0] > em2.shape[0]):
            return em1, v1, em2, v2
        else:
            return em2, v2, em1, v1
    else:
        return em1, v1, em2, v2

def avg_embeddings(em1, em2):
    if (em1.shape[1] != em2.shape[1]):
        cut = em1.iloc[:,em2.shape[1]:]
        avg = np.mean(np.array([em1.iloc[:,:em2.shape[1]].values, em2.values]), axis=0)
        return pd.Series(np.concatenate((avg, cut), axis=1)[0])
    else:
        avg = np.mean(np.array([em1.values, em2.values]), axis=0)[0]
        return pd.Series(avg)

def output_embeddings(efile1, efile2):
    em1, v1 = get_embeddings(efile1)
    em2, v2 = get_embeddings(efile2)
    em1, v1, em2, v2 = compare(em1, v1, em2, v2)

    avgd = pd.DataFrame()

    for v in v1:
        if len(v) < 140:
            try:
                i1 = v1[v1 == v].index[0]
                i2 = v2[v2 == v].index[0]
                avgd_ems = avg_embeddings(em1.iloc[[i1]], em2.iloc[[i2]])
                avgd_vec = avgd_ems.append(pd.Series(v1[i1]), ignore_index=True)
                avgd = avgd.append(avgd_vec, ignore_index=True)
                #print(i1,"\t",avgd.shape)
            except IndexError:
                i1 = v1[v1 == v].index[0]
                srs = em1.iloc[i1].append(pd.Series(v1[i1]), ignore_index=True)
                avgd = avgd.append(srs, ignore_index=True)
                #print(i1,"\t",avgd.shape)

    return avgd

parser = argparse.ArgumentParser(description="""Averages two sets of word embeddings. If size of embeddings is not equal, then averages the dimensions of shorter set and concatenates the remainder. Vocabulary of longer embeddings set is used. Otherwise, averages equally sized embeddings and retains the bigger vocabulary of the two. Embeddings have to be tab separated with embeddings, term by line.""")
parser.add_argument("--efile1", help="path to first embedding file", required=True)
parser.add_argument("--efile2", help="path to second embedding file", required=True)
parser.add_argument("--saveas", help="name for averaged embedding file", required=False)
startTime = datetime.now()

args = parser.parse_args()
efile1 = str(args.efile1)
efile2 = str(args.efile2)
saveas = str(args.saveas)
avgd = output_embeddings(efile1, efile2)
avgd.to_csv(saveas, sep="\t", header=False, index=False)
print("Wrote embeddings of dim",avgd.shape,"to",saveas,".")
print("Took ", datetime.now() - startTime)
