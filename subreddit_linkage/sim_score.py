import json
from collections import defaultdict
import heapq

from scipy import stats

import luigi
import numpy as np
from tqdm import tqdm

from subreddit_linkage.subreddit_graph import SubredditGraphEdges


class SimScore(luigi.Task):
    sub = luigi.Parameter()

    def requires(self):
        return SubredditGraphEdges()

    def score_sub(self, sub_id, given_sums, means, smoothing=0):
        dist = stats.binom(given_sums.sum(), means[sub_id])
        mu = dist.mean()
        sd = dist.std()
        dist = (given_sums[sub_id] - smoothing - mu) / sd

        return dist

    def run(self):
        mat, sub_data, author_data = SubredditGraphEdges().load_recmat()
        given_sub_ix = sub_data[self.sub]

        sum_visit_sub = np.asarray(mat.sum(0)).squeeze()
        p_visit_sub = sum_visit_sub / sum_visit_sub.sum()

        ns = np.nonzero(mat[:, given_sub_ix])[0]
        sum_visit_sub_given = np.asarray(mat[ns].sum(0)).squeeze()

        scorings = {}
        for sub, id in tqdm(sub_data.items(), desc='Calculating scores...'):
            score = self.score_sub(id, sum_visit_sub_given, p_visit_sub)
            scorings[sub] = score

        for item, _ in sorted(scorings.items(), key=lambda v: -v[1])[:6]:
            print(item)

class AllSimScores(luigi.Task):
    def requires(self):
        return SubredditGraphEdges()

    def output(self):
        return luigi.LocalTarget('cache/scores.json')

    @staticmethod
    def score_sub(sub_id, given_sums, means, smoothing=0):
        dist = stats.binom(given_sums.sum(), means[sub_id])
        mu = dist.mean()
        sd = dist.std()
        dist = (given_sums[sub_id] - mu - smoothing) / sd
        return dist

    @staticmethod
    def score_subs(given_sums, visit_prob, smoothing=0):
        n = given_sums.sum()
        means = n * visit_prob
        std = np.sqrt(n * visit_prob * (1-visit_prob))
        dist = (given_sums - means - smoothing) / std
        return dist

    def run(self):
        mat, sub_data, author_data = SubredditGraphEdges().load_recmat()
        inv_sub = {v: k for k, v in sub_data.items()}

        sum_visit_sub = np.asarray(mat.sum(0)).squeeze()
        p_visit_sub = sum_visit_sub / sum_visit_sub.sum()
        top_subs = np.argsort(-p_visit_sub)[:1000]

        sub_scores = {}
        bar = tqdm(total=top_subs.shape[0])
        for sub1, ix1 in sub_data.items():
            if ix1 in top_subs:
                bar.update()
                bar.set_description(sub1)
                ns = np.nonzero(mat[:, ix1])[0]
                sum_visit_sub_given = np.asarray(mat[ns].sum(0)).squeeze()

                scores = self.score_subs(sum_visit_sub_given, p_visit_sub)
                top = np.argsort(-scores)[:11]
                top_scores = scores[top]
                sub_scores[sub1] = {inv_sub[ix]: score for ix, score in zip(top, top_scores) if ix != ix1}

        with self.output().open('w') as f:
            json.dump(sub_scores, f)
