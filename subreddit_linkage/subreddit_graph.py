"""
Build the subreddit/commenter graph.

This is a bipartite graph.
"""
import luigi
import os
import json
import gzip
import numpy as np
import scipy.sparse as sp
import networkx as nx

from tqdm import tqdm


class SubredditGraphEdges(luigi.Task):
    def output(self):
        return luigi.LocalTarget('cache/graph')

    def run(self):
        self.output().makedirs()

        subreddit_ids = {}
        author_ids = {}
        id_counter = 0
        edges = set()
        nodes = set()
        total = 0

        with gzip.open(os.path.expanduser('~/Datasets/reddit/RC_2017-03.gzip'), mode='rt', encoding='ascii') as f:
            for line in tqdm(f, total=79723106):
                total += 1
                line = json.loads(line)
                subreddit = line['subreddit']
                author = line['author']
                if subreddit not in subreddit_ids:
                    subreddit_ids[subreddit] = id_counter
                    nodes.add(id_counter)
                    id_counter += 1
                if author not in author_ids:
                    author_ids[author] = id_counter
                    nodes.add(id_counter)
                    id_counter += 1

                edges.add((subreddit_ids[subreddit], author_ids[author]))

        with self.output().open('w') as f:
            json.dump({
                'edges': list(edges),
                'nodes': list(nodes),
                'subreddit_ids': subreddit_ids,
                'author_ids': author_ids,
            }, f)

    def load_nx(self):
        assert self.complete()
        with self.output().open() as f:
            data = json.load(f)
        G = nx.Graph()
        G.add_node(data['nodes'])
        G.add_edges_from(data['edges'])

        return G, data['subreddit_ids'], data['author_ids']

    def load_recmat(self):
        assert self.complete()
        with self.output().open() as f:
            data = json.load(f)
        print('Data loaded')
        subreddit_remapping = {sid: ix for ix, sid in enumerate(tqdm(data['subreddit_ids'].values(), desc='remapping sub'))}
        author_remapping = {aid: ix for ix, aid in enumerate(tqdm(data['author_ids'].values(), desc='remapping author'))}

        i = np.asarray([author_remapping[auth] for sub, auth in data['edges']])
        j = np.asarray([subreddit_remapping[sub] for sub, auth in data['edges']])
        entries = np.ones(len(data['edges']))

        recmat = sp.coo_matrix((entries, (i, j)), shape=[len(author_remapping), len(subreddit_remapping)])

        sub_mapping = {sub: subreddit_remapping[ix] for sub, ix in data['subreddit_ids'].items()}
        auth_mapping = {auth: author_remapping[ix] for auth, ix in data['author_ids'].items()}
        return sp.csr_matrix(recmat), sub_mapping, auth_mapping
