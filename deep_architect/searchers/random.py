from deep_architect.searchers.common import random_specify, Searcher


class RandomSearcher(Searcher):

    def __init__(self, search_space_fn):
        Searcher.__init__(self, search_space_fn)

    def sample(self):
        inputs, outputs = self.search_space_fn()
        vs = random_specify(outputs.values())
        return inputs, outputs, vs, {}

    def update(self, val, searcher_eval_token):
        pass
