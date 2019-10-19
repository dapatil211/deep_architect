from deep_architect.searchers.common import random_specify
from ..api import Searcher


class RandomSearcher(Searcher):

    def __init__(self, search_space_factory, exp_config):
        Searcher.__init__(self, search_space_factory, exp_config)

    def sample(self):
        inputs, outputs = self.search_space_fn()
        vs = random_specify(outputs.values())
        return inputs, outputs, vs, {}

    def update(self, val, searcher_eval_token):
        pass

    def save_state(self, folderpath):
        pass

    def load_state(self, folderpath):
        pass
