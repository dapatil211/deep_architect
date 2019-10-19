import os
import time
from deep_architect.searchers.common import specify
import deep_architect.utils as ut

from ..api import Searcher


class FixedSearcher(Searcher):

    def __init__(self, search_space_fn, exp_config, filename):
        Searcher.__init__(self, search_space_fn, exp_config)
        self.idx = 0
        while True:
            try:
                self.archs = ut.read_jsonfile(filename)
                break
            except IOError:
                time.sleep(2.0)

    def sample(self):
        vs = self.archs[self.idx]
        inputs, outputs = self.search_space_fn()
        specify(outputs.values(), vs)
        self.idx = (self.idx + 1) % len(self.archs)
        return inputs, outputs, vs, {}

    def update(self, val, searcher_eval_token):
        pass

    def save_state(self, folderpath):
        state = {'idx': self.idx}
        ut.write_jsonfile(state,
                          os.path.join(folderpath, 'fixed_searcher_state.json'))

    def load_state(self, folderpath):
        state = ut.read_jsonfile(
            os.path.join(folderpath, 'fixed_searcher_state.json'))
        self.idx = state['idx']
