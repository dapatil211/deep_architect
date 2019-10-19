import deep_architect.modules as mo
import deep_architect.searchers.common as se


class SearchSpaceFactory(mo.SearchSpaceFactory):

    def __init__(self, search_space_fn, exp_config, reset_scope_upon_get=True):
        mo.SearchSpaceFactory.__init__(
            self, search_space_fn, reset_scope_upon_get=reset_scope_upon_get)

        self.exp_config = exp_config


class Searcher(se.Searcher):

    def __init__(self, search_space_factory, exp_config):
        se.Searcher.__init__(self, search_space_factory.get_search_space)
        self.exp_config = exp_config


class Evaluator(object):

    def __init__(self, exp_config):
        self.exp_config = exp_config

    def pretrain(self):
        raise NotImplementedError()

    def evaluate(self, inputs, outputs, save_fn=None, state=None):
        raise NotImplementedError()
