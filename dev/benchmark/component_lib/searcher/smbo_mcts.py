from deep_architect.searchers.common import random_specify, specify
from deep_architect.surrogates.common import extract_features
import deep_architect.utils as ut
import numpy as np
from .mcts import MCTSSearcher
from ..api import Searcher


# surrogate with MCTS optimization.
# TODO: make sure that can keep the tree while the surrogate changes below me.
# TODO: I would just compute the std for the scores.
class SMBOSearcherWithMCTSOptimizer(Searcher):

    def __init__(self, search_space_fn, exp_config, surrogate_model,
                 num_samples, exploration_prob, tree_refit_interval):
        Searcher.__init__(self, search_space_fn, exp_config)
        self.surr_model = surrogate_model
        self.mcts = MCTSSearcher(self.search_space_fn, exp_config)
        self.num_samples = num_samples
        self.exploration_prob = exploration_prob
        self.tree_refit_interval = tree_refit_interval
        self.cnt = 0

    def sample(self):
        if np.random.rand() < self.exploration_prob:
            inputs, outputs = self.search_space_fn()
            best_vs = random_specify(outputs.values())
        # TODO: ignoring the size of the model here.
        # TODO: needs to add the exploration bonus.
        else:
            best_model = None
            best_vs = None
            best_score = -np.inf
            for _ in range(self.num_samples):
                (inputs, outputs, vs, m_cfg_d) = self.mcts.sample()
                feats = extract_features(inputs, outputs)
                score = self.surr_model.eval(feats)
                if score > best_score:
                    best_model = (inputs, outputs)
                    best_vs = vs
                    best_score = score

                self.mcts.update(score, m_cfg_d)
            inputs, outputs = best_model

        searcher_eval_token = {'vs': best_vs}
        return inputs, outputs, best_vs, searcher_eval_token

    def update(self, val, searcher_eval_token):
        (inputs, outputs) = self.search_space_fn()
        specify(outputs.values(), searcher_eval_token['vs'])
        feats = extract_features(inputs, outputs)
        self.surr_model.update(val, feats)

        self.cnt += 1
        if self.cnt % self.tree_refit_interval == 0:
            self.mcts = MCTSSearcher(self.search_space_fn, self.exp_config)

    # NOTE: this has not been tested.
    def save_state(self, folderpath):
        self.mcts.save_state(folderpath)
        self.surr_model.save_state(folderpath)
        ut.write_jsonfile({"cnt": self.cnt},
                          ut.join_paths([folderpath, "state.json"]))

    def load_state(self, folderpath):
        self.mcts.load_state(folderpath)
        self.surr_model.load_state(folderpath)
        state = ut.read_jsonfile(ut.join_paths([folderpath, "state.json"]))
        self.cnt = state["cnt"]