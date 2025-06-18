from neat_util import NNetPolicy
import pickle
import numpy as np
from game_base import Mark, Result
from sklearn.neural_network import MLPRegressor as MLPC
import logging
from perfect_player import MiniMaxPolicy
from evolve_feedforward import Teacher
from tic_tac_toe import Game


class MiniMaxEvaluator(MiniMaxPolicy):
    def get_policy(self):
        return self._pi
    

class BackpropNet(NNetPolicy):
    def __init__(self, teacher, dataset_file="Minimax_data.pkl",n_hidden=18, encoding='one-hot',n_epochs=200):
        self._model = None
        self.encoding = encoding
        self.player = teacher.player
        self._teacher = teacher
        self.n_hidden = n_hidden
        self._data = self._load_data(dataset_file)
        if self._model is None:
            self._model = self._train(n_epochs)

    def __str__(self):
        return f"BackpropNet(h={self.n_hidden})({self.player.name})"

    def _load_data(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return {'inputs': np.array(data[0]), 'outputs': np.array(data[1])}
    
    def _eval_state(self, state):
        """
        Compare network output to teacher's list of good moves.
        :return: 1 if the networ's pick was in the teacher's list, else .
        """
        net_actions = [act_prob[0] for act_prob in self.recommend_action(state)]
        teacher_actions = [act_prob[0] for act_prob in self._teacher.pi.recommend_action(state)]
        good = [net_action in teacher_actions for net_action in net_actions]
        return np.mean(good)

    def eval(self):
        first, second = self._teacher._states_by_turn['first'], self._teacher._states_by_turn['second']
        first_eval = np.mean([self._eval_state(state) for state in first])
        second_eval = np.mean([self._eval_state(state) for state in second])
        mean_eval = (first_eval + second_eval) / 2.0
        logging.info("Evaluation, net-first: %.2f, net-second: %.2f, mean: %.2f" % (first_eval, second_eval, mean_eval))
        return mean_eval
    
    def _train(self,n_epochs):
        x, y = self._teacher.extract_dataset(encoding=self.encoding)
        x = np.array(x)
        y = np.array(y)
        clf = MLPC(hidden_layer_sizes=(self.n_hidden,), max_iter=n_epochs)
        for epoch in range(n_epochs):
            clf.partial_fit(x, y)
            if epoch % 100 == 0:
                logging.info(f"Epoch {epoch}, Loss: {clf.loss_}")
        return clf
    

    def recommend_action(self, state):

        actions = state.get_actions(flat_inds=True)
        input_state = state.to_nnet_input(method=self.encoding)
        output = self._model.predict([input_state])[0]
        best_valid = np.argmax(output[actions])
        best_act_flat = actions[best_valid]
        best_act = (best_act_flat // 3, best_act_flat % 3)
        return [(best_act, 1.0)]
    

def train_net(n_hidden=18,n_epochs=2000):
    t = Teacher(MiniMaxEvaluator(Mark.X))
    bn = BackpropNet(teacher=t, n_hidden=n_hidden, n_epochs=n_epochs)
    bn.eval()
    return bn


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_net()
    # bn = BackpropNet(mark=Mark.X, dataset_file="Minimax_data.pkl", n_hidden=10)
    # bn.eval()
    # print(bn.recommend_action(state))