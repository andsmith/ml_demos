from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from neat_util import NNetPolicy
import pickle
import numpy as np
from game_base import Mark, Result
from sklearn.neural_network import MLPRegressor as MLPR
import logging
from perfect_player import MiniMaxPolicy
from evolve_feedforward import Teacher
from tic_tac_toe import Game
from copy import deepcopy


class MiniMaxEvaluator(MiniMaxPolicy):
    def get_policy(self):
        return self._pi


ACTIVATION = 'relu'


class ShallowNet(object):
    """
    Convert a keras sequential model to a shallow network.
    """

    def __init__(self, model, activation):
        self._act_fn = activation
        self._weights = [layer.get_weights() for layer in model.layers]
        self._n_layers = len(self._weights)

    def predict(self, input):
        """
        Predict the output for the given input.
        :param input: Input to the model.
        :return: Output of the model.
        """
        return self.eval(input)

    def eval(self, input):
        """
        Evaluate the input using the model.
        :param input: Input to the model.
        :return: Output of the model.
        """
        output = input
        for i, (weights, biases) in enumerate(self._weights):
            output = np.dot(output, weights) + biases
            if (i < len(self._weights) - 1 and self._n_layers > 1) or (self._n_layers == 1):
                if self._act_fn == 'relu':
                    output = np.maximum(0, output)
                elif self._act_fn == 'tanh':
                    output = np.tanh(output)
                elif self._act_fn != 'linear':
                    raise ValueError(f"Unknown activation function: {self._act_fn}")

        return output


class BackpropNet(NNetPolicy):
    def __init__(self, teacher, n_hidden=18, encoding='one-hot', n_epochs=200, weight_alpha=0.0):
        self._model = None
        self.encoding = encoding
        self.weight_alpha = weight_alpha
        self.player = teacher.player
        self._teacher = teacher
        self.n_hidden = n_hidden
        if self._model is None:
            self._model = self._train(n_epochs)

    def __str__(self):
        return f"BackpropNet(in={self.encoding},h={self.n_hidden})({self.player.name})"

    def _get_model(self, use_mlpr=True):
        """
        If there are no hidden units, we need to use a Sequential model.
        If there are hidden units, we can use a MLPR or a Sequential model.

        NOTE: Only the sequential model can train with sample weights

        """

        number_of_outputs = 9
        test_state = self._teacher._states_by_turn['first'][-10]
        test_input = test_state.to_nnet_input(method=self.encoding)
        number_of_features = len(test_input)
        input_shape = (number_of_features,)
        if self.n_hidden > 0:
            if use_mlpr:
                model = MLPR(hidden_layer_sizes=(self.n_hidden,),
                             max_iter=4520*self.n_hidden, batch_size=4520,
                             solver='lbfgs', activation=ACTIVATION)
            else:
                # Define the model sequentially.
                model = Sequential([Input(shape=input_shape),
                                    Dense(units=self.n_hidden, activation=ACTIVATION),
                                    Dense(units=number_of_outputs, activation='linear')])

                model.compile(optimizer='adam', loss='mse')

        else:
            # Define the model sequentially.
            model = Sequential([Input(shape=input_shape),
                                Dense(units=number_of_outputs, activation=ACTIVATION)])

            model.compile(optimizer='adam', loss='mse')

        return model, test_state

    def _load_data(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return {'inputs': np.array(data[0]), 'outputs': np.array(data[1])}

    def _eval_state(self, state, model=None):
        """
        Compare network output to teacher's list of good moves.
        :return: 1 if the networ's pick was in the teacher's list, else .
        """
        net_actions = [act_prob[0] for act_prob in self.recommend_action(state, model=model)]
        teacher_actions = [act_prob[0] for act_prob in self._teacher.pi.recommend_action(state)]
        good = [net_action in teacher_actions for net_action in net_actions]
        return np.mean(good)

    def eval(self, model=None):
        model = self._model if model is None else model
        first, second = self._teacher._states_by_turn['first'], self._teacher._states_by_turn['second']
        first_eval = np.mean([self._eval_state(state, model=model) for state in first])
        second_eval = np.mean([self._eval_state(state, model=model) for state in second])
        mean_eval = (first_eval + second_eval) / 2.0
        # logging.info("Evaluation, net-first: %.2f, net-second: %.2f, mean: %.2f" % (first_eval, second_eval, mean_eval))
        return mean_eval

    def _train(self, n_epochs):
        x, y, w = self._teacher.extract_dataset(encoding=self.encoding)
        x = np.array(x)
        y = np.array(y)
        w = np.array(w) ** self.weight_alpha  # apply weight alpha to the weights

        # Use the MLPR only if there are hidden units but no sample weights.

        model, _ = self._get_model(use_mlpr=(self.n_hidden > 0 and self.weight_alpha == 0.0))

        best_loss = np.inf
        best_model = None
        last_loss = np.inf
        if isinstance(model, Sequential):
            using_sequential = True
        else:
            using_sequential = False
            n_epochs = 1

        model_kind = "Sequential" if using_sequential else "MLPRegressor"
        logging.info("")
        logging.info("Starting training with model:  %s" % model_kind)
        logging.info("\thidden units: %d" % self.n_hidden)
        logging.info("\tencoding: %s" % self.encoding)
        logging.info("\tweight alpha: %.2f" % self.weight_alpha)
        logging.info("\tweight range:  %.6f - %.6f" % (np.min(w), np.max(w)))
        logging.info("")

        for epoch in range(n_epochs):

            if not using_sequential:
                # Using MLPRegressor with lbfgs solver
                model.fit(x, y)
                new_model = model
                loss = model.loss_
                logging.info("\tlbfgs fit - loss: %.7f" % loss)

            else:
                # Using Sequential model with 'adam' optimizer
                model.fit(x, y, sample_weight=w, batch_size=64, verbose=0, shuffle=True)  # has its own print
                new_model = ShallowNet(model, activation=ACTIVATION)
                loss = model.history.history['loss'][0]
                if epoch % 25 == 0 or epoch == n_epochs - 1:
                    logging.info("\tEpoch %i, ADAM fit - loss: %.7f" % (epoch, loss))

            if loss < best_loss:
                best_loss = loss
                best_model = new_model

        model = new_model  # use the shallow net or the MLPRegressor

        final_loss = self.eval(model)
        logging.info("Training done, final loss on all states: %.4f" % final_loss)

        final_loss = self.eval(best_model)
        logging.info("Training done, best loss on all states: %.4f" % final_loss)

        return model

    def recommend_action(self, state, model=None):
        model = self._model if model is None else model

        actions = state.get_actions(flat_inds=True)
        input_state = state.to_nnet_input(method=self.encoding)
        output = model.predict(input_state[None, :])[0]
        best_valid = np.argmax(output[actions])
        best_act_flat = actions[best_valid]
        best_act = (best_act_flat // 3, best_act_flat % 3)
        return [(best_act, 1.0)]


def train_net(n_hidden=18, n_epochs=2000, encoding='enc+free', w_alpha=0.0):
    t = Teacher(MiniMaxEvaluator(Mark.X))
    bn = BackpropNet(teacher=t, n_hidden=n_hidden, n_epochs=n_epochs, encoding=encoding, weight_alpha=w_alpha)
    return bn


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_net()
    # bn = BackpropNet(mark=Mark.X, dataset_file="Minimax_data.pkl", n_hidden=10)

    # print(bn.recommend_action(state))
