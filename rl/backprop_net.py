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

BACKPROP_DIR = "ff_nets"
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


class BackpropPolicy(NNetPolicy):
    def __init__(self, teacher, n_hidden=18, encoding='one-hot', n_epochs=200, weight_alpha=0.0):
        player = teacher.player
        if player != Mark.X and player != Mark.O:
            raise ValueError("mark must be Mark.X or Mark.O")
        self.player = player
        self.opponent = Mark.X if player == Mark.O else Mark.O

        self.winning_result = Result.X_WIN if player == Mark.X else Result.O_WIN
        self.losing_result = Result.O_WIN if player == Mark.X else Result.X_WIN
        self.draw_result = Result.DRAW
        
        self._model = None
        self.encoding = encoding
        self.weight_alpha = weight_alpha
        self.player = teacher.player
        self._teacher = teacher
        self.n_hidden = n_hidden
        if self._model is None:
            self._model = self._train(n_epochs)
    @staticmethod
    def from_file(filename):
        """
        Load a trained BackpropPolicy from a file.
        :param filename: Path to the file containing the trained model.
        :return: An instance of BackpropPolicy.
        """
        with open(filename, 'rb') as f:
            net = pickle.load(f)
        if not isinstance(net, BackpropPolicy):
            raise ValueError("Loaded model is not an instance of BackpropPolicy")
        return net

    def __str__(self):
        return f"BackpropPolicy(in={self.encoding},h={self.n_hidden})({self.player.name})"
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        logging.info(f"Saved trained agent to {filename}")

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
            if True:
                # Define the model sequentially.
                model = Sequential([Input(shape=input_shape),
                                    Dense(units=number_of_outputs, activation=ACTIVATION)])

                model.compile(optimizer='adam', loss='mse')
            else:
                model = MLPR(hidden_layer_sizes=tuple(),
                            max_iter=4520*3, batch_size=4520,
                            solver='lbfgs', activation=ACTIVATION)

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
        return np.mean(good), len(net_actions), good

    def score(self, model=None):
        model = self._model if model is None else model
        first, second = self._teacher._states_by_turn['first'], self._teacher._states_by_turn['second']
        all_states = first + second
        scores = []
        n_acts = []
        perfect = []
        for state in all_states:
            score, n_actions, good = self._eval_state(state, model=model)
            scores.append(score)
            n_acts.append(n_actions)
            perfect.append(np.all(good))

        return np.mean(scores), np.array(n_acts), np.array(perfect)

    def _train(self, n_epochs, remove_no_choice=True):
        x, y, w = self._teacher.extract_dataset(encoding=self.encoding)
        x = np.array(x)
        y = np.array(y)
        w = np.array(w)

        if remove_no_choice:
            valid = w>np.min(w)
            x = x[valid]
            y = y[valid]
            w = w[valid]
            print("  --------------------->  Removing no-choice samples: %i" % np.sum(~valid))

        w = w ** self.weight_alpha
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
        if n_epochs > 1:
            logging.info("\tn_epochs: %d" % n_epochs)
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
                model.fit(x, y, sample_weight=w, batch_size=256, verbose=0, shuffle=True)  # has its own print
                new_model = ShallowNet(model, activation=ACTIVATION)
                loss = model.history.history['loss'][0]
                if epoch % 25 == 0 or epoch == n_epochs - 1:
                    logging.info("\tEpoch %i, ADAM fit - loss: %.7f" % (epoch, loss))

            if loss < best_loss:
                best_loss = loss
                best_model = new_model

        model = new_model  # use the shallow net or the MLPRegressor

        final_results = self.score(model)

        def print_score(results):
            logging.info("\tmean teacher-agreement ratio: %.6f" % results[0])
            logging.info("\tmean deterministic action dist: %.2f" % np.mean(results[1]))
            logging.info("\tnum perfect actions: %d/%d" % (np.sum(results[2]), len(results[2])))
            deterministic = (np.array(results[1]) == 1)
            logging.info("\tmean det & perfect: %.6f" % np.mean(results[2][deterministic]))

        logging.info("")
        logging.info("Training done, final model score:")
        print_score(final_results)
        
        if n_epochs > 1:
            final_loss = self.score(best_model)
            logging.info("best model score:")
            print_score(final_loss)

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


def train_net(n_hidden=18, n_epochs=2000, encoding='enc+free', w_alpha=0.0,save=False):
    t = Teacher(MiniMaxEvaluator(Mark.X))
    bn = BackpropPolicy(teacher=t, n_hidden=n_hidden, n_epochs=n_epochs, encoding=encoding, weight_alpha=w_alpha)
    if save:
        filename = f"backprop_net_{n_hidden}_{encoding}_{n_epochs}_{w_alpha}.pkl"
        bn.save(filename)
    return bn


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_net()
    # bn = BackpropPolicy(mark=Mark.X, dataset_file="Minimax_data.pkl", n_hidden=10)

    # print(bn.recommend_action(state))
