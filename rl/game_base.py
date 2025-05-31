from enum import IntEnum


class Result(IntEnum):
    X_WIN = 1
    O_WIN = 2
    DRAW = 3


class Mark(IntEnum):
    EMPTY = 0
    X = 1  # also denotes the player
    O = 2

# r is known only for these states in adavnce:
TERMINAL_REWARDS = {
    Result.X_WIN: 1.0,
    Result.O_WIN: -1.0,
    Result.DRAW: -.5,
}

def get_reward(state, action, player_mark=Mark.X):
    next_state = state.clone_and_move(action, player_mark)
    result = next_state.check_endstate()
    if result is None:
        return 0.0
    return TERMINAL_REWARDS[result]


WIN_MARKS ={Mark.X: Result.X_WIN, 
            Mark.O: Result.O_WIN}


def get_cell_center(dims, ij):
    """
    Get the  x, y coordinates of the center of the cell at (i, j)"

    :param dims: dict, output of Game.get_image_dims()
    :param ij: tuple, (i, j) location in the 3x3 grid, (0,1,2) for each
    :returns: tuple, (x, y) coordinates of the center of the cell (possibly floats)
    """
    x_span = dims['cells'][ij[0]][ij[1]]['x']
    y_span = dims['cells'][ij[0]][ij[1]]['y']
    return (x_span[0] + (x_span[1]-1)) / 2, (y_span[0] + (y_span[1]-1)) / 2