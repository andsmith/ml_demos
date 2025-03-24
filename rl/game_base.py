from enum import IntEnum
class Result(IntEnum):
    X_WIN = 1
    O_WIN = 2
    DRAW = 3


class Mark(IntEnum):
    EMPTY = 0
    X = 1  # also denotes the player
    O = 2

WIN_MARKS ={Mark.X: Result.X_WIN, 
            Mark.O: Result.O_WIN}