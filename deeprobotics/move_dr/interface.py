from dataclasses import dataclass
from enum import Enum

from actions.base import Interface


class MovementAction(str, Enum):
    STAND_UP = "stand up"
    SIT = "sit"
    SHAKE_PAW = "shake paw"
    DANCE = "dance"
    MOONWALK = "moon walk"
    STAND_STILL = "stand still"
    DO_NOTHING = "stand still"


@dataclass
class MoveInput:
    action: MovementAction


@dataclass
class Move(Interface[MoveInput, MoveInput]):
    """
    A movement to be performed by the agent.

    Effect: Allows the agent to move.
    """

    input: MoveInput
    output: MoveInput
