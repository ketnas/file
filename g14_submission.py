'''
  แก้ code และเพิ่มเติมได้ใน class YourTeamAgent เท่านั้น 
  ตอนส่งไฟล์ ให้แน่ใจว่า YourTeamAgent ไม่มี error และ run ได้
  ส่งแค่ submission.py ไฟล์เดียว
'''
from util import manhattanDistance
from game import Directions
import random, util,copy
from typing import Any, DefaultDict, List, Set, Tuple

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState: GameState,agentIndex=0) -> str:
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument
    is an object of GameState class. Following are a few of the helper methods that you
    can use to query a GameState object to gather information about the present state
    of Pac-Man, the ghosts and the maze.

    gameState.getLegalActions(agentIndex):
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action):
        Returns the successor state after the specified agent takes the action.
        Pac-Man is agent 0 and agent 1.

    gameState.getPacmanState(agentIndex):
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore(agentIndex):
        Returns the score of agentIndex (0 or 1) corresponding to the current state of the game

    gameState.getScores():
        Returns all the scores of the agents in the game as a list where first score corresponds to agent 0
    
    gameState.getFood():
        Returns the food in the gameState

    gameState.getPacmanPosition(agentIndex):
        Returns the pacman (agentIndex 0 or 1) position in the gameState

    gameState.getCapsules():
        Returns the capsules in the gameState

    The GameState class is defined in pacman.py and you might want to look into that for
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions(agentIndex)
    gameState.getScaredTimes(agentIndex)

    # print(legalMoves)
    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action,agentIndex) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best


    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState: GameState, action: str,agentIndex=0) -> float:
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action,agentIndex)
    newPos = successorGameState.getPacmanPosition(agentIndex)
    oldFood = currentGameState.getFood()
    return successorGameState.getScore(agentIndex)


def scoreEvaluationFunction(currentGameState: GameState,agentIndex=0) -> float:
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore(agentIndex)

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'evaluationFunction', depth = '8',agentIndex=0):
    self.index = agentIndex 
    # self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################

class YourTeamAgent(MultiAgentSearchAgent):
    """
    Your team agent
    แก้ เพิ่มเติม ได้ใน class นี้เท่านั้น
    แต่ห้ามแก้ชื่อ class และ getAction method ที่กำหนดให้
    แต่เพิ่ม method เองได้ และเรียกใช้ method ใน class นี้เท่านั้น
    """
    def getAction(self, gameState: GameState, agentIndex=0) -> str:
        
        alpha = -float('inf')
        beta = float('inf')
        bestAction = None

        for action in gameState.getLegalActions(agentIndex):
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            value = self.minimax(successorGameState, 0, self.depth - 1, alpha, beta)
            if value > alpha:
                alpha = value
                bestAction = action
        return bestAction

    def minimax(self, gameState: GameState, agentIndex, depth, alpha, beta):
        
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        if agentIndex == 0:  
            bestValue = -float('inf')
            for action in gameState.getLegalActions(agentIndex):
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                value = self.minimax(successorGameState, 1, depth - 1, alpha, beta)
                bestValue = max(bestValue, value)
                alpha = max(alpha, bestValue)
                if beta <= alpha:
                    break  
            return bestValue
        else:  
            bestValue = float('inf')
            for action in gameState.getLegalActions(agentIndex):
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                value = self.minimax(successorGameState, 0, depth - 1, alpha, beta)
                bestValue = min(bestValue, value)
                beta = min(beta, bestValue)
                if beta <= alpha:
                    break  
            return bestValue

    def evaluationFunction(self, currentGameState: GameState) -> float:
        
        pacmanPosition = currentGameState.getPacmanPosition()
        if pacmanPosition in [(0, 0), (0, currentGameState.data.layout.width - 1),
                              (currentGameState.data.layout.height - 1, 0),
                              (currentGameState.data.layout.height - 1, currentGameState.data.layout.width - 1)]:
            legalMoves = currentGameState.getLegalActions()
            legalMoves.remove(Directions.STOP)  
            if Directions.WEST in legalMoves:
                return 1  
            elif Directions.NORTH in legalMoves:
                return 2  
            else:
                return 0  
        else:
            return currentGameState.getScore()
