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
# implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Minimax agent
  """

  def getAction(self, gameState: GameState,agentIndex = 0) -> str:
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. 

      Here are some method calls that might be useful when implementing minimax.

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent (0 or 1) takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore(agentIndex):
        Returns the score of agentIndex (0 or 1) corresponding to the current state of the game

      gameState.getLegalActions(agentIndex):
        Returns the legal actions for the agent specified (0 or 1). Returns Pac-Man's legal moves by default.

      gameState.getPacmanState(agentIndex):
          Returns an AgentState (0 or 1) object for pacman (in game.py)
          state.configuration.pos gives the current position
          state.direction gives the travel vector

      gameState.getNumAgents():
          Returns the total number of agents in the game

      gameState.getScores():
          Returns all the scores of the agents in the game as a list where first score corresponds to agent 0
      
      gameState.getFood():
          Returns the food in the gameState

      gameState.getPacmanPosition(agentIndex):
          Returns the pacman (agentIndex = 0 or 1) position in the gameState

      gameState.getCapsules()
          Returns the capsules in the gameState

      gameState.getScaredTimes(agentIndex)
          Returns remaining scared time for agentIndex
      
      gameState.getNumFood()
          Returns the number of food in the gameState
      
      gameState.hasFood(x, y)
          Returns True if there is food in the given (x, y) position

      gameState.hasWall(x, y)
          Returns True if there is a wall in the given (x, y) position

      self.depth:
        The depth to which search should continue

    """
    self.index = agentIndex
    bestVal = -float('inf')
    bestAction = None
    scoreStart = copy.deepcopy(gameState.getScores())
    legalMoves = gameState.getLegalActions(agentIndex)
    
    if len(legalMoves) == 1:
      return legalMoves[0]
    else: 
      for action in legalMoves:
        successorGameState = gameState.generateSuccessor(agentIndex, action)
        val = self.minimax(successorGameState,(agentIndex+1)%2,self.depth-1)
        if val > bestVal:
          bestVal = val
          bestAction = action
      # print("score ",gameState.getScore(self.index))
      # print("score ",gameState.getScores())
      return bestAction

  def minimax(self,gameState: GameState, agentIndex,depth):
    if gameState.isWin() or gameState.isLose() or depth == 0:
      return self.evaluationFunction(gameState,agentIndex)
    
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions(agentIndex)
    # print(legalMoves)
    if agentIndex == self.index:
      best = -float('inf')
      for action in legalMoves:
        successorGameState = gameState.generateSuccessor(agentIndex, action)
        best = max(best,self.minimax(successorGameState,(agentIndex+1)%2,depth-1))
      return best
    else:
      best = float('inf')
      for action in legalMoves:
        successorGameState = gameState.generateSuccessor(agentIndex, action)
        best = min(best,self.minimax(successorGameState,(agentIndex+1)%2,depth-1))
      return best

  def evaluationFunction(self, currentGameState: GameState, agentIndex=0) -> float:
    """
    The evaluation function takes in the current
    GameStates (pacman.py) and returns a score of that state.
    """    
    return currentGameState.getScore(agentIndex)

######################################################################################
from util import manhattanDistance, PriorityQueue
from game import Directions, Actions
import random
from typing import List, Tuple
from pacman import GameState

class YourTeamAgent(MultiAgentSearchAgent):
    def getAction(self, gameState, agentIndex=0):
        pacmanPos = gameState.getPacmanPosition(agentIndex)
        foodGrid = gameState.getFood()
        foodPositions = foodGrid.asList()
        ghostPositions = gameState.getGhostPositions()  # Get positions of all ghosts
        capsules = gameState.getCapsules()

        closest_food = min(foodPositions, key=lambda food: self.manhattanDistance(pacmanPos, food)) if foodPositions else None
        closest_ghost = min(ghostPositions, key=lambda ghost: self.manhattanDistance(pacmanPos, ghost)) if ghostPositions else None

        if closest_ghost and self.manhattanDistance(pacmanPos, closest_ghost) <= 4:
            ghostIndex = ghostPositions.index(closest_ghost)
            if gameState.getGhostStates()[ghostIndex].scaredTimer == 0:
                return self.fleeAction(gameState, pacmanPos, closest_ghost)

        if closest_food:
            return self.getActionToPosition(gameState, pacmanPos, closest_food)
        else:
            possible_actions = gameState.getLegalActions(agentIndex)
            return random.choice(possible_actions) if possible_actions else Directions.STOP

    def manhattanDistance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def fleeAction(self, gameState, pacmanPos, goal):
        openSet = PriorityQueue()
        openSet.push((pacmanPos, [], 0), 0)
        closedSet = set()

        dx = goal[0] - pacmanPos[0]
        dy = goal[1] - pacmanPos[1]
        opposite_direction = Directions.STOP
        if dx < 0:
            opposite_direction = Directions.EAST
        elif dx > 0:
            opposite_direction = Directions.WEST
        elif dy < 0:
            opposite_direction = Directions.NORTH
        elif dy > 0:
            opposite_direction = Directions.SOUTH

        while not openSet.isEmpty():
            currentPos, actions, risk = openSet.pop()

            if currentPos in closedSet:
                continue

            closedSet.add(currentPos)

            for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                if direction == opposite_direction:
                    continue

                x, y = currentPos
                dx, dy = Actions.directionToVector(direction)
                nextPos = (int(x + dx), int(y + dy))

                if not gameState.hasWall(nextPos[0], nextPos[1]):
                    newActions = actions + [direction]
                    newRisk = risk + 1
                    openSet.push((nextPos, newActions, newRisk), newRisk + self.manhattanDistance(nextPos, goal))

                    if len(newActions) > 0 and newActions[0] in gameState.getLegalActions(self.index):
                        return newActions[0]

        possible_actions = gameState.getLegalActions(self.index)
        return random.choice(possible_actions) if possible_actions else Directions.STOP

    def getActionToPosition(self, gameState, start, goal):
        dx, dy = goal[0] - start[0], goal[1] - start[1]
        possible_actions = gameState.getLegalActions(self.index)
        valid_actions = []

        if dx < 0 and Directions.WEST in possible_actions and not gameState.hasWall(start[0] - 1, start[1]):
            valid_actions.append(Directions.WEST)
        elif dx > 0 and Directions.EAST in possible_actions and not gameState.hasWall(start[0] + 1, start[1]):
            valid_actions.append(Directions.EAST)

        if dy > 0 and Directions.NORTH in possible_actions and not gameState.hasWall(start[0], start[1] + 1):
            valid_actions.append(Directions.NORTH)
        elif dy < 0 and Directions.SOUTH in possible_actions and not gameState.hasWall(start[0], start[1] - 1):
            valid_actions.append(Directions.SOUTH)

        if valid_actions:
            return random.choice(valid_actions)
        else:
            return random.choice(possible_actions) if possible_actions else Directions.STOP