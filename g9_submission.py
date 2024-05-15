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
    print(successorGameState.getScore(agentIndex))
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

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '8',agentIndex=0):
    self.index = agentIndex 
    self.evaluationFunction = util.lookup(evalFn, globals())
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

      gameState.getCapsules():
          Returns the capsules in the gameState

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

  def minimax(self,gameState: GameState,agentIndex,depth):
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

    return successorGameState.getScore()

######################################################################################
class YourTeamAgent(MultiAgentSearchAgent):
    def __init__(self):
        self.lastPositions = []
        self.dc = None
        self.capsuleTarget = None
        self.foodTarget = None
        self.otherPacmanPosition = None
        self.searchingOtherPacman = False
        self.searchTimeLimit = 10
        self.searchTimer = self.searchTimeLimit
        self.loopCount = 0

    def getAction(self, gameState, agentIndex=0):
        legalMoves = gameState.getLegalActions(agentIndex)
        gameState.getScaredTimes(agentIndex)

        if self.searchingOtherPacman:
            self.searchTimer -= 1

        if self.searchTimer <= 0:
            self.searchingOtherPacman = False
            self.searchTimer = self.searchTimeLimit

        capsules = gameState.getCapsules()
        if len(capsules) <= 2:
            if capsules:
                pacmanPosition = gameState.getPacmanPosition(agentIndex)
                distances = [manhattanDistance(pacmanPosition, capsule) for capsule in capsules]
                closestCapsuleIndex = distances.index(min(distances))
                self.capsuleTarget = capsules[closestCapsuleIndex]
            else:
                self.capsuleTarget = None

        food = gameState.getFood()
        if food and food.count() > 0:
            pacmanPosition = gameState.getPacmanPosition(agentIndex)
            distances = [manhattanDistance(pacmanPosition, foodPos) for foodPos in food.asList()]
            closestFoodIndex = distances.index(min(distances))
            self.foodTarget = food.asList()[closestFoodIndex]
        else:
            self.foodTarget = None

        scores = [self.evaluationFunction(gameState, action, agentIndex) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)

        pacmanPosition = gameState.getPacmanPosition(agentIndex)
        if pacmanPosition in self.lastPositions:
            self.loopCount += 1
            if self.loopCount >= 3:
                self.loopCount = 0
                return self.avoidLoop(gameState, agentIndex, legalMoves)

        self.loopCount = 0
        self.lastPositions.append(pacmanPosition)
        if len(self.lastPositions) > 3:
            self.lastPositions.pop(0)

        # Check if the score increased within 3 seconds
        currentScore = gameState.getScore(agentIndex)
        if self.dc is None or currentScore > self.dc:
            self.dc = currentScore
            self.searchTimer = self.searchTimeLimit
        else:
            self.searchTimer -= 1
            if self.searchTimer <= 0:
                self.searchTimer = self.searchTimeLimit
                return self.foodSearch(gameState, agentIndex, legalMoves)

        if self.capsuleTarget:
            pacmanPosition = gameState.getPacmanPosition(agentIndex)
            for action in legalMoves:
                successorGameState = gameState.generatePacmanSuccessor(action, agentIndex)
                successorPosition = successorGameState.getPacmanPosition(agentIndex)
                if manhattanDistance(successorPosition, self.capsuleTarget) < manhattanDistance(pacmanPosition, self.capsuleTarget):
                    return action

        if self.searchingOtherPacman and self.otherPacmanPosition:
            pacmanPosition = gameState.getPacmanPosition(agentIndex)
            for action in legalMoves:
                successorGameState = gameState.generatePacmanSuccessor(action, agentIndex)
                successorPosition = successorGameState.getPacmanPosition(agentIndex)
                if manhattanDistance(successorPosition, self.otherPacmanPosition) < manhattanDistance(pacmanPosition, self.otherPacmanPosition):
                    return action

        if self.foodTarget:
            pacmanPosition = gameState.getPacmanPosition(agentIndex)
            for action in legalMoves:
                successorGameState = gameState.generatePacmanSuccessor(action, agentIndex)
                successorPosition = successorGameState.getPacmanPosition(agentIndex)
                if manhattanDistance(successorPosition, self.foodTarget) < manhattanDistance(pacmanPosition, self.foodTarget):
                    return action

        wallDirection = self.avoidWalls(gameState, agentIndex)
        if wallDirection:
            return wallDirection

        return legalMoves[chosenIndex]

    def foodSearch(self, gameState, agentIndex, legalMoves):
        pacmanPosition = gameState.getPacmanPosition(agentIndex)
        # Find the closest food
        food = gameState.getFood()
        if food and food.count() > 0:
            distances = [manhattanDistance(pacmanPosition, foodPos) for foodPos in food.asList()]
            closestFoodIndex = distances.index(min(distances))
            self.foodTarget = food.asList()[closestFoodIndex]
        else:
            self.foodTarget = None

        # Choose the action that leads towards the closest food
        if self.foodTarget:
            for action in legalMoves:
                successorGameState = gameState.generatePacmanSuccessor(action, agentIndex)
                successorPosition = successorGameState.getPacmanPosition(agentIndex)
                if manhattanDistance(successorPosition, self.foodTarget) < manhattanDistance(pacmanPosition, self.foodTarget):
                    return action

        return random.choice(legalMoves)

    def evaluationFunction(self, currentGameState, action, agentIndex=0):
        successorGameState = currentGameState.generatePacmanSuccessor(action, agentIndex)
        return successorGameState.getScore(agentIndex)

    def avoidWalls(self, gameState, agentIndex):
        pacmanPosition = gameState.getPacmanPosition(agentIndex)
        legalMoves = gameState.getLegalActions(agentIndex)

        distances = []
        for action in legalMoves:
            successorGameState = gameState.generatePacmanSuccessor(action, agentIndex)
            successorPosition = successorGameState.getPacmanPosition(agentIndex)
            distances.append(manhattanDistance(successorPosition, pacmanPosition))

        maxDistance = max(distances)
        maxDistanceIndices = [index for index in range(len(distances)) if distances[index] == maxDistance]
        chosenIndex = random.choice(maxDistanceIndices)
        return legalMoves[chosenIndex]

    def avoidLoop(self, gameState, agentIndex, legalMoves):
        pacmanPosition = gameState.getPacmanPosition(agentIndex)
        # Try to move ahead
        nextPositions = [(pacmanPosition[0] + dx, pacmanPosition[1] + dy) for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]]
        for action in legalMoves:
            successorGameState = gameState.generatePacmanSuccessor(action, agentIndex)
            successorPosition = successorGameState.getPacmanPosition(agentIndex)
            if successorPosition in nextPositions:
                return action
        # If cannot move ahead, try to