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

class KeyboardAgent(Agent):
  """
  An agent controlled by the keyboard.
  """
  # NOTE: Arrow keys also work.
  WEST_KEY  = 'a' 
  EAST_KEY  = 'd' 
  NORTH_KEY = 'w' 
  SOUTH_KEY = 's'
  STOP_KEY = 'q'

  def __init__( self, index = 0 ):
    
    self.lastMove = Directions.STOP
    self.index = index
    self.keys = []
    
  def getAction( self, state,index=0):
    from graphicsUtils import keys_waiting
    from graphicsUtils import keys_pressed
    keys = keys_waiting() + keys_pressed()
    if keys != []:
      self.keys = keys
    
    self.index = index

    legal = state.getLegalActions(self.index)
    # print(legal)
    move = self.getMove(legal)
    
    if move == Directions.STOP:
      # Try to move in the same direction as before
      if self.lastMove in legal:
        move = self.lastMove
    
    if (self.STOP_KEY in self.keys) and Directions.STOP in legal: move = Directions.STOP

    if move not in legal:
      move = random.choice(legal)
      
    self.lastMove = move
    return move

  def getMove(self, legal):
    move = Directions.STOP
    if   (self.WEST_KEY in self.keys or 'Left' in self.keys) and Directions.WEST in legal:  move = Directions.WEST
    if   (self.EAST_KEY in self.keys or 'Right' in self.keys) and Directions.EAST in legal: move = Directions.EAST
    if   (self.NORTH_KEY in self.keys or 'Up' in self.keys) and Directions.NORTH in legal:   move = Directions.NORTH
    if   (self.SOUTH_KEY in self.keys or 'Down' in self.keys) and Directions.SOUTH in legal: move = Directions.SOUTH
    return move
  
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
class YourTeamAgent(MultiAgentSearchAgent):
    """
    Hybrid agent combining A* search and Minimax algorithms
    """

    def __init__(self, depth=3):
        super().__init__()
        self.depth = depth

    def getAction(self, gameState: GameState, agentIndex=0) -> str:
        """
        Returns the action to take using a hybrid of A* search and Minimax algorithms.
        """
        # Check if there is food left, if not switch to minimax to get capsules
        if any(gameState.getFood().asList()):
            # print("astar")
            return self.a_star_search(gameState, agentIndex)
        else:
            # print("minimax")
            return ReflexAgent().getAction(gameState, agentIndex)

    def heuristic(self, state: GameState, agentIndex) -> float:
      """
      Heuristic function for A* search
      """
      # print("------Pacman1--------")
      pacmanPosition = state.getPacmanPosition(agentIndex)
      # print(pacmanPosition)
      scareMe = state.getScaredTimes(agentIndex)
      # print(scareMe)
      # print("------Pacman2--------")
      foodList = state.getFood().asList()
    
      # Ensure agentIndex is switched to the ghost's index
      ghostIndex = (agentIndex + 1) % 2  # Switch to the other agent's index
      ghostPosition = state.getPacmanPosition(ghostIndex)
      # print(ghostPosition)
      scareEnemy = state.getScaredTimes(ghostIndex)
      # print(scareEnemy)
    
      nearestGhostDist = util.manhattanDistance(pacmanPosition, ghostPosition)
      nearestFoodDist = min([util.manhattanDistance(pacmanPosition, foodPos) for foodPos in foodList]) if foodList else float('inf')

      if scareMe > 0:
        # print("run")
        # print(ghostPosition)
        if nearestGhostDist <= 2:
            return float('inf')
        else:
            return nearestGhostDist
      else:
        return nearestFoodDist


    def random_move(self, gameState: GameState, agentIndex) -> str:
        """
        Function for randomly moving towards nearby food
        """
        legalMoves = gameState.getLegalActions(agentIndex)
        bestMove = None
        bestDistance = float('inf')
        for action in legalMoves:
            successorState = gameState.generatePacmanSuccessor(action, agentIndex)
            successorPosition = successorState.getPacmanPosition(agentIndex)
            h = self.heuristic(successorState, agentIndex)
            if h is not None and h < bestDistance:
                bestMove = action
                bestDistance = h
        return bestMove

    def a_star_search(self, gameState: GameState, agentIndex) -> str:
        """
        A* search algorithm to find the best path to the nearest capsule
        """
        startState = gameState.getPacmanPosition(agentIndex)
        frontier = util.PriorityQueue()
        frontier.push((startState, [], 0), 0)
        visited = set()

        while not frontier.isEmpty():
            currentState, path, cost = frontier.pop()
            if gameState.hasFood(currentState[0], currentState[1]):
                return path[0] if path else Directions.STOP
            if currentState not in visited:
                visited.add(currentState)
                legalMoves = gameState.getLegalActions(agentIndex)
                for action in legalMoves:
                    successorState = gameState.generatePacmanSuccessor(action, agentIndex)
                    successorPosition = successorState.getPacmanPosition(agentIndex)
                    newPath = path + [action]
                    newCost = cost + 1
                    h = self.heuristic(successorState, agentIndex)
                    if h is not None:  # Ensure h is a valid number
                        frontier.push((successorPosition, newPath, newCost), newCost + h)
        return self.random_move(gameState, agentIndex)

    def minimax_search(self, gameState: GameState, agentIndex=0) -> str:
        """
        Minimax search algorithm to find the best action
        """
        bestAction = None
        legalMoves = gameState.getLegalActions(agentIndex)

        if len(legalMoves) == 1:
            return legalMoves[0]

        bestVal = -float('inf')
        for action in legalMoves:
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            val = self.minimax(successorGameState, (agentIndex + 1) % 2, self.depth - 1)
            if val > bestVal:
                bestVal = val
                bestAction = action
        return bestAction

    def minimax(self, gameState: GameState, agentIndex, depth):
        """
        Minimax algorithm
        """
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState, agentIndex)

        legalMoves = gameState.getLegalActions(agentIndex)

        if agentIndex == self.index:
            best = -float('inf')
            for action in legalMoves:
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                best = max(best, self.minimax(successorGameState, (agentIndex + 1) % 2, depth - 1))
            return best
        else:
            if any(gameState.getGhostPosition(i) for i in range(1, gameState.getNumAgents())):
                best = float('inf')
                for action in legalMoves:
                    successorGameState = gameState.generateSuccessor(agentIndex, action)
                    best = min(best, self.minimax(successorGameState, (agentIndex + 1) % 2, depth - 1))
                return best
            else:
                # No ghost nearby, Pacman flees
                return -float('inf')

    def evaluationFunction(self, currentGameState: GameState, agentIndex=0) -> float:
        """
        The evaluation function takes in the current
        GameStates (pacman.py) and returns a score of that state.
        """
        return currentGameState.getScore(agentIndex)