
'''
  แก้ code และเพิ่มเติมได้ใน class YourTeamAgent เท่านั้น 
  ตอนส่งไฟล์ ให้แน่ใจว่า YourTeamAgent ไม่มี error และ run ได้
  ส่งแค่ submission.py ไฟล์เดียว
'''
from util import manhattanDistance,PriorityQueue
from game import Directions
import random, util,copy
from typing import Any, DefaultDict, List, Set, Tuple
import time;
from game import Agent
from pacman import GameState
from game import Actions
import heapq
from typing import List, Tuple, Deque
from collections import deque
from heapq import heappush, heappop
from queue import Queue
from collections import deque

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

    def getAction(self, gameState: GameState, agentIndex=0) -> str:
        legalMoves = gameState.getLegalActions(agentIndex)
        if len(gameState.getFood().asList()) == 1:
            last_food_position = gameState.getFood().asList()[0]
            # Force the move towards the last piece of food
            best_move = min(legalMoves, key=lambda x: manhattanDistance(gameState.generateSuccessor(agentIndex, x).getPacmanPosition(agentIndex), last_food_position))
            print(f"Last food at {last_food_position}, forced move: {best_move}")
            return best_move
        else:
            scores = [self.evaluationFunction(gameState, action, agentIndex) for action in legalMoves]
            bestScore = max(scores)
            bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
            chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
            return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action: str, agentIndex=0) -> float:
        """
        Evaluate the current game state based on various factors and return a score.
        Higher scores indicate better states.

        Args:
        - currentGameState: The current state of the game.
        - action: The action to evaluate.
        - agentIndex: The index of the agent (0 for player 1, 1 for player 2).

        Returns:
        - The score of the evaluated state.
        """

        # Increase the search range to explore more distant areas
        range_to_search = max(currentGameState.data.layout.width, currentGameState.data.layout.height) * 2

        successorGameState = currentGameState.generateSuccessor(agentIndex, action)

        new_agent_pos = successorGameState.getPacmanPosition(agentIndex)

        old_food = currentGameState.getFood()
        new_food = successorGameState.getFood()

        num_food_collected = len(old_food.asList()) - len(new_food.asList())

        # Calculate the number of nearby food pellets with increased search range
        nearby_food = [(x, y) for x in
                        range(new_agent_pos[0] - range_to_search, new_agent_pos[0] + range_to_search + 1)
                        for y in
                        range(new_agent_pos[1] - range_to_search, new_agent_pos[1] + range_to_search + 1)
                        if 0 <= x < new_food.width and 0 <= y < new_food.height and new_food[x][y]]

        # Calculate the number of nearby capsules with increased search range
        nearby_capsules = [(x, y) for x in
                            range(new_agent_pos[0] - range_to_search, new_agent_pos[0] + range_to_search + 1)
                            for y in
                            range(new_agent_pos[1] - range_to_search, new_agent_pos[1] + range_to_search + 1)
                            if 0 <= x < new_food.width and 0 <= y < new_food.height and (x, y) in currentGameState.getCapsules()]

        # Calculate the distance to the nearest food pellet
        closest_food_distance = min(util.manhattanDistance(new_agent_pos, food) for food in nearby_food) if nearby_food else float(
            'inf')

        # Calculate the distance to the nearest capsule
        closest_capsule_distance = min(util.manhattanDistance(new_agent_pos, capsule) for capsule in nearby_capsules) if nearby_capsules else float(
            'inf')

        final_food_bonus = 20 if len(new_food.asList()) <= 1 else 10

        remaining_capsules = len(successorGameState.getCapsules())
        score = (
                successorGameState.getScore(agentIndex) +
                num_food_collected -
                remaining_capsules * 100 -
                closest_food_distance +
                final_food_bonus -
                len(new_food.asList()) * 50
        )

        # Prioritize going towards capsules
        score -= closest_capsule_distance * 10 if closest_capsule_distance != float('inf') else 0

        # Penalize actions that lead to positions with walls
        if currentGameState.hasWall(new_agent_pos[0], new_agent_pos[1]):
            score -= 1000  # Penalize heavily for hitting a wall

        return score
          
    def isAccessible(self, start, end, gameState):
          """
          Check if the end position is accessible from the start position using BFS.

          Args:
          - start: The starting position.
          - end: The ending position to check accessibility to.
          - gameState: The current state of the game.

          Returns:
          - True if the end position is accessible from the start position, False otherwise.
          """
          # Define directions: up, down, left, right
          directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
          
          # Initialize a queue for BFS
          queue = deque([(start, [])])
          
          # Initialize a set to keep track of visited positions
          visited = set([start])
          
          while queue:
              current_pos, path = queue.popleft()
              
              # Check if current position is the end position
              if current_pos == end:
                  return True  # The end is accessible from the start
              
              # Explore neighbors in all four directions
              for direction in directions:
                  next_pos = (current_pos[0] + direction[0], current_pos[1] + direction[1])
                  
                  # Check if the next position is within the bounds of the game grid
                  if not (0 <= next_pos[0] < gameState.data.layout.width and 0 <= next_pos[1] < gameState.data.layout.height):
                      continue
                  
                  # Check if the next position is a wall or has been visited
                  if not gameState.hasWall(next_pos[0], next_pos[1]) and next_pos not in visited:
                      visited.add(next_pos)
                      queue.append((next_pos, path + [direction]))
          
          return False

