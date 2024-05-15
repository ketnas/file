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
# class YourTeamAgent(MultiAgentSearchAgent):
#   """
#     Your team agent
#     แก้ เพิ่มเติม ได้ใน class นี้เท่านั้น
#     แต่ห้ามแก้ชื่อ class หรือ method ที่กำหนดให้
#     แต่เพิ่ม method เองได้ และเรียกใช้ได้ใน method ใน class นี้
#   """
#   def getAction(self, gameState: GameState,agentIndex = 0) -> str:
#     pass
#     # ต้อง return action ที่ดีที่สุดด้วยนะ
#     #  return bestAction

#   def evaluationFunction(self, currentGameState: GameState, action: str,agentIndex=0) -> float:
#     # อาจจะไม่ใช้ก็ได้ แต่ถ้าจะใช้ ให้ return ค่าที่ดีที่สุด
#     pass
#######################################################################################
"""อย่าลืม Import นะครับบบบบบ"""
import heapq
import random

class YourTeamAgent(MultiAgentSearchAgent):
  def getAction(self, gameState, agentIndex=0) -> str:
        """Choose the best action, avoiding other agents or chasing them depending on their state."""
        # Get the current position of our Pacman agent
        current_position = gameState.getPacmanPosition(agentIndex)

        # Get information about the other agent
        other_index = (agentIndex + 1) % gameState.getNumAgents()
        other_position = gameState.getPacmanPosition(other_index)
        other_state = gameState.getPacmanState(other_index)
        food_count = gameState.getNumFood()
        ourscore = gameState.getScore(agentIndex)
        theirscore = gameState.getScore(other_index)
        # Get capsules and food positions
        capsules = gameState.getCapsules()
        food = gameState.getFood().asList()
        width, height = self.get_map_size(gameState)
        if ourscore*1.5 > theirscore:
            distancegap = 13
        elif ourscore > theirscore:
            distancegap = 10
        elif ourscore < theirscore:
            distancegap = 6
        else:
            distancegap = 5
        # Retrieve all legal actions
        legal_actions = gameState.getLegalActions(agentIndex)
        a = util.manhattanDistance(current_position, other_position) 
        # Determine whether we are in "getCapsule" mode or "find" mode
        if capsules and util.manhattanDistance(current_position, other_position) <= distancegap and self.other_agent_closer_to_capsule(gameState, agentIndex, other_index) == False :
            # for capsule in capsules:
            #     capsule = util.manhattanDistance(current_position, capsules)
            #     if capsule <= 3 and :
            mode = "getCapsule"
            targets = capsules
        elif food:
            mode = "getFood"
            targets = food
        else:
            mode = "random"
            targets = []
        if not targets and other_state.scaredTimer == 0:
            return random.choice(legal_actions) if legal_actions else Directions.STOP
        # If no valid targets exist, choose a random legal action
        # If targeting a capsule but the other agent is not vulnerable, avoid them
        if mode == "getCapsule" and other_state.scaredTimer > 0:
                if gameState.getPacmanState(agentIndex).scaredTimer > other_state.scaredTimer and util.manhattanDistance(current_position, other_position) <= distancegap:  
                    # farthest_point = self.find_farthest_point(gameState, current_position, other_position)
                    # path = self.a_star_search(gameState, current_position, [farthest_point])
                    path = self.a_star_search_avoid(gameState, current_position, targets, other_position)
                else:
                    if other_state.scaredTimer <= 3 and other_state.scaredTimer > 1 and self.other_agent_closer_to_capsule(gameState, agentIndex, other_index):
                        # Find the farthest point from the other agent
                        farthest_point = self.find_farthest_point(gameState, current_position, other_position)
                        path = self.a_star_search_avoid(gameState, current_position, targets, other_position)
                        # path = self.a_star_search(gameState, current_position, [farthest_point])
                    else:
                        if util.manhattanDistance(current_position, other_position) <= distancegap:  # Adjust the distance threshold as needed
                         targets = [other_position]
                         path = self.a_star_search(gameState, current_position, targets)
                        else:
                         path = self.a_star_search(gameState, current_position, targets)                      
        elif  gameState.getPacmanState(agentIndex).scaredTimer  > 0:
            if util.manhattanDistance(current_position, other_position) <= distancegap:
                path = self.a_star_search_avoid(gameState, current_position, targets, other_position)
            else:
                path = self.a_star_search(gameState, current_position, targets)
        elif gameState.getPacmanState(agentIndex).scaredTimer < other_state.scaredTimer:
            if food_count > 3 and theirscore > ourscore:
                targets = [other_position]
                path = self.a_star_search(gameState, current_position, targets)
            else:
                path = self.a_star_search(gameState, current_position, targets)
        elif util.manhattanDistance(current_position, other_position) <= distancegap and self.other_agent_closer_to_capsule(gameState, agentIndex, other_index):
            # targets = food
            path = self.a_star_search_avoid(gameState, current_position, targets, other_position)
        else:
            if food_count < 10:
                path = self.a_star_search_avoid_farthest_food(gameState, current_position, targets, other_position)
            else:
                path = self.a_star_search(gameState, current_position, targets)
        if path:
            return path[0]
        else:
            # If no path is found, choose a random legal action
            return random.choice(legal_actions) if legal_actions else Directions.STOP

  
  def get_map_size(self, gameState):
    """Calculate the size of the map."""
    walls = gameState.getWalls()
    width = walls.width
    height = walls.height
    return width, height

  def find_farthest_point(self, gameState, start, other_position):
    """Find the farthest point from the other agent with a minimum cluster of nearby food dots."""
    max_distance = float('-inf')
    farthest_point = None

    width, height = self.get_map_size(gameState)  # Get the size of the map

    walls = gameState.getWalls()
    food = gameState.getFood().asList()
    capsules = gameState.getCapsules()

    # Adjust the distance threshold based on the map size
    distance_threshold = 3

    # If the other agent is too close, prioritize finding the closest food cluster while maintaining distance
    if util.manhattanDistance(start, other_position) <= distance_threshold:  # Adjust the distance threshold as needed
        for x in range(walls.width):
            for y in range(walls.height):
                if not walls[x][y]:
                    distance = util.manhattanDistance((x, y), start)
                    if distance > max_distance:
                        # Check nearby food dots within a certain radius
                        food_count = sum(1 for food_pos in food if util.manhattanDistance((x, y), food_pos) <= 3)
                        if food_count >= 2 and distance > 4:  # Adjust the cluster and distance thresholds as needed
                            max_distance = distance
                            farthest_point = (x, y)
    
    else:
        # If the other agent is not too close, prioritize finding capsules or food within a certain range while maintaining distance
        if capsules:
            for capsule in capsules:
                distance_to_capsule = util.manhattanDistance(start, capsule)
                if distance_to_capsule <= distance_threshold:  # Adjust the distance threshold based on the map size
                    if distance_to_capsule > max_distance:
                        max_distance = distance_to_capsule
                        farthest_point = capsule

        # If no suitable capsules are found or if food is closer than capsules, choose food instead
        if not farthest_point:
            for food_pos in food:
                distance_to_food = util.manhattanDistance(start, food_pos)
                if distance_to_food <= distance_threshold:  # Adjust the distance threshold based on the map size
                    if distance_to_food > max_distance:
                        max_distance = distance_to_food
                        farthest_point = food_pos
    
        # If no suitable capsules or food are found, focus on finding the farthest point from the other agent without considering food clusters
        if not farthest_point:
            for x in range(walls.width):
                for y in range(walls.height):
                    if not walls[x][y]:
                        distance = util.manhattanDistance((x, y), other_position)
                        if distance > max_distance:
                            max_distance = distance
                            farthest_point = (x, y)
    return farthest_point


  def a_star_search(self, gameState, start, targets):
    """A* Search to find the best path to any target."""
    if not targets:
        return None

    walls = gameState.getWalls()
    food = gameState.getFood().asList()

    def heuristic(pos, goals):
        # Calculate the minimum Manhattan distance to any food in the cluster
        return min([util.manhattanDistance(pos, goal) for goal in goals])

    def neighbors(pos):
        x, y = pos
        possible_directions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        deltas = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        neighbors = []

        for direction, (dx, dy) in zip(possible_directions, deltas):
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < walls.width and 0 <= new_y < walls.height and not walls[new_x][new_y]:
                neighbors.append((direction, (new_x, new_y)))
        return neighbors

    frontier = []
    heapq.heappush(frontier, (0, start, []))
    explored = set()

    while frontier:
        cost, current, path = heapq.heappop(frontier)

        if current in explored:
            continue
        explored.add(current)

        if current in targets:
            return path

        for direction, neighbor in neighbors(current):
            new_cost = cost + 1 + heuristic(neighbor, targets)
            heapq.heappush(frontier, (new_cost, neighbor, path + [direction]))

    return None

  def other_agent_closer_to_capsule(self, gameState, agentIndex, other_index):
    """Check if the other agent is closer to the same capsule as our agent."""
    current_position = gameState.getPacmanPosition(agentIndex)
    other_position = gameState.getPacmanPosition(other_index)
    capsules = gameState.getCapsules()

    if not capsules:
        return False

    agent_distance_to_capsules = [util.manhattanDistance(current_position, capsule) for capsule in capsules]
    other_agent_distance_to_capsules = [util.manhattanDistance(other_position, capsule) for capsule in capsules]

    # Compare distances to all capsules
    for agent_distance, other_distance in zip(agent_distance_to_capsules, other_agent_distance_to_capsules):
        if other_distance < agent_distance:
            return True

    return False
  def a_star_search_avoid(self, gameState, start, targets, avoid_pos, avoid_range=4, food_cluster_threshold=2, cluster_radius=2):
    """A* Search to find the best path to any target, avoiding a specific position."""
    agentIndex = 0
    walls = gameState.getWalls()
    other_index = (agentIndex + 1) % gameState.getNumAgents()
    food = gameState.getFood().asList()
    capsules = gameState.getCapsules()
    ourscore = gameState.getScore(agentIndex)
    theirscore = gameState.getScore(other_index)

    # Determine the avoid_range based on the score difference
    # if ourscore*1.5 > theirscore:
    #     avoid_range = 8
    # elif ourscore > theirscore:
    #     avoid_range = 4
    # elif ourscore < theirscore:
    #     avoid_range = 2
    # else:
    #     distancegap = 1

    if not targets:
        # If there are no targets left, move to a position far away from the current position
        farthest_position = None
        max_distance = float('-inf')

        for x in range(walls.width):
            for y in range(walls.height):
                if not walls[x][y]:
                    distance_to_start = util.manhattanDistance((x, y), start)
                    if distance_to_start > max_distance:
                        max_distance = distance_to_start
                        farthest_position = (x, y)

        # Perform A* search to move to the farthest position
        if farthest_position:
            return self.a_star_search(gameState, start, [farthest_position])

        return None

    def heuristic(pos, goal, avoid_pos):
        # Calculate the Manhattan distance to the goal
        h = util.manhattanDistance(pos, goal)
        # If the avoid position is within the same food grid as the goal, prioritize moving away from it
        if util.manhattanDistance(pos, avoid_pos) <= 1:
            # Increase the heuristic value to encourage moving away from the avoid position
            h += 5
        return h

    def neighbors(pos):
        x, y = pos
        possible_directions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        deltas = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        neighbors = []

        for direction, (dx, dy) in zip(possible_directions, deltas):
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < walls.width and 0 <= new_y < walls.height and not walls[new_x][new_y]:
                if (new_x, new_y) != avoid_pos:
                    distance_to_other = util.manhattanDistance((new_x, new_y), avoid_pos)
                    if distance_to_other > avoid_range:
                        neighbors.append((direction, (new_x, new_y)))
        return neighbors

    frontier = []
    heapq.heappush(frontier, (0, start, []))
    explored = set()

    while frontier:
        cost, current, path = heapq.heappop(frontier)

        if current in explored:
            continue
        explored.add(current)

        if current in targets:
            return path

        for direction, neighbor in neighbors(current):
            new_cost = cost + 1 + min([heuristic(neighbor, target, avoid_pos) for target in targets])
            heapq.heappush(frontier, (new_cost, neighbor, path + [direction]))

    # Check if any large food clusters are nearby
    for x in range(0, walls.width, cluster_radius):
        for y in range(0, walls.height, cluster_radius):
            cluster_food = [food_pos for food_pos in food if x <= food_pos[0] < x + cluster_radius and y <= food_pos[1] < y + cluster_radius]
            if len(cluster_food) >= food_cluster_threshold:
                # Check if the other agent is within the food grid * 2 that we are targeting
                other_in_range = any(util.manhattanDistance(food_pos, avoid_pos) < avoid_range * 2 for food_pos in cluster_food)
                if not other_in_range:
                    # Find the farthest food point from the other agent
                    farthest_food = None
                    max_distance = float('-inf')
                    for food_pos in cluster_food:
                        distance_to_other = util.manhattanDistance(food_pos, avoid_pos)
                        if distance_to_other > avoid_range:
                            distance_to_start = util.manhattanDistance(food_pos, start)
                            if distance_to_start > max_distance:
                                max_distance = distance_to_start
                                farthest_food = food_pos
                    if farthest_food:
                        path = self.a_star_search(gameState, start, [farthest_food])
                        if path:
                            return path

    return None

  def a_star_search_avoid_farthest_food(self, gameState, start, targets, avoid_pos, avoid_range=4, food_cluster_threshold=2, cluster_radius=2, agentIndex=0, other_index=0):
    """A* Search to find the best path to any target, avoiding a specific position and prioritizing food farthest from the other agent."""
    walls = gameState.getWalls()
    food = gameState.getFood().asList()
    capsules = gameState.getCapsules()
    other_index = (agentIndex + 1) % gameState.getNumAgents()
    if not targets:
        # If there are no targets left, move to a position far away from the current position
        farthest_position = None
        max_distance = float('-inf')

        for x in range(walls.width):
            for y in range(walls.height):
                if not walls[x][y]:
                    distance_to_start = util.manhattanDistance((x, y), start)
                    if distance_to_start > max_distance:
                        max_distance = distance_to_start
                        farthest_position = (x, y)

        # Perform A* search to move to the farthest position
        if farthest_position:
            return self.a_star_search(gameState, start, [farthest_position])

        return None

    def heuristic(pos, goal, avoid_pos):
        # Calculate the Manhattan distance to the goal
        h = util.manhattanDistance(pos, goal)
        # If the avoid position is within the same food grid as the goal, prioritize moving away from it
        if util.manhattanDistance(pos, avoid_pos) <= 1:
            # Increase the heuristic value to encourage moving away from the avoid position
            h += 5
        return h

    def neighbors(pos):
        x, y = pos
        possible_directions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        deltas = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        neighbors = []

        for direction, (dx, dy) in zip(possible_directions, deltas):
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < walls.width and 0 <= new_y < walls.height and not walls[new_x][new_y]:
                if (new_x, new_y) != avoid_pos:
                    distance_to_other = util.manhattanDistance((new_x, new_y), avoid_pos)
                    if distance_to_other > avoid_range:
                        neighbors.append((direction, (new_x, new_y)))
        return neighbors

    def food_farthest_from_other_agent():
        # Find the farthest food point from the other agent
        farthest_food = None
        max_distance = float('-inf')
        for food_pos in food:
            distance_to_other = util.manhattanDistance(food_pos, avoid_pos)
            if distance_to_other > avoid_range:
                distance_to_start = util.manhattanDistance(food_pos, start)
                if distance_to_start > max_distance:
                    max_distance = distance_to_start
                    farthest_food = food_pos
        return farthest_food

    food_count = gameState.getNumFood()
    our_score = gameState.getScore(agentIndex)
    their_score = gameState.getScore(other_index)

    # If food count is low and the other agent cannot get a score higher than ours, prioritize food farthest from the other agent
    if food_count <= 10 and their_score + (food_count*10) <= our_score:
        farthest_food = food_farthest_from_other_agent()
        if farthest_food:
            return self.a_star_search(gameState, start, [farthest_food])
    elif food_count <= 10 and their_score >= our_score:
            targets = food
            return self.a_star_search(gameState, start, targets)

    frontier = []
    heapq.heappush(frontier, (0, start, []))
    explored = set()

    while frontier:
        cost, current, path = heapq.heappop(frontier)

        if current in explored:
            continue
        explored.add(current)

        if current in targets:
            return path

        for direction, neighbor in neighbors(current):
            new_cost = cost + 1 + min([heuristic(neighbor, target, avoid_pos) for target in targets])
            heapq.heappush(frontier, (new_cost, neighbor, path + [direction]))

    # Check if any large food clusters are nearby
    for x in range(0, walls.width, cluster_radius):
        for y in range(0, walls.height, cluster_radius):
            cluster_food = [food_pos for food_pos in food if x <= food_pos[0] < x + cluster_radius and y <= food_pos[1] < y + cluster_radius]
            if len(cluster_food) >= food_cluster_threshold:
                # Find the farthest food point from the other agent
                farthest_food = food_farthest_from_other_agent()
                if farthest_food:
                    path = self.a_star_search(gameState, start, [farthest_food])
                    if path:
                        return path

    return None
  def find_farthest_food_from_agent(self, gameState, agentIndex, otherAgentIndex):
    """Find the farthest food from the other agent."""
    current_position = gameState.getPacmanPosition(agentIndex)
    other_position = gameState.getPacmanPosition(otherAgentIndex)
    food = gameState.getFood().asList()

    max_distance = float('-inf')
    farthest_food = None

    for food_pos in food:
        distance_to_other = util.manhattanDistance(food_pos, other_position)
        if distance_to_other > max_distance:
            max_distance = distance_to_other
            farthest_food = food_pos

    return farthest_food