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
class YourTeamAgent():
    
  def __init__(self):
    self.oldAction = []
    self.count = 0
    self.savecount = 0
    self.mode = 0

  def getAction(self, gameState: GameState, agentIndex=0) -> str:
    self.index = agentIndex 
    history_pacman_index = 8
    bestAction = self.astar(gameState)
    
    if bestAction is None:
      bestAction = random.choice(gameState.getLegalActions(self.index))
    
    his_action = self.histolypacman(gameState,history_pacman_index)
    if his_action != None:
      bestAction = his_action
    
    self.oldAction.append(bestAction)
    self.count += 1
    return bestAction
    
  def astar(self, gameState: GameState) -> str:
    
      from queue import PriorityQueue
      
      startState = gameState.getPacmanPosition(self.index)
      goalStates = self.findFood(gameState)
      capsulesStates = gameState.getCapsules()

      # A* search
      frontier = PriorityQueue()  # PriorityQueue
      frontier.put((0, startState, []))  # (cost, position, path)
      explored = set()

      while not frontier.empty():
          cost, state, path = frontier.get()

          if state in explored:
              continue
          explored.add(state)

          if state in goalStates:
              return path[0]

          bestAction = None
          bestCost = float('inf')  # ค่า cost ที่มากที่สุด

          for action in gameState.getLegalActions(self.index):
              # จำลองการเคลื่อนที่ 
              successorState = gameState.generateSuccessor(self.index, action)
              successorPosition = successorState.getPacmanPosition(self.index)
              successorPath = path + [action]
              
              pacman_enemy_index = (self.index+1)%2
              
              # เวลากลัวของ pacman
              pacman_Scared_my = gameState.getScaredTimes(self.index)
              pacman_Scared_enemy = gameState.getScaredTimes(pacman_enemy_index)
              
              # ตำแหน่ง pamman และ ระยะห่าง
              pacmanPosition = gameState.getPacmanPosition(self.index)
              pacmanenemyPosition = gameState.getPacmanPosition(pacman_enemy_index)
              
              pacmanDistances = self.get_euclidean_distance(pacmanPosition, pacmanenemyPosition)    
              capsuleDistances =  None
              
              # คำนวณระยะห่างของแคปซูลกับตัวเรา
              if not capsulesStates:  
                  capsuleDistances = 9999
              else:
                  capsuleDistances =  min(self.get_manhattan_distance(pacmanPosition, x) for x in capsulesStates)

              if (pacman_Scared_my > 0 and pacman_Scared_enemy == 0) and (pacmanDistances > capsuleDistances) and (len(capsulesStates) > 0) and pacmanDistances <= 10:
                # ถ้าศัตรูกินแคปและเรายังไม่ได้กิน และศัตรูห่างกับเรามากกว่าแคปซูล และศัตรูใกล้เรามากกว่า 10 ช่อง ให้วิ่งไปหาแคปซูลเพื่อที่จะสวนกลับ
                # print("check1")
                successorCost =  self.heuristic_capsules(successorPosition, capsulesStates)
              elif pacman_Scared_my == 0  and (pacmanDistances < 5 and capsuleDistances < 5 and (len(capsulesStates) > 0)):
                # ถ้าศัตรูไม่กินแคป และ ระยะทางเรากับศัตรูกับเรากับอาหารน้อยกว่า 5 ช่อง ให้ไปกินอาหารมาสู้
                # print("check2")
                successorCost =  self.heuristic_capsules(successorPosition, capsulesStates)
              elif pacman_Scared_my == 0  and (pacmanDistances == capsuleDistances and (len(capsulesStates) > 0)):
                # ถ้าศัตรูไม่กินแคป และ ระยะทางเรากับศัตรูกับเรากับอาหารน้อยกว่าเท่ากัน ให้ไปกินอาหารมาสู้

                # print("check3")
                successorCost =  self.heuristic_capsules(successorPosition, capsulesStates)
              elif (pacman_Scared_enemy > 0 and pacmanDistances <= 5 ):
                # ถ้าเรากินแคปศัตรูกลัว และระยะห่างน้อยกว่า 5 ช่อง เราวิ่งไล่
                # print("check4")
                successorCost =  self.Scared_enemy_evaluationFunction(successorState)
              elif (pacman_Scared_my > 0 and pacmanDistances <= 5)  :
                # ถ้าศัตรูกินแคป และระยะห่างเรากับศัตรูน้อยกว่า 5 ช่อง เราวิ่งหนี
                # print("check5")
                successorCost =  self.Scared_my_evaluationFunction(successorState)
              else :
                if self.count < self.savecount:
                  if self.mode == 1 :
                    # print("check6")
                    #เปลี่ยนโหมดไปกินอาหารไกลที่สุดแทน
                    successorCost =  self.heuristic_max_food(successorPosition, goalStates)
                else : 
                  # print("check7")
                  #กินอาหารใหล้ที่สุดตามปกติ
                  successorCost =  self.heuristic_food(successorPosition, goalStates)

              frontier.put((successorCost, successorPosition, successorPath)) 

              if successorCost < bestCost:  # หากมีค่า cost ที่น้อยกว่า
                  bestAction = action
                  bestCost = successorCost

          if bestAction is not None:  # ถ้ามีการเลือก action ที่ดีที่สุด
              return bestAction
            

  def heuristic_food(self, position, goalStates):
      if not goalStates:  
          return 0  
      else:
          return min(self.get_manhattan_distance(position, goalState) for goalState in goalStates)
        
  def heuristic_max_food(self, position, goalStates):
      if not goalStates:  
          return 0  
      else:
          return max(self.get_euclidean_distance(position, goalState) for goalState in goalStates)
        
  def heuristic_capsules(self, position, goalStates):
      if not goalStates: 
          return 0  
      else:
          return min(self.get_manhattan_distance(position, goalState) for goalState in goalStates)
        
  def Scared_my_evaluationFunction(self, gameState: GameState) -> float: 
    
      pacman_enemy_index = (self.index+1)%2
              
      pacmanPosition = gameState.getPacmanPosition(self.index)
      pacmanenemyPosition = gameState.getPacmanPosition(pacman_enemy_index)
      
      pacmanDistances = self.get_euclidean_distance(pacmanPosition, pacmanenemyPosition)
 
      return -pacmanDistances

  def Scared_enemy_evaluationFunction(self, gameState: GameState) -> float:
      pacman_enemy_index = (self.index+1)%2
              
      pacmanPosition = gameState.getPacmanPosition(self.index)
      pacmanenemyPosition = gameState.getPacmanPosition(pacman_enemy_index)
      
      pacmanDistances = self.get_euclidean_distance(pacmanPosition, pacmanenemyPosition)
      
      return pacmanDistances

  def findFood(self, gameState: GameState):
    foodGrid = gameState.getFood()
    foodPositions = []

    for x in range(foodGrid.width):
      for y in range(foodGrid.height):
        if foodGrid[x][y]:
          foodPositions.append((x, y))
    
    # โดยปกติจะไม่ใส่แคปซูลในลิสอาหาร แต่ถ้าเปลี่ยนโหมดจะใส่
    if self.mode == 1:
      Capsules = gameState.getCapsules()
      for pos_Capsules in Capsules:
        foodPositions.append(pos_Capsules)
      
    return foodPositions

  def get_manhattan_distance(self, pos1: tuple, pos2: tuple) -> int:
    x1, y1 = pos1
    x2, y2 = pos2
    return abs(x1 - x2) + abs(y1 - y2)

  def get_euclidean_distance(self, pos1: tuple, pos2: tuple) -> float:
    import math
    x1, y1 = pos1
    x2, y2 = pos2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
  
  def histolypacman(self,gameState: GameState,history_pacman_index): 
    
    # เช็คย้อนหลัง 8 ก้าว เราไม่ได้เดินเป็นแพทเทินร์ใช่มั้ย ถ้าเป็นให้เปลี่ยนโหมดไปกินอาหารไกลที่สุดแทน 5 ตา เพื่อกันเดินขึ้นๆลงๆ แล้วก็ยัดแคปซูลลงในลิสอาหารด้วยจะได้มีเป้าหมายมากขึ้นไม่เดินเอ๋อ
    
    if len(self.oldAction) > (history_pacman_index-1):
      checkpattern = self.histolypacman_walk()
      checkdirech = self.oldAction[0] == self.oldAction[1] == self.oldAction[2] == self.oldAction[3]
      if checkpattern and checkdirech == False:
      
            self.savecount = self.count+5
            self.oldAction.pop(0)
            self.mode = 1
            return None
   
      self.oldAction.pop(0)
      return None
    
  def histolypacman_walk (self) :
    
      # คำนวณย้อนหลัง 8 ก้าวว่าเป็นแพทเทินมั้ย
      
      status_Odd = True
      status_Even = True
      
      for x in range(len(self.oldAction)):
        if(x%2 == 0):
          status_Odd = (self.oldAction[0] == self.oldAction[x])
        elif(x%2 != 0):
          status_Even = (self.oldAction[1] == self.oldAction[x])
        
        if(status_Odd == False or status_Even == False):
          break
      
      checkpattern = (status_Odd == True ) and (status_Even == True)
      return checkpattern
