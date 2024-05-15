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
class YourTeamAgent(Agent):
  def __init__(self):
    self.lastPositions = []
    self.dc = None
  

  def getAction(self, gameState: GameState,agentIndex=1) -> str:
    #print("-----") #wall check
    #print(gameState.getCapsules()) #item
    #print(gameState.getNumFood()) #all food left
    #print(gameState.hasWall(0, 0)) #check wall
    #print(gameState.hasFood(1, 2)) #check food
    #print(gameState.getPacmanPosition(0)) #position pacman
    #distance
    #print(gameState.getWalls())
    #Information on game
    
    positionE = gameState.getPacmanPosition(0)
    position1 = gameState.getPacmanPosition(1)
    xE = positionE[0]
    x1 = position1[0]
    yE = positionE[1]
    y1 = position1[1]
    x = abs(positionE[0] - position1[0])
    y = abs(positionE[1] - position1[1])
    Wall = gameState.getWalls()
    Food = gameState.getFood()
    Numfood = gameState.getNumFood()
    scare = gameState.getScaredTimes(1)
    #print(scare)
    #print("----->",Numfood)
    #direction = gameState.getPacmanState(1)
    #print("=====")
    #print(Food)
    #if Wall[x1][y1] == False:
      #print("Nowalls")  
    foodpoint = YourTeamAgent.foodpoint(Food)
    halfwf = int(Food.width/2)
    halfhf = int(Food.height/2) 
    bestfoodlocation = YourTeamAgent.bestfood_form_enemie(foodpoint,xE,yE)
    #print(bestfoodlocation)
    #print(legal)
    #print(Point_To_Go)
    #turn = ReflexAgent.nearest_Food(x1,y1,Food)
    #if x>0 :
      #Point_To_Go = ReflexAgent.eat_form_enemie2(xE,yE,foodpoint,bestfoodlocation)
    #else :
    YourTeamAgent.foodzone(Food)
    Point_To_Go = YourTeamAgent.eat_form_me(x1,y1,foodpoint)
    if Food[halfwf][halfhf] == True:
      Point_To_Go = (halfwf,halfhf)
    turn = YourTeamAgent.Go_To_point(x1,y1,Point_To_Go[0],Point_To_Go[1])
    #turn = YourTeamAgent.nearest_Food(x1,y1,Food)
    if (scare > 0):
      if ((x<4)) and ((y<4)):
        turn = YourTeamAgent.runaway(xE,x1,yE,y1,position1)
    #print(Food.height-1)
    action = gameState.getLegalActions(1)
    for i in action:
      if turn == i:
        confirmturn = turn
        break
      if x1 == 1:
        if xE == x1+1 and yE == Food.height-1:
          confirmturn = "South"
        if xE == x1+1 and yE == 1:
          confirmturn = "North"
      if x1 == Food.width-1:
        if xE == x1-1 and yE == Food.height-1:
          confirmturn = "South"
        if xE == x1-1 and yE == 1:
          confirmturn = "North"
      if y1 == 1:
        if  yE == y1+1 and xE == 1:
          confirmturn = "East"
        if yE == y1+1 and xE == Food.width-1:
          confirmturn = "West"
      if y1 == Food.height-1:
        if  yE == y1-1 and xE == 1:
          confirmturn = "East"
        if yE == y1-1 and xE == Food.width-1:
          confirmturn = "West"
      else:
        confirmturn = random.choice(action)
        
    #print(turn)
    #print(distination[0],distination[1])
    #turn = ReflexAgent.Go_To_point(x1,y1,distination[0],distination[1])
    #print(turn)
    return confirmturn
  
  def foodzone(Food):
    foodcolumn = Food.width
    foodrow = Food.height
    fc = foodcolumn/3
    totalfood = 0
    for w in range(foodcolumn):
      if (w+1)%fc == 0 :
        #print(totalfood) 
        totalfood = 0  
      for h in range(foodrow):
        if Food[w][h] == True:
          totalfood+=1
    #print("====")
    return 
  
  def foodpoint(Food):
    foodcolumn = Food.width
    foodrow = Food.height
    #fc = foodcolumn/3
    #totalfood = 0
    allfoodpoint  = []
    for w in range(foodcolumn):
      #if (w+1)%fc == 0 :
        #print(totalfood) 
        #totalfood = 0  
      for h in range(foodrow):
        if Food[w][h] == True:
          #totalfood+=1
          allfoodpoint.append([w,h])
          #return (w,h)
    #print("=======")  
    #print(allfoodpoint)
    return allfoodpoint

  def bestfood_form_enemie(foodpoint,xE,yE):
    LeftBottom = 0
    LeftTop = 0
    RightBottom = 0
    RightTop = 0
    #print(foodpoint)
    #print(foodpoint[7][1])
    for i in foodpoint:
      if i[0] <= xE and i[1] <= yE:
        LeftBottom+=1
      elif i[0] <= xE and i[1] >= yE:
        LeftTop+=1
      elif i[0] >= xE and i[1] <= yE:
        RightBottom+=1
      elif i[0] >= xE and i[1] >= yE:
        RightTop+=1
    #print(LeftTop,RightTop)
    #print(LeftBottom,RightBottom)
    #print("=====")
    best = max(LeftBottom,LeftTop,RightBottom,RightTop)
    if best == LeftBottom:
      return "LeftBottom"
    elif best == LeftTop:
      return "LeftTop"
    elif best == RightBottom:
      return "RightBottom"
    elif best == RightTop:
      return "RightTop"
    return best
  
  def eat_form_enemie2(xE,yE,foodpoint,bestfoodlocation):
    s = []
    slo = []
    for i in foodpoint:
      if i[0] <= xE and i[1] <= yE:
        xdif = xE - i[0]
        ydif = yE - i[1]
        sxsy = xdif+ydif
        #print(sxsy)
        s.append(sxsy)
        slo.append(i)
    if len(s) == 0:
      return (1,1)
    mins = min(s)
    #print(mins)
    for i in range(len(s)):
      #print(i)
      if mins == s[i]:
        return slo[i]
      #print(s,slo)
    return 

  def eat_form_enemie(xE,yE,foodpoint,bestfoodlocation):
    s = []
    slo = []
    if bestfoodlocation == "LeftBottom":
      for i in foodpoint:
        if i[0] <= xE and i[1] <= yE:
          xdif = xE - i[0]
          ydif = yE - i[1]
          sxsy = xdif+ydif
          #print(sxsy)
          s.append(sxsy)
          slo.append(i)
      #print(s,slo)
      if len(s) == 0:
        return (1,1)
      mins = min(s)
      #print(mins)
      for i in range(len(s)):
        #print(i)
        if mins == s[i]:
          return slo[i]
        
    elif bestfoodlocation == "LeftTop":
      for i in foodpoint:
        if i[0] <= xE and i[1] >= yE:
          xdif = xE - i[0]
          ydif = i[1] - yE
          sxsy = xdif+ydif
          #print(sxsy)
          s.append(sxsy)
          slo.append(i)
      #print(s,slo)
      if len(s) == 0:
        return (1,1)
      mins = min(s)
      #print(mins)
      for i in range(len(s)):
        #print(i)
        if mins == s[i]:
          return slo[i]
    
    elif bestfoodlocation == "RightBottom":
      for i in foodpoint:
        if i[0] >= xE and i[1] <= yE:
          xdif = i[0] - xE
          ydif = yE - i[1]
          sxsy = xdif+ydif
          #print(sxsy)
          s.append(sxsy)
          slo.append(i)
      #print(s,slo)
      if len(s) == 0:
        return (1,1)
      mins = min(s)
      #print(mins)
      for i in range(len(s)):
        #print(i)
        if mins == s[i]:
          return slo[i]
        
    elif bestfoodlocation == "RightTop":
      for i in foodpoint:
        if i[0] >= xE and i[1] >= yE:
          xdif = i[0] - xE
          ydif = i[1] - yE
          sxsy = xdif+ydif
          #print(sxsy)
          s.append(sxsy)
          slo.append(i)
      #print(s,slo)
      if len(s) == 0:
        return (1,1)
      mins = min(s)
      #print(mins)
      for i in range(len(s)):
        #print(i)
        if mins == s[i]:
          return slo[i]
    return 
  
  def eat_form_me(x1,y1,foodpoint):
    slo = []
    s = []
    for i in foodpoint:
      absx = abs(x1-i[0])
      absy = abs(y1-i[1])
      xy = absx+absy
      #print(sxsy)
      s.append(xy)
      slo.append(i)
      #print(s,slo)
    if len(s) == 0:
        return (1,1)
    mins = min(s)
    #print(mins)
    for i in range(len(s)):
      #print(i)
      if mins == s[i]:
        return slo[i]
    return ""
  
  
  def Go_To_point(x1,y1,xt,yt):
    if y1 < yt :
      turn = "North"
    elif y1 > yt :
      turn = "South"
    elif x1 > xt :
      turn = "West"
    elif x1 < xt :
      turn = "East"
    else:
      turn = "East"
    return turn
    

  def runaway(x0,x1,y0,y1,position1):
    print("enemie around here")
    if x0<x1:
      turn = "East"
      print("ละกูหลบขวา")
    elif x0>x1:
      turn = "West"
      print("ละกูหลบซ้าย")
    elif y0<y1:
      turn = "North"
      print("ละกูหลบขึ้น")
    elif y0>y1:
      turn = "South"
      print("ละกูหลบลง")
    return turn

  
  def nearest_Food(x1,y1,Food):
    nextNorth = y1+1
    nextSouth = y1-1
    nextEast = x1+1
    nextWest = x1-1
    if Food[x1][nextNorth] == True:
      #print("north have food")
      nf = True
    else:
      nf = False
    if Food[x1][nextSouth] == True:
      #print("South have food")
      sf = True
    else:
      sf = False
    if Food[nextEast][y1] == True:
      #print("East have food")
      ef = True
    else:
      ef = False
    if Food[nextWest][y1] == True:
      #print("West have food")
      wf = True
    else:
      wf = False
    #==================
    turn = ""
    if wf == True:
      turn = "West"
    if ef == True:
      turn = "East"
    if nf == True:
      turn = "North"
    if sf == True:
      turn = "South"  
    #=================
    return turn
  
  
  
    #print(position1,"##",position2)
    #print(x,y)
    
    #if ((-2<=x<=2)) and ((-2<=y<=2)) :
      #print("enemies around here")
  
  #def go_eat_near_enemie():
    #if ((-2<=x<=2)) and ((-2<=y<=2)) :
      #print("enemies around here")
    #else :
      
      
    #print(gameState.getLegalActions(1))
    #print(gameState.getGhostPosition())
    
    #print(gameState.getCapsules())
    #print(gameState.getFood())
    #print(gameState.getWalls())
    
    #print(legalMoves) #wall
    #print(bestIndices) #raya
    #print(chosenIndex) #turn



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
    #successorGameState = currentGameState.generatePacmanSuccessor(action,agentIndex)
    #newPos = successorGameState.getPacmanPosition(agentIndex)
    #print("-----")
    #print(newPos)
    #oldFood = currentGameState.getFood()
    #print("-----")
    #print(oldFood) #food check
    
    successorGameState = currentGameState.generatePacmanSuccessor(action,agentIndex)
    newPos = successorGameState.getPacmanPosition(agentIndex)
    oldFood = currentGameState.getFood()
    #print(oldFood)
    return successorGameState.getScore(agentIndex)
  

def scoreEvaluationFunction(currentGameState: GameState,agentIndex=0) -> float:
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore(agentIndex)