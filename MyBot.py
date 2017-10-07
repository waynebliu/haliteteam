"""
Note: Please do not place print statements here as they are used to communicate with the Halite engine. If you need
to log anything use the logging module.
"""
# Let's start by importing the Halite Starter Kit so we can interface with the Halite engine
import hlt
import numpy as np
import pandas as pd
import logging
logging.basicConfig(filename='mybot.log',level=logging.DEBUG)


# constants
MyPlanetGradientRadius = 10
MyPlanetGradientFactor = 20
UnownedPlanetGradientRadius = 200
UnownedPlanetGradientStrength = 10
EnemyPlanetGradientRadius = 200
EnemyPlanetGradientStrength = 1
LowWeight = -999999


# global just to avoid memory allocations...
Xship = np.outer( range(-7,8), np.ones(15) )
Yship = np.outer( np.ones(15), range(-7,8) )
maxrangemask = None
gradientfield = None
Xmap = None
Ymap = None
pathlists = None
pointsetmaplist = None
angleof = None
speedof = None

game = None

def doOneTurn():
    # TURN START
    # Update the map for the new turn and get the latest version
    game_map = game.update_map()

    # one-time init per game
    initMaps(game_map)

    # analyze positions/strategies
    analysis = Analysis(game_map)

    # command the ships actions
    command_queue = commandShips(game_map, analysis)

    # Send our set of commands to the Halite engine for this turn
    game.send_command_queue(command_queue)
    # TURN END

def initMaps(game_map):
    global Xship,Yship,Xmap,Ymap,pathlists,pointsetmaplist,angleof,speedof,gradientfield,maxrangemask
    if Xmap is None:
        Xmap = np.outer( range(game_map.width+14), np.ones(game_map.height+14) )
        Ymap = np.outer( np.ones(game_map.width+14), range(game_map.height+14) )
        gradientfield = np.zeros([game_map.width+14,game_map.height+14])
        pathlists = genPathLists()
        pointsetmaplist = getPointSetList(pathlists)
        speed = np.sqrt(Xship * Xship + Yship * Yship)
        speedof = speed.reshape(15*15)
        angleof = ( (np.degrees(np.arctan2(Xship,Yship)) + 270 )% 360).reshape(15*15)
        maxrangemask = (speed >7).reshape(15*15)

saved = 1
class Analysis:
    def __init__(self,game_map):
        global gradientfield
        self.me = game_map.get_me()
        # reinitialize
        gradientfield.fill(0)

        # compute various quantities
        self.shipX,self.shipY = getEntityXYs(self.me.all_ships())
        self.closeShips = computeEntitiesDistances(
                self.shipX, self.shipY,
                self.shipX, self.shipY) < 14

        self.planetX,self.planetY = getEntityXYs(game_map.all_planets())
        self.planetDist = computeEntitiesDistances(self.shipX,self.shipY,
                self.planetX, self.planetY)
        self.planetRadius = tuple(p.radius for p in game_map.all_planets())
        planetCloseness = np.outer( np.ones( len(self.me.all_ships() ) ),
                                    self.planetRadius ) + 7
        self.closePlanets = self.planetDist < planetCloseness
        global saved
        #np.savetxt("dist{}.csv".format(saved), self.planetDist, delimiter=",")
        #np.savetxt("closeness{}.csv".format(saved), planetCloseness, delimiter=",")
        #np.savetxt("close{}.csv".format(saved), self.closePlanets, delimiter=",")
        #saved += 1

        enemyships = []
        for player in game_map.all_players():
            enemyships += player.all_ships()
        self.enemyshipX,self.enemyshipY = getEntityXYs(enemyships)
        self.enemyshipDist = computeEntitiesDistances(self.shipX,self.shipY,
                self.enemyshipX, self.enemyshipY)
        self.closeEnemyShips = self.enemyshipDist < 15

        # set up the strategic goals
        self.setupStrategicGradient(game_map,gradientfield)

    def setupStrategicGradient(self,game_map,gradientfield):
        for planet in game_map.all_planets():
            if planet.owner is None:
                applyPlanetField(planet.x, planet.y,planet.radius,
                        gradientfield,
                        UnownedPlanetGradientStrength,
                        planet.num_docking_spots+UnownedPlanetGradientRadius)
            elif planet.owner == self.me:
                strength =  planet.num_docking_spots - len(planet._docked_ship_ids)
                if strength > 0:
                    applyPlanetField(planet.x, planet.y,planet.radius,
                        gradientfield,
                        strength*MyPlanetGradientFactor,
                        planet.num_docking_spots+MyPlanetGradientRadius)
            else:
                applyPlanetField(planet.x, planet.y,planet.radius,
                    gradientfield,
                    EnemyPlanetGradientStrength,
                    planet.num_docking_spots+EnemyPlanetGradientRadius)


def applyPlanetField(x, y, radius, gradientfield, strength, gradientradius):
    #logging.info( ("planet", x, y, radius, strength, gradientradius) )
    gradientfield += ( ( gradientradius / ( 
              np.multiply( Xmap - (x+7), Xmap - (x+7)) 
              + np.multiply( Ymap - (y+7), Ymap - (y+7))
            ) ) * strength / gradientradius ).clip(min=0)
    # don't go to any point on the planet
    planetmask = np.sqrt( 
              np.multiply( Xmap - (x+7), Xmap - (x+7)) 
              + np.multiply( Ymap - (y+7), Ymap - (y+7))
            ) < (radius + 2)
    gradientfield[planetmask] = LowWeight

def getEntityXYs(myships):
    return tuple( s.x for s in myships), tuple( s.y for s in myships )

def computeEntitiesDistances(shipsXs,shipsYs,allshipsXs,allshipsYs):
    ''' my ships are first index, other entities are second index '''
    numships = len(shipsXs)
    numallships = len(allshipsXs)
    shipsXmat = np.outer(shipsXs, np.ones(numallships))
    shipsYmat = np.outer(shipsYs, np.ones(numallships))
    allshipsXmat = np.outer(np.ones(numships),allshipsXs)
    allshipsYmat = np.outer(np.ones(numships),allshipsYs)
    distancesmat = np.sqrt( np.multiply( 
                        allshipsXmat - shipsXmat,
                        allshipsXmat - shipsXmat ) 
                   + np.multiply( 
                        allshipsYmat - shipsYmat,
                        allshipsYmat - shipsYmat )
                   )
    return distancesmat

def commandShips(game_map, analysis):
    # Here we define the set of commands to be sent to the Halite engine at the end of the turn
    turnstate = TurnState()

    # For every ship that I control
    for i,ship in enumerate(analysis.me.all_ships()):
        # If the ship is docked Skip this ship
        if ship.docking_status != ship.DockingStatus.UNDOCKED:
            continue
        # check if I'm near any dockable planets
        dock = False
        for p in np.argwhere(analysis.closePlanets[i]):
            #logging.info( (i,p,analysis.planetDist,analysis.planetRadius) )
                planet = game_map.all_planets()[p[0]]
                if (not ship.can_dock(planet)) or (
                                planet.owner is not None 
                                and planet.owner != analysis.me):
                    continue
                turnstate.addDock(ship,planet)
                dock=True
                break
        if dock:
            continue

        # move to somewhere...
        maneuverShip(ship, i, turnstate, analysis, game_map)

    return turnstate.getCommands()

class TurnState:
    def __init__(self):
        self.command_queue = []
        self.shippaths={}
    def getCommands(self):
        return self.command_queue
    def addDock(self,ship,planet):
        self.command_queue.append(ship.dock(planet))
    def addMove(self,ship,i,ind):
        self.command_queue.append(ship.thrust(speedof[ind],angleof[ind]) )
        self.shippaths[i] = []
    

def maneuverShip(ship, i, turnstate, analysis, game_map):
    movemap = gradientfield[ 
                    int(ship.x-0):int(ship.x+15),
                    int(ship.y-0):int(ship.y+15) 
                ].copy().reshape(15*15)
    #logging.info( movemap.reshape([15,15]).clip(min=0) )
    # check friendly ships
    # we don't care about ramming enemy ships!
    for other in np.argwhere(analysis.closeShips[i]):
        if other == i:
            continue
        path = turnstate.shippaths.get(other[0])
        if path is not None:
            blotout(movemap, path, ship.x, ship.y)
        else:
            othership = analysis.me.all_ships()[other[0]]
            blotout(movemap, [(othership.x,othership.y,i) for i in range(1,8)], ship.x, ship.y)
    # find highest point in movemap and go there
    movemap[maxrangemask] = LowWeight
    target = movemap.argmax()
    #logging.info( ("move", ship.x,ship.y,i,target%15-7,target/15-7) )
    turnstate.addMove( ship, i, target )



def blotout(movemap,path,shipx,shipy):
    ''' blot out any square that's blocked '''
    for x,y,step in path:
        inds = pointsetmaplist.get( (x-shipx,y-shipy,step), [] )
        #logging.info( (x-shipx,y-shipy,step, inds ) )
        movemap[ inds ] = LowWeight
        #logging.info( movemap )


def genPathList(x,y):
    ' assuming distance(x,y) <= moveMax (7) '
    path = []
    for i in range(1,8):
        xloc,yloc = x*i/7,y*i/7
        path.append( 
            set( ( int(xloc-xoff),int(yloc-xoff) ) for xoff in [-.8,0,.8] for yoff in [-.8,0,.8]  )
        )
    return path
def multpath(paths,xm,ym):
    return [ {(x*xm, y*ym) for x,y in pset} for pset in paths ]
def genPathLists():
    pathlists={}
    for x in range(7):
        for y in range(7):
            if x*x+y*y <= 49:
                pathlists[ (x,y) ] = genPathList(x,y)
                pathlists[ (-x,y) ] = multpath( pathlists[ (x,y) ], -1, 1 )
                pathlists[ (x,-y) ] = multpath( pathlists[ (x,y) ], 1, -1 )
                pathlists[ (-x,-y) ] = multpath( pathlists[ (x,y) ], -1, -1 )
    return pathlists

def getPointSetList(pathlists):
    pointsetmap = []
    for (x1,y1),pathlist in pathlists.items():
        for i,pointset in enumerate(pathlist):
            for x,y in pointset:
                pointsetmap.append( (x,y,i+1,(x1+7)+(y1+7)*15) )
    pointsetmapdf = pd.DataFrame(pointsetmap)
    pointsetmapdf.columns = 'x,y,step,ind'.split(',')
    pointsetmaplist = pointsetmapdf.groupby('x y step'.split()).apply(lambda x: x.ind.values)
    return pointsetmaplist

if __name__ == '__main__':
    # GAME START
    # Here we define the bot's name and initialize the game, including communication with the Halite engine.
    game = hlt.Game("Gradient")
    while True:
        doOneTurn()

# GAME END
