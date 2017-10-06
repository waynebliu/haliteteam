"""
Note: Please do not place print statements here as they are used to communicate with the Halite engine. If you need
to log anything use the logging module.
"""
# Let's start by importing the Halite Starter Kit so we can interface with the Halite engine
import hlt
import numpy as np

# constants
MyPlanetGradientRadius = 10
MyPlanetGradientFactor = 20
UnownedPlanetGradientRadius = 200
UnownedPlanetGradientStrength = 10
EnemyPlanetGradientRadius = 200
EnemyPlanetGradientStrength = 1


# global just to avoid memory allocations...
Xship = np.outer( range(-7,8), np.ones(15) )
Yship = np.outer( np.ones(15), range(-7,8) )
movemap = np.zeros(15,15)
Xmap = None
Ymap = None
pathlists = None
pointsetmaplist = None
planetmasks = {}

# GAME START
# Here we define the bot's name and initialize the game, including communication with the Halite engine.
game = hlt.Game("Gradient")

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
    if not Xmap:
        Xmap = np.outer( range(game_map.width), np.ones(game_map.height) )
        Ymap = np.outer( np.ones(game_map.width), range(game_map.height) )
        gradientfield = np.zeros([game_map.width,game_map.height])
        pathlists = genPathLists()
        pointsetmaplist = getPointSetList(pathlists)
        for radius in range(3,7):
            planetmasks[radius] = movemap

class Analysis:
    def __init__(self,game_map):
        self.me = game_map.get_me()
        self.applyPlanetFields(game_map)
        # reinitialize
        gradientfield.fill(0)
        self.shipX,self.shipY = getEntityXYs(self.me.all_ships())
        self.closeShips = computeEntitiesDistances(
                self.shipX, self.shipY,
                self.shipX, self.shipY)

        self.planetX,self.planetY = getEntityXYs(myships,game_map.all_planets())
        self.planetDist = computeEntitiesDistances(self.shipX,self.shipY,
                self.planetX, self.planetY)
        planetRadius = np.array(p.radius for p in game_map.all_planets())
        planetCloseness = np.outer( np.ones( len(self.me.all_ships() ),
                                    planetRadius) ) + 7
        self.closePlanets = self.planetDist < planetCloseness

        enemyships = []
        for player in game_map.all_players():
            enemyships += player.all_ships()
        self.enemyshipX,self.enemyshipY = getEntityXYs(enemyships)
        self.enemyshipDist = computeEntitiesDistances(self.shipX,self.shipY,
                self.enemyshipX, self.enemyshipY)
        self.closeEnemyShips = self.enemyshipDist < 15

    def applyPlanetFields(self,game_map):
        for planet in game_map.all_planets():
            if planet.owner is None:
                applyPlanetField(planet.x, planet.y,
                        gradientfield,
                        UnownedPlanetGradientStrength,
                        planet.num_docking_spots+UnownedPlanetGradientRadius)
            elif planet.owner == self.me:
                strength =  planet.num_docking_spots - len(planet._docked_ship_ids)
                if strength > 0:
                    applyPlanetField(planet.x, planet.y,
                        gradientfield,
                        strength*MyPlanetGradientFactor,
                        planet.num_docking_spots+MyPlanetGradientRadius)
            else:
                applyPlanetField(planet.x, planet.y,
                    gradientfield,
                    EnemyPlanetGradientStrength,
                    planet.num_docking_spots+EnemyPlanetGradientRadius)


def applyPlanetField(x, y, gradientfield, strength, radius):
    gradientfield += ( ( radius - np.sqrt( 
              np.multiply( Xmap - x, Xmap - x) + np.multiply( Ymap - y, Ymap - y)
            ) ) * strength ).clip(min=0)

def getEntityXYs(ships):
    return np.array( s.x for s in myships ), np.array( s.y for s in myships )

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
            planet = game_map.all_planets()[p]
            if ship.can_dock(planet):
                dock=True
                break
        if dock:
            turnstate.addDock(ship,planet)
            continue

        # move to somewhere...
        maneuverShip(ship, turnstate, analysis, game_map)

    return turnstate.getCommands()

class TurnState:
    def __init__(self):
        self.command_queue = []
        self.shippaths={}
    def getCommands(self):
        return self.command_queue
    def addDock(self,ship,planet):
        self.command_queue.append(ship.dock(planet))

def maneuverShip(ship, i, turnstate, analysis, game_map):
    # blot out any square that's blocked
    movemap.fill(0)

    # planets
    for p in np.argwhere(analysis.closePlanets[i]):
        path = [ (planet.x, planet.y, step) for step in range(7)]
        blotout(movemap, analysis, path, ship.x, ship.y)
    # we don't care a whit about ramming enemy ships!

    # friendly ships
    for p in np.argwhere(analysis.closeShips[i]):
        if turnstate.shippaths.get(i):
            blotout(movemap, turnstate.shippaths.get(i))

            

def blotout(movemap,path,shipx,shipy):
    xinds = []
    yinds = []
    for pstep in path:
        xp,yp += pointsetmaplist.get( pstep, ((),()) )
        xinds.append(xp-shipx)
        yinds.append(yp-shipy)
    movemap[ np.concatenate(xinds), np.concatenate(yinds) ] = -1


def genPathList(x,y):
    ' assuming distance(x,y) <= moveMax (7) '
    path = []
    for i in range(1,8):
        xloc,yloc = x*i/7,y*i/7
        path.append( 
            set( ( int(xloc-xoff),int(yloc-xoff) ) for xoff in [-.5,0,.5] for yoff in [-.5,0,.5]  )
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
    for (x1,y1),pathlist in pathLists.items():
        for i,pointset in enumerate(pathlist):
            for x,y in pointset:
                pointsetmap.append( (x,y,i+1,x1,y1) )
                import pandas as pd
    pointsetmapdf = pd.DataFrame(pointsetmap)
    pointsetmapdf.columns = 'x,y,step,x1,y1'.split(',')
    pointsetmaplist = pointsetmapdf.groupby('x y step'.split()).apply(lambda x: (x.x1.values+7,x.y1.values+7) )
    return pointsetmaplist


if __name__ == '__main__':
    while True:
        doOneTurn()

# GAME END
