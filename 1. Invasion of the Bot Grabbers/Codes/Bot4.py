# Importing the necessary libraries
import numpy as np
import pandas as pd
import random
import heapq
import math

# Creating the dataframe
df = pd.DataFrame(columns=['Bot', 'NumAliens', 'CrewSaved', 'TimeAlive', 'IsAlive'])

# Declaring the global variables
D = 50 # Dimension of the ship
ship = np.ones(()) # Layout of the ship
alien_cells = [] # Position of the aliens in the cells
bot_cell = (0, 0) # Current position of the bot cell
crew_cell = (0, 0) # Current position of the crew cell

# Hypothetical variables for the simulated alien movements

hypo_ship = np.ones(()) # Hypothetical ship layout to simulate aliens
hypo_alien_cells = [] # Stores the alien positions for the simulated movement
alien_history_path = [] # Stores the cells where the alien passes through
alien_neighbors = [] # Stores the valid neighbors of the alien cells
heur = {} # Dictionary used to store the frequency of the cells visited by the aliens in the simulated moves
num_alien_steps = 0 # Number of simulated alien steps to be taken
num_alien_sims = 0 # Number of simulations of movements

# Function to display all the basic details
def display():
    print(ship)
    print("Bot at: ", bot_cell)
    print("Crew at: ", crew_cell)
    print("Alien cells: ", alien_cells)


# Checks if the passed cell co-ordinates lie in the range of the ship dimensions
def in_range(x, y):
    return (0 <= x < D) and (0 <= y < D)


# Generates the layout of the ship
def generate_ship_layout():
    global ship
    # Initialize the grid with all cells as blocked (1 -> blocked, 0 -> open)
    ship = np.ones((D, D))

    # Actions to reach neighbors of cell as the adjacent cells in the up/down/right/left direction
    actions = [(-1, 0), (1, 0), (0, 1), (0, -1)]

    # Choose a random cell in the interior to open
    x = random.randint(0, D - 1)
    y = random.randint(0, D - 1)
    ship[x][y] = 0

    # Iteratively open cells with exactly one open neighbor
    while True:
        validBlockedCells = []
        for i in range(D):
            for j in range(D):
                if ship[i][j] == 1 and sum(
                        1 for dx, dy in actions if in_range(i + dx, j + dy) and ship[i + dx][j + dy] == 0) == 1:
                    validBlockedCells.append((i, j))

        if len(validBlockedCells) == 0:
            break

        cell = random.choice(validBlockedCells)
        ship[cell[0]][cell[1]] = 0

    # Identify and open approximately half the dead-end cells
    dead_ends = [(i, j) for i in range(D) for j in range(D) if ship[i][j] == 0 and sum(
        1 for dx, dy in actions if in_range(i + dx, j + dy) and ship[i + dx][j + dy] == 0) == 1]
    random.shuffle(dead_ends)
    for i in range(len(dead_ends) // 2):
        x, y = dead_ends[i]
        neighbors = [(x + dx, y + dy) for dx, dy in actions if in_range(x + dx, y + dy) and ship[x + dx][y + dy] == 1]
        if neighbors:
            nx, ny = random.choice(neighbors)
            ship[nx][ny] = 0


# Function to find a random empty cell on the grid
def find_empty_cell():
    global ship
    while True:
        x = random.randint(0, len(ship) - 1)
        y = random.randint(0, len(ship[0]) - 1)
        if ship[x][y] in [0,9]:
            return (x, y)

def update_alien_neighbors():
    global alien_neighbors
    nlist = []
    actions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    for alien in alien_cells:
        temp = [(alien[0] + dx, alien[1] + dy) for dx, dy in actions]
        for ele in temp:
            nlist.append(ele)

    alien_neighbors.clear()
    alien_neighbors = nlist

# Function to set up the bot
def setup_bot():
    global ship
    global bot_cell
    # Find a random empty cell for the bot
    bot_cell = find_empty_cell()
    ship[bot_cell] = 3


# Function to set up the aliens
def setup_aliens(aliens):
    global ship
    global alien_cells
    # Find a random empty cell for each alien
    num_aliens = aliens
    alien_cells = []
    for x in range(num_aliens):
        temp = find_empty_cell()
        alien_cells.append(temp)
        ship[temp] = 9


# Function to set up the crew member
def setup_crew():
    global ship
    global crew_cell
    # Find a random empty cell for the crew member
    crew_cell = find_empty_cell()
    # ship[crew_cell] = 6

# Function to generate ship, bot, crew and aliens in a single call
def generate_ship_bot_aliens(no_of_aliens):
    # Generate your ship layout
    generate_ship_layout()

    # Set up bot, aliens, and crew
    setup_bot()
    setup_crew()
    setup_aliens(no_of_aliens)
    update_alien_neighbors()

    #display()

# Function to update the position of the bot after it's movement
def update_bot_position(old_bot_cell, new_bot_cell):
    global ship
    ship[old_bot_cell] = 0
    ship[new_bot_cell] = 3


# Updates the alien positions after the movement
# Key variable for differentiating between actual and simulated alien movements
# key=0 --> actual movement; key=1 --> simulated movement
def update_alien_position(old_alien_cell, new_alien_cell, key=0):
    global ship
    global hypo_ship

    if key==0:
        ship[old_alien_cell] = 0
        ship[new_alien_cell] = 9
    else:
        hypo_ship[old_alien_cell] = 0
        hypo_ship[new_alien_cell] = 9

# Function to spawn new crew after one has been saved
def generate_new_crew():
    global ship
    global crew_cell
    ship[crew_cell] = 0
    new_crew_cell = (0, 0)
    while True:
        # Find a random empty cell for the crew member
        new_crew_cell = find_empty_cell()
        if new_crew_cell != crew_cell and ship[new_crew_cell] != 3:
            break
    crew_cell = new_crew_cell
    ship[crew_cell] = 6


# Function to get valid neighbor cells
def get_valid_neighbors(x, y, key=0):
    actions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    if key==0:
        neighbors = [(x + dx, y + dy) for dx, dy in actions if in_range(x + dx, y + dy) and ship[x + dx][y + dy] != 1]
    else:
        neighbors = [(x + dx, y + dy) for dx, dy in actions if in_range(x + dx, y + dy) and hypo_ship[x + dx][y + dy] != 1]
    # Randomize the order of neighbors
    random.shuffle(neighbors)
    return neighbors


# Function to generate alien movements
def generate_alien_movements(key=0):
    global alien_cells
    global hypo_alien_cells
    movements = []
    alien_list = hypo_alien_cells
    if key==0:
        alien_list = alien_cells
    # Shuffle the order of aliens
    random.shuffle(alien_list)
    new_alien_cells = []

    for alien_cell in alien_list:
        x, y = alien_cell
        valid_neighbors = get_valid_neighbors(x, y, key)

        # Choose a random valid neighbor cell or stay in place
        if valid_neighbors:
            new_x, new_y = random.choice(valid_neighbors)
            movements.append((x, y, new_x, new_y))
            update_alien_position(alien_cell, (new_x, new_y), key)
            new_alien_cells.append((new_x, new_y))
        else:
            # Stay in place
            movements.append((x, y, x, y))
            new_alien_cells.append((x, y))

    if key==0:
        alien_cells.clear()
        alien_cells = new_alien_cells
    if key==1:
        hypo_alien_cells.clear()
        hypo_alien_cells = new_alien_cells


    return movements


# Function to calculate manhattan distance between the passed co-ordinates
def manhattan_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return abs(x1 - x2) + abs(y1 - y2)

# Simulates alien movements
def hypo_alien_move():
    global hypo_ship
    global hypo_alien_cells
    global alien_history_path
    hypo_alien_cells = alien_cells.copy()
    hypo_ship = ship.copy()

    for i in range(num_alien_steps):
        temp = generate_alien_movements(1)
        alien_history_path += hypo_alien_cells

# Function for implementation of the A* algorithm
def a_star(key=0):
    global heur
    global bot_cell
    global crew_cell
    global ship

    # Checks validity of cell based on the alien positions and its neighbors
    def is_valid(x, y):
        return 0 <= x < len(ship) and 0 <= y < len(ship[0]) and ship[x][y] == 0 and (x,y) not in alien_neighbors

    # Checks validity of cell based only on the alien positions
    def is_valid2(x, y):
        return 0 <= x < len(ship) and 0 <= y < len(ship[0]) and ship[x][y] == 0

    # Reconstructs the path of the bot till the crew cell
    def reconstruct_path(came_from, current):
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        return path

    came_from = {}
    # Initialization of g score and f score
    g_score = {(i, j): float('inf') for i in range(D) for j in range(D)} # Initialize every cell with infinity
    g_score[bot_cell] = 0 # Zero g score because the bot is already present in that cell
    f_score = {(i, j): float('inf') for i in range(D) for j in range(D)} # Initialize every cell with infinity
    f_score[bot_cell] = manhattan_distance(bot_cell, crew_cell) # Manhattan distance is the selected heuristic measure

    # Open set stores all the possible cell options
    open_set = [(manhattan_distance(bot_cell, crew_cell), manhattan_distance(bot_cell, crew_cell), bot_cell)]

    while open_set:
        _, _, current = heapq.heappop(open_set)
        if current == crew_cell:
            return reconstruct_path(came_from, current)

        # Iterate over all the valid neighbors
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            neighbor = (current[0] + dx, current[1] + dy)

            if key == 0:
                booltemp = is_valid(*neighbor)
            else:
                booltemp = is_valid2(*neighbor)

            if booltemp:
                # Increase g score by 1 because of the step taken to neighbor
                tentative_g_score = g_score[current] + 1
                tentative_f_score = tentative_g_score + manhattan_distance(neighbor, crew_cell)

                # Adding the frequency to the heuristics
                if neighbor in heur:
                    tentative_f_score += heur[neighbor]

                # Update the f score if a lower f score is discovered
                if tentative_f_score < f_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_f_score
                    heapq.heappush(open_set, (f_score[neighbor], manhattan_distance(neighbor, crew_cell), neighbor))


    return None

# Implementation of Bot-4
def bot4(num_aliens):
    global ship
    global bot_cell
    global alien_cells
    global crew_cell
    # num_aliens = 3

    generate_ship_bot_aliens(num_aliens) # Generating the ship layout
    time = 0 # Maintain the time stamp
    crew_saved = 0 # Maintain the number of crew saved
    isAlive = True # Maintain the status of the bot

    heur.clear()

    # Simulate alien movements
    for i in range(num_alien_sims):
        hypo_alien_move()

    # Calculate frequency of cells visited by the aliens
    for ele in alien_history_path:
        if ele in heur:
            heur[ele] += 1
        else:
            heur[ele] = 1

    # Multiply the multiplying factor with the frequency
    for key, values in heur.items():
        heur[key] *= mult_fact

    while (isAlive and time < 1000):
        # Calculate the path using A* algorithm
        path = a_star()
        # print(path)


        if path is None:
            path = a_star(1)
            time += 1
            if path is None:
                time += 1


        else:
            update_bot_position(bot_cell, path[-1])
            time += 1
            bot_cell = path[-1]

        generate_alien_movements()
        update_alien_neighbors()
        # Killing condition when bot encounters an alien
        if (ship[bot_cell] == 9):
            print("Bot Dies!")
            isAlive = False
            return (crew_saved, time, isAlive)

        if bot_cell == crew_cell:
            crew_saved += 1
            bot_cell = crew_cell
            ship[bot_cell] = 3
            setup_crew()


    return (crew_saved, time, isAlive)

result = [] # Stores the result statistics
nums = 5 # Declaring the number of aliens
num_alien_steps = 15
num_alien_sims = 50
mult_fact = 0.05
while nums<=50:
    total = 0
    for i in range(100):
        print(i)
        x, y, z = bot4(nums)
        result.append({'Bot': 4, 'NumAliens': nums, 'CrewSaved': x, 'TimeAlive': y, 'IsAlive': z})
        print(x, y, z)
        total += x
        print("**********************************************")
    nums += 5 # Incrementing the number of aliens by 5

    print(total/100)

df = pd.DataFrame(result)
print(df)
print(heur)

# Saving the results
df.to_excel('results-4.xlsx', index=False)