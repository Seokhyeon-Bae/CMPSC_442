############################################################
# CMPSC 442: Informed Search
############################################################

student_name = "Seokhyeon Bae"

############################################################
# Imports
############################################################

# Include your imports here, if any are used.
import random
from collections import deque
from queue import PriorityQueue
import copy
import math

############################################################
# Section 1: Tile Puzzle
############################################################
def create_tile_puzzle(rows, cols):
    board = [[(r * cols + c + 1) for c in range(cols)] for r in range(rows)]
    board[-1][-1] = 0
    return TilePuzzle(board)


class TilePuzzle(object):
    # Required
    def __init__(self, board):
        self.board = board
        self.rows = len(board)
        if self.rows>0:
            self.cols = len(board[0])
        for row in range(self.rows):
           for col in range(self.cols):
               ## locates the empty tile
               if board[row][col] == 0:
                   self.empty = (row, col)
                   return

    def get_board(self):
        return self.board

    def perform_move(self, direction):
        for i in range(self.rows):
            for j in range(self.cols):
                if self.board[i][j] == 0:
                    if i+1 < self.rows and direction == "down":
                        changing_num = self.board[i+1][j]
                        self.board[i+1][j] = 0
                        self.board[i][j] = changing_num
                        return True
                    if i != 0 and direction == "up":
                        changing_num = self.board[i-1][j]
                        self.board[i-1][j] = 0
                        self.board[i][j] = changing_num
                        return True                                   
                    if j+1 < self.cols and direction == "right":
                        changing_num = self.board[i][j+1]
                        self.board[i][j+1] = 0
                        self.board[i][j] = changing_num
                        return True
                    if j != 0 and direction == "left":
                        changing_num = self.board[i][j-1]
                        self.board[i][j-1] = 0
                        self.board[i][j] = changing_num
                        return True
        return False
                        
    
    def scramble(self, num_moves):
        directions = ["up", "down", "left", "right"]
        for i in range(num_moves):
            self.perform_move(seq=random.choice(directions))

    def is_solved(self):
        correct = [[(r * self.cols + c + 1) % (self.rows * self.cols) for c in range(self.cols)] for r in range(self.rows)]
        return self.board == correct

    def copy(self):
        return TilePuzzle(copy.deepcopy(self.board))

    def successors(self):
        movement = ["down", "up", "right", "left"]
        for moves in movement:
            temp_board = self.copy()
            if temp_board.perform_move(moves):
                yield moves, temp_board

    # Required
    def iddfs_helper(self, limit, moves, visited):
        if self.is_solved():
            yield moves
            return

        if limit == 0:
            return

        state = tuple(map(tuple, self.board))
        visited.add(state)

        for direction, next_puzzle in self.successors():
            next_state = tuple(map(tuple, next_puzzle.board))
            if next_state not in visited:
                yield from next_puzzle.iddfs_helper(limit - 1, moves + [direction], visited)

        visited.remove(state)

    def find_solutions_iddfs(self):
        """Find all optimal solutions using iterative deepening depth-first search."""
        depth = 0
        solutions_found = False

        while not solutions_found:
            visited = set()
            solutions = list(self.iddfs_helper(depth, [], visited))

            if solutions:
                solutions_found = True
                for sol in solutions:
                    yield sol

            depth += 1

    # Required
    def heuristic(self):
        """Calculate the Manhattan distance heuristic."""
        distance = 0
        for r in range(self.rows):
            for c in range(self.cols):
                val = self.board[r][c]
                if val != 0:  # ignore empty tile
                    goal_r, goal_c = divmod(val, self.cols)
                    distance += abs(r - goal_r) + abs(c - goal_c)
        return distance

    def find_solution_a_star(self):    

        initial_state = tuple(map(tuple, self.board))
        visited = set()

        # Priority queue for A* search
        pq = PriorityQueue()
        pq.put((self.heuristic(), 0, initial_state, [], self))  # (f, g, state, path, puzzle)

        while not pq.empty():
            f, g, state, path, puzzle = pq.get()

            # Check if puzzle is solved
            if puzzle.is_solved():
                return path

            # Mark state as visited after extracting from queue
            if state in visited:
                continue
            visited.add(state)

            # Explore successors
            for direction, next_puzzle in puzzle.successors():
                next_state = tuple(map(tuple, next_puzzle.board))
                if next_state not in visited:
                    new_g = g + 1  # Cost increases by 1 for each move
                    h = next_puzzle.heuristic()
                    new_f = new_g + h  # f = g + h
                    pq.put((new_f, new_g, next_state, path + [direction], next_puzzle))

        return None

############################################################
# Section 2: Grid Navigation
############################################################
# helper function
def path_successors(current_position, scene):
    """Generate valid successors (neighbors) for the current position."""
    vertical_limit = len(scene) - 1
    horizontal_limit = len(scene[0]) - 1
    row, col = current_position

    # All 8 possible directions (up, down, left, right, diagonals)
    directions = [
        (-1, 0),  # up
        (1, 0),   # down
        (0, -1),  # left
        (0, 1),   # right
        (-1, -1), # up-left
        (-1, 1),  # up-right
        (1, -1),  # down-left
        (1, 1)    # down-right
    ]

    for dr, dc in directions:
        nr, nc = row + dr, col + dc
        # Check boundaries and obstacles
        if 0 <= nr <= vertical_limit and 0 <= nc <= horizontal_limit and not scene[nr][nc]:
            yield (nr, nc)

# euclidean_distance function
def euclidean_distance(a, b):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

# reconstruct path to find the cheapest path
def reconstruct_path(came_from, current):
    path = []
    while current is not None:
        path.append(current)
        current = came_from.get(current)
    return path[::-1] 

def find_path(start, goal, scene):
    """Find the shortest path from start to goal using A* search."""
    # Check for invalid inputs
    if scene[start[0]][start[1]] or scene[goal[0]][goal[1]]:
        return None

    # Initialize priority queue
    pq = PriorityQueue()
    pq.put((0, start))
    came_from = {start: None}
    g_score = {start: 0}
    visited = set()

    while not pq.empty():
        _, current = pq.get()

        # If goal reached, reconstruct the path
        if current == goal:
            return reconstruct_path(came_from, current)

        if current in visited:
            continue

        visited.add(current)

        # Explore neighbors
        for neighbor in path_successors(current, scene):
            tentative_g_score = g_score[current] + euclidean_distance(current, neighbor)

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + euclidean_distance(neighbor, goal)
                pq.put((f_score, neighbor))
                came_from[neighbor] = current

    return None


############################################################
# Section 3: Linear Disk Movement, Revisited
############################################################



def solve_distinct_disks(length, n):
    # Initial state: first n cells filled with disks 0 to n-1; rest empty (-1)
    initial_state = tuple(i for i in range(n)) + tuple(-1 for _ in range(length - n))
    
    # Goal state: last n cells filled with disks in reverse order
    goal_state = tuple(-1 for _ in range(length - n)) + tuple(i for i in range(n - 1, -1, -1))

    # Helper to get possible moves
    def get_moves(state):
        moves = []
        for i, disk in enumerate(state):
            if disk == -1:
                continue

            # Move 1 step forward
            if i + 1 < length and state[i + 1] == -1:
                moves.append((i, i + 1))

            # Move 2 steps forward (jump) if intermediate cell is occupied
            if i + 2 < length and state[i + 2] == -1 and state[i + 1] != -1:
                moves.append((i, i + 2))

            # Move 1 step backward
            if i - 1 >= 0 and state[i - 1] == -1:
                moves.append((i, i - 1))

            # Move 2 steps backward (jump) if intermediate cell is occupied
            if i - 2 >= 0 and state[i - 2] == -1 and state[i - 1] != -1:
                moves.append((i, i - 2))

        return moves

    # Apply a move to generate a new state
    def apply_move(state, move):
        new_state = list(state)
        disk = new_state[move[0]]
        new_state[move[0]] = -1
        new_state[move[1]] = disk
        return tuple(new_state)

    # Heuristic: sum of distances of each disk from its target position
    def heuristic(state):
        distance = 0
        for i, disk in enumerate(state):
            if disk != -1:
                # Calculate the target position for this disk
                target_pos = length - 1 - disk
                distance += abs(i - target_pos)
        return distance

    # A* search
    pq = PriorityQueue()
    pq.put((heuristic(initial_state), 0, initial_state, []))
    visited = set([initial_state])

    while not pq.empty():
        f, g, current_state, current_moves = pq.get()

        # Check if goal is reached
        if current_state == goal_state:
            return current_moves

        # Generate valid moves
        for move in get_moves(current_state):
            new_state = apply_move(current_state, move)
            if new_state not in visited:
                visited.add(new_state)
                cost = g + 1  # cost is the number of moves so far
                priority = cost + heuristic(new_state)
                pq.put((priority, cost, new_state, current_moves + [move]))

    return None



############################################################
# Section 4: Dominoes Game
############################################################

def create_dominoes_game(rows, cols):
    return DominoesGame([[False for i in range(cols)] for j in range(rows)])

class DominoesGame(object):

    # Required
    def __init__(self, board):
        self.board = board
        self.rows = len(self.board)
        self.cols = len(self.board[0])

    def get_board(self):
        return self.board

    def reset(self):
        self.board = create_dominoes_game(self.rows, self.cols).get_board()

    def is_legal_move(self, row, col, vertical):
        if vertical:
            if row < 0 or row >= self.rows - 1 or col < 0 or col >= self.cols:
                return False
            return not self.board[row][col] and not self.board[row + 1][col]
        else:
            if row < 0 or row >= self.rows or col < 0 or col >= self.cols - 1:
                return False
            return not self.board[row][col] and not self.board[row][col + 1]

    def legal_moves(self, vertical):
        for i in range(self.rows):
            for j in range(self.cols):
                if self.is_legal_move(i, j, vertical):
                    yield (i, j)

    def perform_move(self, row, col, vertical):
        if self.is_legal_move(row, col, vertical):
            if vertical:
                self.board[row][col] = True
                self.board[row+1][col] = True
            else:
                self.board[row][col] = True
                self.board[row][col+1] = True

    def game_over(self, vertical):
        return not any(self.legal_moves(vertical))

    def copy(self):
        return DominoesGame(copy.deepcopy(self.board))

    def successors(self, vertical):
        for moves in list(self.legal_moves(vertical)):
            new_d = self.copy()
            new_d.perform_move(moves[0], moves[1], vertical)
            yield moves, new_d

    def get_random_move(self, vertical):
        movements = self.legal_moves(vertical)
        random_move = random.choice(list(movements))
        row, col = random_move
        self.perform_move(row, col, vertical)

    def evaluate(self, vertical):
        current_moves = len(list(self.legal_moves(vertical)))
        opponent_moves = len(list(self.legal_moves(not vertical)))
        if vertical: 
            return current_moves - opponent_moves
        else:
            return opponent_moves - current_moves
        
        
    def alpha_beta(self, vertical, limit, min_val, max_val,):
                if limit == 0 or self.game_over(vertical):
                    return self.evaluate(vertical), None, 1
                best_move = float("-inf") if vertical else float ("inf")
                best_action = None
                leaf_nodes = 0
                for move, new_game in self.successors(vertical):
                    value, _, leaves = new_game.alpha_beta(not vertical, limit - 1, min_val, max_val)
                    leaf_nodes += leaves
                    if vertical:
                        if value > best_move:
                            best_move, best_action = value, move
                        min_val = max(min_val, value)
                    else:
                        if value < best_move:
                            best_move, best_action = value, move
                        max_val = min(max_val, value) 
                    if max_val <= min_val:
                        break
                return best_move,best_action, leaf_nodes

    # Required
    def get_best_move(self, vertical, limit):
        best_move, value, leaf_nodes = self.alpha_beta(vertical, limit, float("-inf"), float("inf"))
        return value, best_move if vertical else -best_move, leaf_nodes

d = create_dominoes_game(3, 3)
best_move, value, leaf_nodes = d.get_best_move(True, 2)
print(f"Best Move: {best_move}, Value: {value}, Leaf Nodes Visited: {leaf_nodes}")