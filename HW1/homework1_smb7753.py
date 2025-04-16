############################################################
# CMPSC 442: Uninformed Search
############################################################

student_name = "Seokhyeon Bae"

############################################################
# Imports
import math
import random
from collections import deque
############################################################

# Include your imports here, if any are used.



############################################################
# Section 1: N-Queens
############################################################

def num_placements_all(n):
    return math.factorial(n*n)/(math.factorial(n) * math.factorial(n*(n-1)))

def num_placements_one_per_row(n):
    return n**n

def n_queens_valid(board):
    n = len(board)
    for i in range(n):
        for j in range(i + 1, n):
            if board[i] == board[j]:
                return False
            if abs(i - j) == abs(board[i] - board[j]):  # check diagonally valid
                return False
    return True


# Using DFS, yield solutions
def n_queens_helper(n, board=[], row=0):
    """Depth-First Search for N-Queens solutions."""
    if row == n:  
        yield board[:]
        return

    for col in range(n):
        board.append(col)
        if n_queens_valid(board):  
            yield from n_queens_helper(n, board, row + 1)
        board.pop()


def n_queens_solutions(n):
    """Yields all valid solutions for the N-Queens problem using DFS."""
    yield from n_queens_helper(n)


print(len(list(n_queens_solutions(8))))
############################################################
# Section 2: Lights Out
############################################################

class LightsOutPuzzle(object):

    def __init__(self, board):
        self.board = board
        self.rows = len(board)
        if self.rows>0:
            self.cols = len(board[0])

    def get_board(self):
        return self.board

    def perform_move(self, row, col):
        """Toggle the boolean value at (row, col) and its adjacent positions (up, down, left, right)."""
        if 0 <= row < self.rows and 0 <= col < self.cols: 
            self.board[row][col] = not self.board[row][col]  

        if row > 0:
            self.board[row - 1][col] = not self.board[row - 1][col]

        if row < self.rows - 1:
            self.board[row + 1][col] = not self.board[row + 1][col]

        if col > 0:
            self.board[row][col - 1] = not self.board[row][col - 1]

        if col < self.cols - 1:
            self.board[row][col + 1] = not self.board[row][col + 1]


    def scramble(self):
        for i in range(self.rows):
            for j in range(self.cols):
                if (random.random() < 0.5):
                    self.perform_move(i, j)


    def is_solved(self):
        for row in self.board:
            for col in row:
                if col == True:
                    return False
        return True

    def copy(self):
        p2 = create_puzzle(self.rows, self.cols)
        for row in range(self.rows):
            for col in range(self.cols):
                if self.board[row][col] != p2.board[row][col]:
                    p2.board[row][col] = not p2.board[row][col]
        return p2


    def successors(self):
        for i in range(self.rows):
            for j in range(self.cols):
                new_board = self.copy()
                new_board.perform_move(i,j)
                yield (i,j), new_board


    def find_solution(self, return_list=[]):
        initial_state = tuple(map(tuple, self.board))
        if self.is_solved():
            return []
        queue = deque([(self, [])])
        visited = set()
        visited.add(initial_state)         

        while queue:  
            current, moves = queue.popleft()
            for move, new in current.successors():
                new_state = tuple(map(tuple, new.board))
                if new_state in visited:
                    continue
                
                if new.is_solved():
                    return moves + [move]
                
                queue.append((new, moves+[move]))
                visited.add(new_state)
            
        return None

def create_puzzle(rows, cols):
    return LightsOutPuzzle([[False for i in range(cols)] for j in range(rows)])

############################################################
# Section 3: Linear Disk Movement
############################################################

def solve_identical_disks(length, n):
    initial_state = tuple([1]*n + [0]*(length-n)) 
    target_state = tuple([0]*(length-n)+[1]*n)  
    queue = deque([(initial_state, [])])  
    visited = set()
    visited.add(initial_state)
    while queue:
        current_state, moves = queue.popleft()
        if current_state == target_state:
            return moves
        current_list = list(current_state)
        for i in range(length):
            if current_list[i] == 1:
                if i + 1 < length and current_list[i + 1] == 0:
                    new_list = current_list[:]
                    new_list[i], new_list[i + 1] = 0, 1 
                    new_state = tuple(new_list)
                    if new_state not in visited:
                        queue.append((new_state, moves + [(i, i + 1)]))
                        visited.add(new_state)
                if i + 2 < length and current_list[i + 1] == 1 and current_list[i + 2] == 0:
                    new_list = current_list[:]
                    new_list[i], new_list[i + 2] = 0, 1 
                    new_state = tuple(new_list)
                    if new_state not in visited:
                        queue.append((new_state, moves + [(i, i + 2)]))
                        visited.add(new_state)
    return []

def solve_distinct_disks(length, n):
# Start state with distinct disks numbered from 0 to n-1 at the beginning
    initial_state = tuple(i for i in range(n)) + tuple(-1 for _ in range(length-n))
    goal_state = tuple(-1 for _ in range(length-n)) + tuple(i for i in range(n-1, -1, -1))  
    def get_moves(state):
        moves = []
        for i, disk in enumerate(state):
            if disk == -1:
                continue
            if i+1 < length and state[i+1] == -1:
                moves.append((i, i+1))
            if i+2 < length and state[i+2] == -1 and state[i+1] != -1:
                moves.append((i, i+2))
            if i-1 >= 0 and state[i-1] == -1:
                moves.append((i, i-1))
            if i-2 >= 0 and state[i-2] == -1 and state[i-1] != -1:
                moves.append((i, i-2))
        return moves
    def apply_move(state, move):
        new_state = list(state)
        disk = new_state[move[0]]
        new_state[move[0]] = -1  
        new_state[move[1]] = disk  
        return tuple(new_state)
    queue = deque([(initial_state, [])]) 
    visited = set([initial_state])
    while queue:
        current_state, current_moves = queue.popleft()
        if current_state == goal_state:
            return current_moves
        for move in get_moves(current_state):
            new_state = apply_move(current_state, move)
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, current_moves+[move]))
