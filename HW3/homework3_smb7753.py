############################################################
# CMPSC 442: Homework 3
############################################################

student_name = "Seokhyeon Bae"

############################################################
# Imports
############################################################

# Include your imports here, if any are used.
from collections import deque
import copy
############################################################
# Section 1: Sudoku
############################################################

def sudoku_cells():
    return [(i, j) for i in range(9) for j in range(9)]

def sudoku_arcs():
    cells = sudoku_cells()
    arcs = set()
    for cell in cells:
        r, c = cell
        for cj in range(9):
            if cj != c:
                arcs.add((cell, (r, cj)))
        for ri in range(9):
            if ri != r:
                arcs.add((cell, (ri, c)))
        block_r = (r // 3) * 3
        block_c = (c // 3) * 3
        for dr in range(3):
            for dc in range(3):
                neighbor = (block_r + dr, block_c + dc)
                if neighbor != cell:
                    arcs.add((cell, neighbor))
    return list(arcs)


def read_board(path):
    board = {}
    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    for i, line in enumerate(lines[:9]):
        for j, ch in enumerate(line[:9]):
            if ch == '*':
                board[(i, j)] = list(range(1, 10))
            else:
                board[(i, j)] = [int(ch)]
    for i in range(9):
        for j in range(9):
            board.setdefault((i, j), list(range(1, 10)))
    return Sudoku(board)

class Sudoku(object):

    CELLS = sudoku_cells()
    ARCS = sudoku_arcs()

    def __init__(self, board):
        self.board = board
        
    def __getitem__(self, cell):
        return self.board[cell]
    
    def __setitem__(self, cell, value):
        """Allow direct assignment to Sudoku cells, ensuring values are stored as lists."""
        if isinstance(value, list):
            self.board[cell] = value
        elif isinstance(value, int):
            self.board[cell] = [value]
        elif isinstance(value, set):
            self.board[cell] = list(value)
            
    def get_values(self, cell):
        return set(self.board[cell])


    
    def remove_inconsistent_values(self, cell1, cell2):
        if len(self.board[cell2]) == 1:
            value = self.board[cell2][0]
            if value in self.board[cell1]:
                self.board[cell1].remove(value)
                return True
        return False

    def get_neighbors(self, cell):
        neighbors = set()
        (row, col) = cell
        for i in range(9):
            if i != col:
                neighbors.add((row, i))
            if i != row:
                neighbors.add((i, col))
        box_i, box_j = (row // 3) * 3, (col // 3) * 3
        for m in range(3):
            for n in range(3):
                neighbor = (box_i + m, box_j + n)
                if neighbor != cell:
                    neighbors.add(neighbor)
        return neighbors

    def infer_ac3(self):
        queue = deque(self.ARCS)
        while queue:
            (cell1, cell2) = queue.popleft()
            if self.remove_inconsistent_values(cell1, cell2):
                if len(self.board[cell1]) == 0:
                    return False
                for neighbor in self.get_neighbors(cell1):
                    if neighbor != cell2:
                        queue.append((neighbor, cell1))
        return True

    
    def infer_improved(self):
        if not self.infer_ac3():
            return False
        progress = True
        while progress:
            progress = False
            for num in range(1, 10):
                
                for row in range(9):
                    cells_with_num = [cell for cell in self.CELLS if cell[0] == row and num in self.board[cell]]
                    if len(cells_with_num) == 1:
                        cell = cells_with_num[0]
                        if len(self.board[cell]) != 1 or self.board[cell][0] != num:
                            self.board[cell] = [num]
                            progress = True
                            
                for col in range(9):
                    cells_with_num = [cell for cell in self.CELLS if cell[1] == col and num in self.board[cell]]
                    if len(cells_with_num) == 1:
                        cell = cells_with_num[0]
                        if len(self.board[cell]) != 1 or self.board[cell][0] != num:
                            self.board[cell] = [num]
                            progress = True

                for box_i in range(3):
                    for box_j in range(3):
                        cells_with_num = []
                        for i in range(3):
                            for j in range(3):
                                cell = (box_i * 3 + i, box_j * 3 + j)
                                if num in self.board[cell]:
                                    cells_with_num.append(cell)
                        if len(cells_with_num) == 1:
                            cell = cells_with_num[0]
                            if len(self.board[cell]) != 1 or self.board[cell][0] != num:
                                self.board[cell] = [num]
                                progress = True
            if progress:
                if not self.infer_ac3():
                    return False
        return True

    def infer_with_guessing(self):
        self.infer_improved()  
        if all(len(self.board[cell]) == 1 for cell in Sudoku.CELLS):
            return True
        for cell in Sudoku.CELLS:
            if len(self.board[cell]) > 1:
                guess_cell = cell
                break

        backup = copy.deepcopy(self.board)

        for value in list(self.board[guess_cell]):
            self.board[guess_cell] = {value}
            if self.infer_with_guessing():
                return True 
            self.board = copy.deepcopy(backup)
        return False
    
