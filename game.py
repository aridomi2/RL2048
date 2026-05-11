import numpy as np
from environment import Environment

class TwntyFrtyEight(Environment):
    WINNING_VALUE = 2048
    # define winning state number as the max state number + 1
    WINNING_STATE = np.log2(WINNING_VALUE)**16
    # map state number to board configuration
    visited_states = {}
    
    @staticmethod
    def play():
        board = TwntyFrtyEight.new_board(row_count=4, col_count=4) # create new game
        score = 0
        while True:
            print(board)
            print(f'{score=}')
            # ask user for input
            try: direction = int(input(f'Direction (0 left, 1 up, 2 right, 3 down): '))
            except ValueError: continue
            if direction not in TwntyFrtyEight.get_valid_actions(TwntyFrtyEight.board_to_state(board)):
                print(f'Invalid direction. Please input a valid direction.')
                continue
            # transition the board to next state
            # done flag is used to check if the direction is a valid move
            board, points = TwntyFrtyEight.move(board, direction)
            score += points
            # add new tile of value 2 to random position on the board
            board = TwntyFrtyEight.add_two(board)
            if TwntyFrtyEight.get_board_status(board) == 'win':
                print('Win')
                break
            if TwntyFrtyEight.get_board_status(board) == 'lose':
                print('Lose')
                break
        
    @staticmethod
    def new_board(row_count, col_count):
        empty_board = np.zeros((row_count, col_count), dtype=int)
        board = TwntyFrtyEight.add_two(TwntyFrtyEight.add_two(empty_board))
        return board
        
    @staticmethod
    def add_two(board: np.ndarray, position: tuple[int, int]=None):
        if position is None:
            while True:
                position = np.unravel_index(np.random.randint(np.prod(board.shape)), board.shape)
                if board[position] == 0:
                    break
        next_board = np.copy(board)
        next_board[position] = 2
        return next_board
    
    @staticmethod
    def get_board_status(board: np.ndarray):
        # check for win cell
        if np.any(board==2048):
            return 'win'
        # check for any zero entries
        if np.any(board==0):
            return 'not over'
        # check for same cells that touch each other
        for i in range(len(board)-1):
            # intentionally reduced to check the row on the right and below
            # more elegant to use exceptions but most likely this will be their solution
            for j in range(len(board[0])-1):
                if board[i][j] == board[i+1][j] or board[i][j+1] == board[i][j]:
                    return 'not over'
        for k in range(len(board)-1):  # to check the left/right entries on the last row
            if board[len(board)-1][k] == board[len(board)-1][k+1]:
                return 'not over'
        for j in range(len(board)-1):  # check up/down entries on last column
            if board[j][len(board)-1] == board[j+1][len(board)-1]:
                return 'not over'
        return 'lose'
    
    @staticmethod
    def get_state_status(state: int):
        board = TwntyFrtyEight.state_to_board(state)
        return np.max(board)
    
    @staticmethod
    def get_valid_actions(state: int):
        board = TwntyFrtyEight.state_to_board(state)
        all_actions = range(4)
        valid_actions = []
        
        for action in all_actions:
            next_board, points = TwntyFrtyEight.move(board, action)
            if np.any(board != next_board):
                valid_actions.append(action)
        
        return valid_actions
    
    @staticmethod
    def compress(board: np.ndarray):
        '''
        compress the board towards the left
        vectorized algorithm from
        https://stackoverflow.com/a/43011036
        '''
        valid_mask = board!=0
        flipped_mask = valid_mask.sum(1,keepdims=1) > np.arange(board.shape[1]-1,-1,-1)
        flipped_mask = flipped_mask[:,::-1]
        board[flipped_mask] = board[valid_mask]
        board[~flipped_mask] = 0
            
        return board
    
    @staticmethod
    def merge(board: np.ndarray):
        points = 0
        for i in range(board.shape[0]):
            for j in range(board.shape[1]-1):
                if board[i][j] != 0 and board[i][j] == board[i][j+1]:
                    board[i][j] *= 2
                    points += board[i][j]
                    board[i][j+1] = 0
        return board, points
    
    @staticmethod
    def move(board: np.ndarray, action: int):
        '''
        helper function for right, up, left, down functions below
        '''
        next_board = np.copy(board)
        # rotate board
        next_board = np.rot90(next_board, action)
        # compress, merge, compress towards left
        next_board = TwntyFrtyEight.compress(next_board)
        next_board, points = TwntyFrtyEight.merge(next_board)
        next_board = TwntyFrtyEight.compress(next_board)
        # rotate back to original
        next_board = np.rot90(next_board, -action)
        
        return next_board, points
    
    @staticmethod
    def left(board: np.ndarray):
        '''
        Shift Board Left
        '''
        return TwntyFrtyEight.move(board, action=0)
    
    @staticmethod
    def up(board: np.ndarray):
        '''
        Shift Board Up
        '''
        return TwntyFrtyEight.move(board, action=1)
    
    @staticmethod
    def right(board: np.ndarray):
        '''
        Shift Board Up
        '''
        return TwntyFrtyEight.move(board, action=2)
    
    @staticmethod
    def down(board: np.ndarray):
        '''
        Shift Board Down
        '''
        return TwntyFrtyEight.move(board, action=3)
    
    @staticmethod
    def get_initial_state():
        return TwntyFrtyEight.board_to_state(TwntyFrtyEight.new_board(4, 4))
    
    @staticmethod
    def state_to_board(state: int):
        '''
        converts state number s denoted by an integer 
        in the range [0, 11^16-1] to a 4x4 board
        '''
        if state in TwntyFrtyEight.visited_states:
            return TwntyFrtyEight.visited_states[state]
        
        base = int(np.log2(TwntyFrtyEight.WINNING_VALUE))
        if not 0 <= state < TwntyFrtyEight.WINNING_STATE: return None
        
        # Convert state number to 16 digit base 11 representation
        # where each digit represents a tile
        digits = [int(digit, base) for digit in np.base_repr(state, base).zfill(16)]
        board = digits
        # Rearrange digits in a 4x4 grid
        board = np.array(board).reshape((4,4))
        # Convert digit values to tile values
        board = np.power(2, board)
        # Place blank tiles
        board[board==1] = 0
        TwntyFrtyEight.visited_states[state] = board
        
        return board
    
    @staticmethod
    def board_to_state(board):
        '''
        Hashes a 4x4 board to state number s in the range [0, 11^16-1]
        This function should be the inverse of stateToGrid()
        
        Works by first converting the grid to a base 11 representation
        then calculating the actual value of the base 11 representation
        '''
        arr = np.copy(board)
        if np.any(board==TwntyFrtyEight.WINNING_VALUE): return TwntyFrtyEight.WINNING_STATE
        # Replace blank tiles with 1
        arr[arr==0] = 1
        # Convert to base 11 representation
        digits = np.log2(arr.flatten()).astype('int64')
        base = int(np.log2(TwntyFrtyEight.WINNING_VALUE))
        values_per_digit = np.power(base, np.arange(len(digits)-1, -1, -1, dtype='int64'))
        state = digits.dot(values_per_digit)
        
        TwntyFrtyEight.visited_states[state] = board
        
        return state
    
    @staticmethod
    def get_feature_vector(state: int, action: int):
        '''
        Converts state number to feature vector
        '''
        board = TwntyFrtyEight.state_to_board(state)
        compressed_board, points = TwntyFrtyEight.move(board, action)
        def mean(): return compressed_board.mean()
        def std(): return compressed_board.std()
        def emptiness(): return np.count_nonzero(compressed_board==0)

        def distance_to_corner():
            '''
            Calculates manhattan distance of the largest tile to the nearest corner
            '''
            row_count, col_count = compressed_board.shape
            corners = np.array([(0,0), (0, col_count-1), (row_count-1, 0), (row_count-1, col_count-1)])
            max_pos = np.unravel_index(np.argmax(compressed_board), compressed_board.shape)
            return np.min(np.linalg.norm(corners-max_pos, ord=1, axis=1))

        def center_sum(): return np.sum(compressed_board[1:-1, 1:-1])
        def perimeter_sum(): return np.sum(compressed_board)-center_sum()
        
        def tile_delta():
            row_dif = np.sum(compressed_board[0:-1]-compressed_board[1:])
            col_dif = np.sum(compressed_board[:, 0:-1]-compressed_board[:, 1:])
            return np.min([row_dif, col_dif])
        
        def tile_delta_max():
            row_dif = np.max(compressed_board[0:-1]-compressed_board[1:])
            col_dif = np.max(compressed_board[:, 0:-1]-compressed_board[:, 1:])
            return np.max([row_dif, col_dif])
        
        def perimeter_diff(x: int, y: int):
            shifts = [-1, 0, 1]
            perimeter_scores = []
            for horizontal_shift in shifts:
                for vertical_shift in shifts:
                    if horizontal_shift == vertical_shift == 0:
                        continue
                    try:
                        if board[y+vertical_shift][x+horizontal_shift] != 0:
                            perimeter_scores.append(board[y+vertical_shift][x+horizontal_shift])
                    except:
                        continue
            return np.log(board[y][x])-np.average(np.log(perimeter_scores))
        
        #for each existing tile, calculate log average of perimeter
        #subtract sum from log of tile to obtain a difference score, lower the smoother
        #add/average the difference of all tiles

        def smoothness():
            smooth_scores = []
            for x in range(4):
                for y in range(4):
                    if board[y][x] != 0:
                        smooth_scores.append(perimeter_diff(x, y))
            return np.average(smooth_scores)

        def max_tile():
            tile_value = np.max(compressed_board)
            tile_value = 1 if tile_value == 0 else tile_value
            return np.log2(tile_value)

        def roughness():
            '''Standard deviation of differences between adjacent tiles (measures board jaggedness)'''
            row_diffs = np.diff(compressed_board, axis=1).flatten()
            col_diffs = np.diff(compressed_board, axis=0).flatten()
            return np.std(np.concatenate([row_diffs, col_diffs]))

        def monotonicity():
            '''Length of longest monotonically non-decreasing tile sequence in any row/col direction'''
            max_run = 1
            for row in compressed_board:
                for direction in [row, row[::-1]]:
                    run = 1
                    for j in range(1, len(direction)):
                        if direction[j] >= direction[j - 1]:
                            run += 1
                        else:
                            run = 1
                        max_run = max(max_run, run)
            for col in compressed_board.T:
                for direction in [col, col[::-1]]:
                    run = 1
                    for i in range(1, len(direction)):
                        if direction[i] >= direction[i - 1]:
                            run += 1
                        else:
                            run = 1
                        max_run = max(max_run, run)
            return max_run

        def std_vertical_dif():
            '''Standard deviation of vertical (row-to-row) tile differences'''
            return np.std(compressed_board[:-1] - compressed_board[1:])

        def std_horizontal_dif():
            '''Standard deviation of horizontal (col-to-col) tile differences'''
            return np.std(compressed_board[:, :-1] - compressed_board[:, 1:])

        X = np.array([points,
                      emptiness(),
                      roughness(),
                      monotonicity(),
                      std_vertical_dif(),
                      std_horizontal_dif(),
                      tile_delta(),
                      mean(),
                      std(),
                      distance_to_corner(),
                      center_sum(),
                      perimeter_sum(),
                      max_tile()])
        return X
        
    @staticmethod
    def get_all_next_states(state: int, action: int):
        next_states = []
        board = TwntyFrtyEight.state_to_board(state)
        compressed_board, points = TwntyFrtyEight.move(board, action)
        
        if np.any(compressed_board != board):
            for position in np.argwhere(compressed_board == 0):
                position = tuple(position)
                next_board = TwntyFrtyEight.add_two(compressed_board, position)
                next_state = TwntyFrtyEight.board_to_state(next_board)
                next_states.append(next_state)
        else:
            next_states.append(state)
            
        return np.array(next_states)
    
    @staticmethod
    def reward(current_state: int, current_action: int, next_state: int):
        '''
        Calculate reward based on current state, current action, and next state
        '''
        # reward winning
        if next_state == TwntyFrtyEight.WINNING_STATE:
            return 0
        
        if 0 <= next_state < TwntyFrtyEight.WINNING_STATE:
            current_board = TwntyFrtyEight.state_to_board(current_state)
            next_board = TwntyFrtyEight.state_to_board(next_state)
            # punish losing or choosing an action that does nothing
            if TwntyFrtyEight.get_board_status(next_board) == 'lose' or current_state == next_state:
                return 0
            # punish valid moves by -1
            else:
                # reward combining tiles
                compressed_board, points = TwntyFrtyEight.move(current_board, current_action)
                bonus = np.max(compressed_board) if np.max(compressed_board) > np.max(current_board) else 0
                return points + bonus
            
    @staticmethod
    def transition(state: int, action: int):
        board = TwntyFrtyEight.state_to_board(state)
        next_board, points = TwntyFrtyEight.move(board, action)
        
        if np.any(next_board != board):
            board = TwntyFrtyEight.add_two(next_board)
            
        return TwntyFrtyEight.board_to_state(board)
    
    @staticmethod
    def is_terminal_state(state: int):
        if state == TwntyFrtyEight.WINNING_STATE:
            return True

        if 0 <= state < TwntyFrtyEight.WINNING_STATE:
            board = TwntyFrtyEight.state_to_board(state)
            if TwntyFrtyEight.get_board_status(board) == 'lose':
                return True
            else:
                return False

    @staticmethod
    def heuristic_score(board: np.ndarray) -> float:
        '''
        Heuristic board evaluation for beam search.
        Combines four factors from Li & Peng 2021:
          1. Empty tiles  (more is better)
          2. Max tile     (higher is better)
          3. Smoothness   (lower adjacent log-difference is better)
          4. Monotonicity (more ordered rows/cols is better)
        Weights tuned to balance exploration vs. greedy corner-stacking.
        '''
        # 1. Empty tiles
        empty = np.count_nonzero(board == 0)

        # 2. Max tile in log2 space
        max_val = np.max(board)
        max_score = np.log2(max_val) if max_val > 0 else 0

        # 3. Smoothness: negative sum of |log2 differences| between adjacent tiles
        with np.errstate(divide='ignore'):
            log_board = np.where(board > 0, np.log2(board.astype(float)), 0)
        smoothness = -np.sum(np.abs(np.diff(log_board, axis=1)))
        smoothness -= np.sum(np.abs(np.diff(log_board, axis=0)))

        # 4. Monotonicity: count adjacent pairs that are in order (row/col, both directions)
        mono = 0
        for row in log_board:
            mono += max(
                sum(row[i] >= row[i + 1] for i in range(len(row) - 1)),
                sum(row[i] <= row[i + 1] for i in range(len(row) - 1))
            )
        for col in log_board.T:
            mono += max(
                sum(col[i] >= col[i + 1] for i in range(len(col) - 1)),
                sum(col[i] <= col[i + 1] for i in range(len(col) - 1))
            )

        return empty * 2.7 + max_score + smoothness * 0.1 + mono