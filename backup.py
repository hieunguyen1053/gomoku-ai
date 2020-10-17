import pygame
import numpy as np
import time
import os


class Player:
    WHITE = 1
    BLACK = 2


class Board:
    def __init__(self, board=None, matrix=None, size=None):
        if board is not None:
            self.matrix = np.array(board.matrix)
        if matrix is not None:
            self.matrix = np.array(matrix)
        if size is not None:
            self.matrix = np.zeros((size, size))
        self.size = self.matrix.shape[0]

    def __check_row(self, color):
        neg_color = 1 if color == 2 else 2
        for i in range(self.size):
            for j in range(self.size-4):
                window = self.matrix[i, j:j+5]
                if np.equal(window, np.ones(5) * color).all():
                    if j > 0 and j + 4 < self.size:
                        if self.matrix[i, j-1] == neg_color and self.matrix[i, j+5] == neg_color:
                            continue
                    return True
        return False

    def __check_col(self, color):
        neg_color = 1 if color == 2 else 2
        for i in range(self.size-4):
            for j in range(self.size):
                window = self.matrix[i:i+5, j]
                if np.equal(window, np.ones(5) * color).all():
                    if i > 0 and i + 4 < self.size:
                        if self.matrix[i-1, j] == neg_color and self.matrix[i+5, j] == neg_color:
                            continue
                    return True
        return False

    def __check_diagonal(self, color):
        neg_color = 1 if color == 2 else 2
        for i in range(self.size-4):
            for j in range(self.size-4):
                window = self.matrix[i:i+5, j:j+5]
                if np.equal(window.diagonal(), np.ones(5) * color).all():
                    if i > 0 and i + 4 < self.size:
                        if self.matrix[i-1, j-1] == neg_color and self.matrix[i+5, j+5] == neg_color:
                            continue
                    return True
                if np.equal(np.fliplr(window).diagonal(), np.ones(5) * color).all():
                    if self.matrix[i-1, j+5] == neg_color and self.matrix[i+5, j-1] == neg_color:
                        continue
                    return True
        return False

    def check_win(self, color):
        if self.__check_row(color):
            return True
        if self.__check_col(color):
            return True
        if self.__check_diagonal(color):
            return True
        return False

    def generate_moves(self):
        moves = []
        for i in range(self.size):
            for j in range(self.size):
                if self.matrix[i, j] > 0:
                    continue
                if i > 0:
                    if j > 0:
                        if self.matrix[i-1][j-1] > 0 or self.matrix[i][j-1] > 0:
                            moves.append((i, j))
                            continue
                    if j < self.size - 1:
                        if self.matrix[i-1][j+1] > 0 or self.matrix[i][j+1] > 0:
                            moves.append((i, j))
                            continue
                    if self.matrix[i-1][j] > 0:
                        moves.append((i, j))
                        continue
                if i < self.size - 1:
                    if j > 0:
                        if self.matrix[i+1][j-1] > 0 or self.matrix[i][j-1] > 0:
                            moves.append((i, j))
                            continue
                    if j < self.size - 1:
                        if self.matrix[i+1][j+1] > 0 or self.matrix[i][j+1] > 0:
                            moves.append((i, j))
                            continue
                    if self.matrix[i+1][j] > 0:
                        moves.append((i, j))
                        continue
        return moves

    def draw(self, move, is_black):
        color = 2 if is_black else 1
        self.matrix[move] = color

    def create_str(self):
        txt = ""
        for i in range(self.size):
            for j in range(self.size):
                txt += str(int(self.matrix[i, j]))
        return txt


class Engine:
    evaluation_count = 0
    win_score = 100000000

    @classmethod
    def __get_patterns(cls, line, pattern_dict, is_black):
        color = 2 if is_black else 1
        neg_color = 1 if is_black else 2
        s = ''
        old = 0

        for i, c in enumerate(line):
            if c == color:
                if old == neg_color:
                    s += ('O' if color == 1 else 'X')
                s += ('X' if color == 1 else 'O')
            if c != color or i == len(line)-1:
                if c == neg_color and len(s) > 0:
                    s += ('O' if color == 1 else 'X')
                if s in pattern_dict.keys():
                    pattern_dict[s] += 1
                else:
                    pattern_dict[s] = 1
                s = ''
            old = c

    @classmethod
    def __get_patterns_row(cls, board: Board, pattern_dict, is_black):
        size = board.size
        matrix = board.matrix
        for i in range(size):
            cls.__get_patterns(matrix[i], pattern_dict, is_black)

    @classmethod
    def __get_patterns_col(cls, board: Board, pattern_dict, is_black):
        size = board.size
        matrix = board.matrix
        for i in range(size):
            cls.__get_patterns(matrix[:, i], pattern_dict, is_black)

    @classmethod
    def __get_patterns_diagonal(cls, board: Board, pattern_dict, is_black):
        size = board.size
        matrix1 = board.matrix
        matrix2 = matrix1[::-1, :]
        for i in range(-size+1, size):
            cls.__get_patterns(matrix1.diagonal(i), pattern_dict, is_black)
            cls.__get_patterns(matrix2.diagonal(i), pattern_dict, is_black)

    @classmethod
    def evaluate_board(cls, board: Board, is_black_turn: bool):
        cls.evaluation_count += 1
        black_score = cls.get_score(board, True, is_black_turn)
        white_score = cls.get_score(board, False, is_black_turn)
        if black_score == 0:
            black_score = 1.0
        return white_score / black_score

    @classmethod
    def get_score(cls, board: Board, is_black: bool, is_black_turn: bool):
        pattern_dict = {}
        cls.__get_patterns_row(board, pattern_dict, is_black)
        cls.__get_patterns_col(board, pattern_dict, is_black)
        cls.__get_patterns_diagonal(board, pattern_dict, is_black)
        print(pattern_dict)
        for pattern in pattern_dict:
            print(pattern)
        matrix = board.matrix
        value_horizontal = cls.__evaluate_horizontal(matrix, is_black, is_black_turn)
        value_vertical = cls.__evaluate_vertical(matrix, is_black, is_black_turn)
        value_diagonal = cls.__evaluate_diagonal(matrix, is_black, is_black_turn)
        return value_horizontal + value_vertical + value_diagonal

    @classmethod
    def __evaluate_horizontal(cls, matrix: np.ndarray, is_black, is_black_turn):
        consecutive = 0
        blocks = 2
        score = 0
        size, size = matrix.shape

        for i in range(size):
            for j in range(size):
                if matrix[i, j] == (2 if is_black else 1):
                    consecutive += 1
                elif matrix[i, j] == 0:
                    if consecutive > 0:
                        blocks -= 1
                        score += cls.get_consecutive_score(
                            consecutive, blocks, is_black == is_black_turn)
                        consecutive = 0
                        blocks = 1
                    else:
                        blocks = 1
                elif consecutive > 0:
                    score += cls.get_consecutive_score(
                        consecutive, blocks, is_black == is_black_turn)
                    consecutive = 0
                    blocks = 2
                else:
                    blocks = 2
            if consecutive > 0:
                score += cls.get_consecutive_score(
                    consecutive, blocks, is_black == is_black_turn)
            consecutive = 0
            blocks = 2
        return score

    @classmethod
    def __evaluate_vertical(cls, matrix: np.ndarray, is_black, is_black_turn):
        consecutive = 0
        blocks = 2
        score = 0
        size, size = matrix.shape

        for j in range(size):
            for i in range(size):
                if matrix[i, j] == (2 if is_black else 1):
                    consecutive += 1
                elif matrix[i, j] == 0:
                    if consecutive > 0:
                        blocks -= 1
                        score += cls.get_consecutive_score(
                            consecutive, blocks, is_black == is_black_turn)
                        consecutive = 0
                        blocks = 1
                    else:
                        blocks = 1
                elif consecutive > 0:
                    score += cls.get_consecutive_score(
                        consecutive, blocks, is_black == is_black_turn)
                    consecutive = 0
                    blocks = 2
                else:
                    blocks = 2
            if consecutive > 0:
                score += cls.get_consecutive_score(
                    consecutive, blocks, is_black == is_black_turn)
            consecutive = 0
            blocks = 2
        return score

    @classmethod
    def __evaluate_diagonal(cls, matrix: np.ndarray, is_black, is_black_turn):
        consecutive = 0
        blocks = 2
        score = 0
        size, size = matrix.shape

        for k in range(2 * size - 3):
            iStart = max(0, k - size + 1)
            iEnd = min(size - 1, k)

            for i in range(iStart, iEnd+1):
                j = k - i
                if matrix[i, j] == (2 if is_black else 1):
                    consecutive += 1
                elif matrix[i, j] == 0:
                    if consecutive > 0:
                        blocks -= 1
                        score += cls.get_consecutive_score(
                            consecutive, blocks, is_black == is_black_turn)
                        consecutive = 0
                        blocks = 1
                    else:
                        blocks = 1
                elif consecutive > 0:
                    score += cls.get_consecutive_score(
                        consecutive, blocks, is_black == is_black_turn)
                    consecutive = 0
                    blocks = 2
                else:
                    blocks = 2
            if consecutive > 0:
                score += cls.get_consecutive_score(
                    consecutive, blocks, is_black == is_black_turn)
            consecutive = 0
            blocks = 2

        for k in range(1-size, size):
            iStart = max(0, k)
            iEnd = min(size + k - 1, size - 1)

            for i in range(iStart, iEnd+1):
                j = i - k
                if matrix[i, j] == (2 if is_black else 1):
                    consecutive += 1
                elif matrix[i, j] == 0:
                    if consecutive > 0:
                        blocks -= 1
                        score += cls.get_consecutive_score(
                            consecutive, blocks, is_black == is_black_turn)
                        consecutive = 0
                        blocks = 1
                    else:
                        blocks = 1
                elif consecutive > 0:
                    score += cls.get_consecutive_score(
                        consecutive, blocks, is_black == is_black_turn)
                    consecutive = 0
                    blocks = 2
                else:
                    blocks = 2
            if consecutive > 0:
                score += cls.get_consecutive_score(
                    consecutive, blocks, is_black == is_black_turn)
            consecutive = 0
            blocks = 2
        return score

    @classmethod
    def get_consecutive_score(cls, count, blocks, current_turn):
        WIN_SCORE = 1000000
        if blocks == 2 and count < 5:
            return 0
        if count == 5:
            return WIN_SCORE
        if count == 4:
            if current_turn:
                return WIN_SCORE
            else:
                if blocks == 0:
                    return WIN_SCORE//4
                else:
                    return 200
        if count == 3:
            if blocks == 0:
                if current_turn:
                    return 50000
                else:
                    return 200
            else:
                if current_turn:
                    return 10
                else:
                    return 5
        if count == 2:
            if blocks == 0:
                if current_turn:
                    return 7
                else:
                    return 5
            else:
                return 3
        if count == 1:
            return 1
        end = time.time()
        return WIN_SCORE * 2

    @classmethod
    def find_next_move(cls, board: Board, depth, is_black):
        inp = board.create_str()
        if is_black:
            inp = inp.replace("1", "3")
            inp = inp.replace("2", "1")
            inp = inp.replace("3", "2")

        try:
            move = os.popen(f"java Engine {inp} {depth}").read()
            row, col = move.split(" ")
            row, col = int(row), int(col)
        except Exception as e:
            import random
            row, col = random.randint(0, 20), random.randint(0, 20)
        return (row, col)
        # start = time.time()
        # value, best_move = cls.__search_winning_move(board)
        # if best_move is not None:
        #     move = best_move
        # else:
        #     value, best_move = cls.minimax_alphabeta(
        #         board, depth, -1.0, cls.win_score, True)
        #     if best_move is None:
        #         move = None
        #     else:
        #         move = best_move
        # end = time.time()
        # print("Phep tinh: ", cls.evaluation_count, "thoi gian:", end-start)
        # cls.evaluation_count = 0
        # return move

    @classmethod
    def minimax_alphabeta(cls, board: Board, depth, alpha, beta, is_max):
        if depth == 0:
            return cls.evaluate_board(board, not is_max), None

        all_possible_moves = board.generate_moves()[::-1]

        if len(all_possible_moves) == 0:
            return cls.evaluate_board(board, not is_max), None

        best_move = None

        if is_max:
            best_value = -1.0
            for move in all_possible_moves:
                dumm_board = Board(board=board)
                dumm_board.draw(move, False)
                value, temp_move = cls.minimax_alphabeta(
                    dumm_board, depth-1, alpha, beta, not is_max)
                if value > alpha:
                    alpha = value
                if value >= beta:
                    return value, temp_move
                if value > best_value:
                    best_value = value
                    best_move = move
        else:
            best_value = 100000000
            best_move = all_possible_moves[0]
            for move in all_possible_moves:
                dumm_board = Board(board=board)
                dumm_board.draw(move, True)
                value, temp_move = cls.minimax_alphabeta(
                    dumm_board, depth-1, alpha, beta, not is_max)
                if value < beta:
                    beta = value
                if value <= alpha:
                    return value, temp_move
                if value < best_value:
                    best_value = value
                    best_move = move
        return best_value, best_move

    @classmethod
    def __search_winning_move(cls, board: Board):
        all_possible_moves = board.generate_moves()

        for move in all_possible_moves:
            cls.evaluation_count += 1
            dumm_board = Board(board=board)
            dumm_board.draw(move, False)
            if cls.get_score(dumm_board, False, False) >= cls.win_score:
                return (None, move)
        return (None, None)


class Game:
    def __init__(self, board: Board):
        self.grid_size = 30
        self.start_x, self.start_y = 30, 50
        self.edge_size = self.grid_size / 2

        self.piece = 2

        self.board = board

        self.black_turn = True
        self.game_over = False
        self.winner = 0

        self.black_turn = True
        self.black_score = 0
        self.white_score = 0

    def handle_key_event(self, e):
        origin_x = self.start_x - self.edge_size
        origin_y = self.start_y - self.edge_size
        size = (self.board.size - 1) * self.grid_size + self.edge_size * 2
        pos = e.pos
        if origin_x <= pos[0] <= origin_x + size and origin_y <= pos[1] <= origin_y + size:
            x = pos[0] - origin_x
            y = pos[1] - origin_y
            row = int(y // self.grid_size)
            col = int(x // self.grid_size)
            self.set_piece(row, col)
        self.black_turn = not self.black_turn

    def ai_play(self, player):
        is_black = player == 2
        row, col = Engine.find_next_move(self.board, 3, is_black)
        self.set_piece(row, col)
        self.black_turn = not self.black_turn

    def set_piece(self, row, col):
        if self.board.matrix[row][col] == 0:
            self.board.matrix[row][col] = self.piece

            if self.piece == 1:
                self.piece = 2
            else:
                self.piece = 1

            return True
        return False

    def check_win(self):
        if self.board.check_win(1):
            self.winner = 1
            self.game_over = True
        if self.board.check_win(2):
            self.winner = 2
            self.game_over = True

    def draw(self, screen):
        pygame.draw.rect(screen, (185, 122, 87), [self.start_x - self.edge_size, self.start_y - self.edge_size, (self.board.size - 1)
                                                  * self.grid_size + self.edge_size * 2, (self.board.size - 1) * self.grid_size + self.edge_size * 2], 0)

        for r in range(self.board.size):
            y = self.start_y + r * self.grid_size
            pygame.draw.line(screen, (0, 0, 0), [self.start_x, y], [
                             self.start_x + self.grid_size * (self.board.size - 1), y], 2)

        for c in range(self.board.size):
            x = self.start_x + c * self.grid_size
            pygame.draw.line(screen, (0, 0, 0), [x, self.start_y], [
                             x, self.start_y + self.grid_size * (self.board.size - 1)], 2)

        for r in range(self.board.size):
            for c in range(self.board.size):
                piece = self.board.matrix[r][c]
                if piece != 0:
                    if piece == 1:
                        color = (255, 255, 255)
                    else:
                        color = (0, 0, 0)

                    x = self.start_x + c * self.grid_size
                    y = self.start_y + r * self.grid_size
                    pygame.draw.circle(
                        screen, color, [x, y], self.grid_size // 2)


class GomokuUI():
    PVP = 0
    PVC = 1
    CVC = 2

    def __init__(self, width, height, name="Gomoku", mode=1):
        pygame.init()
        pygame.display.set_caption(name)
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.font = pygame.font.SysFont("arial", 24)
        board = Board(size=20)

        self.mode = mode
        self.going = True
        self.clock = pygame.time.Clock()
        self.game = Game(board)

    def loop(self):
        while self.going:
            self.update()
            self.draw()
            self.clock.tick(60)

        pygame.quit()

    def handle_event(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.going = False
            elif e.type == pygame.MOUSEBUTTONDOWN:
                self.game.handle_key_event(e)

    def update(self):
        self.game.check_win()
        if self.game.game_over:
            return

        self.game.black_score = Engine.get_score(
            self.game.board, True, self.game.black_turn)
        self.game.white_score = Engine.get_score(
            self.game.board, False, not self.game.black_turn)
        if self.mode == self.PVC:
            if self.game.black_turn:
                self.handle_event()
            else:
                self.game.ai_play(Player.WHITE)
        elif self.mode == self.PVP:
            self.handle_event()
        elif self.mode == self.CVC:
            self.handle_event()
            if self.game.black_turn:
                self.game.ai_play(Player.BLACK)
            else:
                self.game.ai_play(Player.WHITE)

    def draw(self):
        self.screen.fill((255, 255, 255))
        self.screen.blit(self.font.render("FPS: {0:.2F}".format(
            self.clock.get_fps()), True, (0, 0, 0)), (10, 10))
        self.screen.blit(self.font.render("Black: {0}".format(
            self.game.black_score), True, (0, 0, 0)), (10, self.height - 50))
        self.screen.blit(self.font.render("White: {0}".format(
            self.game.white_score), True, (0, 0, 0)), (10, self.height - 100))

        self.game.draw(self.screen)
        if self.game.game_over:
            self.screen.blit(self.font.render("{0} Win".format(
                "Black" if self.game.winner == 2 else "White"), True, (0, 0, 0)), (500, 10))

        pygame.display.update()


if __name__ == '__main__':
    game = GomokuUI(650, 750, "Gomoku", GomokuUI.PVP)
    game.loop()
