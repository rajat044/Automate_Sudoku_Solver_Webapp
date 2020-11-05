def canput(grid, x, y, n):
    for i in range(9):
        if grid[i][y] == n: return False
    for i in range(9):
        if grid[x][i] == n: return False
    x0, y0 = (x//3)*3, (y//3)*3

    for i in range(3):
        for j in range(3):
            if grid[i+x0][j+y0] == n: return False
    return True
    
def find_em(board):
	for i in range(9):
		for j in range(9):
			if board[i][j] == 0: return (i, j)


def solve(board):
	cor = find_em(board)
	if not cor: return True
	else: x, y = cor
	for i in range(1, 10):
		if canput(board, x, y, i):
			board[x][y] = i

			if solve(board): return True

			board[x][y] = 0
	return False

	

