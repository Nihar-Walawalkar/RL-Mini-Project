import numpy as np
from collections import deque

GRID_SIZE = 10

UP    = 0
DOWN  = 1
LEFT  = 2
RIGHT = 3

class SnakeEnv:
    def __init__(self, grid_size=GRID_SIZE):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        mid = self.grid_size // 2
        self.snake = deque([(mid, mid)])
        self.direction = RIGHT
        self.place_food()
        self.done = False
        self.steps = 0
        self.max_steps = self.grid_size * self.grid_size * 2
        return self._get_state()

    def place_food(self):
        empty = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if (r, c) not in self.snake
        ]
        self.food = empty[np.random.randint(len(empty))] if empty else None

    def _get_state(self):
        head = self.snake[0]
        # Danger: straight, right, left relative to current direction
        danger_straight = self._is_danger(self._next_pos(head, self.direction))
        danger_right    = self._is_danger(self._next_pos(head, self._turn_right(self.direction)))
        danger_left     = self._is_danger(self._next_pos(head, self._turn_left(self.direction)))
        # Food direction relative to head
        food_up    = int(self.food[0] < head[0]) if self.food else 0
        food_down  = int(self.food[0] > head[0]) if self.food else 0
        food_left  = int(self.food[1] < head[1]) if self.food else 0
        food_right = int(self.food[1] > head[1]) if self.food else 0
        return (
            int(danger_straight), int(danger_right), int(danger_left),
            self.direction,
            food_up, food_down, food_left, food_right
        )

    def _next_pos(self, pos, direction):
        r, c = pos
        if direction == UP:    return (r - 1, c)
        if direction == DOWN:  return (r + 1, c)
        if direction == LEFT:  return (r, c - 1)
        if direction == RIGHT: return (r, c + 1)

    def _is_danger(self, pos):
        r, c = pos
        return (r < 0 or r >= self.grid_size or
                c < 0 or c >= self.grid_size or
                pos in self.snake)

    def _turn_right(self, d):
        return [RIGHT, LEFT, UP, DOWN][d]

    def _turn_left(self, d):
        return [LEFT, RIGHT, DOWN, UP][d]

    def step(self, action):
        # action: 0=straight, 1=turn right, 2=turn left
        if action == 1:
            self.direction = self._turn_right(self.direction)
        elif action == 2:
            self.direction = self._turn_left(self.direction)

        head = self.snake[0]
        new_head = self._next_pos(head, self.direction)
        self.steps += 1

        # Collision
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size or
            new_head in self.snake or
            self.steps >= self.max_steps):
            self.done = True
            return self._get_state(), -10, True

        self.snake.appendleft(new_head)

        if new_head == self.food:
            reward = 10
            self.place_food()
        else:
            self.snake.pop()
            # Small reward for moving toward food
            reward = self._distance_reward(new_head)

        return self._get_state(), reward, False

    def _distance_reward(self, head):
        if self.food is None:
            return 0
        dist = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
        max_dist = 2 * self.grid_size
        return (max_dist - dist) / max_dist * 0.1

    def get_grid(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for i, (r, c) in enumerate(self.snake):
            grid[r][c] = 2 if i == 0 else 1  # 2=head, 1=body
        if self.food:
            grid[self.food[0]][self.food[1]] = 3
        return grid

    @property
    def score(self):
        return len(self.snake) - 1
