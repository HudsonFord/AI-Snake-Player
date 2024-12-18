import pygame
import random
import numpy as np
from collections import defaultdict

# Initialize Pygame
pygame.init()

# Screen settings
WIDTH, HEIGHT = 500, 500
CELL_SIZE = 20
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake RL Environment")

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

# Snake game class
class SnakeGame:
    def __init__(self):
        self.snake = [[100, 100], [90, 100], [80, 100]]  # Snake starting positions
        self.food = [random.randint(0, WIDTH // CELL_SIZE - 1) * CELL_SIZE,
                     random.randint(0, HEIGHT // CELL_SIZE - 1) * CELL_SIZE]
        self.direction = "RIGHT"
        self.score = 0

    def step(self, action):
        # Update direction based on action
        if action == "UP" and self.direction != "DOWN":
            self.direction = "UP"
        elif action == "DOWN" and self.direction != "UP":
            self.direction = "DOWN"
        elif action == "LEFT" and self.direction != "RIGHT":
            self.direction = "LEFT"
        elif action == "RIGHT" and self.direction != "LEFT":
            self.direction = "RIGHT"

        # Move the snake
        head = self.snake[0][:]
        if self.direction == "UP":
            head[1] -= CELL_SIZE
        elif self.direction == "DOWN":
            head[1] += CELL_SIZE
        elif self.direction == "LEFT":
            head[0] -= CELL_SIZE
        elif self.direction == "RIGHT":
            head[0] += CELL_SIZE

        self.snake.insert(0, head)
        reward = 0

        # Check if the snake ate the food
        if head == self.food:
            reward = 10
            self.score += 1
            self.food = [random.randint(0, WIDTH // CELL_SIZE - 1) * CELL_SIZE,
                         random.randint(0, HEIGHT // CELL_SIZE - 1) * CELL_SIZE]
        else:
            self.snake.pop()

        # Check for collisions
        if (head in self.snake[1:] or
                head[0] < 0 or head[0] >= WIDTH or
                head[1] < 0 or head[1] >= HEIGHT):
            return -10, True  # Negative reward for game over

        return reward, False

    def render(self):
        screen.fill(WHITE)
        for block in self.snake:
            pygame.draw.rect(screen, GREEN, (*block, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(screen, RED, (*self.food, CELL_SIZE, CELL_SIZE))
        pygame.display.flip()

# Q-Learning agent class
class QLearningAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.actions = actions  # List of possible actions the agent can take (e.g., UP, DOWN, LEFT, RIGHT)
        self.lr = learning_rate  # Learning rate (alpha): How fast the agent learns from new information.
        self.gamma = discount_factor  # Discount factor (gamma): How much future rewards matter compared to immediate rewards.
        self.epsilon = epsilon  # Exploration rate: Controls how often the agent chooses a random action instead of exploiting known actions.
        self.epsilon_decay = epsilon_decay  # Decay factor for epsilon (how fast the agent will stop exploring over time).
        self.min_epsilon = min_epsilon  # Minimum value for epsilon (the lowest exploration rate).
        self.q_table = defaultdict(float)  # The Q-table: A dictionary storing Q-values for (state, action) pairs.

    def get_state(self, snake, food, direction):
        head = snake[0]
        food_dir = (np.sign(food[0] - head[0]), np.sign(food[1] - head[1]))  # Relative position of food to snake head
        return (tuple(head), direction, food_dir)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)  # Explore (choose a random action)
        else:
            q_values = [self.q_table[(state, action)] for action in self.actions]
            return self.actions[np.argmax(q_values)]  # Exploit (choose the best-known action)

    def update_q_table(self, state, action, reward, next_state):
        best_next_q = max([self.q_table[(next_state, a)] for a in self.actions])  # Best Q-value for next state
        current_q = self.q_table[(state, action)]  # Current Q-value for the (state, action) pair
        # Update Q-value using the Bellman equation
        self.q_table[(state, action)] = current_q + self.lr * (reward + self.gamma * best_next_q - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

# Main training loop
def train_agent():
    actions = ["UP", "DOWN", "LEFT", "RIGHT"]
    agent = QLearningAgent(actions)
    game = SnakeGame()

    clock = pygame.time.Clock()

    # Run the game loop
    for episode in range(1000):  # You can increase the number of episodes
        game.__init__()  # Reset the game at the start of each episode
        done = False
        total_reward = 0

        while not done:
            state = agent.get_state(game.snake, game.food, game.direction)
            action = agent.choose_action(state)
            reward, done = game.step(action)
            total_reward += reward

            next_state = agent.get_state(game.snake, game.food, game.direction)
            agent.update_q_table(state, action, reward, next_state)
            agent.decay_epsilon()

            game.render()

            clock.tick(10)

        print(f"Episode {episode+1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

if __name__ == "__main__":
    train_agent()
