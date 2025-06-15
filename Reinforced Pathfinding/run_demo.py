# run_demo.py
import pygame
import torch
import sys
import random
from environment import GridWorld
from dqn_agent import DQNAgent

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

CELL_SIZE = 15

def draw_grid(env, surface):
    surface.fill(WHITE)
    for row in range(env.height):
        for col in range(env.width):
            color = WHITE
            if (row, col) == env.goal_pos:
                color = GREEN
            elif (row, col) == env.agent_pos:
                color = BLUE
            elif (row, col) in env.obstacles:
                color = BLACK
            pygame.draw.rect(surface, color, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1))
    pygame.display.flip()

if __name__ == "__main__":
    pygame.init()

    model_name = input("Enter model filename to run (e.g., final_model.pth): ").strip()
    use_obstacles = input("Include obstacles? (y/n): ").strip().lower() == 'y'

    WIDTH, HEIGHT = 50, 50
    screen = pygame.display.set_mode((WIDTH * CELL_SIZE, HEIGHT * CELL_SIZE))
    pygame.display.set_caption("DQN GridWorld Visualization")
    clock = pygame.time.Clock()

    env = GridWorld(width=WIDTH, height=HEIGHT, obstacle_ratio=0.1 if use_obstacles else 0.0)
    env.agent_pos = (0, 0)
    env.goal_pos = (HEIGHT - 1, WIDTH - 1)

    state_size = 4
    action_size = env.action_size
    agent = DQNAgent(state_size, action_size)

    try:
        agent.load_model(model_name)
        print(f"Model loaded successfully from {model_name}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("No metadata found. Running model anyway.")

   
    agent.epsilon = 0.1
   
    EPISODES = 5
    for ep in range(EPISODES):
        env.agent_pos = (0, 0)
        state = [0, 0, HEIGHT - 1, WIDTH - 1]
        done = False
        step_count = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            draw_grid(env, screen)
            clock.tick(10)

            action = agent.act(state)
            next_state, reward, done = env.step(action)

            sx, sy = env.agent_pos
            gx, gy = env.goal_pos
            state = [sx, sy, gx, gy]

            step_count += 1
            if step_count > WIDTH * HEIGHT:
                print(f"Episode {ep + 1}: Timed out after {step_count} steps.")
                break

        if env.agent_pos == env.goal_pos:
            print(f"Episode {ep + 1}: Reached goal in {step_count} steps!")
        else:
            print(f"Episode {ep + 1}: Failed to reach goal.")

    pygame.quit()
