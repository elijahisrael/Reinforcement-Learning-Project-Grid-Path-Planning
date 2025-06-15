# train_dqn.py
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
from environment import GridWorld
from tqdm import tqdm
import pygame
import sys
import os
import random

sys.tracebacklimit = 1000

WIDTH = int(input("Enter grid width (e.g., 20): "))
HEIGHT = int(input("Enter grid height (e.g., 20): "))
EPISODES = int(input("Enter number of episodes to train (e.g., 500): "))
show_every = int(input("Visualize every N episodes (e.g., 10): "))
visualize = input("Show Pygame visuals during training? (y/n): ").strip().lower() == 'y'
model_name = input("Enter name to save model as (e.g., my_model): ").strip()


CELL_SIZE = 15
if visualize:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH * CELL_SIZE, HEIGHT * CELL_SIZE))
    pygame.display.set_caption("DQN GridWorld Visualization")
    clock = pygame.time.Clock()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

def draw_grid(env):
    screen.fill(WHITE)
    for row in range(env.height):
        for col in range(env.width):
            color = WHITE
            if (row, col) == env.agent_pos:
                color = BLUE
            elif (row, col) == env.goal_pos:
                color = GREEN
            elif (row, col) in env.obstacles:
                color = BLACK
            pygame.draw.rect(screen, color, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1))
    pygame.display.flip()

def handle_exit():
    print("\nExiting and saving model...")
    filename = model_name if model_name.endswith(".pth") else model_name + ".pth"
    agent.save_model(filename)
    print(f"Model saved as {filename}")
    if visualize:
        pygame.quit()
    sys.exit()


env = GridWorld(width=WIDTH, height=HEIGHT, obstacle_ratio=0.1)
state_size = 4
agent = DQNAgent(state_size, env.action_size, epsilon=0.999)

goals_reached = 0
rewards_per_episode = []

try:
    for episode in tqdm(range(EPISODES), desc="Training Progress", dynamic_ncols=True):
        env.agent_pos = (0, 0)
        env.goal_pos = (HEIGHT - 1, WIDTH - 1)
        state = env.reset()
        total_reward = 0
        done = False
        step_count = 0

        render = visualize and (episode == 0 or episode % show_every == 0)

        while not done:
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        handle_exit()
                draw_grid(env)
                clock.tick(30)

            sx, sy = env.agent_pos
            gx, gy = env.goal_pos
            full_state = [sx, sy, gx, gy]

            action = agent.act(full_state)
            next_state, _, done = env.step(action)
            nsx, nsy = env.agent_pos

            if (nsx, nsy) == (sx, sy):
                reward = -10

            if env.agent_pos == env.goal_pos:
                print(f"GOAL REACHED on step {step_count}")
                reward = 500
            else:
                prev_dist = abs(sx - gx) + abs(sy - gy)
                new_dist = abs(nsx - gx) + abs(nsy - gy)
                if new_dist < prev_dist:
                    reward = 5
                elif new_dist > prev_dist:
                    reward = -3
                else:
                    reward = -1

            agent.remember(full_state, action, reward, [nsx, nsy, gx, gy], done)
            if step_count % 10 == 0:
                agent.replay()

            state = next_state
            total_reward += reward
            step_count += 1

            if step_count > WIDTH * HEIGHT:
                done = True
                total_reward -= 50

        
        if env.agent_pos == env.goal_pos:
            goals_reached += 1
            agent.epsilon = min(1.0, agent.epsilon - 0.01)
        if agent.epsilon == 1:
            agent.epsilon = max(agent.epsilon_min, agent.epsilon - 0.002)
        if agent.epsilon < agent.epsilon_min:
            agent.epsilon = max(agent.epsilon_min, agent.epsilon + 0.01)

            

        rewards_per_episode.append(total_reward)
        avg_reward = sum(rewards_per_episode[-10:]) / min(len(rewards_per_episode), 10)
        if total_reward < avg_reward:
            agent.epsilon = max(agent.epsilon_min, agent.epsilon + 0.001)
        if total_reward > avg_reward:
            agent.epsilon = max(agent.epsilon_min, agent.epsilon - 0.001)
        print(f"Ep {episode + 1} | Steps: {step_count} | Total: {total_reward:.1f} | Avg(10): {avg_reward:.1f} | Eps: {agent.epsilon:.3f}", flush=True)

        if episode > 0 and episode % 10 == 0:
            print(f"Goals reached last 10 eps: {goals_reached}/10")
            goals_reached = 0


except KeyboardInterrupt:
    print("\nTraining interrupted by user.")

finally:
    save = input("Save model? (y/n): ").strip().lower()
    if save == 'y':
        filename = model_name if model_name.endswith(".pth") else model_name + ".pth"
        agent.save_model(filename)

    if visualize:
        pygame.quit()

    plt.figure(figsize=(10, 5))
    plt.plot(rewards_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"DQN Learning Curve: {WIDTH}x{HEIGHT} Grid")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
