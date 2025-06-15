# environment.py
import random

class GridWorld:
    def __init__(self, width=10, height=10, obstacle_ratio=0.1):
        self.width = width
        self.height = height
        self.obstacle_ratio = obstacle_ratio
        self.action_space = [0, 1, 2, 3] 
        self.action_size = len(self.action_space)
        self.agent_pos = (0, 0)
        self.goal_pos = (height - 1, width - 1)
        self.obstacles = set()
        self.generate_obstacles()


    def reset(self):
        return self._get_state()

    def generate_obstacles(self):
        self.obstacles = set()
        total_cells = self.width * self.height
        num_obstacles = int(total_cells * self.obstacle_ratio)

        while len(self.obstacles) < num_obstacles:
            pos = (random.randint(0, self.height - 1), random.randint(0, self.width - 1))
            if pos != self.agent_pos and pos != self.goal_pos and not self._blocks_agent(pos):
                self.obstacles.add(pos)

    def _blocks_agent(self, new_obstacle):
        y, x = self.agent_pos
        neighbors = [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
        count = 0
        for ny, nx in neighbors:
            if 0 <= ny < self.height and 0 <= nx < self.width:
                if (ny, nx) in self.obstacles or (ny, nx) == new_obstacle:
                    count += 1
        return count >= len(neighbors)

    def _get_state(self):
        return [
            self.agent_pos[0] / (self.height - 1),
            self.agent_pos[1] / (self.width - 1),
            self.goal_pos[0] / (self.height - 1),
            self.goal_pos[1] / (self.width - 1)
        ]

    def step(self, action):
        y, x = self.agent_pos
        if action == 0 and y > 0: y -= 1       
        elif action == 1 and y < self.height - 1: y += 1  
        elif action == 2 and x > 0: x -= 1     
        elif action == 3 and x < self.width - 1: x += 1    

        new_pos = (y, x)
        if new_pos not in self.obstacles:
            self.agent_pos = new_pos

        done = self.agent_pos == self.goal_pos
        dist = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
        max_dist = self.width + self.height - 2

        
        if done:
            reward = 100
        else:
            reward = (max_dist - dist) / max_dist * 10
            if new_pos == self.agent_pos:
                reward -= 1

        return self._get_state(), reward, done
