import numpy as np
import gym

from enum import IntEnum
from gym import spaces
from tensorflow.keras.losses import CategoricalCrossentropy
from grid_drawer import AgentSprite, Drawer


class DyadicConvnetGymEnv(gym.Env):

    metadata = {'render-modes': ['human']}

    class Actions(IntEnum):
        down = 10
        up_top_left = 11
        up_bottom_left = 12
        up_top_right = 13
        up_bottom_right = 14

    def __init__(self, dataset, images, labels, layers, max_steps, visualize=False, training=False, num_classes=10, tile_width=10,
                 class_penalty=0.1, correct_class=1.0, illegal_mov=0.5, same_position=0.01, non_classified=3.0,
                 step_reward_multiplier=0.01):
        super(DyadicConvnetGymEnv, self).__init__()
        self.episodes_count = 0
        self.dataset = dataset
        self.images = images if visualize else None
        self.labels = labels
        self.layers = layers
        self.dataset_length = len(self.dataset)
        self.num_layers = len(self.layers)
        self.max_layer = np.max(self.layers)
        self.min_layer = np.min(self.layers)
        # Extracting current training image from dataset
        self.train_image = None
        self.image_class = None
        # CNN representation of the extracted image
        self.features = None
        # Will need this for computing the reward
        self.actions = DyadicConvnetGymEnv.Actions
        # Single action space
        """self.action_space = spaces.Dict({'classification': spaces.Discrete(len(self.ground_truth)),
                                         'movement': spaces.Discrete(len(self.actions))
                                         })"""
        self.num_actions = num_classes + len(self.actions)
        self.action_space = spaces.Discrete(n=self.num_actions)
        # 64 conv features
        self.observation_space = spaces.Dict({'features': spaces.Box(low=0.0, high=1.0, shape=(147,), dtype=np.float32)
                                              })
        self.step_count = 0
        self.agent_pos = None
        self.ep_visited = None
        self.agent_classification = None
        self.one_hot_action = None
        self.max_steps = max_steps
        self.agent_reward_loss = CategoricalCrossentropy()
        self.class_reward = 0.0
        self.mov_reward = 0.0
        self.last_reward = [0.0]
        self.last_action = None
        self.training = training
        # Reward setup
        self.class_penalty = class_penalty
        self.correct_class = correct_class
        self.illegal_mov = illegal_mov
        self.same_position = same_position
        self.non_classified = non_classified
        self.step_reward_multiplier = step_reward_multiplier
        # Drawing
        if training:
            self.visualize = False
            self.agent_sprite = None
            self.drawer = None
        else:
            self.visualize = visualize
            self.agent_sprite = AgentSprite(rect_width=tile_width, layers=self.layers, pos=(0, 0, 0)) if self.visualize else None
            self.drawer = Drawer(self.agent_sprite, layers=self.layers, tile_width=tile_width) if self.visualize else None

    def step(self, action):
        self.step_count += 1
        done = False
        self.class_reward = 0.0
        self.mov_reward = 0.0
        old_pos = self.agent_pos
        # New agent position based on the movement action
        if action == self.actions.down:
            if self.agent_pos[0] < self.max_layer:
                self.agent_pos = (self.agent_pos[0] + 1,
                                  int(self.agent_pos[1]/2),
                                  int(self.agent_pos[2]/2))
        elif action == self.actions.up_top_left:
            if self.agent_pos[0] > self.min_layer:
                self.agent_pos = (self.agent_pos[0] - 1,
                                  2*self.agent_pos[1],
                                  2*self.agent_pos[2])
        elif action == self.actions.up_top_right:
            if self.agent_pos[0] > self.min_layer:
                self.agent_pos = (self.agent_pos[0] - 1,
                                  2*self.agent_pos[1] + 1,
                                  2*self.agent_pos[2])
        elif action == self.actions.up_bottom_left:
            if self.agent_pos[0] > self.min_layer:
                self.agent_pos = (self.agent_pos[0] - 1,
                                  2*self.agent_pos[1],
                                  2*self.agent_pos[2] + 1)
        elif action == self.actions.up_bottom_right:
            if self.agent_pos[0] > self.min_layer:
                self.agent_pos = (self.agent_pos[0] - 1,
                                  2*self.agent_pos[1] + 1,
                                  2*self.agent_pos[2] + 1)
        # If agent classifies well, end the episode
        else:
            self.class_reward = self.correct_class if action == self.image_class else -self.class_penalty
            done = True

        if self.agent_pos not in self.ep_visited:
            self.ep_visited.append(self.agent_pos)
            new_state = 0.05
        else:
            new_state = 0.0

        if self.visualize:
            self.agent_sprite.move(self.agent_pos)
            self.drawer.render(agent=self.agent_sprite, img=self.train_image, label=int(self.image_class),
                               predicted=action if int(action) < 10 else None, first_step=False)

        if self.step_count >= self.max_steps:
            done = True

        # Punishing the agent for illegal actions
        if old_pos[0] == self.layers[0] and action in [self.actions.up_bottom_right, self.actions.up_top_right,
                                                       self.actions.up_top_left, self.actions.up_bottom_left]:
            self.mov_reward = -self.illegal_mov
        elif old_pos[0] == self.layers[self.num_layers - 1] and action == self.actions.down:
            self.mov_reward = -self.illegal_mov
        else:
            self.mov_reward = 0.0
        # If agent classifies well but stays in the same position it's ok, while if classification is wrong, it is encouraged to move
        if self.agent_pos == old_pos and action != self.image_class:
            self.mov_reward -= self.same_position

        reward = self.class_reward + self.mov_reward + self.step_reward_multiplier*self.step_count + new_state
        if self.step_count == self.max_steps and int(action) >= 10:
            reward -= self.non_classified
        # Adjusting parameters for new observation
        self.one_hot_action = [1.0 if x == action else 0.0 for x in range(self.num_actions)]
        self.last_reward = [reward]
        self.last_action = action
        obs = self.gen_obs()
        # Why {}?
        return obs, reward, done, {}

    def reset(self):
        # New image extraction
        self.features = self.dataset[self.episodes_count]
        if self.visualize:
            self.train_image = self.images[self.episodes_count]
        self.image_class = int(self.labels[self.episodes_count])
        # Go to next index
        self.episodes_count = (self.episodes_count + 1) % self.dataset_length
        # Agent starting position encoded as (layer, x, y)
        starting_layer = np.random.choice(self.layers)
        starting_x = np.random.randint(0, self.features[starting_layer].shape[0] - 1) if starting_layer != 4 else 0
        starting_y = np.random.randint(0, self.features[starting_layer].shape[0] - 1) if starting_layer != 4 else 0
        #starting_x = pow(2, len(self.layers) - starting_layer - 1)
        #starting_y = pow(2, len(self.layers) - starting_layer - 1)
        self.agent_pos = (starting_layer, starting_x, starting_y)
        self.ep_visited = []
        self.ep_visited.append(self.agent_pos)
        self.last_reward = [0.0]
        self.one_hot_action = [0.0 for x in range(self.num_actions)]
        self.step_count = 0
        obs = self.gen_obs()
        if self.visualize:
            self.agent_sprite.move(self.agent_pos)
            self.drawer.render(agent=self.agent_sprite, img=self.train_image, label=int(self.image_class),
                               predicted=None, first_step=True)

        return obs

    def gen_obs(self):
        # Adding random gaussian noise to features vector
        feats = self.features[self.agent_pos[0]][self.agent_pos[1]][self.agent_pos[2]]
        obs = {
            'features': np.concatenate((feats, self.features[4][0][0], self.agent_pos, self.one_hot_action, self.last_reward),
                                       axis=0)
        }

        return obs

    def set_agent_classification(self, value):
        if self.agent_classification is not None:
            self.right_old_class = np.max(self.agent_classification) if np.argmax(self.agent_classification) < 10 else 0.0
        else:
            self.right_old_class = 0.0
        self.agent_classification = value
