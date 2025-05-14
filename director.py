# director.py
"""
Module: director.py
Description: Defines the DirectorAgent class and the StoryEnv environment for multi-agent storytelling.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from agents import PlotAgent, CharacterAgent, DialogueAgent, FactCheckerAgent
from utils import FANTASY_KEYWORDS

class StoryEnv(gym.Env):
    """Custom Gym environment for the storytelling multi-agent collaboration."""
    def __init__(self, agents: dict = None, max_steps: int = 10, training: bool = False):
        super().__init__()
        self.max_steps = max_steps
        self.training = training
        stub = self.training
        self.used_actions = set()
        self.action_history = []

        if agents:
            self.plot_agent = agents.get('plot') or PlotAgent(use_stub=stub)
            self.char_agent = agents.get('char') or CharacterAgent(use_stub=stub)
            self.dial_agent = agents.get('dialogue') or DialogueAgent(use_stub=stub)
            self.fact_checker = agents.get('fact') or FactCheckerAgent(use_stub=stub)
        else:
            self.plot_agent = PlotAgent(use_stub=stub)
            self.char_agent = CharacterAgent(use_stub=stub)
            self.dial_agent = DialogueAgent(use_stub=stub)
            self.fact_checker = FactCheckerAgent(use_stub=stub)
        # Story state
        self.story = ""
        self.characters = set()
        self.used_fantasy_terms = set()
        self.step_count = 0
        self.total_issues = 0 
        self.last_action = -1
        # Define action and observation spaces
        # Actions: 0 = PlotAgent, 1 = CharacterAgent, 2 = DialogueAgent, 3 = EndStory
        self.action_space = spaces.Discrete(4)
        # Observation: [last_agent_id, steps, char_count, contradictions_count, fantasy_term_count]
        low = np.array([-1, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([2, max_steps, max_steps, max_steps, len(FANTASY_KEYWORDS)], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        # Reset story state
        self.story = "In the land of Eldoria, a prophecy foretold the rise of a forgotten heir. "
        self.characters = set("Kael")
        self.used_fantasy_terms = set()
        self.step_count = 0
        self.total_issues = 0
        self.last_action = -1
        # Return initial observation
        obs = np.array([self.last_action, self.step_count, len(self.characters), 0, len(self.used_fantasy_terms)], dtype=np.float32)
        return obs, {}

    def step(self, action: int):
        terminated = False
        truncated = False
        reward = 0.0
        info = {}
        # If action corresponds to an agent contributing to story
        if action in [0, 1, 2]:
            if action in self.used_actions:
                reward -= 0.5
            self.used_actions.add(action)
            self.action_history.append(action)
            # Determining which agent to invoke
            if action == 0:
                agent = self.plot_agent
                new_text = agent.contribute(self.story)
            elif action == 1:
                agent = self.char_agent
                new_text = agent.contribute(self.story, characters=list(self.characters))
            elif action == 2:
                agent = self.dial_agent
                new_text = agent.contribute(self.story, characters=list(self.characters))
            # Append the new content to the story
            self.story += (" " if self.story and not self.story.endswith("\n") else "") + new_text
            # Update character list by finding new proper-names in the added text
            for word in new_text.split():
                if word.istitle():
                    self.characters.add(word.strip("\"',.:!?;"))
            # Use FactChecker to check consistency of this new segment
            is_consistent, issues = self.fact_checker.check_consistency(new_text, self.story)
            if not is_consistent:
                # penalty for each inconsistency detected
                reward += -1.0 * len(issues)
                self.total_issues += len(issues)
            # eward introduction of new fantasy elements
            text_lower = new_text.lower()
            for term in FANTASY_KEYWORDS:
                if term in text_lower and term not in self.used_fantasy_terms:
                    reward += 0.5
                    self.used_fantasy_terms.add(term)
            # Increment step count for content addition
            self.step_count += 1
            # If we've reached max_steps without an explicit end, truncate the episode
            if self.step_count >= self.max_steps:
                truncated = True
        elif action == 3:
            # End of story action
            terminated = True
            # add a concluding to the story.
            self.story += "\nTHE END."
        else:
            pass
        # Compute observation for next state
        self.last_action = action if action in [0,1,2] else self.last_action
        obs = np.array([float(self.last_action), float(self.step_count), float(len(self.characters)), float(self.total_issues), float(len(self.used_fantasy_terms))], dtype=np.float32)
        # If episode is ending (either terminated or truncated)
        if terminated or truncated:
            # Final evaluation of story quality
            fantasy_count = len(self.used_fantasy_terms)
            char_count = len(self.characters)
            issues_count = self.total_issues
            # Reward diversity of agents used
            reward += 1.2 * len(set(self.used_actions))  # up to +3.6 if all 3 used
            # Penalize repetitive behavior
            if len(self.action_history) >= 3 and self.action_history[-1] == self.action_history[-2] == self.action_history[-3]:
                reward -= 1.0  # repetition penalty
            # Reward for use of fantasy elements
            reward += 1.0 * fantasy_count
            # Reward for introducing multiple characters
            if char_count > 1:
                reward += 0.5 * (char_count - 1)
            # Penalty for consistency issues
            reward += -1.0 * issues_count
            # Encourage adequate story length and proper conclusion
            if truncated:
                # Penalty if story ended due to reaching max length without explicit ending
                reward += -3.0
            if self.step_count < 3:
                # Story too short penalty
                reward += -1.0
            elif self.step_count >= 5:
                # Small bonus for creating a reasonably long narrative
                reward += 1.0
            info['story'] = self.story
            info['episode_reward'] = reward
        return obs, reward, terminated, truncated, info

class DirectorAgent:
    """Reinforcement learning-based Director agent that coordinates the story generation process."""
    def __init__(self, env: StoryEnv):
        self.env = env
        self.model = None
    def train(self, timesteps: int = 10000):
        # Train the Director agent using PPO on the storytelling environment.
        check_env(self.env)
        # Initialize PPO model and train
        self.model = PPO("MlpPolicy", self.env, verbose=1, ent_coef=0.05)
        self.model.learn(total_timesteps=timesteps)
    def save_policy(self, path: str = "director_policy"):
        # Save the trained policy model to the given file path
        if self.model:
            self.model.save(path)
    def load_policy(self, path: str):
        # Load a trained policy from file and attach it to the environment
        self.model = PPO.load(path, env=self.env)
    def act(self, observation):
        # Decide an action given the current observation (story state)
        if self.model:
            action, _state = self.model.predict(observation, deterministic=True)
            return int(action)
        else:
            # Default heuristic: cycle through agents and end after a certain length
            last_agent = int(observation[0])
            step_count = int(observation[1])
            # if story long enough, end it
            if step_count >= 8:
                return 3  # End story
            # Otherwise, cycle Plot -> Character -> Dialogue -> back to Plot
            if last_agent == -1 or last_agent == 2:
                return 0  # Next, use PlotAgent (if none previous or after Dialogue)
            elif last_agent == 0:
                return 1  # after Plot, use Character
            elif last_agent == 1:
                return 2  # after Character, use Dialogue
            else:
                return 0
