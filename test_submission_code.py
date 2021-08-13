import gym
import torch as th
from basalt_baselines.bc import bc_baseline, WRAPPERS as bc_wrappers
from stable_baselines3.common.utils import get_device
import numpy as np
import os 

class EpisodeDone(Exception):
    pass


class Episode(gym.Env):
    """A class for a single episode."""
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self._done = False

    def reset(self):
        if not self._done:
            return self.env.reset()

    def step(self, action):
        s, r, d, i = self.env.step(action)
        if d:
            self._done = True
            raise EpisodeDone()
        else:
            return s, r, d, i

    def wrap_env(self, wrappers):
        for wrapper, kwargs in wrappers:
            self.env = wrapper(self.env, **kwargs)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space


class MineRLAgent():
    """
    To compete in the competition, you are required to implement the two
    functions in this class:
        - load_agent: a function that loads e.g. network models
        - run_agent_on_episode: a function that plays one game of MineRL

    By default this agent behaves like a random agent: pick random action on
    each step.
    """

    def load_agent(self):
        """
        This method is called at the beginning of the evaluation.
        You should load your model and do any preprocessing here.
        THIS METHOD IS ONLY CALLED ONCE AT THE BEGINNING OF THE EVALUATION.
        DO NOT LOAD YOUR MODEL ANYWHERE ELSE.
        """
        # This is a random agent so no need to do anything
        # YOUR CODE GOES HERE
        pass

    def run_agent_on_episode(self, single_episode_env: Episode):
        """This method runs your agent on a SINGLE episode.

        You should just implement the standard environment interaction loop here:
            obs  = env.reset()
            while not done:
                env.step(self.agent.act(obs))
                ...

        Args:
            env (gym.Env): The env your agent should interact with.
        """
        # An implementation of a random agent
        # YOUR CODE GOES HERE
        _ = single_episode_env.reset()
        done = False
        steps = 0
        min_steps = 500
        while not done:
            random_act = single_episode_env.action_space.sample()
            if steps < min_steps and random_act['equip'] == 'snowball':
                random_act['equip'] = 'air'
            single_episode_env.step(random_act)
            steps += 1

class MineRLBehavioralCloningAgent(MineRLAgent):
    def load_agent(self):
        # TODO not sure how to get us to be able to load the policy from the right agent here
        self.policy = th.load(f"train/MineRLBasalt{os.getenv('MINERL_TRACK')}-v0.pt", map_location=th.device(get_device('auto')))
        self.policy.eval()

    def run_agent_on_episode(self, single_episode_env : Episode):
        # TODO Get wrappers actually used in BC training, and wrap environment with those
        single_episode_env.wrap_env(bc_wrappers)
        obs = single_episode_env.reset()
        done = False
        while not done:

            action, _, _ = self.policy.forward(th.from_numpy(obs.copy()).unsqueeze(0).to(get_device('auto')))
            try:
                if action.device.type == 'cuda':
                    action = action.cpu()
                obs, reward, done, _ = single_episode_env.step(np.squeeze(action.numpy()))
            except EpisodeDone:
                done = True
                continue
