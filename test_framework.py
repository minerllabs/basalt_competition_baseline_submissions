
import os

import gym
import minerl
from basalt_baselines.bc import bc_baseline, WRAPPERS as bc_wrappers

from test_submission_code import MineRLAgent, Episode, EpisodeDone, MineRLBehavioralCloningAgent
from basalt_utils.utils import wrap_env
import torch as th
# import coloredlogs
#coloredlogs.install(logging.DEBUG)


MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLBasaltFindCave-v0')
MINERL_MAX_EVALUATION_EPISODES = int(os.getenv('MINERL_MAX_EVALUATION_EPISODES', 5))
# We only use one evaluation thread
EVALUATION_THREAD_COUNT = 1

####################
# EVALUATION CODE  #
####################

def main():
    agent = MineRLBehavioralCloningAgent()
    agent.load_agent()

    assert MINERL_MAX_EVALUATION_EPISODES > 0
    assert EVALUATION_THREAD_COUNT > 0

    env = gym.make(MINERL_GYM_ENV)

    # Ensure that videos are closed properly

    wrapped_env = wrap_env(env, bc_wrappers)
    obs = wrapped_env.reset()
    done = False
    policy = th.load("train/trained_policy.pt")
    policy.eval()
    while not done:
        # TODO this is currently erroring
        obs_tensor = th.from_numpy(obs.copy()).unsqueeze(0)
        action, _, _ = policy.forward(obs_tensor)
        try:
            obs, reward, done, _ = wrapped_env.step(th.squeeze(action))
        except EpisodeDone:
            done = True
            continue
    #agent.run_agent_on_episode(Episode(env))
    # @atexit.register
    # def cleanup_env():
    #     env.close()
    #
    # # A simple function to evaluate on episodes!
    # def evaluate(i, env):
    #     print("[{}] Starting evaluator.".format(i))
    #     for i in range(MINERL_MAX_EVALUATION_EPISODES):
    #         try:
    #             agent.run_agent_on_episode(Episode(env))
    #         except EpisodeDone:
    #             print("[{}] Episode complete".format(i))
    #             pass
    #
    # thread = threading.Thread(target=evaluate, args=(0, env))
    # thread.start()
    # thread.join()


if __name__ == "__main__":
    main()
