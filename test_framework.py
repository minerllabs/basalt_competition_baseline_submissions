import json
import logging
import os
import threading
import atexit

import aicrowd_helper
import gym
import minerl

from test_submission_code import MineRLAgent, Episode, EpisodeDone

import coloredlogs
coloredlogs.install(logging.DEBUG)

MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV')
MINERL_MAX_EVALUATION_EPISODES = int(os.getenv('MINERL_MAX_EVALUATION_EPISODES', 5))
# We only use one evaluation thread
EVALUATION_THREAD_COUNT = 1

####################
# EVALUATION CODE  #
####################


def main():
    agent = MineRLAgent()
    agent.load_agent()

    assert MINERL_MAX_EVALUATION_EPISODES > 0
    assert EVALUATION_THREAD_COUNT > 0

    env = gym.make(MINERL_GYM_ENV)

    # Ensure that videos are closed properly
    @atexit.register
    def cleanup_env():
        env.close()

    # A simple function to evaluate on episodes!
    def evaluate(i, env):
        print("[{}] Starting evaluator.".format(i))
        for i in range(MINERL_MAX_EVALUATION_EPISODES):
            try:
                agent.run_agent_on_episode(Episode(env))
            except EpisodeDone:
                print("[{}] Episode complete".format(i))
                pass

    thread = threading.Thread(target=evaluate, args=(0, env))
    thread.start()
    thread.join()


if __name__ == "__main__":
    main()
