import logging
import os

import numpy as np
import aicrowd_helper
import gym
import minerl
from utility.parser import Parser
from basalt_baselines.bc import bc_baseline
import cv2

import coloredlogs
coloredlogs.install(logging.DEBUG)


# You need to ensure that your submission is trained by launching less than MINERL_TRAINING_MAX_INSTANCES instances
MINERL_TRAINING_MAX_INSTANCES = int(os.getenv('MINERL_TRAINING_MAX_INSTANCES', 5))
# The dataset is available in data/ directory from repository root.
MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')
# You need to ensure that your submission is trained within allowed training time.
MINERL_TRAINING_TIMEOUT = int(os.getenv('MINERL_TRAINING_TIMEOUT_MINUTES', 4 * 24 * 60))
# You need to ensure that your submission is trained by launching less than MINERL_TRAINING_MAX_INSTANCES instances
MINERL_TRAINING_MAX_INSTANCES = int(os.getenv('MINERL_TRAINING_MAX_INSTANCES', 5))

BASALT_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLBasaltFindCave-v0')


# Optional: You can view best effort status of your instances with the help of parser.py
# This will give you current state like number of steps completed, instances launched and so on.
# Make your you keep a tap on the numbers to avoid breaching any limits.
parser = Parser(
    'performance/',
    maximum_instances=MINERL_TRAINING_MAX_INSTANCES,
    raise_on_error=False,
    no_entry_poll_timeout=600,
    submission_timeout=MINERL_TRAINING_TIMEOUT * 60,
    initial_poll_timeout=600
)


def play_video(arr):
    # Check if camera opened successfully
    for frame in arr:
        # Display the resulting frame
        cv2.imshow('Frame', frame)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    # Closes all the frames
    cv2.destroyAllWindows()

def basic_train():
    """
    This function will be called for training phase.
    This should produce and save same files you upload during your submission.
    """
    aicrowd_helper.training_start()
    # How to sample minerl data is document here:
    # http://minerl.io/docs/tutorials/data_sampling.html
    data = minerl.data.make('MineRLBasaltFindCave-v0', data_dir=MINERL_DATA_ROOT)

    # Sample code for illustration, add your training code below
    env = gym.make('MineRLBasaltFindCave-v0')

    # For an example, lets just run one episode of MineRL for training
    _ = env.reset()
    done = False
    video_frames = []
    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
        video_frames.append(obs['pov'])
        # Do your training here

        # To get better view in your training phase, it is suggested
        # to register progress continuously, example when 54% completed
        # aicrowd_helper.register_progress(0.54)

        # To fetch latest information from instance manager, you can run below when you want to know the state
        #>> parser.update_information()
        #>> print(parser.payload)
    video_array = np.array(video_frames, dtype=np.uint8)
    play_video(video_array)
    should_proceed = input(f"You will be asked to watch this video of size {len(video_frames)}.Press Y to play it?")
    should_proceed = input(f"Watch this video of size {len(video_frames)}. Should we continue (Y/N)?")
    if should_proceed != "Y":
        raise ValueError
    # Save trained model to train/ directory
    # For a demonstration, we save some dummy data.
    np.save("./train/parameters.npy", np.random.random((10,)))

    # Training 100% Completed
    aicrowd_helper.register_progress(1)
    env.close()


def main():
    basic_train()
    # Documentation for BC Baseline can be found in train_bc.py
    # TODO make this configurable once we have multiple baselines

if __name__ == "__main__":
    main()
