# NeurIPS 2021: MineRL BASALT Behavioral Cloning Baseline 

[![Discord](https://img.shields.io/discord/565639094860775436.svg)](https://discord.gg/BT9uegr)

This repository provides an example of how to integrate a solution based on Behavioral Cloning into the submission kit 
for the 2021 MineRL BASALT competition.

MineRL BASALT is a competition on solving human-judged tasks. The tasks in this competition do not have a pre-defined reward function: the goal is to produce trajectories that are judged by real humans to be effective at solving a given task.

See [the homepage](https://minerl.io/basalt/) of the competition for further details.

**This repository contains**:
*  **Documentation** on how to submit your agent to the leaderboard
*  **The procedure** for Round 1 and Round 2
*  **Starter code** that uses the [imitation](https://github.com/HumanCompatibleAI/imitation/) implementation of Behavioral Cloning to train a simple agent.

**Other Resources**:
- [AICrowd competition page](https://www.aicrowd.com/challenges/neurips-2021-minerl-basalt-competition) - Main registration page & leaderboard.
- [MineRL Documentation](http://minerl.io/docs) - Documentation for the `minerl` package!
- [Imitation Documentation](https://imitation.readthedocs.io/en/latest/) - Documentation for the `imitation` package, which 
trains imitation-based models in the style of Stable Baselines 3 
- [Sacred Documentation](https://sacred.readthedocs.io/en/stable/) - Documentation for the `sacred` package, which is used to 
structure and define configurations for experiments 

# Code Structure 
## basalt_utils 
This section of the repo is structured as a small utility package, containing wrappers, tools, and compatibility wrappers 
that allow us to more easily train on top of BASALT environments. It should be automatically installed by following the 
setup instructions and originating a conda environment from `environment.yml`, but can also be installed manually by navigating 
into the directory and calling `pip install .`
## basalt_baselines 
This section of the repo is where the actual logic of a Behavioral Cloning training procedure is laid out, specifically in 
`basalt_baselines/bc.py`; other baselines are current works in progress.  

The training procedure is structure as a Sacred experiment. The most salient things to know about this are: 
1. Configuration values specified in the method decorated with `@bc_baseline.config` (in this case, `default_config`), 
are automatically made available to any functions decorated with `@bc_baseline.capture`, `@bc_baseline.main`, or 
`bc_baseline.automain`. 
2. If you want to run the testing or training code directly, you can call `basalt_baselines/bc.py with mode='train'`
or `with mode='test'`. You can also experiment with different configuration parameters on the command line by specifying 
a new value of anything defined in the config method (mentioned in (1)). For example, you could call 
`basalt_baselines.bc.py with batch_size=16` 

This BC baseline is meant to be simple and minimal, and, as such, it tries to make the simplest design choices that allow 
it to handle the structure of a Minecraft environment. These include: 
- Extracting only the pixel POV observation, and using a CNN on that observation as the input to the BC model 
- Turning the continuous camera action into discretely chunked left/right and up/down movements, since otherwise 
the scale of the log likelihood for the continuous space swamps the discrete spaces 
- Constructing separate action distributions for each of the actions that make up a joint Minecraft action (which can be
a mix of Discrete and Box), and combining those action distributions into one MultiModalActionDistribution that is used 
to predict the actions being predicted for BC (by sampling independently from each action space's distribution)
- This action distribution architecture works by learning a single latent vector, and then feeding that representation 
to a head mapping it into the parameters required by each action's distribution 
# How to Submit a Model on AICrowd.

In brief: you define your Python environment using Anaconda environment files, and AICrowd system will build a Docker image and run your code using the docker scripts inside the `utility` directory.

You submit pretrained models, the evaluation code and the training code. Training code should produce the same models you upload as part of your submission.

Your evaluation code (`test_submission_code.py`) only needs to control the agent and accomplish the environment's task. The evaluation server will handle recording of videos.

You specify tasks you want to submit agent for with `aicrowd.json` file, `tags` field (see below).

## Setup
1.  **Clone the github repository** or press the "Use this Template" button on GitHub!

    ```
    git clone https://github.com/minerllabs/basalt_competition_baseline_submissions.git
    ```

2. **Install** the Java JDK! **Make sure you have the [JDK 8 installed first](http://minerl.io/docs/tutorials/getting_started.html)!**
    -> Go to http://minerl.io/docs/tutorials/getting_started.html



3. **Specify** your specific submission dependencies (PyTorch, Tensorflow, etc)

    * **Anaconda Environment**. To run this baseline code on your local machine, you will need to
     create an environment with the correct dependencies on your local machine. We recommend `anaconda` for this 
     purpose, and have included an `environment.yml` file specifying necessary dependencies to run our BC baseline. 
     Make sure at least version `4.5.11`  of `anaconda` is installed (By following instructions [here](https://www.anaconda.com/download)). 
     
     Also, if you are not on a machine with NVIDIA drivers that can support `cudatoolkit=10.2`, remove that dependency 
     from the `environment.yml` file before trying to install. Then:
     
      * **Create your new conda environment**
       Use the following command: 
            ```
            conda-env create -f environment.yml
            conda activate basalt
            ```
            
          This will install the `minerl` environment (containing all of the competition environments), as well as 
          dependencies used in the training of the baselines themselves. 

      * **Your code specific dependencies**
        Add your own dependencies to the `environment.yml` file. **Remember to add any additional channels**. PyTorch requires the channel `pytorch`, for example.
        You can also install them locally using
        ```sh
        conda install <your-package>
        ```

    * **Pip Packages** If you need pip packages (not on conda), you can add them to the `environment.yml` file (see the currently populated version):

    * **Apt Packages** If your training procedure or agent depends on specific Debian (Ubuntu, etc.) packages, add them to `apt.txt`.
    


These files are used to construct both the **local and AICrowd docker containers** in which your agent will train. 

If above are too restrictive for defining your environment, see [this Discourse topic for more information](https://discourse.aicrowd.com/t/how-to-specify-runtime-environment-for-your-submission/2274).

### Common Setup Issues 
- Some users reported having issues installing this set of dependencies on Mac, and hit some variant of [this error](https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial). Our 
current belief is that this is a system-level setup issue, and there is not a single solution that works for all Mac OS versions and CUDA versions, 
which is why we do not provide a specific suggested workaround here. 
- If you're running the test code on a machine that doesn't have a native display (like a headless linux server
you're connecting to via SSH), we recommend installing `xvfb` and running code according to the pattern 
of `xvfb-run -a python test_framework.py`). If you hit an error that resembles the following, 
we recommend following the instructions in [this blog post](https://davidsanwald.github.io/2016/11/13/building-tensorflow-with-gpu-support.html) for installing 
CUDA without GL options. 
```
There was an error with Malmo"/"No OpenGL context found"/"Couldn't set pixel formal"
```




## What should my code structure be like ?

Please follow the example structure shared in the starter kit for the code structure.
The different files and directories have following meaning:

```
.
├── aicrowd.json             # Submission meta information like your username
├── apt.txt                  # Packages to be installed inside docker image
├── data                     # The downloaded data, the path to directory is also available as `MINERL_DATA_ROOT` env variable
├── test_submission_code.py  # IMPORTANT: Your testing/inference phase code. NOTE: This is NOT the the entry point for testing phase!
├── train                    # Your trained model MUST be saved inside this directory
├── train_submission_code.py # IMPORTANT: Your training code. Running this should produce the same agent as you upload as part of the agent.
├── test_framework.py        # The entry point for the testing phase, which sets up the environment. Your code DOES NOT go here.
└── utility                  # The utility scripts which provide a smoother experience to you.
    ├── debug_build.sh
    ├── docker_run.sh
    ├── environ.sh
    ├── evaluation_locally.sh
    ├── parser.py
    ├── train_locally.sh
    └── verify_or_download_data.sh
```

Finally, **you must specify an AIcrowd submission JSON in `aicrowd.json` to be scored!** 

The `aicrowd.json` of each submission should contain the following content:

```json
{
  "challenge_id": "neurips-2021-minerl-basalt-competition",
  "authors": ["your-aicrowd-username"],
  "description": "sample description about your awesome agent",
  "tags": "FindCave",
  "license": "MIT",
  "gpu": false
}
```

This JSON is used to map your submission to the said challenge, so please remember to use the correct `challenge_id` as specified above.

You **need to** specify the task of the submission with the `tags` field with one of the following: `{"FindCave", "MakeWaterfall", "CreateVillageAnimalPen", "BuildVillageHouse"}`. You need to create one submission per task to cover all tasks.

Please specify if your code will use a GPU or not for the evaluation of your model. If you specify `true` for the GPU, a **NVIDIA Tesla K80 GPU** will be provided and used for the evaluation.

### Dataset location

You **don't** need to upload the MineRL dataset in submission and it will be provided in online submissions at `MINERL_DATA_ROOT` path, should you need it. For local training and evaluations, you can download it once in your system via `python ./utility/verify_or_download_data.py` or place manually into the `./data/` folder.

## How to submit!

To make a submission, you will have to create a private repository on [https://gitlab.aicrowd.com/](https://gitlab.aicrowd.com/).

You will have to add your SSH Keys to your GitLab account by following the instructions [here](https://docs.gitlab.com/ee/gitlab-basics/create-your-ssh-keys.html).
If you do not have SSH Keys, you will first need to [generate one](https://docs.gitlab.com/ee/ssh/README.html#generating-a-new-ssh-key-pair).

Then you can create a submission by making a _tag push_ to your repository on [https://gitlab.aicrowd.com/](https://gitlab.aicrowd.com/).
**Any tag push (where the tag name begins with "submission-") to your private repository is considered as a submission**  
Then you can add the correct git remote, and finally submit by doing :

```
cd competition_submission_starter_template
# Add AIcrowd git remote endpoint
git remote add aicrowd git@gitlab.aicrowd.com:<YOUR_AICROWD_USER_NAME>/basalt_competition_submission_template.git
git push aicrowd master

# Create a tag for your submission and push
git tag submission-v0.1
git push aicrowd master
git push aicrowd submission-v0.1

# Note : If the contents of your repository (latest commit hash) does not change,
# then pushing a new tag will **not** trigger a new evaluation.
```

You now should be able to see the details of your submission at: `https://gitlab.aicrowd.com/<YOUR_AICROWD_USER_NAME>/basalt_competition_submission_template/issues/`

**Best of Luck** :tada: :tada:

# Ensuring that your code works.

You can perform local training and evaluation using utility scripts shared in this directory. To mimic the online training phase you can run `./utility/train_locally.sh` from the repository root, you can specify `--verbose` for complete logs.

For local evaluation of your code, you can use `./utility/evaluation_locally.sh`, add `--verbose` if you want to view complete logs. **Note** that you do not need to record videos in your code! AICrowd server will handle this. Your code only needs to play the games.

For running/testing your submission in a docker environment (identical to the online submission), you can use `./utility/docker_train_locally.sh` and `./utility/docker_evaluation_locally.sh`. You can also run docker image with bash entrypoint for debugging on the go with the help of `./utility/docker_run.sh`. These scripts respect following parameters:

* `--no-build`: To skip docker image build and use the last build image
* `--nvidia`: To use `nvidia-docker` instead of `docker` which include your nvidia related drivers inside docker image


# Team

The quick-start kit was authored by 
[Anssi Kanervisto](https://github.com/Miffyli) and [Shivam Khandelwal](https://twitter.com/skbly7) with help from [William H. Guss](http://wguss.ml)

The BASALT competition is organized by the following team:

* [Rohin Shah](https://rohinshah.com) (UC Berkeley)
* Cody Wild (UC Berkeley)
* Steven H. Wang (UC Berkeley)
* Neel Alex (UC Berkeley)
* Brandon Houghton (OpenAI and Carnegie Mellon University)
* [William H. Guss]((http://wguss.ml)) (OpenAI and Carnegie Mellon University)
* Sharada Mohanty (AIcrowd)
* Anssi Kanervisto (University of Eastern Finland)
* [Stephanie Milani](https://stephmilani.github.io/) (Carnegie Mellon University)
* Nicholay Topin (Carnegie Mellon University)
* Pieter Abbeel (UC Berkeley)
* Stuart Russell (UC Berkeley)
* Anca Dragan (UC Berkeley)
