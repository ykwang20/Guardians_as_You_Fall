# GYF: Active Mode Transition for Safe Falling #
Code for training and onboard deployment.
********

### Code Base 

This repository is based on legged_gym https://leggedrobotics.github.io/legged_gym/.  


### Installation ###
1. Create a new python virtual env with python 3.6, 3.7 or 3.8 (3.8 recommended)
2. Install pytorch 1.10 with cuda-11.3:
    - `pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 tensorboard==2.8.0 pybullet==3.2.1 opencv-python==4.5.5.64 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
3. Install Isaac Gym
   - Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym
   - `cd isaacgym/python && pip install -e .`
   - Try running an example `cd examples && python 1080_balls_of_solitude.py`
   - For troubleshooting check docs `isaacgym/docs/index.html`)
4. Install rsl_rl (PPO implementation)
   - Clone https://github.com/leggedrobotics/rsl_rl
   -  `cd rsl_rl && pip install -e .` 
5. Install legged_gym
    - Clone this repository
   - `cd legged_gym && pip install -e .`

### Code Structure ###
1. Each environment is defined by an env file (`*env_name*.py`) and a config file (`*env_name*_config.py`). The config file contains two classes: one conatianing all the environment parameters (`*env_name*Cfg`) and one for the training parameters (`*env_name*CfgPPo`).  
2. Both env and config classes use inheritance.  
3. Each non-zero reward scale specified in `cfg` will add a function with a corresponding name to the list of elements which will be summed to get the total reward.  
4. Tasks must be registered using `task_registry.register(name, EnvClass, EnvConfig, TrainConfig)`. This is done in `envs/__init__.py`, but can also be done from outside of this repository.  



### Framework

![917_overview](https://github.com/wangyiji20/Fall_Recovery_control/assets/75070200/26855baf-42c3-4ab0-bb8b-8311600be72f)




1. Working policy: the standing policy, trained with `go1_stand.py`.
2. Transition controller: the policy for falling control, trained with `curr.py`, which provides mixed terrains, mixed tasks and training curriculum.
3. recovery controller, the policy for convert the quadrupedal from reversed mode to regular mode, trained with `go1_fall.py`.
4. Planner: trained with `selector.py`, which is trained to select the appropriate low-level policy. Trained from demonstration.

The rl algorithm of each environment is defined in the corresponding `_config.py` files.

### Usage ###
1. Train:  
    ```python issacgym_anymal/scripts/train.py --task=back --headless```
    -  To run on CPU add following arguments: `--sim_device=cpu`, `--rl_device=cpu` (sim on CPU and rl on GPU is possible).
    -  To run headless (no rendering) add `--headless`.
    - **Important**: To improve performance, once the training starts press `v` to stop the rendering. You can then enable it later to check the progress.
    - The trained policy is saved in `./logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`. Where `<experiment_name>` and `<run_name>` are defined in the train config.
    -  The following command line arguments override the values set in the config files:
     - --task TASK: Task name.
     - --resume:   Resume training from a checkpoint
     - --experiment_name EXPERIMENT_NAME: Name of the experiment to run or load.
     - --run_name RUN_NAME:  Name of the run.
     - --load_run LOAD_RUN:   Name of the run to load when resume=True. If -1: will load the last run.
     - --checkpoint CHECKPOINT:  Saved model checkpoint number. If -1: will load the last checkpoint.
     - --num_envs NUM_ENVS:  Number of environments to create.
     - --seed SEED:  Random seed.
     - --max_iterations MAX_ITERATIONS:  Maximum number of training iterations.
2. Play a trained policy:  
   ```python ./legged_gym/scripts/play.py --task=back```
    - By default the loaded policy is the last model of the last run of the experiment folder.
    - Other runs/model iteration can be selected by setting `load_run` and `checkpoint` in the train config.

### Adding a new environment ###
The base environment `legged_robot` implements a rough terrain locomotion task. The corresponding cfg does not specify a robot asset (URDF/ MJCF) and no reward scales. 

1. Add a new folder to `envs/` with `'<your_env>_config.py`, which inherit from an existing environment cfgs  
2. If adding a new robot:
    - Add the corresponding assets to `resourses/`.
    - In `cfg` set the asset path, define body names, default_joint_positions and PD gains. Specify the desired `train_cfg` and the name of the environment (python class).
    - In `train_cfg` set `experiment_name` and `run_name`
3. (If needed) implement your environment in <your_env>.py, inherit from an existing environment, overwrite the desired functions and/or add your reward functions.
4. Register your env in `isaacgym_anymal/envs/__init__.py`.
5. Modify/Tune other parameters in your `cfg`, `cfg_train` as needed. To remove a reward set its scale to zero. Do not modify parameters of other envs!


### Troubleshooting ###
1. If you get the following error: `ImportError: libpython3.8m.so.1.0: cannot open shared object file: No such file or directory`, do: `sudo apt install libpython3.8`

### Known Issues ###

1. The function `gym.set_*_tensor()` or `gym.set_*_tensor_indexed()` can take effect only once in a simulation step. If multiple times of reset of joints or root states is needed, a solution is to record the indices of envs need to be reset in this step, reset all the environments recorded at the end of each step.

2. The contact forces reported by `net_contact_force_tensor` are unreliable when simulating on GPU with a triangle mesh terrain. A workaround is to use force sensors, but the force are propagated through the sensors of consecutive bodies resulting in an undesireable behaviour. However, for a legged robot it is possible to add sensors to the feet/end effector only and get the expected results. When using the force sensors make sure to exclude gravity from trhe reported forces with `sensor_options.enable_forward_dynamics_forces`. Example:

```
    sensor_pose = gymapi.Transform()
    for name in feet_names:
        sensor_options = gymapi.ForceSensorProperties()
        sensor_options.enable_forward_dynamics_forces = False # for example gravity
        sensor_options.enable_constraint_solver_forces = True # for example contacts
        sensor_options.use_world_frame = True # report forces in world frame (easier to get vertical components)
        index = self.gym.find_asset_rigid_body_index(robot_asset, name)
        self.gym.create_asset_force_sensor(robot_asset, index, sensor_pose, sensor_options)
    (...)

    sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
    self.gym.refresh_force_sensor_tensor(self.sim)
    force_sensor_readings = gymtorch.wrap_tensor(sensor_tensor)
    self.sensor_forces = force_sensor_readings.view(self.num_envs, 4, 6)[..., :3]
    (...)

    self.gym.refresh_force_sensor_tensor(self.sim)
    contact = self.sensor_forces[:, :, 2] > 1.
```
*********
## Guidelines for Onboard Deployment
### Connect user PC to onboard mini PC
1. Set the user's PC IP to: 192.168.123.xxx, so that the user's PC and the onboard PC can be integrated into the same local network.
2. Connect to one of the mini PC via ethernet or ssh. The recommendation is the Jetson NX: 192.168.123.15

### Configure the Libtorch
The main tools being used are libtorch(C++ version of Pytorch) and other dependencies listed in https://github.com/unitreerobotics/unitree_legged_sdk. Ideally, the required environment is pre-installed. Check the version of pre-installed Pytorch. If the version is >=1.8.0, skip to **Deploy a Custom Model**. Otherwise, you need to manully install Libtorch on the mini PC. There are several choices:
1. Download the source code of Libtorch to the mini PC and compile onboard. *Downloading the official pre-built version is not feasible because all mini PCs are based on the ARM architecture*.
2. Download the pre-built PyTorch pip wheel installers for Jetson.https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-11-now-available/72048
3. Build a docker image containing the required libtorch and other dependencies, and install the docker on the mini PC. Could use the docker provided by https://github.com/Improbable-AI/walk-these-ways.git directly. Then edit the `Dockerfile` to mount the directory 'unitree_legged_sdk` into docker, as follows:
```
RUN useradd -m $USERNAME && \
        echo "$USERNAME:$USERNAME" | chpasswd && \
        usermod --shell /bin/bash $USERNAME && \
        usermod -aG sudo $USERNAME && \
        mkdir /etc/sudoers.d && \
        echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME && \
        chmod 0440 /etc/sudoers.d/$USERNAME && \
        # Add this line
        # Replace 1000 with your user/group id
        usermod  --uid 1000 $USERNAME && \
        groupmod --gid 1000 $USERNAME
```
Then enter the docker by `cd ./go1_gym_deploy/docker && sudo make run` .

### Deploy a Custom Model
1. Export a TorchScript model of your custom model. `play.py` could accomplish this by setting `EXPORT_POLICY = True`.
2. Write your source code `source.cpp` to receive sensor data （filter the data if needed), inference, and perform action. An example can be found here.
3. Edit the `CMakeLists.txt`:
```
cmake_minimum_required(VERSION 2.8.3)
project(unitree_legged_sdk)

find_package(Torch REQUIRED) #Add this line

# check arch and os
message("-- CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "x86_64.*")
  set(ARCH amd64)
endif()
if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "aarch64.*")
  set(ARCH arm64)
endif()

include_directories(include)
link_directories(lib/cpp/${ARCH})

option(PYTHON_BUILD "build python wrapper" OFF)
if(PYTHON_BUILD)
  add_subdirectory(python_wrapper)
endif()

set(EXTRA_LIBS -pthread libunitree_legged_sdk.a)
set(CMAKE_CXX_FLAGS "-O3 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 14)

add_executable(executable_name path/to/source.cpp)
target_link_libraries(executable_name ${TORCH_LIBRARIES} ${EXTRA_LIBS})     # to configure your source code
```

