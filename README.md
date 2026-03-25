# Template for Isaac Lab Projects

## Overview

This project/repository serves as a template for building projects or extensions based on Isaac Lab.
It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.

**Keywords:** extension, template, isaaclab

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda or uv installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/qmini

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        ```

    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/zero_agent.py --task=<TASK_NAME>
            ```
        - Random-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/random_agent.py --task=<TASK_NAME>
            ```
## Tasks

### Qmini-Velocity-Flat-v0 (Velocity Tracking Locomotion)

평지에서 주어진 속도 명령을 추적하며 안정적으로 걷는 태스크입니다.

| 항목 | 값 |
|------|-----|
| 환경 타입 | Manager-based (`ContactVisRLEnv`) |
| RL 프레임워크 | RSL-RL (PPO) |
| 제어 주기 | 50Hz (dt=0.005s, decimation=4) |
| 에피소드 길이 | 20초 |

- **Action** (10 dim): Joint Position Control — `default_pos + action * 0.5`
- **Observation** (42 dim): base velocity(3), base angular velocity(3), projected gravity(3), velocity commands(3), joint pos relative(10), joint vel relative(10), last action(10)
- **Reward**: 선속도/각속도 추적(+1.0), feet air time(+0.25), 종료 패널티(-200.0), 몸체 수평(-1.0), 토크/가속도/액션 변화율 패널티 등
- **Termination**: 시간 초과(20초), base_link 접촉(넘어짐)

```bash
# 학습
python scripts/rsl_rl/train.py --task Qmini-Velocity-Flat-v0 --headless

# 멀티 GPU 학습
python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 \
  scripts/rsl_rl/train.py --task Qmini-Velocity-Flat-v0 --distributed --headless

# 평가
python scripts/rsl_rl/play.py --task Qmini-Velocity-Flat-Play-v0
```

> 상세 설명은 [Task README](source/qmini/qmini/tasks/manager_based/locomotion/velocity/config/qmini/README.md)를 참고하세요.

### Qmini-Locomotion-v0 (Basic Locomotion)

기본적인 바이페드 보행 태스크입니다. 속도 명령 없이 서있기/걷기를 학습합니다.

| 항목 | 값 |
|------|-----|
| 환경 타입 | Manager-based (`ManagerBasedRLEnv`) |
| 제어 주기 | 50Hz (dt=0.005s, decimation=4) |

- **Action** (10 dim): Joint Position Control
- **Observation** (39 dim): base velocity(3), base angular velocity(3), projected gravity(3), joint pos relative(10), joint vel relative(10), last action(10)
- **Reward**: alive(+1.0), 종료 패널티(-5.0), 수직 속도/각속도/토크/가속도/액션 변화율/수평 유지 패널티

```bash
python scripts/rsl_rl/train.py --task Qmini-Locomotion-v0 --headless
```

### Qmini-Locomotion-Direct-v0 (Direct RL)

`Qmini-Locomotion-v0`와 동일한 구성이지만 Direct RL 방식으로 구현되어 있습니다. 추가로 몸체 기울기 > 1.0 rad 시 종료됩니다.

```bash
python scripts/rsl_rl/train.py --task Qmini-Locomotion-Direct-v0 --headless
```

### Template-Qmini-Marl-Direct-v0 (MARL)

Direct MARL 템플릿 태스크입니다. (현재 CartDouble Pendulum 기반 레거시 코드)

```bash
python scripts/rsl_rl/train.py --task Template-Qmini-Marl-Direct-v0 --headless
```

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/qmini/qmini/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/qmini"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```