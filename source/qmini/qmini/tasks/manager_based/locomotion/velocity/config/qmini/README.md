# Qmini-Velocity-Flat-v0

Qmini 바이페드 로봇의 **평지 속도 추적(Velocity Tracking)** 태스크입니다.
주어진 속도 명령(선속도, 각속도)을 추적하며 안정적으로 걷는 것을 목표로 합니다.

## 환경 개요

| 항목 | 값 |
|------|-----|
| 환경 ID | `Qmini-Velocity-Flat-v0` (학습), `Qmini-Velocity-Flat-Play-v0` (평가) |
| 환경 타입 | Manager-based (`ContactVisRLEnv`) |
| RL 프레임워크 | RSL-RL (PPO) |
| 시뮬레이션 dt | 0.005s |
| Decimation | 4 (제어 주기 = 0.02s, 50Hz) |
| 에피소드 길이 | 20초 |
| 기본 환경 수 | 4096 |

## 로봇

Qmini 바이페드 (10 DOF, 총 질량 ~9.6kg)

| 관절 | Joint 이름 | Effort 한계 |
|------|-----------|------------|
| Hip Yaw | `LL_joint1`, `RL_joint1` | 20N |
| Hip Roll | `LL_joint2`, `RL_joint2` | 60N |
| Hip Pitch | `LL_joint3`, `RL_joint3` | 20N |
| Knee | `LL_joint4`, `RL_joint4` | 20N |
| Ankle | `LL_joint5`, `RL_joint5` | 20N |

## Action (10 dim)

**Joint Position Control** — 각 관절의 위치 목표를 출력합니다.

```
target = default_joint_pos + action * scale(0.5)
```

PD 컨트롤러(Implicit Actuator)가 위치 목표를 토크로 변환하여 로봇을 구동합니다.

## Observation (42 dim)

| 항목 | 차원 | 노이즈 |
|------|------|--------|
| Base Linear Velocity | 3 | Uniform ±0.1 |
| Base Angular Velocity | 3 | Uniform ±0.2 |
| Projected Gravity | 3 | Uniform ±0.05 |
| Velocity Commands (vx, vy, wz) | 3 | - |
| Joint Position (relative) | 10 | Uniform ±0.01 |
| Joint Velocity (relative) | 10 | Uniform ±1.5 |
| Last Action | 10 | - |

학습 시 observation에 노이즈가 추가되며(`enable_corruption = True`), 평가 시에는 비활성화됩니다.

## Velocity Command

속도 명령은 일정 주기(10초)마다 균일 분포에서 리샘플링됩니다.

| 항목 | 범위 |
|------|------|
| Linear velocity X | 0.0 ~ 1.0 m/s |
| Linear velocity Y | 0.0 m/s (고정) |
| Angular velocity Z | -1.0 ~ 1.0 rad/s |
| Heading | -π ~ π |

## Reward

| 항목 | Weight | 설명 |
|------|--------|------|
| `track_lin_vel_xy_exp` | +1.0 | 선속도 추적 (yaw-frame, exponential) |
| `track_ang_vel_z_exp` | +1.0 | 각속도 추적 (exponential) |
| `feet_air_time` | +0.25 | 발 공중 시간 (바이페드 보행 유도) |
| `feet_slide` | -0.25 | 발 미끄러짐 패널티 |
| `termination_penalty` | -200.0 | 종료(넘어짐) 패널티 |
| `flat_orientation_l2` | -1.0 | 몸체 수평 유지 |
| `dof_torques_l2` | -1e-5 | 관절 토크 패널티 |
| `dof_acc_l2` | -1.25e-7 | 관절 가속도 패널티 |
| `action_rate_l2` | -0.005 | 액션 변화율 패널티 |
| `joint_deviation_hip` | -0.2 | Hip yaw/roll 편차 패널티 |
| `dof_pos_limits` | -1.0 | Ankle 관절 한계 패널티 |

## Termination

| 조건 | 설명 |
|------|------|
| `time_out` | 에피소드 시간(20초) 초과 |
| `base_contact` | base_link 접촉 힘 > 1.0N (넘어짐 감지) |

## Domain Randomization (Event)

| 항목 | 시점 | 내용 |
|------|------|------|
| Physics Material | startup | 마찰 계수 고정 (static 0.8, dynamic 0.6) |
| Base Mass | startup | base_link 질량 ±1.0kg 랜덤 |
| External Force/Torque | reset | base_link 외력 (현재 0으로 비활성) |
| Base Reset | reset | 위치 ±0.5m, yaw ±π 랜덤 |
| Joint Reset | reset | 기본 관절 위치로 리셋 |

## Contact Visualization

GUI 모드에서 실행 시, `ContactVisRLEnv`가 base_link의 접촉 상태를 시각화합니다.
- 접촉 발생 시: base_link 메시가 **빨간색**으로 변경
- 접촉 해제 시: 기본 색상(회색)으로 복원

## 학습 실행

```bash
# 단일 GPU
python scripts/rsl_rl/train.py --task Qmini-Velocity-Flat-v0 --headless

# 멀티 GPU (예: 2장)
python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 \
  scripts/rsl_rl/train.py --task Qmini-Velocity-Flat-v0 --distributed --headless
```

## 평가 실행

```bash
python scripts/rsl_rl/play.py --task Qmini-Velocity-Flat-Play-v0
```

## PPO 하이퍼파라미터

| 항목 | 값 |
|------|-----|
| Network | MLP [128, 128, 128] (ELU) |
| Steps per env | 24 |
| Mini-batches | 4 |
| Learning epochs | 5 |
| Learning rate | 1e-3 (adaptive) |
| Clip param | 0.2 |
| Entropy coef | 0.005 |
| Gamma | 0.99 |
| Lambda (GAE) | 0.95 |
| Max iterations | 1500 |

## 파일 구조

```
velocity/config/qmini/
├── __init__.py              # 환경 등록 (gym.register)
├── flat_env_cfg.py          # 환경 설정 (Scene, MDP, Reward 등)
├── contact_vis_env.py       # 접촉 시각화 커스텀 환경
└── agents/
    └── rsl_rl_ppo_cfg.py    # RSL-RL PPO 학습 설정
```
