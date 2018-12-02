---
title: "Curiosity-driven Exploration (2017) Summary"
date: 2018-11-25
tags: 
  - reinforcement learning
  - curiosity
  - exploration
categories: 
  - rl
---

Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017, May). **Curiosity-driven exploration by self-supervised prediction.** In International Conference on Machine Learning (ICML) (Vol. 2017).

## 논문 링크

- [https://arxiv.org/pdf/1705.05363.pdf](https://arxiv.org/pdf/1705.05363.pdf)

## Abstract

- Curiosity는 agent가 environment를 탐험(exploration)하기 위한 intrinsic reward signal로 작용 가능.
- 여기서는 curiosity를 agent가 어떤 visual feature space에서 그들의 action에 대한 결과를 예측하는 능력에 대한 error로 formulation 함.
    - 여기서 visual feature space는 self-supervised inverse dynamics model로 학습됨.
- 본 formulation은 image와 같은 high-dimensional continuous state space에도 적용 가능하며, agent에 영향을 미치지 않는 environment를 무시할 수도 있음.
- 본 formulation을 실험해 본 세가지 세팅은 다음과 같음.
    - (1) sparse extrinsic reward
    - (2) exploration with no extrinsic reward
    - (3) generalization to unseen scenarios

## Introduction

- 대부분의 real-world scenario에서는 agent에게 주어지는 extrinsic reward가 상당히 sparse하거나 missing. → reward function을 만들기 어려움.
- Extrinsic rewards가 sparse한 environment에서 Agent가 학습을 해야 한다면, intrinsic motivation이 상당히 중요한 역할을 함.
    - Intrinsic motivation(혹은 curiosity) 는 environment를 탐험(exploration)하고, novel state를 발견하는 원동력이 됨.
- Intrinsic reward는 2가지 종류로 구분해 볼 수 있음.
    - (1) agent가 "novel" state를 탐험하도록 장려
        - "novelty"를 측정하기 위해서는 **environmental state의 distribution에 대한 statistical model**이 필요함.
    - (2) agent가 그들의 action에 대한 결과를 예측하는 능력에 대한 error / uncertainty를 줄일 수 있도록 그들의 action을 선택하도록 장려
        - prediction error를 측정하기 위해서는 t 시점에 주어진 state와 action에 대해 t+1시점에서의 state를 예측할 수 있는 **environmental dynamics model**이 필요함.
- 그러나, Intrinsic reward를 위한 model을 만드는 것은 쉽지 않음.
    - Image와 같은 high-dimensional continuous state space에서는 model을 만들기 어려움.
    - 추가적으로 Agent-environment system 자체의 stochasticity를 다루는 것도 어려움.
        - agent가 작동하는(혹은 구동되는) 과정에서의 noise
        - environment 자체에 내재된 stochasticity
- 하나 가능한 솔루션은 agent가 predict하기 어렵지만 "learnable"하다고 판단되는 state를 마주쳤을 때 reward를 받도록 하는 것.
    - 그럼 learnable한 건 어떻게 판단??
- 본 연구는 agent가 그들의 action에 대한 결과를 예측하기 어려워하는 정도에 따라 Intrinsic reward singal을 생성하는 방법의 일환임.
    - 본 연구의 접근법은 Environment 변화 중에서 agent의 action에 의한 변화 혹은 agent에 영향을 주는 변화만을 예측. 나머지는 무시.
    - HOW? Sensory input (e.g., pixels)을 feature space로 transform 함. 그 feature space는 agent가 수행한 action과 연관된 info와 연관되어 있음.
    - 해당 feature space는 self-supervision 방법으로 학습함.
        - 현재와 next state가 주어졌을 때 agent의 action을 prediction하는 'proxy inverse dynamic task'에 대한 neural network를 학습함.
    - 이 feature space는 forward dynamics model을 학하는데 쓰임.
        - 해당 model은 current state와 action 정보가 주어졌을 때 next state의 feature representation을 예측하는 model임.
    - 이 forward dynamics model의 prediction error를 agent에게 intrinsic reward로 제공함.
        - prediction error가 클수록 curiosity를 장려.
- Curiosity의 주요 역할로는, (1) agent가 새로운 지식을 얻기 위해 환경을 탐험하도록 돕는다는 것, (2) agent가 future scenario에 도움이 될 만한 skill을 배우는 mechanism이라는 것.
- 본 논문에서는 Viz-Doom이라는 환경에서 A3C agent를 대상으로 curiosity 적용 여부에 따른 결과를 비교하여, intrinsic reward의 효과를 보임.
- Super Mario Bros 환경에서는 특정 scenario(e.g., Level1)에서 학습한 exploration policy는 새로운 scenario(e.g., Level2)에서도 agent가 빠르게 환경을 탐험할 수 있도록 돕는다는 것을 보임.
    - 이는 본 논문에서 제안한 방법이 agent로 하여금 generalizable skill을 배울 수 있도록 돕는다고 볼 수 있음.

## Curiosity-driven Exploration

- agent는 두 sub-system으로 구성됨.
    - (1) reward generator
        - ICM, Intrinsic Curiosity Module
        - intrinsic reward signal($r^i_t$) 생성
    - (2) policy
        - reward signal($r_t$)을 최대화할 수 있는 action sequence를 생성
        - 여기서 reward signal($r_t$)은 intrinsic reward($r^i_t$)와 extrinsic reward($r^e_t$)의 sum으로 구성.

![]({{site.url}}/assets/img/2018-11-25-imgs/Untitled-41745d8e-774f-401f-813b-eabd5f26af26.png)

- ICM에 대한 부연 설명
    - raw state $s_t$와 $s_{t+1}$을 feature vector $\phi (s_t)$와 $\phi (s_{t+1})$로 encoding을 함.
        - 이러한 접근법은 image와 같은 high-dimensional continuous state space에도 적용 가능함.
    - Inverse dynamics model($g$)은 두 feature $\phi (s_t)$와 $\phi (s_{t+1})$를 이용하여 $a_t$를 예측함.
        - $\hat{a_t} = g(s_t, s_{t+1}; \theta_I)$
        - $a_t$가 discrete이면, g의 output은 모든 possible action에 대한 soft-max distribution
    - Forward model($f$)은 t시점에서의 feature $\phi (s_t)$와 action $a_t$를 이용하여 t+1시점에서의 feature $\phi (s_{t+1})$을 예측함.
        - $\hat{\phi}(s_{t+1}) = f(\phi(s_t), a_t; \theta_F)$
    - Feature space에서의 prediction error는 intrinsic reward ($r^i_t$)로 사용됨.
        - $r^i_t = \frac{\eta}{2} \lVert \hat{\phi}(s_{t+1}) - \phi(s_{t+1}) \rVert ^2 _2$
    - 결과적으로 Feature $\phi (s_t)$는 agent에 영향을 주거나 혹은 agent의 action에 영향을 받는 부분만 feature로 encoding 됨. → agent의 exploration strategy가 상당히 robust 해짐.
- Overall optimization problem
    - $\beta \in [0, 1]$ and $\lambda > 0$

        $min_{\theta_P, \theta_I, \theta_F} [-\lambda \mathbb{E}_{\pi(s_t;\theta_P)}[\Sigma_t r_t] + (1-\beta)L_I + \beta L_F]$


## Experimental Setup

- Environment
    - Env#1 VizDoom
        - Doom 3D navigation task
            - 'DoomMyWayHome-v0'
            - OpenAI Gym에서 제공
        - Four discrete actions
            - move forward, left, right, no-action
        - (Sparse) Terminal reward
            - Vest를 찾으면, +1
            - 2100 time steps를 넘으면, 0
    - Env#2 Super Mario Bros
        - Re-parameterization action space → 14 unique actions
        - No reward from the game
- Training details
    - trained using visual inputs
    - RGB images → gray-scale & 42X42 size
    - 4개의 연속된 frame을 엮어서 state ($s_t$) 형성
    - VizDoom 환경에서는 4번의 action repeat(?), Mario 환경에서는 6번의 action repeat을 사용.
    - A3C에서의 asynchronous training protocol 적용
        - 20 workers, SGD로 training, worker간 parameter sharing 없이 ADAM optimizer 사용
- A3C architecture
    - 4개의 순차적인 Convolution layers
        - 32 filters, 3X3 kernel size, 2 stride, 1 padding, ELU 사용.
    - 마지막 Conv layer에서 output → LSTM
        - 256 units으로 구성된 LSTM
    - 2개의 분리된 fully connected layers는 LSTM feature representation으로부터 value function과 action을 예측하는 용도로 사용됨.
- ICM architecture
    - Inverse model
        - Step#1 $s_t \rightarrow \text{CNN} \rightarrow \phi(s_t)$
            - CNN: 4 conv layers, 32 filters, kernel size 3x3, 2 stride, 1 padding, ELU activation ftn
            - output dim $\phi(s_t)$ : 288
        - Step#2 concat $\phi(s_t), \phi(s_{t+1})$, denseNet (256, 4)
    - Forward model
        - $\phi(s_t), a_t \rightarrow denseNet(256, 288)$
    - $\beta = 0.2, \lambda = 0.1$
- Baseline Methods
    - (1) ICM + A3C agent
    - (2) vanilla A3C (with $\epsilon$-greedy exploration)
    - (3) ICM-pixels + A3C
        - forward 모델이 next observation을 pixel space에서 예측
    - (4) TRPO-VIME(Variational Information Maximization), Houthoof et al., 2016

## Experiments (정리 예정)

- Sparse Extrinsic Reward Setting
    - Varying the degree of reward sparsity
    - Robustness to uncontrollable dynamics
    - Comparison to TRPO-VIME
- No Reward Setting
    - VizDoom: Coverage during Exploration
    - Mario: Learning to play with no rewards
- Generalization to Novel Scenarios
    - Evaluate 'AS-IS'
    - Fine-tuning with curiosity only
    - Fine-tuning with extrinsic rewards

## Related Work

- RL에서 Curiosity-driven exploration에 대한 연구는 예전부터 지속적으로 수행되어 왔고, 리뷰 논문도 많음.
- Intrinsic rewards를 설정하는 방법이 다양함.
    - Surprise & compression progress
    - Information gain based on entropy of actions
    - Prediction error in the feature space of AE
    - State visitation counts
    - Information gain about the agent's belief of the environment's dynamics
        - 본 연구에서는 semantic feature embedding을 이용했다는 것이 차별점임.
- 최근에는 data efficiency를 향상시키기 위한 연구가 진행되고 있음.
    - pre-training을 통해 supervision을 generating하기도 하고, self-supervised prediction을 이용하기도 함.

## Discussion

- 정리 예정