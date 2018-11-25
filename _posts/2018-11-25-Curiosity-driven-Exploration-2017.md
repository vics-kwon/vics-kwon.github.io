---
title: "Curiosity-driven Exploration (2017) Summary"
date: 2018-11-25
tags: 
  - rl
  - curiosity
  - exploration
categories: 
  - reinforcement learning
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

- 정리 예정

## Experimental Setup

- 논문 참조

## Experiments

- 논문 참조

## Related Work

- 정리 예정

## Discussion

- 정리 예정