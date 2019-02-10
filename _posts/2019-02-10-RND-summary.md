---
title: "Exploration by RND (2018) Summary"
date: 2019-02-10
tags: 
  - reinforcement learning
  - curiosity
  - exploration
  - random network distillation
categories: 
  - rl
---

Burda, Y., Edwards, H., Storkey, A., & Klimov, O. (2018). **Exploration by random network distillation**. arXiv preprint arXiv:1810.12894.

# 논문 링크

- [https://arxiv.org/abs/1810.12894](https://arxiv.org/abs/1810.12894)

# 소개자료

- [https://blog.openai.com/reinforcement-learning-with-prediction-based-rewards/](https://blog.openai.com/reinforcement-learning-with-prediction-based-rewards/)
- [https://github.com/openai/random-network-distillation](https://github.com/openai/random-network-distillation)

# Abstract

- Deep RL을 위한 Exploration bonus 소개함
    - Exploration Bonus는 Fixed randomly initialized NN이 준 Observation들의 Feature를 예측하는 NN의 error를 의미함
- Intrinsic & extrinsic reward를 합치는 방법도 소개함
- Random Network Distillation(RND) bonus는 Hard exploration Atari game(특히 Montezuma's Revenge)에서 Human level보다 좋은 성과를 보임

# Introduction

- Policy의 Expected Return을 최대화하는 RL 알고리즘은 Random Action으로부터 쉽게 Reward를 얻을 수 있는 환경에서 잘 작동함. 그러나, Reward가 Sparse한 환경에서는 실패하는 경우가 많음
- 최근 RL의 접근법은 동일한 환경을 복제하여 parallel 환경으로 구성하고, 이 환경으로부터 방대한 Sample 데이터를 빠르게 얻어서 Agent를 학습시킴. 이에 맞게 Exploration 방법도 parallel 환경에 적합하게 개발되어야 함
- 본 논문에서 제안하는 Exploration bonus는 손쉽게 구현할 수 있고, High-dimensional Obs에도 적용 가능하며, 어떠한 policy optimizaiton algorithm과도 함께 사용 가능함
- 본 논문에서 제안하는 Exploration bonus는 새로운 경험에 대한 Novelty를 수치화하기 위해 과거 경험을 학습한 NN의 Prediction Error를 사용함
    - 여기서의 Prediction problem은 지금의 Obs와 Agent의 Action이 주어졌을 때 Next Obs를 예측하는 문제를 의미함
    - Prediction Error를 최대화하기 위해 Agent가 Stochastic transition을 찾으려는 성향을 가질 수 있어서, 지금의 Obs에 대한 Fixed randomly initialized NN의 Output을 예측하는 NN의 Prediction Error를 사용함.
- Montezuma's Revenge 게임은 대표적인 Exploration game이며 Sparse reward를 제공함. 본 논문에서는 RND Exploration bonus와 Extrinsic Reward를 결합하고 이를 PPO에 접목하여 Level 1을 통과함.

# Method

## Exploration Bonuses

- $r_t = e_t + i_t$
    - $e_t$: extrinsic reward
    - $i_t$: intrinsic reward, exploration bonus associated with the transition at time t
- novel state 방문을 장려하기 위해서는 자주 방분하는 state보다 novel state에서의 $i_t$ 값이 높아야 함
- $i_t$를 (1) Agent의 State 방문 정도를 고려하여 정의하는 방법도 있고, (2) Agent의 transition에 대한 predictin error로 정의하는 방법도 있음

## Random Network Distillation

- 본 연구에서는 Prediction problem이 Randomly generated되는 접근법을 취하며, 두 NNs로 구성됨
    - **NN#1. Fixed & Randomly Initialized 'Target' Network**
        - Prediction problem을 세팅하는 NN
        - Obs를 k차원으로 임베딩 실시
    - **NN#2. 'Predictor' Network**
        - Target network와 동일하게 Obs를 k차원으로 임베딩 실시. Target Network를 그대로 따라하는 것이 본 Predictor Network의 목표
        - Agent가 수집한 데이터로 NN의 결과값이 target network 결과값과의 expected MSE가 최소화되도록 학습 진행
    - Predictor Network는 주어진 state가 이미 학습한 state와 다를수록, 즉 novel state일수록 prediction error가 높아지게 됨

**Sources of prediction errors**

- (1) Amount of training data: 유사한 example이 적을수록 prediction error는 증가하게 됨
- (2) Stochasticity: target function이 stochastic하기에 prediciton error가 높음
- (3) Model misspecification: 필요한 정보가 빠져있거나, model class가 target function의 complexity를 표현하기에 너무 제한적인 경우에 prediction error가 높음
- (4) Learning dynamics: Optimization process가 target function을 근사하는 model class에 대한 predictor를 찾지 못한 경우.
- 의미 해석
    - Factor 1은 Prediction error를 Exploration bonus로 사용할 수 있도록 함
    - Factor 2&3은 agent가 forward dynamics model의 prediction error를 reward로 사용할 경우 문제를 야기함.
    - RND는 Target Network가 Deterministic하고 Predictor Network의 Model-class 안에 있어서 Factor 2&3 영향을 제거 가능함.

**Relation to uncertainty quantification**

Combining intrinsic & extrinsic returns 

Reward & Observation Normalization 


# Experiments

Pure Exploration 

Combining Episodic & Non-episodic Returns 

Discount Factors

Scaling-up Training

Recurrence 

Comparison to Baselines

Qualitative Analysis: Dancing with Skulls 


# Discussion

- 실험을 통해 발견한 사항 중 하나는 Intrinsic rewards에 대한 Stream과 Extrinsic rewards에 대한 Stream을 분리하여 다루는 방법이 더 효과적임
- RND Exploration Bonus는 Local exploration(e.g. 적을 피해야 하는지, 특정 object와 interaction 해야 하는지 등)에 효과적임
- But, Global exploration은 충분히 커버하지 못함. Montezuma's Revenge 게임에서도 Level 1을 깨기 위해서 Key 관리가 중요한데, 이러한 부분들을 RND Exploration Bonus만으로는 커버하지 못함. High-level Exploration이 필요함. 이 부분은 Future work으로 남김

**Terms**

- Stochastic, deterministic function of the input
- Generic, non-generic prediction problem
- Forward, inverse dynamics