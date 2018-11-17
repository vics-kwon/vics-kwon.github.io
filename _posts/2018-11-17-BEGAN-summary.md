---
title: "BEGAN (2017) Summary"
date: 2018-11-17
tags: 
  - GAN
  - BEGAN
categories: 
  - GAN
---

Berthelot, D., Schumm, T., & Metz, L. (2017). **BEGAN: boundary equilibrium generative adversarial networks**. arXiv preprint arXiv:1703.10717.

## 논문 링크

- [https://arxiv.org/abs/1703.10717](https://arxiv.org/abs/1703.10717)

## Abstract

- Equilibrium enforcing method
    - paired with a loss derived from the Wasserstein distance
    - for training AE based GAN
- Image generation task에 집중.
- Visual quality 측면에서의 milestone.

## Introduction

- GAN은 pixel-wise loss를 이용하여 AE로 생성한 이미지보다 더 sharp한 이미지를 만들어 냄.
- Difficulties
    - Training이 잘 안됨.
    - 올바른 hyper-parameter selection이 중요함.
    - Image diversity를 controll하기 어려움.
    - G와 D의 convergence를 밸런싱하기 어려움 (트레이닝 초반에 D가 너무 쉽게 이겨버림).
- BEGAN
    - (1) Training하는 동안 Generator와 Discriminator의 균형을 조정해주는 equilibrium concept 제시.
    - (2) Fast & stable convergence 가능한 standard training procedure & Robust Arch 제시.
    - (3) Image diversity와 visual quality 간의 trade-off 조절하는 방법 제시.
    - (4) Convergence Measure 제시.

## Related Work 

- Energy Based GAN (EBGAN)
    - D(x)를 energy function으로 modeling
    - D를 Auto-encoder로 구현. per-pixel error 사용.
    - 좀더 stable하게 converge & hyper-parameter variation에 좀더 robust함
    - ⇒ BEGAN에서도 D를 Auto-encoder로 구현.
- Wasserstein GAN (WGAN)
    - convergence measure로 사용할 수 있는 loss 제시
    - slow training, but more stable & better mode converge

## (잠깐..) Discriminator가 Auto-encoder? 

- 입력 이미지 x와 AE를 통과한 이미지 간의 픽셀 별 차이(pixel-wise loss)를 구함.
- Discriminator를 input 이미지가 real일 확률을 계산하는 함수로 보기보다는 energy function으로 봄.
    - data manifold 안에 속하는 데이터는 low energy를 가지고, data manifold 밖에 속하는 데이터는 high energy를 가짐.
- Generator는 Low energy를 가지는 data들을 생성하는 쪽으로 학습이 진행 됨.

  ![]({{site.url}}/assets/img/2018-11-17-imgs/Untitled-a4b40cae-43a8-46c9-bdd5-5d2281ab3421.png)

## BEGAN

- Core Concept
    - **Auto-encoder의 loss distribution를 기준으로 D와 G를 학습하고자 함.**
    - 이때 Wasserstein distance로부터 유도한 loss를 사용.
- Lower Bound to Wasserstein Distance
    - Wassertein distance는 distribution 간의 distance를 의미함. 
      - dist. #1: 진짜 image의 pixel-wise loss $\mathcal{L}(x)$ 의 분포 $\mu _1$
      - dist. #2: G가 생성한 image의 pixel-wise loss $\mathcal{L} (G(z))$ 의 분포 $\mu _2$
    - Jensen's inequality를 이용하면 이 Distance의 Lower bound를 구할 수 있음. 

  ![]({{site.url}}/assets/img/2018-11-17-imgs/Untitled-ec59964c-ef85-4783-9622-fd46f85c7649.png)

- GAN Objective: Maximize lower bound of wasserstein distance

    ![]({{site.url}}/assets/img/2018-11-17-imgs/Untitled-c471c7a3-bb5e-493a-bccb-beb9c33b3e92.png)

    - Lower bound를 Maximize 하는 방법 중에서 (b)를 선택
        - $m_1$을 minimize하는 것 = 진짜 image의 auto-encoder loss $\mathcal{L} (x)$를 줄이는 것과 동일.
    - Discriminator / Generator Loss

    ![]({{site.url}}/assets/img/2018-11-17-imgs/Untitled-d3c21a59-5cae-4422-8f22-2992c5d251ee.png)

    - WGAN과의 차이점
        - WGAN은 sample간의 distribution을 맞추지만, BEGAN에서는 loss 간의 distribution을 맞춤
        - D가 K-Lipschitz 함수(?) 일 필요가 없어짐.

## Equilibrium Concept
  - **G와 D 각각의 loss에 대한 Expectation이 동일하면, equilibrium 상태로 볼 수 있음.**
  - 만약 D가 generated sample과 real one을 구분하지 못한다면, 그들의 error에 대한 distribution은 동일함.
  - diversity ratio ($\gamma$): equilibrium 상태를 조절하는 hyper-parameter
      - Discriminator (Auto-encoder)기준, 두 loss 간의 비율
        - $\mathcal{L} (G(z))$ = G가 생성한 image를 Auto-encoder에 넣었을때 pixel-wise loss
        - $\mathcal{L}(x)$ = 진짜 image를 Auto-encoder에 넣었을때 pixel-wise loss 
      - gamma 값이 작을수록, G가 생성한 image quality는 real에 가까워지고 image diversity는 떨어짐. (vise versa)

  ![]({{site.url}}/assets/img/2018-11-17-imgs/Untitled-0cd605b1-f2cd-490d-af27-d3a3b2b40afd.png)

## Boundary Equilibrium GAN
  - Gradient descent 동안, Equilibrium(diversity ratio)을 유지하기 위해 Proportional Control Theory를 이용함.
  - $\mathcal{L} (G(z))$에 두는 비중을 k 값을 이용하여 Control함 (일종의 learning rate). 여기서 $\lambda$는 k의 proportional gain을 의미함.
  - k가 diversity ratio를 유지하는 일종의 closed-loop feedback control로 볼 수 있음.
  - Training 초반에 G는 AE를 위한 easy-to-reconstruct data를 만들어 냄. $\mathcal{L} (z)$ > $\mathcal{L} (G(z))$
    - 이 두 loss는 equilibrium constraint에 의해 유지됨.
  - 기존의 GAN에서는 D와 G를 번갈아가면서 학습을 하였으나, BEGAN에서는 그럴 필요가 없음.
    - D와 G 각각에 대해 Adam optimizer를 적용하고, 이를 통해 얻은 loss로 $\theta$를 업데이트하면 됨.

![]({{site.url}}/assets/img/2018-11-17-imgs/Untitled-076bfd47-bbf0-427c-9259-d93cc2d6834c.png)

## Convergence measure
  - 기존 GAN의 convergence를 결정하는 일은 어려웠음. zero-sum game. 그래서 epoch 횟수나 직접 image를 보면서 training 정도를 가늠함.
  - equilibrium concept을 이용하면 global convergence measure를 유도할 수 있음.
      - 진짜 이미지에 대한 loss $\mathcal{L} (x)$와 proportion control algorithm의 error 합의 최소화 -> G와 D의 균형 유지를 위해 수렴점을 찾는 것
  - 이 measure를 이용하면, network가 final state에 도달했는지 혹은 model 이 collapse했는지 알 수 있음.

![]({{site.url}}/assets/img/2018-11-17-imgs/Untitled-5387ace1-c472-46c2-a148-cb2845a85e3c.png)

## Model architecture
  - D: Convolutional DNN을 가진 Auto-encoder
      - 기존 GAN trick들을 사용하지 않음 (e.g., batch normalization, dropout, etc.)
      - 각 layer는 두번씩 반복
      - down-sampling 할때마다 convolution filter linearly 증가
  - G: D의 decoder와 동일한 architecture 이용. but, D와 별도로 구성.

![]({{site.url}}/assets/img/2018-11-17-imgs/Untitled-a69d3d3c-02e8-475e-8b07-9a15141b7a8a.png)

## Optional Improvement
  - vanishing residuals을 이용하여 network를 init함
  - gradient propagation을 돕기 위해 hidden state와 upsampling layer 간의 skip connection을 이용함.

## Experimentation 
  - 논문 참조 

## Conclusion
  - D는 auto-encoder여야 하는가?
  - dataset 을 고려할때 최적의 latent space size는 무엇?
  - input에 noise를 넣어주는 시기는? 얼마나 많이?
  - 일반적인 auto-encoder 이외에 VAE를 적용했을 때의 이점은?

## 참고 자료

- [1] 초짜 대학원생의 입장에서 이해하는 BEGAN [(1)](http://jaejunyoo.blogspot.com/2017/04/began-boundary-equilibrium-gan-1.html?m=1) / [(2)](http://jaejunyoo.blogspot.com/2017/04/began-boundary-equilibrium-gan-2.html)
- [2] 초짜 대학원생의 입장에서 이해하는 [EBGAN](http://jaejunyoo.blogspot.com/2018/02/energy-based-generative-adversarial-nets-1.html)
- [3] 기계학습 펀치라인 - [EBGAN](http://blog.soundcorset.kr/2017/07/ebgan-energy-based-genarative.html?m=1)
- [4] 로이Y's Blog - [EBGAN](http://dogfoottech.tistory.com/m/183)