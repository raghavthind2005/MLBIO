### Multimodal reasoning with direct image input

Consider a multimodal VQA task with an image $I$, a question $x$, and an answer $y$.

A standard multimodal model directly answers the question conditioned on the image:

$$
y \sim \pi_\theta(\cdot \mid I,x).
$$

The model must extract the relevant visual information from $I$ and use it to solve the task.


### Multimodal reasoning through a caption

Instead of directly answering the question, we first ask the model to convert the image into a textual representation:

$$
c
\sim
\pi_\theta
\left(
\cdot \mid I,x,q_{\mathrm{cap}}
\right),
$$

where $q_{\mathrm{cap}}$ is the caption instruction.

The model then answers the question using only the generated caption:

$$
\tilde y
\sim
\pi_\theta(\cdot \mid c,x).
$$

Since the caption is generated conditioned on the question $x$, it can be viewed as a **question-conditioned textual representation of the image**.

Ideally, replacing the image $I$ with the caption $c$ should preserve the model's reasoning behavior:

$$
\pi(\cdot \mid c,x)
\approx
\pi(\cdot \mid I,x).
$$


### Caption-distortion objective

We measure how much the model's answer distribution changes after replacing the image with the generated caption.

For a caption $c$, define the distortion as

$$D(c)=
D_{\mathrm{KL}}
\left(
\pi_{\theta_{\mathrm{old}}}(\cdot \mid c,x)
\;\middle\|\;
\pi_{\theta_{\mathrm{old}}}(\cdot \mid I,x)
\right).
$$

The goal is to train captions to **preserve downstream reasoning behavior**. Here we use $\theta_{old}$ because the distortion loss is computed only w.r.t the captioning process  $c \sim \pi_\theta \left( \cdot \mid I,x,q_{\mathrm{cap}} \right)$.

Note here we're using reverse KL, but forward KL or JS-divergence can be explored depending on the training and estimation strategy.


### Estimating the distortion

Suppose an answer

$$
\tilde y=(y_1,\ldots,y_T)
$$

is sampled from the caption-conditioned policy:

$$
\tilde y
\sim
\pi_{\theta_{\mathrm{old}}}(\cdot \mid c,x).
$$

The reverse-KL distortion can be estimated as

$$\widehat D(c) = \sum_{j=1}^{T}
\left[ \log \pi_{\theta_{\mathrm{old}}} (y_j \mid c,x,y_{<j}) -
\log
\pi_{\theta_{\mathrm{old}}}
(y_j \mid I,x,y_{<j})
\right].
$$

Intuitively, for the same answer trajectory, we measure how much the token probabilities change when the original image is replaced by the caption. 



Therefore, with the caption loss:

$$ J_{\mathrm{cap}}(\theta) = - \mathbb E
\left[
\operatorname{sg}
\left[
\widehat D(c)
\right]
\right],
$$

where $c \sim \pi_\theta(\cdot \mid I,x,q_{\mathrm{cap}})$ and $\operatorname{sg}[\cdot]$ denotes stop-gradient.

The corresponding policy gradient is

$$\nabla_\theta J_{\mathrm{cap}} = -
\mathbb E
\left[
\operatorname{sg}
\left[
\widehat D(c)
\right]
\nabla_\theta
\log
\pi_\theta
(c \mid I,x,q_{\mathrm{cap}})
\right].
$$

Thus, distortion provides a **caption-level learning signal**. The policy evaluations used to compute $\widehat D(c)$ are treated as fixed, and gradients are applied only to the generated caption.


### Task-success objective

The final goal is still to solve the multimodal reasoning task correctly.

For caption-based reasoning, we define

$$ J_{\mathrm{success}}(\theta) = \mathbb E \left[
R(y)
\right],
$$

where

$$ 
\ y
\sim
\pi_\theta(\cdot \mid I,x),
$$

and $R( y)$ measures task success.


### Final objective

The caption-distortion objective can be combined with task-level RL:

$$ J(\theta) = J_{\mathrm{success}}(\theta)
+
\lambda J_{\mathrm{cap}}(\theta),
$$

where $\lambda \ge 0$ controls the strength of the distortion objective.

Alternatively, $J_{\mathrm{cap}}$ can be used as a pretraining objective before RL training.

The central idea is:

> **A good visual caption is one that preserves the model's downstream reasoning behavior.**

The full-image policy therefore provides a natural supervision signal for learning task-relevant perception without requiring human-written captions.
