# Probability & Statistics — Five LLM-Stumping Tasks (KaTeX)

**Subdomain:** Mathematics — Probability & Statistics

> Rendering note: prose is Markdown; all mathematics is written in KaTeX and is
> designed to render on a Markdown + KaTeX platform. Each task is **fully specified**,
> **requires genuine domain expertise**, and has **exactly one verifiable answer**.
> Each also carries a well-documented *trap* that language models routinely fall into
> (the common wrong answer is listed with every solution).

---

## Task 1 — Gamma-Mixed Poisson: Variance of an Aggregate Loss

**Prompt.**
An insurance portfolio uses a **mixed compound Poisson** model. The claim count $N$ is
Poisson *conditional* on an unobserved risk parameter $\Lambda$, which is itself random:

$$
N \mid \Lambda = \lambda \sim \operatorname{Poisson}(\lambda), \qquad
\Lambda \sim \operatorname{Gamma}(\alpha = 3,\ \beta = 2)
$$

with rate parametrization $f_\Lambda(\lambda)=\dfrac{\beta^\alpha}{\Gamma(\alpha)}\lambda^{\alpha-1}e^{-\beta\lambda}$.
Claim severities $X_i$ are i.i.d. with $\mathbb{E}[X_i]=500$ and $\operatorname{Var}(X_i)=30000$,
independent of one another and of $N$. Let $S=\sum_{i=1}^{N} X_i$ (with $S=0$ if $N=0$).
Find $\operatorname{Var}(S)$ as a single exact number.

**Answer:** $\boxed{607500}$.

**Solution.** Gamma-mixed Poisson $\Rightarrow$ $N$ is negative binomial, so use the law of total variance:

$$
\mathbb{E}[N]=\mathbb{E}[\Lambda]=\tfrac{\alpha}{\beta}=1.5,\qquad
\operatorname{Var}(N)=\mathbb{E}[\Lambda]+\operatorname{Var}(\Lambda)=\tfrac{\alpha}{\beta}+\tfrac{\alpha}{\beta^{2}}=1.5+0.75=2.25.
$$

Compound-variance identity:

$$
\operatorname{Var}(S)=\mathbb{E}[N]\operatorname{Var}(X)+\operatorname{Var}(N)\,\mathbb{E}[X]^2
=(1.5)(30000)+(2.25)(500)^2=45000+562500=607500.
$$

**Common LLM trap:** using $\operatorname{Var}(N)=\mathbb{E}[N]=1.5$ (pure Poisson),
giving the wrong $1.5\cdot30000+1.5\cdot250000=420000$.

---

## Task 2 — Conditional Expectation of a Uniform Given an Ordering

**Prompt.**
Let $X$ and $Y$ be independent $\operatorname{Uniform}(0,1)$ random variables. Compute

$$
\mathbb{E}\big[\,X \mid X < Y\,\big].
$$

Give a single exact value.

**Answer:** $\boxed{\dfrac{1}{3}}$.

**Solution.** Conditioned on $\{X<Y\}$, $X$ is the **minimum** of two independent
$\operatorname{Uniform}(0,1)$ variables, and $\mathbb{E}[\min(X,Y)]=\tfrac{1}{3}$.
Directly:

$$
\mathbb{E}[X\mid X<Y]=\frac{\displaystyle\int_0^1\!\!\int_x^1 x\,dy\,dx}{\Pr(X<Y)}
=\frac{\int_0^1 x(1-x)\,dx}{1/2}=\frac{1/6}{1/2}=\frac13.
$$

**Common LLM trap:** answering $\tfrac12$ (ignoring the conditioning) or $\tfrac14$
(confusing $\mathbb{E}[X\mid X<Y]$ with $\mathbb{E}[X\mid X<Y]\cdot\Pr$-type quantities).

---

## Task 3 — Expected Waiting Time for the Pattern HTH

**Prompt.**
A fair coin is tossed repeatedly. Let $T$ be the number of tosses until the pattern
**H, T, H** first appears as three consecutive outcomes. Compute $\mathbb{E}[T]$.

**Answer:** $\boxed{10}$.

**Solution.** By the Conway leading-numbers / martingale method, the expected waiting
time for a pattern equals the sum of $2^{k}$ over each length-$k$ prefix that is also a
suffix (an overlap) of the pattern. For $\text{HTH}$: the whole pattern (length 3) matches
itself $\to 2^{3}$, and the length-1 prefix "H" equals the length-1 suffix "H" $\to 2^{1}$;
the length-2 prefix "HT" $\neq$ suffix "TH" $\to$ no contribution. Hence

$$
\mathbb{E}[T]=2^{3}+2^{1}=8+2=10.
$$

**Common LLM trap:** answering $8=2^3$ (treating all length-3 patterns as equal) — but a
non-self-overlapping pattern like $\text{HTT}$ has $\mathbb{E}[T]=8$, whereas the
self-overlapping $\text{HTH}$ is $10$ and $\text{HHH}$ is $2^3+2^2+2^1=14$.

---

## Task 4 — Posterior Probability from a Uniform Prior (Beta Posterior)

**Prompt.**
A coin has unknown bias $p$ with prior $p \sim \operatorname{Uniform}(0,1)$. It is flipped
$5$ times and lands **heads on all $5$ flips**. Given this data, compute the posterior
probability that the coin is biased toward heads, i.e. $\Pr(p > \tfrac{1}{2} \mid \text{data})$.

**Answer:** $\boxed{\dfrac{63}{64}}$.

**Solution.** With a $\operatorname{Uniform}(0,1)=\operatorname{Beta}(1,1)$ prior and
$5$ heads, $0$ tails, the posterior is $\operatorname{Beta}(6,1)$ with density
$f(p)=6p^{5}$ on $(0,1)$. Therefore

$$
\Pr\!\left(p>\tfrac12 \mid \text{data}\right)=\int_{1/2}^{1} 6p^{5}\,dp
=\Big[p^{6}\Big]_{1/2}^{1}=1-\left(\tfrac12\right)^{6}=1-\tfrac{1}{64}=\tfrac{63}{64}.
$$

**Common LLM trap:** reporting the **rule-of-succession** predictive
$\tfrac{6}{7}$ (that is $\Pr(\text{next flip heads})$, a different question) or guessing
$\tfrac12$.

---

## Task 5 — Maximum of Two Exponentials with Unequal Rates

**Prompt.**
Let $X$ and $Y$ be independent exponential random variables with rates $\lambda_X = 1$
and $\lambda_Y = 2$ (means $1$ and $\tfrac12$). Compute $\mathbb{E}\big[\max(X,Y)\big]$.

**Answer:** $\boxed{\dfrac{7}{6}}$.

**Solution.** Use $\max(X,Y)=X+Y-\min(X,Y)$. For independent exponentials,
$\min(X,Y)\sim\operatorname{Exponential}(\lambda_X+\lambda_Y)$, so
$\mathbb{E}[\min]=\dfrac{1}{\lambda_X+\lambda_Y}=\dfrac{1}{3}$. Then

$$
\mathbb{E}[\max(X,Y)]=\mathbb{E}[X]+\mathbb{E}[Y]-\mathbb{E}[\min(X,Y)]
=1+\tfrac12-\tfrac13=\tfrac{7}{6}.
$$

**Common LLM trap:** answering $\tfrac32=\mathbb{E}[X]+\mathbb{E}[Y]$ (forgetting the
subtraction) or reusing the equal-rate shortcut
$\mathbb{E}[\max]=\tfrac{1}{\lambda}(1+\tfrac12)$, which is only valid when the rates are equal.

---

### Answer Key (single verifiable value each)

| # | Task | Answer |
|---|------|--------|
| 1 | Variance of Gamma-mixed compound loss | $607500$ |
| 2 | $\mathbb{E}[X \mid X<Y]$, $X,Y\sim U(0,1)$ | $\tfrac{1}{3}$ |
| 3 | Expected tosses to see HTH | $10$ |
| 4 | $\Pr(p>\tfrac12\mid 5\text{ heads})$, uniform prior | $\tfrac{63}{64}$ |
| 5 | $\mathbb{E}[\max(X,Y)]$, rates $1,2$ | $\tfrac{7}{6}$ |
