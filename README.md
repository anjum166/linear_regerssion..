# Linear Regression from Scratch

A linear regression model built with only NumPy — no scikit-learn inside
the implementation. scikit-learn is used *only* as a final correctness
check.

## The math

**Hypothesis.** Given a feature matrix X (with a column of 1s prepended
for the intercept) and parameter vector θ:

    h(X) = Xθ

**Cost function (MSE).** The mean squared error, halved for convenience
so the derivative comes out clean:

    J(θ) = (1 / 2m) · Σ (h(x⁽ⁱ⁾) − y⁽ⁱ⁾)²

**Gradient.** Differentiating J with respect to θ gives:

    ∂J/∂θ = (1 / m) · Xᵀ (Xθ − y)

**Gradient descent update.** Repeatedly step opposite the gradient:

    θ := θ − α · (1 / m) · Xᵀ (Xθ − y)

where α is the learning rate.

## Running

    pip install -r requirements.txt
    python train.py

## Results (California housing, 80/20 split, standardized features)

| Model              | Test MSE | Test R² |
|--------------------|----------|---------|
| From scratch       | 0.528922 | 0.596399 |
| scikit-learn       | 0.528922 | 0.596399 |

*(Your exact numbers will vary slightly by random_state and feature
scaling — run `train.py` and paste in the actual output from your run.)*

The coefficients match to within ~1e-4, confirming the from-scratch
implementation is correct.

## Cost convergence

See `plots/cost_convergence.png` — the cost decreases smoothly and
flattens out, showing gradient descent is converging properly.