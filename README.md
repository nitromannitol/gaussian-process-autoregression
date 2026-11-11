# gaussian-process-autoregression

Julia code that runs Gaussian process regression on the price of Bitcoin. The script
[`anyOrderFar.jl`](anyOrderFar.jl) generates all the figures of the paper *Gaussian process
regression*, a final project for the course Stats 370 (Bayesian Statistics).

The script reads the daily closing prices in [`dataset/close.json`](dataset/close.json) and
takes their base-10 logarithm. It uses the first four fifths of the series for training and
the last fifth for testing, and regresses each value on the previous `p = 10` values
(an autoregression of order `p`) by a Gaussian process with a squared exponential covariance
`C(x₁, x₂) = h² exp(−‖x₁ − x₂‖² / (2λ²))`. The hyperparameters (the noise level `σ`, and
the output scales `h`, input scales `λ` and linear coefficients `β` for each lag) are fitted
by gradient steps on random subsets of the training points. The script then plots the
predictions on the training and test data, and two sequential predictions. The helper
functions (the gradients of the covariance and of the objective, the covariance matrix and the
autoregression matrix) are in [`anyOrderFunctions.jl`](anyOrderFunctions.jl).

## Running

The code was written in March 2016 for a Julia release before 1.0 and does not run on
current Julia: it uses `srand` ([`anyOrderFar.jl`](anyOrderFar.jl) line 328), which was
removed in Julia 1.0, and the old PyPlot syntax `ax[:set_xlim]`. The packages the code loads
are `JSON`, `DataStructures`, `PyPlot` (matplotlib through PyCall) and `Distributions`.

With a pre-1.0 Julia and these packages installed, run from the repository directory:

```julia
include("anyOrderFar.jl")
```

The figures are saved as EPS files in [`figures/`](figures/); the directory also contains the
PDF conversions of the figures.

## Data

[`dataset/close.json`](dataset/close.json) contains 2040 daily values, from 2010-07-18 to
2016-02-16, under the key `bpi`. The `disclaimer` field of the file states that "This data
was produced from the CoinDesk Bitcoin Price Index. BPI value data returned as USD." The
`time` field records that the file was updated on 2016-02-17.

## Repository layout

| Path | Contents |
|---|---|
| [`anyOrderFar.jl`](anyOrderFar.jl) | data processing, hyperparameter fit, prediction, figures |
| [`anyOrderFunctions.jl`](anyOrderFunctions.jl) | helper functions |
| [`dataset/`](dataset/) | the price data |
| [`figures/`](figures/) | the figures of the paper, as EPS and PDF |

## Authors and citation

The code is by **Ahmed Bou-Rabee**.
