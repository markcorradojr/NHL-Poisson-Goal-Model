# NHL Poisson Goal Prediction Model

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Model](https://img.shields.io/badge/Model-Poisson-lightgrey)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

This repository contains a complete end-to-end statistical model for predicting NHL scoring outcomes using Poisson distributions.  
It is designed to demonstrate quantitative reasoning, model construction, clean code structure, and applied statistics — all highly relevant to sportsbook analytics, data science, and quantitative research roles.

## Table of Contents
1. [Project Summary](#project-summary)
2. [Repository Structure](#repository-structure)
3. [Model Overview](#model-overview)
4. [Mathematical Framework](#mathematical-framework)
5. [Example Output](#example-output)
6. [How to Run the Model](#how-to-run-the-model)
7. [Strengths of the Approach](#strengths-of-the-approach)
8. [Limitations](#limitations)
9. [Future Work](#future-work)
10. [Use Cases](#use-cases)

## Project Summary

This project implements a Poisson-based model to forecast NHL goal distributions and fair betting odds.  
The model:

- Computes expected goals (λ) for each team based on offensive and defensive strength  
- Builds Poisson goal distributions for each team  
- Convolves them to generate a full total-goals probability distribution  
- Estimates Over/Under probabilities for any total line  
- Computes regulation win probabilities using joint PMFs  
- Converts probabilities into no-vig American odds  
- Produces interpretable predictions suitable for betting analytics or research

All logic is explained in a 3+ page technical PDF for employer review.

## Repository Structure

| File | Description |
|------|-------------|
| **NHL Poisson Goal Model – Full Technical Report.pdf** | Full technical write-up (3–4 pages). |
| **nhl_poisson_goals_model.py** | Core Python implementation. |
| **nhl_team_stats.xlsx** | Team scoring inputs (GF/G, GA/G). |

## Model Overview

1. Load team scoring data (GF/G, GA/G).  
2. Compute attack and defense strength relative to league averages.  
3. Derive expected goals using strength factors and home-ice adjustments.  
4. Build Poisson goal distributions.  
5. Convolve distributions for total-goals probabilities.  
6. Compute Over/Under probabilities.  
7. Compute regulation win/tie probabilities via joint PMFs.  
8. Convert to fair American odds.

## Mathematical Framework

### Expected Goals
- λ_home = league_avg_GF × attack_home × defense_away × 1.05  
- λ_away = league_avg_GF × attack_away × defense_home × 0.95  

### Poisson Distribution
P(k goals) = (λ^k * e^(−λ)) / k!

### Regulation Win Probabilities
Derived via joint PMF.

### Fair Odds Conversion
Standard no-vig American odds formulas.

## Example Output

**Expected goals:**  
Rangers: 3.952  
Devils: 2.930  

**Over/Under 6.5:**  
Over probability: 0.531  
Under probability: 0.469  

## How to Run the Model

### Requirements
- Python 3.8+  
- numpy  
- pandas  
- matplotlib  

### Example

```
from nhl_poisson_goals_model import estimate_matchup_probs
estimate_matchup_probs("New York Rangers", "New Jersey Devils", line=6.5)
```

## Strengths of the Approach

- Transparent and interpretable  
- Full distribution modeling  
- No-vig odds calculations  
- Modular structure  
- Demonstrates analytics and statistical reasoning

## Limitations

- Poisson independence assumption  
- No player-level adjustments  
- No rest/travel effects  
- Deterministic strengths (not fit by ML)

## Future Work

- Fit strengths using MLE  
- Add Dixon–Coles adjustments  
- Add goalie metrics (GSAA, xG_saved)  
- Add Monte Carlo simulations  
- Add OT/shootout modeling  
- Automate NHL data scraping

## Use Cases

Suitable for:

- Sportsbook analytics  
- Data analyst or data science roles  
- Quantitative research  
- Risk/forecasting teams  

Demonstrates Python, probability modeling, and clear communication.
