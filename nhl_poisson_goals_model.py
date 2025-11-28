"""
NHL Poisson Goals Model
----------------------

Core statistical functions for estimating:
- Expected goals (lambda)
- Goal distributions (Poisson)
- Total-goals distribution (via convolution)
- Over/Under probabilities
- Regulation win probabilities
- Fair American odds (no vig)

This file is meant to be imported into a notebook or script.
"""

import numpy as np
import pandas as pd
from math import exp, factorial
import matplotlib.pyplot as plt


# ============================
#   POISSON & DISTRIBUTIONS
# ============================

def poisson_pmf(k, lam):
    """Probability of scoring k goals given expected goals lam."""
    return (lam ** k) * exp(-lam) / factorial(k)


def build_goal_distribution(lam, max_goals=10):
    """Returns ks (0..max_goals) and probabilities for Poisson(λ)."""
    ks = np.arange(0, max_goals + 1)
    probs = np.array([poisson_pmf(k, lam) for k in ks])
    probs = probs / probs.sum()  # normalize tail
    return ks, probs


def convolve_goal_distributions(probs_home, probs_away):
    """Convolve two Poisson goal PMFs into a total-goals distribution."""
    max_goals = len(probs_home) - 1
    totals = np.arange(0, 2 * max_goals + 1)
    probs_total = np.zeros_like(totals, dtype=float)

    for i, p_i in enumerate(probs_home):
        for j, p_j in enumerate(probs_away):
            probs_total[i + j] += p_i * p_j

    return totals, probs_total / probs_total.sum()


# ============================
#   ODDS & OVER/UNDER
# ============================

def over_under_prob(probs_total, totals, line):
    """Probability of Over/Under a half-goal total like 6.5."""
    p_over = probs_total[totals > line].sum()
    p_under = probs_total[totals < line].sum()
    return p_over, p_under


def prob_to_american_odds(p):
    """Convert a probability to fair American odds (no vig)."""
    if p == 0:
        return None
    if p == 1:
        return None
    if p > 0.5:
        return -round(100 * p / (1 - p))
    return round(100 * (1 - p) / p)


# ============================
#   MAIN MODEL FUNCTIONS
# ============================

def load_standings_csv(path="nhl_team_stats.csv"):
    """
    Loads local NHL standings CSV.
    Must contain:
    team_name, games_played, goals_for, goals_against.
    """
    df = pd.read_csv(path)
    df["gf_per_game"] = df["goals_for"] / df["games_played"]
    df["ga_per_game"] = df["goals_against"] / df["games_played"]
    return df


def estimate_matchup_probs(
    home_team_name,
    away_team_name,
    standings,
    line=6.5,
    home_adv_factor=1.05,
    away_adv_factor=0.95,
    max_goals=10
):
    """
    Compute all matchup probabilities using local standings DataFrame.
    """

    df = standings.copy()

    league_avg_gf = df["gf_per_game"].mean()
    league_avg_ga = df["ga_per_game"].mean()

    df["attack_factor"] = df["gf_per_game"] / league_avg_gf
    df["defense_factor"] = df["ga_per_game"] / league_avg_ga

    home = df.loc[df["team_name"] == home_team_name].iloc[0]
    away = df.loc[df["team_name"] == away_team_name].iloc[0]

    lam_home = league_avg_gf * home["attack_factor"] * away["defense_factor"] * home_adv_factor
    lam_away = league_avg_gf * away["attack_factor"] * home["defense_factor"] * away_adv_factor

    ks_home, probs_home = build_goal_distribution(lam_home, max_goals=max_goals)
    ks_away, probs_away = build_goal_distribution(lam_away, max_goals=max_goals)
    totals, probs_total = convolve_goal_distributions(probs_home, probs_away)

    # Over/Under
    p_over, p_under = over_under_prob(probs_total, totals, line)

    # Regulation win probabilities
    joint = np.outer(probs_home, probs_away)
    p_home_win = np.tril(joint, k=-1).sum()
    p_away_win = np.triu(joint, k=1).sum()
    p_tie = np.trace(joint)

    return {
        "lam_home": lam_home,
        "lam_away": lam_away,
        "p_over": p_over,
        "p_under": p_under,
        "p_home_reg": p_home_win,
        "p_away_reg": p_away_win,
        "p_tie_reg": p_tie,
        "fair_odds_over": prob_to_american_odds(p_over),
        "fair_odds_under": prob_to_american_odds(p_under),
        "fair_odds_home_ml_simple": prob_to_american_odds(p_home_win + 0.5 * p_tie),
        "fair_odds_away_ml_simple": prob_to_american_odds(p_away_win + 0.5 * p_tie),
        "totals": totals,
        "probs_total": probs_total,
        "probs_home": probs_home,
        "probs_away": probs_away,
        "ks_home": ks_home,
        "ks_away": ks_away,
    }


# ============================
#   SUMMARY & PLOTTING
# ============================

def summarize_matchup(home_team_name, away_team_name, standings, line=6.5):
    """
    Print a clean summary + show total-goals distribution plot.
    """

    res = estimate_matchup_probs(
        home_team_name,
        away_team_name,
        standings,
        line=line
    )

    print(f"\nMatchup: {home_team_name} (home) vs {away_team_name} (away)\n")

    print("Expected goals:")
    print(f"  {home_team_name}: {res['lam_home']:.3f}")
    print(f"  {away_team_name}: {res['lam_away']:.3f}\n")

    print(f"Over/Under {line}:")
    print(f"  P(Over)  = {res['p_over']:.3f}")
    print(f"  P(Under) = {res['p_under']:.3f}")
    print(f"  Fair odds → Over {res['fair_odds_over']}, Under {res['fair_odds_under']}\n")

    print("Regulation win probabilities:")
    print(f"  {home_team_name}: {res['p_home_reg']:.3f}")
    print(f"  {away_team_name}: {res['p_away_reg']:.3f}")
    print(f"  Tie: {res['p_tie_reg']:.3f}\n")

    print("Moneyline-style fair odds (splitting ties):")
    print(f"  {home_team_name}: {res['fair_odds_home_ml_simple']}")
    print(f"  {away_team_name}: {res['fair_odds_away_ml_simple']}\n")

    # Total-goals distribution plot
    totals = res["totals"]
    probs_total = res["probs_total"]

    plt.figure(figsize=(8, 5))
    plt.bar(totals, probs_total)
    plt.xlabel("Total Goals")
    plt.ylabel("Probability")
    plt.title(f"Total Goals Distribution: {home_team_name} vs {away_team_name}")
    plt.grid(True)
    plt.show()


# ============================
#   OPTIONAL EXAMPLE USAGE
# ============================

if __name__ == "__main__":
    print("Loading standings CSV...")
    standings = load_standings_csv("nhl_team_stats.csv")

    summarize_matchup(
        "New York Rangers",
        "New Jersey Devils",
        standings,
        line=6.5
    )
