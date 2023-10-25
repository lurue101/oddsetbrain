import os

import numpy as np
import pandas as pd


def outcome_to_numeric_class(outcome: pd.Series):
    def outcome_if_function(outcome):
        if outcome == "h":
            return 0
        elif outcome == "d":
            return 1
        elif outcome == "a":
            return 2

    y_classes = outcome.apply(lambda outcome: outcome_if_function(outcome))
    return y_classes


def calculate_probs_from_odds(odds_array):
    probs_array = 1 / odds_array
    return np.round(probs_array, decimals=2)


def bet_from_highest_prob_difference(pred_probs, odds_probs):
    prob_diff = pred_probs - odds_probs
    outcome_highest_diff = np.argmax(prob_diff, axis=1)
    return outcome_highest_diff


def bet_from_highest_model_prob(model_pred):
    return np.argmax(model_pred, axis=1)


def place_bet(pred_outcome, outcome, odds, money_placed=1):
    """
    Calculates the net win/loss of a bet. The 0.95 multiplicator is the tax, that needs to be paid
    :param pred_outcome:
    :param outcome:
    :param odds:
    :param money_placed:
    :return:
    """
    if pred_outcome == outcome:
        return (money_placed * odds * 0.95) - money_placed
    else:
        return -money_placed


def concat_analysis_files(include_str, rel_path="analysis") -> pd.DataFrame:
    base_path = os.getcwd()
    dir_path = os.path.join(base_path, rel_path)
    df = pd.DataFrame()
    for file_name in os.listdir(dir_path):
        if include_str not in file_name:
            continue
        full_path = os.path.join(dir_path, file_name)
        df_temp = pd.read_csv(full_path)
        df = pd.concat((df, df_temp))
    return df


def calc_payback_rate(odds):
    return 1 / ((1 / odds[0]) + (1 / odds[1]) + (1 / odds[2]))
