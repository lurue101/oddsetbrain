import pandas as pd
from NNInputPreparpator import NNInputPreparator
from test_constants import test_xg_df

def test_fill_missing_player_stats_if_long_available():
    """
    Tests filling behavior if a player has stats in long df, but not in form.
    Here Hasebe (NA in form df) is filled with the values from Hasebe in long df
    """
    mean_stats_form =  pd.DataFrame.from_dict({'shirtnumber': {'almamy touré': 18.0,
  'ansgar knauff': 36.0,
  'christopher lenz': 25.0,
  'djibril sow': 8.0,
  'faride alidou': 11.0,
  'filip kostić': 10.0,
  'jesper lindstrøm': 29.0,
  'kevin trapp': 1.0,
  'kristijan jakić': 6.0,
  'makoto hasebe': 20.0},
 'xg': {'almamy touré': 0.0,
  'ansgar knauff': 0.2,
  'christopher lenz': 0.0,
  'djibril sow': 0.0,
  'faride alidou': 0.0,
  'filip kostić': 0.0,
  'jesper lindstrøm': 0.5,
  'kevin trapp': 0.0,
  'kristijan jakić': 0.0,
  'makoto hasebe': pd.NA},
 'progressive_passes': {'almamy touré': 2.0,
  'ansgar knauff': 0.0,
  'christopher lenz': 3.0,
  'djibril sow': 3.0,
  'faride alidou': 0.0,
  'filip kostić': 4.0,
  'jesper lindstrøm': 0.0,
  'kevin trapp': 0.0,
  'kristijan jakić': 1.0,
  'makoto hasebe': pd.NA}})
    mean_stats_long = pd.DataFrame.from_dict({'shirtnumber': {'almamy touré': 18.0,
  'ansgar knauff': 36.0,
  'christopher lenz': 25.0,
  'djibril sow': 8.0,
  'faride alidou': 11.0,
  'filip kostić': 10.0,
  'jesper lindstrøm': 29.0,
  'kevin trapp': 1.0,
  'kristijan jakić': 6.0,
  'makoto hasebe': 20.0},
 'xg': {'almamy touré': 0.05,
  'ansgar knauff': 0.05000000000000001,
  'christopher lenz': 0.0,
  'djibril sow': 0.016666666666666666,
  'faride alidou': 0.025,
  'filip kostić': 0.0,
  'jesper lindstrøm': 0.25,
  'kevin trapp': 0.0,
  'kristijan jakić': 0.016666666666666666,
  'makoto hasebe': 0.0},
 'progressive_passes': {'almamy touré': 3.0,
  'ansgar knauff': 1.3333333333333333,
  'christopher lenz': 2.8,
  'djibril sow': 6.5,
  'faride alidou': 1.25,
  'filip kostić': 4.0,
  'jesper lindstrøm': 1.1666666666666667,
  'kevin trapp': 0.0,
  'kristijan jakić': 4.166666666666667,
  'makoto hasebe': 1.0}})
    nn_ip = NNInputPreparator(pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),3,5,["xg","progressive_passes"],[])
    mean_stats_form_filled , _ = nn_ip.fill_missing_player_stats(mean_stats_form, mean_stats_long)
    mean_stats_form_expected =  mean_stats_form.copy()
    mean_stats_form_expected.loc["makoto hasebe",:] = mean_stats_long.loc["makoto hasebe",:]
    assert mean_stats_form_expected.equals(mean_stats_form_filled)

def test_filling_player_stats_neither_form_nor_long():
    mean_stats_form = pd.DataFrame.from_dict({'shirtnumber': {'almamy touré': 18.0,
  'ansgar knauff': 36.0,
  'christopher lenz': 25.0},
 'xg': {'almamy touré': 0.05,
  'ansgar knauff': pd.NA,
  'christopher lenz': 0.0},
 'progressive_passes': {'almamy touré': 3.0,
  'ansgar knauff': pd.NA,
  'christopher lenz': 2.8}})
    mean_stats_long = pd.DataFrame.from_dict({'shirtnumber': {'almamy touré': 18.0,
  'ansgar knauff': 36.0,
  'christopher lenz': 25.0},
 'xg': {'almamy touré': 0.4,
  'ansgar knauff': pd.NA,
  'christopher lenz': 0.02},
 'progressive_passes': {'almamy touré': 3.0,
  'ansgar knauff': pd.NA,
  'christopher lenz': 2.3}})
    nn_ip = NNInputPreparator(pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),3,5,["xg","progressive_passes"],[], nan_fill_method="mean")
    mean_stats_form_filled, mean_stats_long_filled = nn_ip.fill_missing_player_stats(mean_stats_form,mean_stats_long)
    mean_stats_form_expected, mean_stats_long_expected = mean_stats_form.copy(), mean_stats_long.copy()
    mean_stats_form_expected.loc["ansgar knauff","xg"] = (0.05 + 0)/2
    mean_stats_form_expected.loc["ansgar knauff", "progressive_passes"] = (3.0 + 2.8) / 2
    mean_stats_form_expected["xg"], mean_stats_form_expected["progressive_passes"] = mean_stats_form_expected["xg"].astype("float64"),mean_stats_form_expected["progressive_passes"].astype("float64")
    mean_stats_long_expected.loc["ansgar knauff","xg"] = (0.4 + 0.02)/2
    mean_stats_long_expected.loc["ansgar knauff","progressive_passes"] = (3.0 + 2.3 ) / 2
    mean_stats_long_expected["xg"], mean_stats_long_expected["progressive_passes"] = mean_stats_long_expected["xg"].astype("float64"),mean_stats_long_expected["progressive_passes"].astype("float64")

    assert mean_stats_form_filled.equals(mean_stats_form_filled, )
    assert mean_stats_long_filled.equals(mean_stats_long_expected)

def test_fill_keeper_stats_player_hasnt_played():
    df_keeper = pd.DataFrame.from_dict({'shirtnumber':[99,77],"player":["der titan","gianluigi buffon"], "gk_psxg":[3,7]})
    nn_ip = NNInputPreparator(pd.DataFrame(),pd.DataFrame(),df_keeper,3,5,[],["gk_psxg"], nan_fill_method="mean")
    mean_stats_form = pd.DataFrame()
    mean_stats_long = pd.DataFrame()
    mean_stats_form_filled, mean_stats_long_filled = nn_ip.fill_missing_keeper_stats(mean_stats_form,mean_stats_long,"kevin trapp")
    mean_stats_form_expected  = pd.DataFrame.from_dict({
                                              'gk_psxg': {'kevin trapp': 5.0}})
    assert mean_stats_form_filled.equals(mean_stats_form_expected)
    assert mean_stats_long_filled.equals(mean_stats_form_expected)

def test_fill_keeper_stats_from_long_into_form():
    df_keeper = pd.DataFrame.from_dict(
        {'shirtnumber': [99, 77], "player": ["der titan", "gianluigi buffon"], "gk_psxg": [3, 7]})
    nn_ip = NNInputPreparator(pd.DataFrame(), pd.DataFrame(), df_keeper, 3, 5, [], ["gk_psxg"], nan_fill_method="mean")
    mean_stats_form = pd.DataFrame.from_dict({
        'gk_psxg': {'kevin trapp': pd.NA}})
    mean_stats_long = pd.DataFrame.from_dict({
        'gk_psxg': {'kevin trapp': 5.0}})
    mean_stats_form_filled, mean_stats_long_filled = nn_ip.fill_missing_keeper_stats(mean_stats_form, mean_stats_long,
                                                                                     "kevin trapp")
    mean_stats_form_expected = pd.DataFrame.from_dict({
        'gk_psxg': {'kevin trapp': 5.0}})
    assert mean_stats_form_filled.equals(mean_stats_form_expected)

def test_calc_mean_xg_as():
    meta_info_dict = {'home_team': 'SC Freiburg',
     'away_team': 'FC Augsburg',
     'outcome': 'h',
     'date': '2018-05-12',
     'season': '2017-2018',
     'gameweek': 34}
    nn = NNInputPreparator(test_xg_df, pd.DataFrame(), pd.DataFrame(), 2, 5, [], [])
    mean_xgs_home,mean_xgas_home,mean_xgs_away,mean_xgas_away = nn.calc_mean_team_xgs(meta_info_dict)
    assert mean_xgs_home == [2.65 ,1.7]
    assert mean_xgas_home == [1.85,1.6]
    assert mean_xgs_away == [0.8, 1.0]
    assert mean_xgas_away == [1.85, 1.36]
