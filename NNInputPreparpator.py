import numpy as np
import pandas as pd
from config import (
    STATS_ABBREVIATIONS,
    META_COLUMNS,
    ODDS_COLUMNS,
    DEFENSIVE_STATS,
    OFFENSIVE_STATS,
)


class NNInputPreparator:
    """
    A class that combines the infos and prepares them for the ml model, such that for every match the stats
    are prepared in the following way:
    For the specified columns (stats), each player in the starting lineup gets a form and a long average,
    such that the model input will be 2 * stats_chosen_players * 20 + 2 * stats_chosen_keepers * 2
    Stats for players and keepers are separate
    """

    def __init__(
        self,
        df_fixtures: pd.DataFrame,
        df_player_stats: pd.DataFrame,
        df_keeper_stats: pd.DataFrame,
        nr_matches_form: int,
        nr_matches_long: int,
        player_stat_column_names: list[str],
        keeper_stat_column_names: list[str],
        nan_fill_method: str = "median",
    ):
        self.df_fixtures = df_fixtures
        self.df_player_stats = df_player_stats
        self.df_keeper_stats = df_keeper_stats
        self.nr_matches_form = nr_matches_form
        self.nr_matches_long = nr_matches_long
        self.player_stat_column_names = player_stat_column_names
        self.keeper_stat_column_names = keeper_stat_column_names
        self.nan_fill_method = nan_fill_method

    def prepare_ml_df(self):
        nn_input_columns = self.create_list_of_stats_columns()
        self.add_def_and_off_summary_stats()
        self.split_lineup_column_into_keeper_and_field_player_columns()
        # if gameweek > nr_matches_form, then there's not enough games played that season to calc the average
        mask_gameweek_big_enough = self.df_fixtures["gameweek"] > self.nr_matches_form
        all_inputs = np.empty(
            shape=(
                len(self.df_fixtures.loc[mask_gameweek_big_enough, :]),
                len(nn_input_columns),
            )
        )
        for row_idx, fixture in (
            self.df_fixtures.loc[mask_gameweek_big_enough, :].reset_index().iterrows()
        ):
            meta_info_dict = fixture[
                fixture.index.isin(META_COLUMNS + ODDS_COLUMNS)
            ].to_dict()
            # Read Lineup
            (
                keeper_home,
                players_home,
                keeper_away,
                players_away,
            ) = self.check_and_retrieve_lineup(fixture)
            (
                mean_xgs_team_home,
                mean_xgas_team_home,
                mean_xgs_team_away,
                mean_xgas_team_away
            ) = self.calc_mean_team_xgs(meta_info_dict)
            # For each player get form and long average
            mean_player_stats_home = self.calc_mean_player_stats(
                players_home, meta_info_dict
            )
            mean_player_stats_away = self.calc_mean_player_stats(
                players_home, meta_info_dict
            )
            mean_keeper_stats_home = self.calc_mean_keeper_stats(
                keeper_home, meta_info_dict
            )
            mean_keeper_stats_away = self.calc_mean_keeper_stats(
                keeper_away, meta_info_dict
            )
            all_stats = np.concatenate(
                [
                    mean_xgs_team_home,
                    mean_xgas_team_home,
                    mean_player_stats_home,
                    mean_keeper_stats_home,
                    mean_xgs_team_away,
                    mean_xgas_team_away,
                    mean_player_stats_away,
                    mean_keeper_stats_away,
                ]
            )
            all_inputs[row_idx, :] = all_stats
        df_input_and_meta_info = pd.DataFrame(all_inputs, columns=nn_input_columns)
        df_input_and_meta_info.loc[
            :, META_COLUMNS + ODDS_COLUMNS
        ] = self.df_fixtures.loc[
            mask_gameweek_big_enough, META_COLUMNS + ODDS_COLUMNS
        ].reset_index()
        return df_input_and_meta_info

    def calc_mean_player_stats(self, player_lineup, meta_info_dict: dict):
        mask_games_form, mask_games_long = self.get_relevant_games_masks(
            meta_info_dict, "player"
        )
        df_players = self.df_with_split_shirtnumber_player_from_lineup_tupe(
            player_lineup
        )
        df_players_form = df_players.merge(
            self.df_player_stats.loc[mask_games_form, :],
            how="left",
            on=["shirtnumber", "player"],
        )
        df_players_long = df_players.merge(
            self.df_player_stats.loc[mask_games_long, :],
            how="left",
            on=["shirtnumber", "player"],
        )
        mean_stats_form = df_players_form.groupby("player").mean(numeric_only=True)
        mean_stats_long = df_players_long.groupby("player").mean(numeric_only=True)
        mean_stats_form, mean_stats_long = self.fill_missing_player_stats(
            mean_stats_form, mean_stats_long
        )

        mean_stats = np.concatenate(
            [
                mean_stats_form[self.player_stat_column_names].stack().values,
                mean_stats_long[self.player_stat_column_names].stack().values,
            ]
        )
        return mean_stats

    def get_relevant_games_masks(self, meta_info_dict: dict, player_or_keeper: str):
        if player_or_keeper == "player":
            df = self.df_player_stats
        elif player_or_keeper == "keeper":
            df = self.df_keeper_stats
        else:
            raise ValueError("please input player or keeper only")
        mask_season = df["season"] == meta_info_dict["season"]
        mask_max_week = df["gameweek"] < meta_info_dict["gameweek"]
        mask_min_week_form = (
            df["gameweek"] >= meta_info_dict["gameweek"] - self.nr_matches_form
        )
        mask_min_week_long = (
            df["gameweek"] >= meta_info_dict["gameweek"] - self.nr_matches_form
        )
        mask_relevant_games_form = mask_max_week * mask_season * mask_min_week_form
        mask_relevant_games_long = mask_max_week * mask_season * mask_min_week_long
        return mask_relevant_games_form, mask_relevant_games_long

    def df_with_split_shirtnumber_player_from_lineup_tupe(self, player_tuple):
        shirt_numbers, player_names = map(list, zip(*player_tuple))
        df_players = pd.DataFrame.from_dict(
            {"shirtnumber": shirt_numbers, "player": player_names}
        )
        df_players.shirtnumber = df_players.shirtnumber.astype("int64")
        return df_players

    def split_lineup_column_into_keeper_and_field_player_columns(self):
        self.df_fixtures.loc[
            :, ["keeper_starting_home", "players_starting_home"]
        ] = self.df_fixtures["starting_lineup_home"].apply(
            lambda x: pd.Series(
                {"keeper_starting_home": x[0][1], "players_starting_home": x[1:]}
            )
        )
        self.df_fixtures.loc[
            :, ["keeper_starting_away", "players_starting_away"]
        ] = self.df_fixtures["starting_lineup_away"].apply(
            lambda x: pd.Series(
                {"keeper_starting_away": x[0][1], "players_starting_away": x[1:]}
            )
        )

    def calc_mean_keeper_stats(self, keeper_name, meta_info_dict):
        mask_games_form, mask_games_long = self.get_relevant_games_masks(
            meta_info_dict, "keeper"
        )
        mask_keeper = self.df_keeper_stats["player"] == keeper_name
        mean_stats_form = (
            self.df_keeper_stats.loc[mask_keeper * mask_games_form]
            .groupby("player")
            .mean(numeric_only=True)
        )
        mean_stats_long = (
            self.df_keeper_stats.loc[mask_keeper * mask_games_long]
            .groupby("player")
            .mean(numeric_only=True)
        )

        mean_stats_form, mean_stats_long = self.fill_missing_keeper_stats(
            mean_stats_form, mean_stats_long, keeper_name
        )
        mean_stats = np.concatenate(
            [
                mean_stats_form[self.keeper_stat_column_names].stack().values,
                mean_stats_long[self.keeper_stat_column_names].stack().values,
            ]
        )
        return mean_stats

    def create_list_of_stats_columns(self):
        home_xg_a_s = ["home_xg_f", "home_xg_l","home_xga_f","home_xga_l"]
        away_xg_a_s = ["away_xg_f", "away_xg_l","away_xga_f","away_xga_l"]
        home_player_stat_column_names = self.list_of_player_stats("home")
        away_player_stat_column_names = self.list_of_player_stats("away")
        keeper_home_stat_column_names = self.list_of_keeper_stats("home")
        keeper_away_stat_column_names = self.list_of_keeper_stats("away")
        column_list = (
            home_xg_a_s
            + home_player_stat_column_names
            + keeper_home_stat_column_names
            + away_xg_a_s
            + away_player_stat_column_names
            + keeper_away_stat_column_names
        )
        return column_list

    def list_of_player_stats(self, home_or_away: str):
        player_stats_column_names = []
        for form_or_long in ["form", "long"]:
            for idx_player in range(10):
                for stat_name in self.player_stat_column_names:
                    player_stats_column_names.append(
                        f"p{idx_player}_{STATS_ABBREVIATIONS.get(stat_name,stat_name)}_{form_or_long}_{home_or_away}"
                    )
        return player_stats_column_names

    def list_of_keeper_stats(self, home_or_away: str):
        keeper_stats_column_names = []
        for form_or_long in ["form", "long"]:
            for stat_name in self.keeper_stat_column_names:
                keeper_stats_column_names.append(
                    f"k_{STATS_ABBREVIATIONS.get(stat_name,stat_name)}_{form_or_long}_{home_or_away}"
                )
        return keeper_stats_column_names

    def fill_missing_player_stats(
        self, mean_stats_form: pd.DataFrame, mean_stats_long: pd.DataFrame
    ):
        """
        Fills stats of players, where stats are missing due to one reason or the other. Most likely in the form df they
        are missing because the player hasn't played in the last X games. For the long df, either due to newly bought
        player or just hasn't played yet (for example young players)

        :param mean_stats_form:
        :param mean_stats_long:
        :return:
        """
        nr_of_players_with_missing_stats_form = 10 - len(
            mean_stats_form.loc[:, self.player_stat_column_names].dropna(
                axis=0, how="all"
            )
        )
        nr_of_players_with_missing_stats_long = 10 - len(
            mean_stats_long.loc[:, self.player_stat_column_names].dropna(
                axis=0, how="all"
            )
        )

        # Case 1: No player has stats entirely missing
        if nr_of_players_with_missing_stats_form == 0:
            # Case 1.1 no missing stats at all - nothing to do
            if (
                mean_stats_form.loc[:, self.player_stat_column_names].isna().sum().sum()
                == 0
            ):
                return mean_stats_form, mean_stats_long
            else:
                if self.nan_fill_method == "mean":
                    mean_stats_form = mean_stats_form.fillna(mean_stats_form.mean())
                    mean_stats_long = mean_stats_long.fillna(mean_stats_long.mean())
                elif self.nan_fill_method == "median":
                    mean_stats_form = mean_stats_form.fillna(mean_stats_form.median())
                    mean_stats_long = mean_stats_long.fillna(mean_stats_long.median())
        # Case 2: In the long format there are no missing, so we use these stats for the form
        elif nr_of_players_with_missing_stats_long == 0:
            player_name_missing_stats = mean_stats_form.index.difference(
                mean_stats_form.loc[:, self.player_stat_column_names]
                .dropna(axis=0, how="all")
                .index
            )
            mean_stats_form.loc[
                player_name_missing_stats, self.player_stat_column_names
            ] = mean_stats_long.loc[
                player_name_missing_stats, self.player_stat_column_names
            ]
        # Case 3: Stats are missing in form and long df -> fill by calculation from other players
        elif (
            nr_of_players_with_missing_stats_form
            == nr_of_players_with_missing_stats_long
        ):
            if self.nan_fill_method == "mean":
                mean_stats_form = mean_stats_form.fillna(mean_stats_form.mean())
                mean_stats_long = mean_stats_long.fillna(mean_stats_long.mean())
            elif self.nan_fill_method == "median":
                mean_stats_form = mean_stats_form.fillna(mean_stats_form.median())
                mean_stats_long = mean_stats_long.fillna(mean_stats_long.median())
            else:
                raise ValueError(
                    f"fill_method: {self.nan_fill_method} is not a valid method (yet)"
                )
        mean_stats_form[self.player_stat_column_names] = mean_stats_form[
            self.player_stat_column_names
        ].astype("float64")
        mean_stats_long[self.player_stat_column_names] = mean_stats_long[
            self.player_stat_column_names
        ].astype("float64")
        return mean_stats_form, mean_stats_long

    def fill_missing_keeper_stats(
        self,
        mean_stats_form: pd.DataFrame,
        mean_stats_long: pd.DataFrame,
        keeper_name: str,
    ):
        # FILL NA
        # Case 1 - keeper hasn't played this season -> take mean of all keepers
        if mean_stats_form.empty and mean_stats_long.empty:
            mean_stats_form.loc[keeper_name, :], mean_stats_long.loc[keeper_name, :] = (
                pd.NA,
                pd.NA,
            )
            mean_stats_form.loc[
                keeper_name, self.keeper_stat_column_names
            ] = self.df_keeper_stats.loc[:, self.keeper_stat_column_names].mean(axis=0)
            mean_stats_long.loc[
                keeper_name, self.keeper_stat_column_names
            ] = self.df_keeper_stats.loc[:, self.keeper_stat_column_names].mean(axis=0)
        # Case 2 - No NA -> Nothing to do
        elif (
            mean_stats_form.loc[:, self.keeper_stat_column_names].isna().sum().sum()
            == mean_stats_long.loc[:, self.keeper_stat_column_names].isna().sum().sum()
            == 0
        ):
            pass
        # Case 3 - NA in form but not in long -> fill form with long stats
        elif (
            mean_stats_form.loc[:, self.keeper_stat_column_names].isna().sum().sum()
            >= mean_stats_long.loc[:, self.keeper_stat_column_names].isna().sum().sum()
        ):
            mask_form_stats_na = mean_stats_form.loc[
                :, self.keeper_stat_column_names
            ].isna()
            mean_stats_form[mask_form_stats_na] = mean_stats_long[mask_form_stats_na]
        mean_stats_form[self.keeper_stat_column_names] = mean_stats_form[
            self.keeper_stat_column_names
        ].astype("float64")
        mean_stats_long[self.keeper_stat_column_names] = mean_stats_long[
            self.keeper_stat_column_names
        ].astype("float64")
        return mean_stats_form, mean_stats_long

    def check_and_retrieve_lineup(self, fixture):
        keeper_home, players_home = (
            fixture["keeper_starting_home"],
            fixture["players_starting_home"],
        )
        keeper_away, players_away = (
            fixture["keeper_starting_away"],
            fixture["players_starting_away"],
        )
        if len(players_home) != 10:
            raise ValueError(
                f"lineup of {fixture.home_team} does not have 10 players."
                f"Season: {fixture.season}, Gameweek: {fixture.gameweek}"
            )
        elif len(players_away) != 10:
            raise ValueError(
                f"lineup of {fixture.away_team} does not have 10 players."
                f"Season: {fixture.season}, Gameweek: {fixture.gameweek}"
            )
        else:
            return keeper_home, players_home, keeper_away, players_away

    def add_def_and_off_summary_stats(self):
        self.df_player_stats.loc[:, "off_summary"] = self.df_player_stats.loc[
            :, OFFENSIVE_STATS
        ].sum(axis=1)
        self.df_player_stats.loc[:, "def_summary"] = self.df_player_stats.loc[
            :, DEFENSIVE_STATS
        ].sum(axis=1)

    def calc_mean_team_xgs(self, meta_info_dict):
        mask_season = self.df_fixtures["season"] == meta_info_dict["season"]
        mask_max_week = self.df_fixtures["gameweek"] < meta_info_dict["gameweek"]
        mask_min_week_form = (
            self.df_fixtures["gameweek"]
            >= meta_info_dict["gameweek"] - self.nr_matches_form
        )
        mask_min_week_long = (
            self.df_fixtures["gameweek"]
            >= meta_info_dict["gameweek"] - self.nr_matches_long
        )
        mask_form_games = mask_season * mask_max_week * mask_min_week_form
        mask_long_games = mask_season * mask_max_week * mask_min_week_long
        mean_xgs_home, mean_xgas_home = self.calc_mean_xg_for_team(
            mask_form_games, mask_long_games, meta_info_dict["home_team"]
        )
        mean_xgs_away, mean_xgas_away = self.calc_mean_xg_for_team(
            mask_form_games, mask_long_games, meta_info_dict["away_team"]
        )
        return mean_xgs_home,mean_xgas_home, mean_xgs_away, mean_xgas_away

    def calc_mean_xg_for_team(self, mask_form_games, mask_long_games, team_name):
        mask_team_home_games = self.df_fixtures["home_team"] == team_name
        mask_team_away_games = self.df_fixtures["away_team"] == team_name

        # xg-form
        home_games_xg_form = self.df_fixtures.loc[
            mask_form_games * mask_team_home_games
        ].home_xg.to_numpy()
        away_games_xg_form = self.df_fixtures.loc[
            mask_form_games * mask_team_away_games
            ].away_xg.to_numpy()
        xg_form = np.mean(np.concatenate((home_games_xg_form, away_games_xg_form)))

        # xg-long
        home_games_xg_long = self.df_fixtures.loc[
            mask_long_games * mask_team_home_games
            ].home_xg.to_numpy()
        away_games_xg_long = self.df_fixtures.loc[
            mask_long_games * mask_team_away_games
            ].away_xg.to_numpy()
        xg_long = np.mean(np.concatenate((home_games_xg_long, away_games_xg_long)))

        # xga-form
        # Since we don't have xga col - we use the xg of the opponents
        home_games_xga_form = self.df_fixtures.loc[
            mask_form_games * mask_team_home_games
        ].away_xg.to_numpy()
        away_games_xga_form = self.df_fixtures.loc[
            mask_form_games * mask_team_away_games
            ].home_xg.to_numpy()
        xga_form = np.mean(np.concatenate((home_games_xga_form, away_games_xga_form)))

        # xga-long
        home_games_xga_long = self.df_fixtures.loc[
            mask_long_games * mask_team_home_games
            ].away_xg.to_numpy()
        away_games_xga_long = self.df_fixtures.loc[
            mask_long_games * mask_team_away_games
            ].home_xg.to_numpy()
        xga_long = np.mean(np.concatenate((home_games_xga_long, away_games_xga_long)))
        return [xg_form, xg_long], [xga_form, xga_long]
