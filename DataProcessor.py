import os

import pandas as pd
import re
from utils import slugify_columns, seaon_start_end_in_four_digits, rename_teams
from config import  league_mapping, FOOTBALL_DATA_INTERESTING_COLUMNS, DEFENSIVE_STATS



class DataProcessor:


    def prepare_football_data_df(
        self, season_start_year, season_end_year, league_name
    ) -> pd.DataFrame:
        football_data = pd.read_csv(
            f"data/{league_mapping[league_name]}/football-data/fd_{season_start_year}_{season_end_year}.csv"
        )
        football_data = football_data.rename(
            {"HomeTeam": "home_team", "AwayTeam": "away_team"}, axis=1
        )
        football_data = slugify_columns(football_data)
        football_data = rename_teams(football_data)
        football_data = self.standardize_date_format(football_data)
        football_data.date = pd.to_datetime(
            football_data.date,
            format="%d/%m/%Y",
        )
        return football_data

    # Join Football Data + Fixtures + Lineups
    def prepare_fixtures_lineups(
        self, season_start_year: int, season_end_year: int, league_name: str
    ):
        df_fixtures = pd.read_csv(
            f"data/{league_mapping[league_name]}/fbref/fixtures_{season_start_year}_{season_end_year}.csv"
        )
        df_lineups = pd.read_csv(
            f"data/{league_mapping[league_name]}/fbref/lineups_{season_start_year}_{season_end_year}.csv",
            converters={
                "starting_lineup_home": pd.eval,
                "starting_lineup_away": pd.eval,
                "bench_lineup_home": pd.eval,
                "bench_lineup_away": pd.eval,
            },
        )
        df_fixtures_and_lineups = pd.merge(
            left=df_fixtures,
            right=df_lineups,
            on=[
                "gameweek",
                "dayofweek",
                "date",
                "start_time",
                "season",
                "home_team",
                "away_team",
            ],
            how="inner",
        )
        df_fixtures_and_lineups["date"] = pd.to_datetime(
            df_fixtures_and_lineups["date"], format="%Y-%m-%d"
        )
        if not len(df_fixtures) == len(df_lineups) == len(df_fixtures_and_lineups):
            raise ValueError(
                f"Length of merged Dataframe of fixtures as lineups has changed"
            )
        df_fixtures_and_lineups = \
            rename_teams(df_fixtures_and_lineups)
        return df_fixtures_and_lineups

    def merge_fbref_football_data(
        self, season_start_year: int, season_end_year: int, league_name: str
    ) -> pd.DataFrame:
        fbref = self.prepare_fixtures_lineups(
            season_start_year, season_end_year, league_name
        )
        fbref = self.add_outcome_column(fbref)
        football_data = self.prepare_football_data_df(
            season_start_year, season_end_year, league_name
        )
        fbref_and_football_data = pd.merge(
            left=fbref,
            right=football_data.loc[:, FOOTBALL_DATA_INTERESTING_COLUMNS],
            on=[
                "date",
                "home_team",
                "away_team",
            ],
            how="inner",
            suffixes=("_fbref", "_fd"),
        )
        return fbref_and_football_data

    def concat_all_seasons_of_one_file_type_and_save(
        self, league_name: str, folder_name: str, filename_must_contain: str
    ) -> None:
        df_all = pd.DataFrame()
        base_data_path = f"data/{league_mapping[league_name]}/{folder_name}"
        min_season_start_year = 3000
        max_season_start_year = 1000
        for filename in os.listdir(base_data_path):
            if filename_must_contain not in filename:
                continue
            else:
                season_start_year = int(filename.split("_")[-2])
                if season_start_year < min_season_start_year:
                    min_season_start_year = season_start_year
                if season_start_year > max_season_start_year:
                    max_season_start_year = season_start_year
                df = pd.read_csv(os.path.join(base_data_path, filename))
                df_all = pd.concat([df_all, df])
        filename_merged_file = os.path.join(
            base_data_path,
            f"{filename_must_contain}_{seaon_start_end_in_four_digits(min_season_start_year)}_"
            f"{seaon_start_end_in_four_digits(max_season_start_year)}",
        )
        df_all.to_csv(filename_merged_file + ".csv", index=False)
        #df_all.to_pickle(filename_merged_file)

    def standardize_date_format(self, football_data):
        def change_2_digit_year_to_4_digit_year(date):
            year = date.split("/")[-1]
            if len(year) == 2:
                four_digit_year = "20" + year
                new_date = date[:6] + four_digit_year
                return new_date
            elif len(year) == 4:
                return date
            else:
                raise ValueError("unbekanntes Format")
        football_data["date"] = football_data["date"].astype("str")
        football_data["date"] = football_data["date"].apply(
            lambda date: change_2_digit_year_to_4_digit_year(date)
        )
        return football_data

    def add_outcome_column(self, df_fixtures):
        def calc_outcome_from_score(score: str):
            split_score = re.findall("[0-9]", score)
            if len(split_score) > 2:
                raise ValueError("A Team scored 10 goals ?")
            score_home = split_score[0]
            score_away = split_score[1]
            if score_home > score_away:
                return "h"
            elif score_home == score_away:
                return "d"
            elif score_home < score_away:
                return "a"
            else:
                raise ValueError("Weird result")

        df_fixtures.loc[:, "outcome"] = df_fixtures["score"].apply(
            lambda score: calc_outcome_from_score(score)
        )
        return df_fixtures

