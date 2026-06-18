import soccerdata as sd
import os

os.makedirs("data/raw", exist_ok=True)

fbref = sd.FBref(leagues="Big 5 European Leagues Combined",
                 seasons=["2019-2020", "2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025"],
                 no_cache=False
                )

df_standard_raw = fbref.read_player_season_stats(stat_type="standard")
df_shooting_raw = fbref.read_player_season_stats(stat_type="shooting")
df_playingtime_raw = fbref.read_player_season_stats(stat_type="playing_time")

df_standard_raw.to_csv("data/raw/standard.csv")
df_shooting_raw.to_csv("data/raw/shooting.csv")
df_playingtime_raw.to_csv("data/raw/playingtime.csv")

print("Scraping complete! CSV saved to data/raw/")