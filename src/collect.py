import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

SEASONS = [f"{y}-{str(y+1)[-2:]}" for y in range(2015, 2025)]  # 2015-16 ... 2024-25

frames = []
for season in SEASONS:
    log = leaguegamelog.LeagueGameLog(
        season=season, season_type_all_star="Regular Season"
    ).get_data_frames()[0]
    log["SEASON"] = season
    frames.append(log)
    print(f"Pulled {season}: {len(log)} rows")
    time.sleep(2) 

df = pd.concat(frames, ignore_index=True)
df.to_csv("data/raw/game_logs.csv", index=False)