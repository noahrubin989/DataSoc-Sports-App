import re
import pandas as pd
import numpy as np
import streamlit as st


def find_index(string):
    """Helper function for `separate_team_and_name`"""
    match = re.search(r"[a-z]", string[::-1])
    if match:
        return len(string) - match.start()
    else:
        return -1

def separate_team_and_name(ds):
    # Find the index of the first lowercase letter
    ds['name_splitting_point'] = ds['Name'].apply(lambda x: find_index(x))
    ds['Team'] = ds.apply(lambda row: row['Name'][row['name_splitting_point']::1], axis=1)
    ds['Name'] = ds.apply(lambda row: row['Name'].replace(row['Team'], ''), axis=1)
    ds.drop(columns=['name_splitting_point', "RK"], inplace=True)
    return ds


def reorder_cols(ds):
    cols = ds.columns.tolist()
    for c in ['Team', 'Year']:
        cols.remove(c)

    cols.insert(2, 'Team')
    cols.insert(3, 'Year')

    # Reindex the DataFrame with the new column order
    ds = ds.reindex(columns=cols)
    return ds


def clean_team_names(ds):
    ds['Team'] = ds.Team.str.replace('.', '', regex=True)
   
   
def rename_columns(ds):
    renamed = {
        'POS': 'Position',
        'GP': 'Games Played',
        'MIN': 'Minutes Per Game',
        'PTS': 'Points Per Game',
        'FGM': 'Average Field Goals Made',
        'FGA': 'Average Field Goals Attempted',
        'FG%': 'Field Goal Percentage',
        '3PM': 'Average 3-Point Field Goals Made',
        '3PA': 'Average 3-Point Field Goals Attempted',
        '3P%': '3-Point Field Goal Percentage',
        'FTM': 'Average Free Throws Made',
        'FTA': 'Average Free Throws Attempted',
        'FT%': 'Free Throw Percentage',
        'REB': 'Rebounds Per Game',
        'AST': 'Assists Per Game',
        'STL': 'Steals Per Game',
        'BLK': 'Blocks Per Game',
        'TO': 'Turnovers Per Game',
        'DD2': 'Double Double',
        'TD3': 'Triple Double'
    }

    ds.rename(columns=renamed, inplace=True)
    return ds

def main():
    df = pd.read_csv('../data/basketball_data.csv')
    df = separate_team_and_name(df)
    df = reorder_cols(df)
    clean_team_names(df)
    df = rename_columns(df)
    
    df.to_csv("../data/basketball_data_cleaned.csv", index=False)
    

if __name__ == "__main__":
    main()