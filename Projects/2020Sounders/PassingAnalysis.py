# -*- coding: utf-8 -*-
"""
Created on Sun May 10 02:25:36 2020

@author: Daniel Andrade
"""

import Metrica_IO as mio
import Metrica_Viz_S as mviz
import Metrica_Velocities as mvel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Metrica_PitchControl as mpc


DATADIR = '/Users/User/Documents/BigData/FOT/Metrica/data'
game_id = 1

events = mio.read_event_data(DATADIR,game_id)

# Using Metricas function to prep tracking_data that will allow us to find: single playing direction and passing probabilities towards the end.
tracking_home = mio.tracking_data(DATADIR,game_id,'Home')
tracking_away = mio.tracking_data(DATADIR,game_id,'Away')

tracking_home = mio.to_metric_coordinates(tracking_home)
tracking_away = mio.to_metric_coordinates(tracking_away)
events = mio.to_metric_coordinates(events)

tracking_home,tracking_away,events = mio.to_single_playing_direction(tracking_home,tracking_away,events)

tracking_home = mvel.calc_player_velocities(tracking_home,smoothing=True, filter_ = 'moving average')
tracking_away = mvel.calc_player_velocities(tracking_away,smoothing=True, filter_ = 'moving average')

shots_map = events[(events['Type'] == 'SHOT') & (events['Team'] == 'Home')]
mviz.plot_events( goals_map, color = 'k', indicators = ['Marker'], annotate = True)

# Defenders in front of ball for Home team shots
defenders = []
for i in shots_map.index:
    ball = tracking_away.loc[i, 'ball_x']
   
    players_count = [player for player in tracking_away.columns if player[-2:] == '_x' and tracking_away.loc[i,player] > ball ]
    defenders.append(len(players_count))

average_front = sum(defenders) / len(shots_map)

#Let's look at Home team passes that manage to cut through formation lines

index = events[(events.Type == 'PASS') & (events.Team == 'Home')].reset_index().drop('index', axis = 1)
breaks = 0
index['Break'] = ""
for i in tracking_away.index:
    if breaks < len(index):
        if i >= index.loc[breaks, 'Start Frame'] and i <= index.loc[breaks, 'End Frame']:
            players = [c for c in tracking_away.columns if c[-2:]== '_x' and tracking_away.loc[i, c] <= tracking_away.loc[i, 'ball_x'] and tracking_away.loc[index.loc[breaks, 'Start Frame'], c] > index.loc[breaks, 'Start X'] ]
            if i == index.loc[breaks, 'End Frame']:
                num = len(players) 
                index.loc[breaks, 'Break'] = num
                breaks += 1
    

index['Break'] = index['Break'].astype('float')

# Average defenders overplayed per player
players_break = pd.DataFrame(index.groupby('From').sum()['Break'] / index.groupby('From').count()['Break'])

players_averages = players_break.merge(index.groupby('From').count()['Type'], left_index = True, right_index = True, how = 'left' )

players_averages.columns = ['Overplayed per pass', 'Total Passes']

players_averages = players_averages.sort_values('Overplayed per pass', ascending = False)

# We want the top 5 not including the goalie (Player 11)
top_passers = players_averages[(players_averages['Overplayed per pass'] >= 1.8) & (players_averages.index != 'Player11')]


#We can plot the passes from each player in the top 5. Only passes that broke through 3 away opponents.
for i in top_passers.index:
    pass_success_probability = []
    player_passing = index[(index.From == i) & (index.Break >= 3)]
    for i,row in player_passing.iterrows():
        pass_start_pos = np.array([row['Start X'], row['Start Y']])
        pass_target_pos = np.array([row['End X'], row['End Y']])
        pass_frame = row['Start Frame']
        
        attacking_players = mpc.initialise_players(tracking_home.loc[pass_frame], 'Home', params)
        defending_players = mpc.initialise_players(tracking_away.loc[pass_frame], 'Away', params)
    
        Patt, _ = mpc.calculate_pitch_control_at_target(pass_target_pos, attacking_players, defending_players, pass_start_pos, params)
    
        pass_success_probability.append( (i,Patt))

    fig, ax = plt.subplots()
    ax.hist( [p[1] for p in pass_success_probability], bins = 8)
    ax.set_xlabel('Pass success probability')
    ax.set_ylabel('Frequency')

    pass_success_probability = sorted(pass_success_probability, key = lambda x: x[1])
    
    risky_passes = index.loc[ [p[0] for p in pass_success_probability if p[1] < 0.5]]
    
    player_passing = player_passing.drop(risky_passes.index, axis = 0)
    
    mviz.plot_events( player_passing, risky_passes, color = 'b', indicators = ['Marker', 'Arrow'], annotate = False)
    


