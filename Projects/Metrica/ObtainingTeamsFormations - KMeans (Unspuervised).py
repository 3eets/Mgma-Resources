# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 17:53:28 2020

@author: Daniel Andrade @Mgmaplus
"""

import pandas as pd 
from sklearn.cluster import KMeans
import Metrica_IO as mio
import Metrica_Viz as mviz



DATAPATH = '/Users/User/Documents/BigData/FOT/Metrica/data'

game_id = 1

events = mio.read_event_data(DATAPATH, game_id) 

tracking_home = mio.tracking_data(DATAPATH, game_id, 'Home')
tracking_away = mio.tracking_data(DATAPATH, game_id, 'Away')

tracking_home, tracking_away, events = mio.to_single_playing_direction(tracking_home, tracking_away, events)

tracking_home = mio.to_metric_coordinates(tracking_home)
tracking_away = mio.to_metric_coordinates(tracking_away)

#Teams dataframes with frames of defensive actions and attacking actions
#Starting to create dataframes of possible formations

defense_home_formations = tracking_home.loc[events[(events.Team == 'Away') & (events.Type != 'BALL LOST') & (events.Type == 'PASS')]['Start Frame']]
attacking_home_formations = tracking_home.loc[events[(events.Team == 'Home') & (events.Type != 'BALL LOST') & (events.Type == 'PASS')]['Start Frame']]

defense_away_formations = tracking_away.loc[events[(events.Team == 'Home') & (events.Type != 'BALL LOST') & (events.Type == 'PASS')]['Start Frame']]
attacking_away_formations = tracking_away.loc[events[(events.Team == 'Away') & (events.Type != 'BALL LOST') & (events.Type == 'PASS')]['Start Frame']]

import numpy as np
def frames(df):
    " Takes frames from 3 seconds apart"
    df['diff'] = df['Time [s]'].diff()
    formations = df[df['diff'] >= 3].copy()
    formations.drop(['Period','Time [s]', 'ball_x','ball_y', 'diff'], axis = 1, inplace = True)
    return formations

defense_home, attack_home = frames(defense_home_formations), frames(attacking_home_formations)
defense_away, attacking_away = frames(defense_away_formations), frames(attacking_away_formations)


def substitutes(df):
    "Adjusts for substitutions made during the match to keep a df of 11 features at all times"
    new_df = df.iloc[:,0:22].copy()
    cols = new_df.columns[new_df.isna().any()]
    subs = df.columns[22:].tolist()
    k = 0
    for i in cols:
        frames = new_df[i].isna().sum()
        player1 = new_df[i].dropna().copy()
        if frames == len(df[subs[k]].dropna()):
            player2 = df[subs[k]].dropna().copy()
            item = subs[0]
            subs.remove(item)
        else:
            if frames == len(df[subs[k+2]].dropna()):
                player2 = df[subs[k+2]].dropna().copy()
                item = subs[2]
                subs.remove(item)
            else:
                player2 = df[subs[k+4]].dropna().copy()
                item = subs[4]
                subs.remove(item)
        new_df[i] = player1.append(player2)
    return new_df

defense_home = substitutes(defense_home)
attack_home = substitutes(attack_home)
defense_away = substitutes(defense_away)
attack_away = substitutes(attacking_away)
 

inertias = []
for i in range(2, 10):
    model = KMeans(n_clusters = i )
    model.fit(defense_away)
    inertias.append(model.inertia_)

import matplotlib.pyplot as plt
plt.plot(inertias)

model = KMeans(n_clusters = 1)
model.fit(defense_home)
centroids = model.cluster_centers_
labels = model.predict(defense_home)
defense_home['labels'] = labels

model = KMeans(n_clusters = 1)
model.fit(attack_home)
centroids = model.cluster_centers_
labels = model.predict(attack_home)
attack_home['labels'] = labels

model = KMeans(n_clusters = 3)
model.fit(attack_away)
centroids = model.cluster_centers_
labels_away = model.predict(attack_away)
attack_away['labels'] = labels_away

model = KMeans(n_clusters = 1)
model.fit(defense_away)
centroids = model.cluster_centers_
labels_away = model.predict(defense_away)
defense_away['labels'] = labels_away


defense_home_means = defense_home.mean(axis = 0)[0:22] 
attack_home_means = attack_home.mean(axis = 0)[0:22]

attack_away_1 = attack_away[attack_away.labels == 0]
attack_away_2 = attack_away[attack_away.labels == 1]
attack_away_3 = attack_away[attack_away.labels == 2]

attack_away_1_means = attack_away_1.mean(axis = 0)[0:22] 
attack_away_2_means = attack_away_2.mean(axis = 0)[0:22]
attack_away_3_means = attack_away_3.mean(axis = 0)[0:22]

defense_away_means = defense_away.mean(axis = 0)[0:22] 

vectors = [['Home_4_x','Home_3_x', 'Home_2_x', 'Home_7_x', 'Home_8_x','Home_6_x', 'Home_5_x'], ['Home_4_y','Home_3_y','Home_2_y', 'Home_7_y','Home_8_y','Home_6_y', 'Home_5_y'], ['Away_17_x','Away_15_x', 'Away_16_x', 'Away_20_x', 'Away_18_x','Away_19_x', 'Away_21_x', 'Away_22_x'], ['Away_17_y','Away_15_y','Away_16_y', 'Away_20_y','Away_18_y','Away_19_y', 'Away_21_y', 'Away_22_y']]

mviz.plot_frame_centroids(defense_home_means, attack_away_1_means, vectors, annotate = True)
mviz.plot_frame_centroids(defense_home_means, attack_away_2_means, vectors, annotate = True)
mviz.plot_frame_centroids(defense_home_means, attack_away_3_means, vectors, annotate = True, switch = 2)
mviz.plot_frame_centroids(attack_home_means, defense_away_means, vectors, annotate = True)

