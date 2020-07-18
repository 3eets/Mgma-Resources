# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 17:53:28 2020

@author: Daniel Andrade @Mgmaplus
"""

import pandas as pd 
from sklearn.cluster import KMeans
import Metrica_IO as mio
import Metrica_Viz as mviz


## We look at the formations in Metrica Data. Indicating the path of our Game Sample Data.
DATAPATH = '/Users/User/Documents/BigData/FOT/Metrica/data'

game_id = 1

# It preprocess data using Laurie's FOT metrica tidy tracking data functions
events = mio.read_event_data(DATAPATH, game_id) 

tracking_home = mio.tracking_data(DATAPATH, game_id, 'Home')
tracking_away = mio.tracking_data(DATAPATH, game_id, 'Away')

tracking_home = mio.to_metric_coordinates(tracking_home)
tracking_away = mio.to_metric_coordinates(tracking_away)

tracking_home, tracking_away, events = mio.to_single_playing_direction(tracking_home, tracking_away, events)

#Teams dataframes with frames of defensive actions and attacking actions
#Starting to create dataframes of possible formations (attacking - defending)

defense_home_formations = tracking_home.loc[events[(events.Team == 'Away') & (events.Type == 'PASS')]['Start Frame']]
attacking_home_formations = tracking_home.loc[events[(events.Team == 'Home') & (events.Type == 'PASS')]['Start Frame']]

defense_away_formations = tracking_away.loc[events[(events.Team == 'Home') & (events.Type == 'PASS')]['Start Frame']]
attacking_away_formations = tracking_away.loc[events[(events.Team == 'Away') & (events.Type == 'PASS')]['Start Frame']]

# Selects or filters dataframes for each entry to be 3 seconds apart
import numpy as np
def frames(df):
    " Takes frames from 3 seconds apart"
    df['diff'] = df['Time [s]'].diff()
    formations = df[df['diff'] >= 3].copy()
    formations.drop(['Period','Time [s]', 'ball_x','ball_y', 'diff'], axis = 1, inplace = True)
    return formations

defense_home, attack_home = frames(defense_home_formations), frames(attacking_home_formations)
defense_away, attacking_away = frames(defense_away_formations), frames(attacking_away_formations)

# The following adjusts the data for the columns of subs during the match to always work with a df of 11 players or 22 columns.

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
        elif frames == len(df[subs[k+2]].dropna()):
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
 
# By doing the elbow analysis we find inertia to be highest before inertia starts decreasing in a linear fashion.
inertias = []
for i in range(2, 10):
    model = KMeans(n_clusters = i, n_init = 100)
    model.fit(attack_away)
    inertias.append(model.inertia_)
 
import matplotlib.pyplot as plt
plt.plot(inertias)

# Also focus on looking at the silhouette analysis to find how different number of clusters perform vs the average value of their clusters, that is
# the average distance of the samples from a neighbouring cluster
# Done for the 4 different dfs

import matplotlib.cm as cm
for i in range(2,10) :
    fig, ax1 = plt.subplots(figsize=(18,7))
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(attack_away) + (i+1) *10])
    model = KMeans(n_clusters = i, n_init = 100)
    labels = model.fit_predict(attack_away)
    silhouette_avg = silhouette_score(attack_away, labels)
    sample_silhouette_values = silhouette_samples(attack_away, labels)
    y_lower = 10
    for k in range(i):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == k]
        ith_cluster_silhouette_values.sort()
        size_cluster_k = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_k
        color = cm.nipy_spectral(float(k) / i)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor = color, edgecolor = color, alpha = 0.7)
        y_lower = y_upper + 10 
    ax1.axvline( x = silhouette_avg, color = "red", linestyle = "--")
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        
plt.show()

# For our different dfs, we find k = 1 for three of our dfs and k = 3 for a single one (attack_away).

model = KMeans(n_clusters = 1, random_state = 120)
model.fit(defense_home)
centroids = model.cluster_centers_
labels = model.predict(defense_home)
defense_home['labels'] = labels

model = KMeans(n_clusters = 1, random_state = 120)
model.fit(attack_home)
centroids = model.cluster_centers_
labels = model.predict(attack_home)
attack_home['labels'] = labels

model = KMeans(n_clusters = 3, random_state = 120)
model.fit(attack_away)
centroids = model.cluster_centers_
labels_away = model.predict(attack_away)
attack_away['labels'] = labels_away

model = KMeans(n_clusters = 1, random_state = 120)
model.fit(defense_away)
centroids = model.cluster_centers_
labels_away = model.predict(defense_away)
defense_away['labels'] = labels_away

# Filters the dataframes for each label that is the tracking data of the 11 players on the pitch for the home and away formations (from attack and defend dfs).

defense_home_means = defense_home.mean(axis = 0)[0:22] 

attack_home_means = attack_home.mean(axis = 0)[0:22]

attack_away_1 = attack_away[attack_away.labels == 0]
attack_away_2 = attack_away[attack_away.labels == 1]
attack_away_3 = attack_away[attack_away.labels == 2]

attack_away_1_means = attack_away_1.mean(axis = 0)[0:22] 
attack_away_2_means = attack_away_2.mean(axis = 0)[0:22]
attack_away_3_means = attack_away_3.mean(axis = 0)[0:22]

defense_away_means = defense_away.mean(axis = 0)[0:22] 

# The following vector looks at the different players whom we can plot with vectors to picture the formation lines of defenders and midfielders.
vectors = [['Home_4_x','Home_3_x', 'Home_2_x', 'Home_7_x', 'Home_8_x','Home_6_x', 'Home_5_x'], 
           ['Home_4_y','Home_3_y','Home_2_y', 'Home_7_y','Home_8_y','Home_6_y', 'Home_5_y'], 
           ['Away_17_x','Away_15_x', 'Away_16_x', 'Away_20_x', 'Away_18_x','Away_19_x', 'Away_21_x', 'Away_22_x'], 
           ['Away_17_y','Away_15_y','Away_16_y', 'Away_20_y','Away_18_y','Away_19_y', 'Away_21_y', 'Away_22_y']]

# Plots
title = 'Home defense 4-1-3-2 / Away attack (1) 3-5-2  model = KNN'
mviz.plot_frame_centroids(defense_home_means, attack_away_1_means, vectors, title, annotate = True, switch = 2)
title = 'Home defense 4-1-3-2 / Away attack (2) 4-4-2  model = KNN'
mviz.plot_frame_centroids(defense_home_means, attack_away_2_means, vectors, title, annotate = True)
title = 'Home defense 4-1-3-2 / Away attack (3) 4-4-2  model = KNN'
mviz.plot_frame_centroids(defense_home_means, attack_away_3_means, vectors, title, annotate = True)
title = 'Home attack 4-1-3-2 / Away defense 4-4-2  model = KNN'
mviz.plot_frame_centroids(attack_home_means, defense_away_means, vectors, title, annotate = True)

# The code below uses Spectral Clustering to plot the same number of clusters for the different dataframes 
# It allows us to find these clusters with non-linear bounderies

from sklearn.cluster import SpectralClustering

model = SpectralClustering(n_clusters=1, affinity='nearest_neighbors',
                           assign_labels='kmeans')
labels = model.fit_predict(defense_home)
defense_home['labels'] = labels

model = SpectralClustering(n_clusters=1, affinity='nearest_neighbors',
                           assign_labels='kmeans')
labels = model.fit_predict(attack_home)
attack_home['labels'] = labels

model = SpectralClustering(n_clusters=3, affinity='nearest_neighbors',
                           assign_labels='kmeans')
labels = model.fit_predict(attack_away)
attack_away['labels'] = labels

model = SpectralClustering(n_clusters=1, affinity='nearest_neighbors',
                           assign_labels='kmeans')
labels = model.fit_predict(defense_away)
defense_away['labels'] = labels

# Filters the labels for each dataframe to plot each label on a diffetent plot
defense_home_means = defense_home.mean(axis = 0)[0:22] 

attack_home_means = attack_home.mean(axis = 0)[0:22]

attack_away_1 = attack_away[attack_away.labels == 0]
attack_away_2 = attack_away[attack_away.labels == 1]
attack_away_3 = attack_away[attack_away.labels == 2]

attack_away_1_means = attack_away_1.mean(axis = 0)[0:22] 
attack_away_2_means = attack_away_2.mean(axis = 0)[0:22]
attack_away_3_means = attack_away_3.mean(axis = 0)[0:22]

defense_away_means = defense_away.mean(axis = 0)[0:22] 

# Plots
title = 'Home defense 4-1-3-2 / Away attack (1) 3-5-2 model = Spectral Clustering'
mviz.plot_frame_centroids(defense_home_means, attack_away_1_means, vectors, title, annotate = True, switch = 3)
title = 'Home defense 4-1-3-2 / Away attack (2) 4-4-2 model = Spectral Clustering'
mviz.plot_frame_centroids(defense_home_means, attack_away_2_means, vectors, title, annotate = True)
title = 'Home defense 4-1-3-2 / Away attack (3) 3-1-4-2 model = Spectral Clustering'
mviz.plot_frame_centroids(defense_home_means, attack_away_3_means, vectors, title, annotate = True, switch = 2)
title = 'Home attack 4-1-3-2 / Away defense 4-4-2 model = Spectral Clustering'
mviz.plot_frame_centroids(attack_home_means, defense_away_means, vectors, title, annotate = True)
