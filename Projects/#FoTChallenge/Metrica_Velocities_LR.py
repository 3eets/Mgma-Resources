"""
Created on May  8th 2020
Module initially created for reading in Metrica sample data but adapted for #FoT challenge
Data can be found at: https://github.com/Friends-of-Tracking-Data-FoTD/Last-Row (Not Metrica data :p
@author: Laurie Shaw (@EightyFivePoint)
adapted by: Daniel Andrade
"""

import numpy as np
import scipy.signal as signal
import pandas as pd

def calc_player_velocities(team, smoothing=True, filter_='Savitzky-Golay', window=7, polyorder=1, maxspeed = 12):
    """ calc_player_velocities( tracking_data )
    
    Calculate player velocities in x & y direciton, and total player speed at each timestamp of the tracking data
    
    Parameters
    -----------
        team: the tracking DataFrame for home or away team
        smoothing: boolean variable that determines whether velocity measures are smoothed. Default is True.
        filter: type of filter to use when smoothing the velocities. Default is Savitzky-Golay, which fits a polynomial of order 'polyorder' to the data within each window
        window: smoothing window size in # of frames
        polyorder: order of the polynomial for the Savitzky-Golay filter. Default is 1 - a linear fit to the velcoity, so gradient is the acceleration
        maxspeed: the maximum speed that a player can realisitically achieve (in meters/second). Speed measures that exceed maxspeed are tagged as outliers and set to NaN. 
        
    Returrns
    -----------
       team : the tracking DataFrame with columns for speed in the x & y direction and total speed added

    """
    # remove any velocity data already in the dataframe
    team = remove_player_velocities(team)
    
    # Get the player ids
    player_ids = np.unique( [ c[:-2] for c in team.columns if c[-2:].lower()=='_x' and c != 'ball_x'] )

    # Calculate the timestep from one frame to the next. Should always be 0.04 within the same half
    dt = team['Time [s]'].diff()
    
    # estimate velocities for players in team
    for player in player_ids: # cycle through players individually
        # difference player positions in timestep dt to get unsmoothed estimate of velicity
        vx = team[player+"_dx"] / dt
        vy = team[player+"_dy"] / dt
        vx = list(vx)
        vy = list(vy)
            
        if smoothing:
            if filter_=='Savitzky-Golay':
                # calculate first half velocity
                vx = signal.savgol_filter(vx,window_length=window,polyorder=polyorder)
                vy = signal.savgol_filter(vy,window_length=window,polyorder=polyorder)        
                # calculate second half velocity
                vx = signal.savgol_filter(vx,window_length=window,polyorder=polyorder)
                vy = signal.savgol_filter(vy,window_length=window,polyorder=polyorder)
            elif filter_=='moving average':
                ma_window = np.ones( window ) / window 
                # calculate first half velocity
                vx = np.convolve( vx , ma_window, mode='same' ) 
                vy = np.convolve( vy , ma_window, mode='same' )      
                # calculate second half velocity
                vx = np.convolve( vx , ma_window, mode='same' ) 
                vy = np.convolve( vy , ma_window, mode='same' ) 
                
        
        # put player speed in x,y direction, and total speed back in the data frame
        team[player + "_vx"] = vx
        team[player + "_vy"] = vy
        team[player + "_speed"] = np.sqrt( np.power(vx,2) + np.power(vy,2) )

    return team

def remove_player_velocities(team):
    # remove player velocoties and acceleeration measures that are already in the 'team' dataframe
    columns = [c for c in team.columns if c.split('_')[-1] in ['vx','vy','ax','ay','speed','acceleration']] # Get the player ids
    team = team.drop(columns=columns)
    return team

def eventing(data, play):
    #turning our tracking data into events to separate important frames 
    cols = [c for c in data.columns if (c[-2:] in ['_x', '_y'] and c[:4] != 'ball') ]
    index = data.index[1:] 
    frame = [frame for frame in index if data.loc[frame, 'ball_y'] and data.loc[frame, 'ball_x'] in list(data.loc[frame, cols])]
    j = 0
    k = 0
    temp = pd.DataFrame(columns = ['Type', 'Start Frame', 'Start Time [s]','End Frame', 'End Time [s]', 'From', 'Start X', 'Start Y','End X', 'End Y', 'End Z', 'Match'])
    for i in frame:
        options = data.loc[i]
        player = [ c for c in cols if options[c] == data.loc[i, 'ball_y'] or options[c] == data.loc[i, 'ball_x']]
        if len(player) == 2 and player[0].split('_')[0] == player[1].split('_')[0]:
            if i == frame[-1]:
                temp.loc[k,'Start Frame'] = i
                temp.loc[k,'Start Time [s]'] = options['Time [s]']
                end = [c for c in data.index if data.loc[c,'ball_x'] >= 53 or data.loc[c, 'ball_x'] <= -53]
                temp.loc[k,'End Frame'] = end[0] 
                temp.loc[k,'End Time [s]'] = data.loc[end[0], 'Time [s]']
                temp.loc[k,'Type'] = 'SHOT'
                temp.loc[k,'From'] = player[0].split('_')[0]
                temp.loc[k,'Start X'] = options[player[0]]
                temp.loc[k,'Start Y'] = options[player[1]]
                temp.loc[k,'End X'] = data.loc[end[0], 'ball_x']
                temp.loc[k,'End Y'] = data.loc[end[0], 'ball_y']
                temp.loc[k,'End Z'] = data.loc[end[0], 'ball_z']
                temp.loc[k, 'Match'] = play
                k +=1
            else: 
                options_1 = data.loc[frame[j +1]]
                next_p = [ c for c in cols if options_1[c] == data.loc[frame[j + 1], 'ball_y'] or options_1[c] == data.loc[frame[j + 1], 'ball_x']]
                if len(next_p) == 2 and next_p[0].split('_')[0] == next_p[1].split('_')[0]:
                    if player[0] != next_p[0]:
                        temp.loc[k,'Type'] = 'PASS'
                        temp.loc[k,'Start Frame'] = i
                        temp.loc[k,'Start Time [s]'] = options['Time [s]']
                        temp.loc[k,'End Time [s]'] = data.loc[frame[j+1], 'Time [s]']
                        temp.loc[k,'End Frame'] = frame[j+1]
                        temp.loc[k,'Start X'] = options[player[0]]
                        temp.loc[k,'Start Y'] = options[player[1]]
                        temp.loc[k,'From'] = player[0].split('_')[0]
                        temp.loc[k,'End X'] = options_1[next_p[0]]
                        temp.loc[k,'End Y'] = options_1[next_p[1]]
                        temp.loc[k, 'Match'] = play
                        k += 1
        j += 1
    return temp
