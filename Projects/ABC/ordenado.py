'''
Set de funciones para procesar datos extraidos del
modelo YOLOv3-ABC
'''

import pandas as pd
import numpy as np
import math

def ordenado(file):
    with open(file, 'r') as f:
        df = pd.read_csv(f)
        df['image'] = df['image'].apply(lambda x: x.split('.')[0].replace('scene', '')).astype('int')
        df.sort_values('image', inplace = True)
    return df


def court(df):
    #net label 5
    half_c = df[df['label'] == 5]
    pixel_net = (half_c['ymax'].mean())
    #court label 4
    court_df = df[df['label'] == 4]
    #lenght in pixels
    line = court_df['ymax'].min() - pixel_net
    #best fit pixels per mtr
    metro_pixel_base = (court_df['xmax'].min() - court_df['xmin'].max()) / 10.97
    metro_pixel_net = (half_c['xmax'].mean() - half_c['xmin'].mean()) / 12.79
    #coordinates pixels
    c_left = court_df['xmin'].max() + (metro_pixel_base * 1.372)
    c_right = court_df['xmax'].min() - (metro_pixel_base * 1.372)
    net_left = half_c['xmin'].mean() + (metro_pixel_net * 2.28)
    net_right = half_c['xmax'].mean() - (metro_pixel_net * 2.28)
    #length per mtr
    metro_pixel_length = 11.885 / line
    diff = (net_left - c_left)

    return pixel_net, c_left, c_right, net_left, net_right, metro_pixel_length, diff, line

def to_plot_coordinates(pixel_net, c_left, c_right, net_left, net_right, metro_pixel_length, diff, line, x, y):

    new_x = []
    new_y = []

    for i in range(len(y)):
        sample = (y[i] - pixel_net)
        if sample >= 0:
            new_y.append((y[i] - pixel_net) * metro_pixel_length)
            plus = (1 - (sample / line)) * diff * .8
            temp_left = c_left + plus
            temp_right = c_right - plus
        else:
            new_y.append((y[i] - pixel_net) * metro_pixel_length * 2.9)
            plus = (np.abs(sample) / (line/1.5)) * (diff/1.3)
            temp_left = net_left + plus
            temp_right = net_right - plus
        new_ratio = 8.23 / (temp_right - temp_left)
        switch = (8.23 / 2) - ((x[i] - temp_left) * new_ratio)
        new_x.append(switch)

    return new_x, new_y

def ball_new(ball_df):
    ball_df['x'] = (ball_df['xmax'].values + ball_df['xmin'].values) / 2
    ball_df['y'] = (ball_df['ymax'].values + ball_df['ymin'].values) / 2
    return ball_df

def duplicates_ball(ball_df):
    duplicates = ball_df[ball_df.duplicated(['x', 'y'], keep=False)].index
    idx = [i for i in ball_df.index if i not in duplicates]
    b_df = ball_df.loc[idx]
    new_ball_df = b_df.copy()
    duplicates = new_ball_df[new_ball_df.duplicated('image', keep=False)]
    unique = duplicates['image'].unique()
    idx = []
    for img in unique:
        duplicate = b_df[b_df.duplicated('image', keep='first')]
        duplicate = duplicate[duplicate['image'] == img].index
        duplicate_ = b_df[b_df.duplicated('image', keep='last')]
        duplicate_ = duplicate_[duplicate_['image'] == img].index
        play = new_ball_df.drop(duplicate, 'index')
        play2 = new_ball_df.drop(duplicate_, 'index')
        shot = play[play['image'] == img]
        prev_y = play[play['image'] <= img]['y'].values
        post_y = play[play['image'] >= img]['y'].values
        testa = np.abs(prev_y[-2] - shot['y'].values) + np.abs(post_y[2] - shot['y'].values)
        shot2 = play2[play2['image'] == img]
        prev_y2 = play2[play2['image'] <= img]['y'].values
        post_y2 = play2[play2['image'] >= img]['y'].values
        testa_2 = np.abs(prev_y2[-2] - shot2['y'].values) + np.abs(post_y2[2] - shot2['y'].values)
        if testa > testa_2:
            idx.extend(shot.index)
        else:
            idx.extend(shot2.index)

    idx = [i for i in new_ball_df.index if i not in set(idx)]
    b_df = b_df.loc[idx]
    return b_df


def duplicates_player(player1_df, pixel_net):
    idx = [i for i in player1_df.index if player1_df.loc[i, 'ymax'] < pixel_net]
    new_df = player1_df.loc[idx]
    idx = [i for i in new_df.index if new_df.loc[i, 'xmin'] > 95]
    new_df = new_df.loc[idx]
    return new_df


def player_new(player_df):
    player_df['x'] = (player_df['xmax'].values + player_df['xmin'].values) / 2
    player_df['y'] = (player_df['ymax'].values * .95)
    return player_df


def ball_in_play(new_ball_df, player2_df, pixel_net, line, c_left, c_right):
    zipped = zip(new_ball_df['x'], new_ball_df['y'])
    zipped = list(zipped)
    #define list of shots
    index_end_plays = []
    index_start_plays = []
    ball_idx = []
    i = 0

    # detect start of shot
    while i in range(len(new_ball_df) - 2):
        #see if ball crossed the net
        if zipped[i][1] >= pixel_net and (zipped[i][1] < pixel_net + (line/2)):
            if zipped[i][0] > c_left and zipped[i][0] < c_right:
                print(i)
                dist = zipped[i+1][1] - zipped[i][1]
                if dist > 2:
                #check ball prox to player2 to discard ball toss in serves
                    to_p2 = [b for b in player2_df.index if player2_df.loc[b,'image'] >= new_ball_df.iloc[i]['image']]
                    pos_p2_0 = np.abs(player2_df.loc[to_p2[0], 'ymin'] - new_ball_df.iloc[i]['ymax'])
                #pos_p2_1 = player2_df.loc[to_p2[1], 'ymin'] - ball_df.loc[idx, 'ymax']
                    if pos_p2_0 > 7:
                        j = i
                        index_start_plays.append(j)
                        dist0 = 0
                    #ignore if ball comes from the side of the net (with court coordinates)
                # detect end of shot
                        while j in range(len(new_ball_df) - 2) and dist >= dist0 + 2:
                            ball_idx.append(j)
                            dist_0 = zipped[j+1][1] - zipped[j][1]
                            dist_1 = zipped[j+2][1] - zipped[j+1][1]
                            dist = (dist_0+dist_1)
                            j += 1
                        i = j
                        index_end_plays.append(j)
        i += 1
    #merge lists (im tired)
    index_plays = zip(index_start_plays, index_end_plays)
    index_plays = list(index_plays)
    return index_plays, ball_idx



def bounce_pick(ball_df, index_plays, rac_df, d = .1):
    '''
    methods to pick bounces are:
    1)direction on ball 
    2)racket and ball within close dist
    3)ball and player within close dist (to be added)
    4)ball out of court (wide or long)

    at least two if already two exit

    :param d:
    :param rac_df:
    :param ball_df:
    :param index_plays: 
    :return: 
    '''
    plays_tbd = []
    bounces = []
    for i, j in iter(index_plays):
        if j-i > 2:
            count = 0
        # 1
            angles = []
        # 2
            closest = 1000
        #:(

        #success

            can1 = np.random.random_integers(i, high=j)
            bounce= np.random.random_integers(i, high=j)
            for ball in range(i,j):
            #print(i)
            #print(j)
                print(ball)
                print(ball_df.iloc[ball]['image'])
            #print(ball_df.iloc[i:j+1]['image'])
            #index_ball = [i for i in ball_df.index if ball < i < j+1]
            #print(index_ball)
                x_dis = ball_df.iloc[ball+1]['x'] - ball_df.iloc[ball]['x']
                y_dis = ball_df.iloc[ball+1]['y'] - ball_df.iloc[ball]['y']
                length = math.hypot(x_dis , y_dis)
                angle_join = math.degrees(math.asin(y_dis / length))
                if angle_join > 0:
                    angles.append((angle_join, ball))
                mini_r = detect_rak(rac_df, ball_df, i, j)
                if len(mini_r) > 1:
                    mini_r['dist'] = mini_r.ymin.apply(lambda y: np.abs( y - ball_df.iloc[ball]['y']))
                    choice = mini_r['dist'].min()
                    if choice < closest:
                        closest = choice
                        b = ball_df.iloc[ball:].index
                        can1 = b[0]
                if closest < 50 :
                    mini_b = ball_df.iloc[i:j]
                    if len(mini_b) > 2:
                        bounce = mini_b.iloc[-2:].index[0]
                        count +=1

            can = angle_check(angles, d)
            if can:
                can = can[0][1]
                can = ball_df.iloc[can:].index[0]
                bounces.append(can)
            if count == 0:
                plays_tbd.append((i,j))
            else:
                bounces.append(bounce)

    return bounces, plays_tbd

def detect_rak(r_df, ball_df, a, z):
    idx = ball_df.iloc[a:z+1]
    a,z = idx.iloc[0]['image'], idx.iloc[-1]['image']
    idx = [i for i in r_df.index if a -1 <= r_df.loc[i, 'image'] <= z +1]
    return r_df.loc[idx]


def angle_check(angles_list, d, add = 0.05):
    d += add
    if d > .8:
        return None
    angles = []
    for i in angles_list:
        angles.append(i[0])
    can = [i for i in angles_list if (i[0] / np.mean(angles) <= (1-d)) or (i[0] / np.mean(angles) >= (1+d))]
    if len(can) == 1:
        return can
    else:
        return angle_check(angles_list, d)