import Metrica_IOLR as mio
import Metrica_Viz_LR as mviz
import Metrica_Velocities_LR as mvel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Metrica_Velocities_LR as mvel

DATADIR = '/Users/User/Documents/BigData/FOT/Last-Row-master/datasets/positional_data'
data = pd.read_csv('~/Documents/BigData/FOT/Last-Row-master/datasets/positional_data/liverpool_2019.csv', index_col=('play', 'frame'))

# Dataframe of goals
eventos = pd.DataFrame()

# List for defenders in front of the ball when goals are shot
defenders = []

plays = data.index.get_level_values('play').unique()
PLOTDIR = DATADIR

for play in plays:
    #kits, player_number fix
    df = data.loc[play].reset_index()

    pivot = df.pivot(index = 'frame', columns = 'player', values = ['player_num','team'])
    pivot.columns.name = None
    home_players = pd.DataFrame(index = pivot.index)
    key = pivot.loc[0,'team'] == 'attack'
    i = pivot.columns.levels[0][0]
    df = pivot.loc[:,i].copy()
    temp = df.loc[:,key].copy()
    home_players = pd.concat([home_players, temp], axis = 1).dropna(how='all', axis = 1)
    
    # plays 
    df = data.loc[play].reset_index()

    pivot = df.pivot(index = 'frame', columns = 'player', values = ['x','y','z','dx','dy','team'])
    pivot.columns.name = None

    home_tracking = pd.DataFrame(index = pivot.index)
    key = pivot.loc[0,'team'] == 'attack'
    for i in pivot.columns.levels[0].drop('team'):
        index = i
        df = pivot.loc[:,index].copy()
        temp = df.loc[:,key].copy()
        col = []
        for j in range(len(temp.columns)):
            col.append(str(temp.columns[j]) + '_' + index)
        temp.columns = col
        home_tracking = pd.concat([home_tracking, temp], axis = 1)
   
    away_tracking = pd.DataFrame(index = pivot.index)
    key = pivot.loc[0, 'team'] == 'defense'
    for i in pivot.columns.levels[0].drop('team'):
        index = i
        df = pivot.loc[:,index].copy()
        temp = df.loc[:,key].copy()
        col = []
        for j in range(len(temp.columns)):
            col.append(str(temp.columns[j]) + '_' + index)
        temp.columns = col
        away_tracking = pd.concat([away_tracking, temp], axis = 1)
    
    for i in pivot.columns.levels[0].drop(['team']):
        index = i
        df = pivot.loc[:,index].copy()
        temp = df.loc[:,0].copy()
        temp.name = 'ball' + '_' + index
        home_tracking, away_tracking = pd.concat([home_tracking, temp], axis = 1), pd.concat([away_tracking, temp], axis = 1)

    time = []
    for i in range(0,len(pivot.index)):
        time.append(i * 0.05)
    home_tracking['Time [s]'] = time
    away_tracking['Time [s]'] = time
        
    tracking_home = mio.to_metric_coordinates(home_tracking)
    tracking_away = mio.to_metric_coordinates(away_tracking)

# Formulate velocities for all players in both home and away tracking dfs
    tracking_home = mvel.calc_player_velocities(tracking_home.astype('float'),smoothing=True, filter_ = 'moving average')
    tracking_away = mvel.calc_player_velocities(tracking_away.astype('float'),smoothing=True, filter_ = 'moving average')

# Drop NAs columns
    tracking_away = tracking_away.dropna(how = 'all', axis = 1)
    tracking_home = tracking_home.dropna(how = 'all', axis = 1)
    
    print(play)

    df_events = mvel.eventing(tracking_home, play)
    
    kits_cols = [c for c in tracking_home.columns if c.split('_')[0] in home_players.columns.astype('str')]
    
    cols = []
    for i in range(len(tracking_home.columns)):
        if tracking_home.columns[i] in kits_cols:
            cols.append(str(int(home_players.loc[0,int(tracking_home.columns[i].split('_')[0])])) + '_' + tracking_home.columns[i].split('_')[1])
        else: 
            cols.append(tracking_home.columns[i])

    tracking_home.columns = cols
    
    for i in range(len(df_events)):
        df_events.loc[i,'From'] = int(home_players.loc[0, int(df_events.loc[i, 'From'])])
    
    import Metrica_PitchControl_LR as mpc

    params = mpc.default_model_params(3)
    
    tracking_home = tracking_home.astype('float')
    tracking_away = tracking_away.astype('float')
    
    ix = int(df_events[df_events.Type == 'SHOT']['Start Frame'])
    ball = tracking_away.loc[ix, 'ball_x']
    if df_events[df_events.Type == 'SHOT']['Start X'].values > 0:
        players_count = [player for player in tracking_away.columns if player[-2:] == '_x' and tracking_away.loc[ix,player] > ball ]
    else:
        players_count = [player for player in tracking_away.columns if player[-2:] == '_x' and tracking_away.loc[ix,player] < ball ]
    
    defenders.append(len(players_count))
     
# PCF clip 
    name = "%s_goal_break" % play
    mviz.save_match_clip_pcf_break(df_events, tracking_home,tracking_away, PLOTDIR, params, fname = name, include_player_velocities=True)
    
    eventos = pd.concat([eventos, df_events], axis = 0)


# Goal maps

eventos = eventos.reset_index().drop('index', axis = 1)
new = eventos[['Type','Start X','Start Y', 'From']]

i = 0
while i in new.index:
    while new.loc[i, 'Type'] == 'PASS':
        i += 1
    if new.loc[i, 'Start X'] < 0:
        new.loc[i, 'Start X'] = -1 * new.loc[i, 'Start X']
        new.loc[i, 'Start Y'] = -1 * new.loc[i, 'Start Y']
        j = i -1
        while new.loc[j, 'Type'] == 'PASS':
            x, y = new.loc[j, 'Start X'], new.loc[j, 'Start Y']
            new.loc[j, 'Start X'] = -1 * x
            new.loc[j, 'Start Y'] = -1 * y
            if j == 0:
                j = 3
            j -= 1
    i += 1
    

# Looking at some plots and average stats for the data
# Goals
goals_map = new[new['Type'] == 'SHOT']
mviz.plot_events( goals_map, color = 'k', indicators = ['Marker'], annotate = True)
from matplotlib.patches import Arc

H_Goal=np.histogram2d(goals_map['Start X'], goals_map['Start Y'],bins=50,range=[[0, 53],[-34, 34]])



#Plot the number of shots from different points
(fig,ax) = mviz.createGoalMouth()
pos=ax.imshow(H_Goal[0], extent=[0,65,0,54], aspect='auto',cmap=plt.cm.Reds)
fig.colorbar(pos, ax=ax)
ax.set_title('19 goals by Liverpool from Last Row data (\'19 season)')
plt.xlim((-1,66))
plt.ylim((-3,35))
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

fig.savefig('NumberOfShots.png', dpi=None, bbox_inches="tight")   

# Who scored?
top_scorers = eventos[eventos.Type == 'SHOT'].groupby('From').count()['Type'].sort_values(ascending = False)

# Organizing data b assists
index = new[new.Type == 'SHOT'].index - 1
assists_df = new.loc[index,:]
for i in assists_df.index:
    assists_df.loc[i, 'Type'] = 'ASSIST'

assists = assists_df.groupby('From').count()['Type'].sort_values(ascending = False)
mviz.plot_events( assists_df, color = 'k', indicators = ['Marker'], annotate = True)

# Breaking down the play by passes
pass_before_assist = [c - 1  for c in index if new.loc[c-1, 'Type'] == 'PASS']
pass_to_assist_df = new.loc[pass_before_assist, :]
mviz.plot_events( pass_to_assist_df, color = 'k', indicators = ['Marker'], annotate = True)

# Overplayed defenders by pass
average_breaks = eventos[eventos.Type != 'SHOT']
average = average_breaks.Break.sum() / len(average_breaks)
players_break = average_breaks.groupby('From').sum()['Breaks'] / average_breaks.groupby('From').count()['Breaks']
players_break.sort_values(ascending = False)

# Defenders in front of the ball when shot is taken
average_front = sum(defenders) / 19

# Saving tables
players_kits = {4:'Virgil van Dijk', 5:'Georginio Wijnaldum', 7:'James Milner', 8:'Naby Keita', 9:'Roberto Firmino', 10:'Sadio Mane', 11:'Mohamed Salah', 14:'Jordan Henderson', 20:'Adam Lallana', 23:'Xherdan Shaqiri', 26:'Andrew Robertson', 27:'Divock Origi', 66:'Trent Alexander-Arnold'}
kits = pd.DataFrame(players_kits.values(), index = players_kits.keys(), columns =['Name'])

one = kits.merge(np.round(players_break, 1), how= 'right', left_index = True, right_index = True)
one.columns = ['Player', 'Overplayed']
one = one.sort_values('Overplayed', ascending = False)

one.to_csv('/Users/User/Documents/BigData/FOT/Last-Row-master/datasets/positional_data/one.csv', index = True)

#goals export
two = kits.merge(top_scorers, how='right', left_index = True, right_index = True)
two.columns = ['Player','Goals']
two.to_csv('/Users/User/Documents/BigData/FOT/Last-Row-master/datasets/positional_data/two.csv', index = True)

#Assists export
three = kits.merge(assists, how= 'right', left_index = True, right_index = True)
three.columns = ['Player', 'Assists']
three.to_csv('/Users/User/Documents/BigData/FOT/Last-Row-master/datasets/positional_data/three.csv', index = True)

#Overplayed per assist
eventos = eventos.drop('index', axis = 1)
assists_breaks = eventos.loc[assists_df.index]
assists_breaks = assists_breaks.groupby('From').sum()['Break'] / assists_breaks.groupby('From').count()['Break']
four = kits.merge(np.round(assists_breaks, 1), how = 'right', left_index= True, right_index = True)
four.columns = ['Player', 'Overplayed']
four.to_csv('/Users/User/Documents/BigData/FOT/Last-Row-master/datasets/positional_data/four.csv', index = True)

#exports events
eventos.to_csv('/Users/User/Documents/BigData/FOT/Last-Row-master/datasets/positional_data/eventos_liverpool.csv', index = True)
