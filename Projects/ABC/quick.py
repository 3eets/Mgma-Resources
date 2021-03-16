'''
Creado por: Daniel Andrade Tamayo

Algoritmo (ABC)

Procesa datos de tracking por modelo YOLOv3 personalizado con las siguientes etiquetas
labels { 0 : player1,
         1 : ball,
         2 : racket,
         3 : player2,
         4 : court,
         5 : net
        }
El modelo YOLOv3 es una adaptación del código en el repositorio de Anton Mu. https://github.com/AntonMu/TrainYourOwnYOLO
'''

#librerias requeridas importar
from ordenado import (
ordenado,
ball_new,
player_new,
duplicates_ball,
duplicates_player,
court,
to_plot_coordinates,
ball_in_play,
bounce_pick
)

from draw_court import plot_pitch
import pandas as pd

# Archivo con resultados de las detecciones de YOLO
file = 'C:\\Users\\User\\Downloads\\Detection_Results.csv'

def main(file):

    #leer los datos y ordenar los resultados en una Dataframe ordenada por imagen
    df = ordenado(file)

    #filtrar la dataframe principal por la categoria "ball"
    ball_df = df[df['label'] == 1]
    new_ball_df = ball_new(ball_df)  #promedio de xmin y xmax y ymin y ymax

    b_df = duplicates_ball(new_ball_df) #Para los casos cuando hay mas de una pelota detectada en la imagen
    # Se extraen las variables y las coordenadas de la cancha de acuerdo a las imagenes
    pixel_net, c_left, c_right, net_left, net_right, metro_pixel_length, diff, line = court(df)

    #Se remueven duplicados de jugador 1
    player_df = duplicates_player(df[df['label'] == 0], pixel_net)

    player_df = player_new(player_df) #un punto x y y para el jugador 1 ("x" es promedio de xmin y xmax y "y" ymax(90%))

    player2_df = df[df['label'] == 3]
    player2_df = player_new(player2_df) #nuevas coordenadas para x y y (de xmin, xmax, ymin y ymax) igual que jugador 1

    #Se obtienen registros con la pelota en juego
    plays, ball_idx = ball_in_play(b_df, player2_df, pixel_net, line, c_left, c_right)

    # Se filtra la dataframe de la categoria pelota por optar con los registros que están en juego
    ball_valid = b_df.iloc[ball_idx,:]

    plays_df = pd.DataFrame()
    count = 1
    # Con el index de las jugadas de la misma forma creamos el dataframe de eventos agregando la columna play_id
    for i in plays:
        temp = b_df.iloc[i[0]:i[1],:]
        temp['play_id'] = count
        plays_df = pd.concat([plays_df, temp])
        count +=1

    #Dataframe con registros pertenecientes a la etiqueta raqueta
    racket_df = df[df['label'] == 2]
    bounces, plays_tbd = bounce_pick(b_df, plays, racket_df) #Genera registros de botes como entrega un index de las
                                                             #jugadas que no se han podido registrar un bote
    #Dataframe de botes
    bounce_df = df.loc[bounces]
    bounce_df = ball_new(bounce_df) #se encuentra un solo punto x y un solo punto y

                    ################################################
                    ######### VISUALIZACION EN CANCHA ##############
                    ################################################

    #Conocemos con que imágenes filtrar las dataframes de jugador1, jugador2 y pelota
    plays_for_df = plays_df['image'].unique()
    dataframes [player2_df, player_df, ball_valid]
    for data in dataframes:
        idx = [z for z in data.index if data.loc[z, 'image'] in plays_for_df]
        data = data.loc[idx, :]

    #Se plasman las columnas x y y a coordenadas para cancha de tenis en dos dimensiones
    player2_df['x'], player2_df['y'] = to_plot_coordinates(pixel_net, c_left, c_right, net_left, net_right,metro_pixel_length, diff, line, player2_df['x'].values, player2_df['y'].values)
    player_df['x'], player_df['y'] = to_plot_coordinates(pixel_net, c_left, c_right, net_left, net_right,metro_pixel_length, diff, line, player_df['x'].values, player_df['y'].values)
    ball_valid['x'], ball_valid['y'] = to_plot_coordinates(pixel_net, c_left, c_right, net_left, net_right,metro_pixel_length, diff, line, ball_valid['x'].values, ball_valid['y'].values)
    bounce_df['x'], bounce_df['y'] = to_plot_coordinates(pixel_net, c_left, c_right, net_left, net_right,metro_pixel_length, diff, line, bounce_df['x'].values, bounce_df['y'].values)

    #Se grafican todas las jugadas del set original de datos
    fig,ax = plot_pitch()
    ax.plot( player2_df['x'], player2_df['y'], 'o', color = 'orange')
    ax.plot( player_df['x'], player_df['y'],'o', color = 'green')
    ax.plot( ball_valid['x'], ball_valid['y'], 'o', color = 'yellow')
    ax.plot(bounce_df['x'], bounce_df['y'], 'o', color = 'brown')

    fig.savefig('jugadas.png')

    # jugadas = [(3343,3355), (4873, 4888), (12421,12433), (21197,22114), (27532,27544), (64621,64657), (65455,65464), (70723, 70738)]
    # title = [10,18,40,75,93,117,122, 140]
    # h = 0
    # for jugada in jugadas:
    # fig,ax = plot_pitch()
    ##tit = title[h]
    # h += 1
    # p2 = player2_df[(player2_df['image'] >= jugada[0]) & (player2_df['image'] <= jugada[1]) ]
    ##p1 = player_df[(player_df['image'] >= jugada[0]) & (player_df['image'] <= jugada[1])]
    # b = ball_valid[(ball_valid['image'] >= jugada[0]) & (ball_valid['image'] <= jugada[1])]
    # bnce = bounce_df[(bounce_df['image'] >= jugada[0]) & (bounce_df['image'] <= jugada[1])]
    # ax.plot( p2['x'], p2['y'], 'o', color = 'orange')
    # ax.plot( p1['x'], p1['y'],'o', color = 'green')
    # ax.plot( b['x'], b['y'],'o', color = 'yellow')
    # ax.plot( bnce['x'], bnce['y'],'o', color = 'brown')
    # fig.savefig('jugada {}.png'.format(tit))

if __name__ == "__main__":
    main(file)