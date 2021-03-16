'''
Set de funciones para visualizar los datos
del algoritmo YOLOv3-ABC

Por desarrollar save_match_clip para
crear videos de las jugadas
'''


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
#import Metrica_PitchControl as mpc
#import Metrica_IO as mio


def plot_pitch(field_dimen=(8.23, 23.77), field_color='blue', linewidth=2, markersize=15):
    """ plot_pitch

    Plots a tennis court. All distance units converted to meters.

    Parameters
    -----------
        field_dimen: (length, width) of field in meters. Default is (106,68)
        field_color: color of field. options are {'green','white'}
        linewidth  : width of lines. default = 2
        markersize : size of markers (e.g. penalty spot, centre spot, posts). default = 20

    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)

    """
    fig, ax = plt.subplots(figsize=(10, 18))  # create a figure
    # decide what color we want the field to be. Default is green, but can also choose white
    if field_color == 'blue':
        ax.set_facecolor('#3392FF')
        lc = 'whitesmoke'  # line color
        pc = 'w'  # 'spot' colors
    elif field_color == 'white':
        lc = 'k'
        pc = 'k'
    # ALL DIMENSIONS IN m
    border_dimen = (3, 3)  # include a border around of the field of width 3m

    half_pitch_width= field_dimen[0] / 2.  # width of half pitch
    half_pitch_length = field_dimen[1] / 2.  # length of half pitch
    signs = [-1, 1]
    # Soccer field dimensions typically defined in yards, so we need to convert to meters
    # plot half way line # center circle
    ax.plot([-half_pitch_width, half_pitch_width], [0, 0], lc, linewidth=linewidth)
    ax.plot([0, 0], [(-half_pitch_length + 5.49), (half_pitch_length - 5.49)], lc, linewidth=linewidth)
    for s in signs:  # plots each line seperately
        # plot pitch boundary
        ax.plot([-half_pitch_width, half_pitch_width], [s * half_pitch_length, s * half_pitch_length], lc,
                linewidth=linewidth)
        ax.plot([s * half_pitch_width, s * half_pitch_width], [-half_pitch_length, half_pitch_length], lc,
                linewidth=linewidth)
        # half-line serve length
        ax.plot([-half_pitch_width, half_pitch_width], [s * (half_pitch_length - 5.49), s* (half_pitch_length - 5.49)], lc,
                linewidth=linewidth)

    # remove axis labels and ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    # set axis limits
    xmax = field_dimen[0] / 2. + border_dimen[0]
    ymax = field_dimen[1] / 2. + border_dimen[1]
    ax.set_xlim([-xmax, xmax])
    ax.set_ylim([-ymax, ymax])
    ax.set_axisbelow(True)
    return fig, ax


def save_match_clip(hometeam, awayteam, fpath, fname='clip_test', figax=None, frames_per_second=25,
                    team_colors=('r', 'b'), field_dimen=(106.0, 68.0), include_player_velocities=False,
                    PlayerMarkerSize=10, PlayerAlpha=0.7):
    """ save_match_clip( hometeam, awayteam, fpath )

    Generates a movie from Metrica tracking data, saving it in the 'fpath' directory with name 'fname'

    Parameters
    -----------
        hometeam: home team tracking data DataFrame. Movie will be created from all rows in the DataFrame
        awayteam: away team tracking data DataFrame. The indices *must* match those of the hometeam DataFrame
        fpath: directory to save the movie
        fname: movie filename. Default is 'clip_test.mp4'
        fig,ax: Can be used to pass in the (fig,ax) objects of a previously generated pitch. Set to (fig,ax) to use an existing figure, or None (the default) to generate a new pitch plot,
        frames_per_second: frames per second to assume when generating the movie. Default is 25.
        team_colors: Tuple containing the team colors of the home & away team. Default is 'r' (red, home team) and 'b' (blue away team)
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        include_player_velocities: Boolean variable that determines whether player velocities are also plotted (as quivers). Default is False
        PlayerMarkerSize: size of the individual player marlers. Default is 10
        PlayerAlpha: alpha (transparency) of player markers. Defaault is 0.7

    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)

    """
    # check that indices match first
    assert np.all(hometeam.index == awayteam.index), "Home and away team Dataframe indices must be the same"
    # in which case use home team index
    index = hometeam.index
    # Set figure and movie settings
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='Tracking Data', artist='Matplotlib', comment='Metrica tracking data clip')
    writer = FFMpegWriter(fps=frames_per_second, metadata=metadata)
    fname = fpath + '/' + fname + '.mp4'  # path and filename
    # create football pitch
    if figax is None:
        fig, ax = plot_pitch(field_dimen=field_dimen)
    else:
        fig, ax = figax
    fig.set_tight_layout(True)
    # Generate movie
    print("Generating movie...", end='')
    with writer.saving(fig, fname, 100):
        for i in index:
            figobjs = []  # this is used to collect up all the axis objects so that they can be deleted after each iteration
            for team, color in zip([hometeam.loc[i], awayteam.loc[i]], team_colors):
                x_columns = [c for c in team.keys() if
                             c[-2:].lower() == '_x' and c != 'ball_x']  # column header for player x positions
                y_columns = [c for c in team.keys() if
                             c[-2:].lower() == '_y' and c != 'ball_y']  # column header for player y positions
                objs, = ax.plot(team[x_columns], team[y_columns], color + 'o', MarkerSize=PlayerMarkerSize,
                                alpha=PlayerAlpha)  # plot player positions
                figobjs.append(objs)
                if include_player_velocities:
                    vx_columns = ['{}_vx'.format(c[:-2]) for c in x_columns]  # column header for player x positions
                    vy_columns = ['{}_vy'.format(c[:-2]) for c in y_columns]  # column header for player y positions
                    objs = ax.quiver(team[x_columns], team[y_columns], team[vx_columns], team[vy_columns], color=color,
                                     scale_units='inches', scale=10., width=0.0015, headlength=5, headwidth=3,
                                     alpha=PlayerAlpha)
                    figobjs.append(objs)
            # plot ball
            objs, = ax.plot(team['ball_x'], team['ball_y'], 'ko', MarkerSize=6, alpha=1.0, LineWidth=0)
            figobjs.append(objs)
            # include match time at the top
            frame_minute = int(team['Time [s]'] / 60.)
            frame_second = (team['Time [s]'] / 60. - frame_minute) * 60.
            timestring = "%d:%1.2f" % (frame_minute, frame_second)
            objs = ax.text(-2.5, field_dimen[1] / 2. + 1., timestring, fontsize=14)
            figobjs.append(objs)
            writer.grab_frame()
            # Delete all axis objects (other than pitch lines) in preperation for next frame
            for figobj in figobjs:
                figobj.remove()
    print("done")
    plt.clf()
    plt.close(fig)