from dash import Dash
import dash_core_components as dcc
import dash_html_components as html 
from dash.dependencies import Input, Output
import dash_daq as daq
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

import math 
import re
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.collections import PatchCollection
from matplotlib import cm
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
import sqlite3
import json

import flask
import glob
import os

# ---- Misc Globals ----
# columns we will use
loc_col_list = ["location_1", "location_2", "location_3", "location_4", "location_5", "location_6", "location_7", "location_8"]
cycle_col_list = ['hub_upper_score','hub_lower_score','hub_upper_miss','hub_lower_miss',
                       'location_1', 'location_2', 'location_3', 'location_4', 'location_5',
                       'location_6', 'location_7', 'location_8']

color_right_side_app_UI='blue' #"red" or "blue" 

goal_to_analyze = 1 #0 = unknown, 1 = high, 2=low   

# ---- Utility Functions ----
# Utility Functions and Classes
def digit_cnt(n):
    if n > 0:
        return int(math.log10(n))+1
    elif n == 0:
        return 1
    elif n < 0:
        return int(math.log10(-n))+2
    
# Plotting
def colorFromBivariateData(Z1,Z2,A1,A2,cmap1 = plt.cm.RdBu, cmap2 = plt.cm.PRGn):
    ''' #Find departures Departures from Mean, 
        #Normaize by global values for that location,
        #Rescale values to fit into colormap range (0->255)
    '''
    Z1_plot = np.array(128+(255*((np.subtract(A1,Z1)/A2))), dtype=int)
    Z2_plot = np.array(128+(255*((np.subtract(A2,Z2)/A2))), dtype=int)

    Z1_color = cmap1(Z1_plot)
    Z2_color = cmap2(Z2_plot)

    # Color for each point
    Z_color = np.sum([Z1_color, Z2_color], axis=0)/2.0

    return Z_color, np.vstack((Z1_plot/255, Z2_plot/255))

def filter_data(df_user_results_e, df_tba_results_alliances, teams_to_include=[], teams_to_exclude=['frc000'], goal_to_analyze=1):
    teams_with_data = df_user_results_e['team_key'].unique()
    #print(teams_with_data)

    if not bool(teams_to_exclude):
        teams_to_exclude = ['frc000']    #Need to do a check such that we are always excluding frc000

    if bool(teams_to_include):      
        teams_to_include = [x for x in teams_to_include if x not in teams_to_exclude]
    else:
        teams_to_include = [x for x in teams_with_data if x not in teams_to_exclude]


    print(f'teams to include {teams_to_include} ; teams to exclude:{teams_to_exclude}')

    df_user_results_teams = df_user_results_e[df_user_results_e['team_key'].isin(teams_to_include)]

    #Grab stats for all locations from teams querried
    all_cycle_locs = pd.DataFrame()
    for idx, d in df_user_results_teams.iterrows():
        match = int(''.join(map(str,re.findall('^.*qm([0-9]+)$',d['match_key'])))) #strip the match number from the end of the match key, the only gaurentee we have is the matches are orginized by the left most digit, easier to just snag the match number and link it to an alliance color 
        team = d['team_key']
        df_match = df_tba_results_alliances.loc[match-1]    #fix out 1 indexed match keys to 0 indexed match data frame
        alliance_color = df_match.loc[(df_match['team1'] == team) | \
                                      (df_match['team2'] == team) | \
                                      (df_match['team3'] == team)].index[0]
      
        m = d['data']
        df = pd.DataFrame()
        try:
            df = pd.DataFrame(m['cycles'])
        except:
            print("An exception occurred w/ lack of cycle key (no cycles in this match)")
            continue
            
        if df.empty:                                               # we need to contitnue and jump this iteration of the for loop if there are no cycles
            #print(f"No cycles to parse!")
            continue
        df = df[cycle_col_list]
        df.replace({False: 0, True: 1}, inplace=True)                # cast boolean values across all columns to int
        
        df['loc'] = df[loc_col_list[::-1]].astype(str).sum(axis=1).astype(int).apply(digit_cnt)  # we invert the list w/ [::-1] because we want 1 to be on the left, not 8
        df['loc_checksum'] = df[loc_col_list].sum(axis=1)
        df.loc[df['loc_checksum'] != 1, 'loc'] = 0
        
        df['target_goal'] = df[cycle_col_list[:4][::-1]].astype(str).sum(axis=1).astype(int).apply(digit_cnt) #0 is unknown, 1 is upper hub goal, 2 is lower hub goal
        df['missed_shot'] = df['target_goal'].map({1:0, 2:0, 3:1, 4:1}).fillna(1)    # map target goal vals based on a dict to create the missed shot column
        df['target_goal'] = df['target_goal'].map({1:1, 2:2, 3:1, 4:2}).fillna(0)    # map target goal to true target, regardless of make or miss
        df['target_checksum'] = df[loc_col_list].sum(axis=1)
        df.loc[df['target_checksum'] != 1, 'target_goal'] = 0
        
        df['defended_shot'] = 0  #temporary, as we are not collecting defended shots, this will trickle down and create a bunch of extra json output, but it is easier vs pulling all the defense code out
                                        #this key would need to be added back into cycle column list in order to be used again
    

        loc_swap_dict = {1:4 , 2:3 , 3:2 , 4:1 , 5:8 , 6:7 , 7:6 , 8:5 }
        if alliance_color == color_right_side_app_UI:
            df.replace({"loc":loc_swap_dict}, inplace=True)  
        
        df['match_idx'] = match
        
        all_cycle_locs = all_cycle_locs.append(df[['loc','target_goal','missed_shot','defended_shot','match_idx']])

    all_locs_data = pd.DataFrame(np.arange(0,10,1)).drop(index=0)                  #create df w/ index column w/ ALL POSSIBLE locations
    all_locs_data['scored_by_loc_raw'] = all_cycle_locs['loc'].where((all_cycle_locs['target_goal'] == goal_to_analyze)&(all_cycle_locs['missed_shot'] == 0)).value_counts()
    all_locs_data['missed_by_loc_raw'] = all_cycle_locs['loc'].where((all_cycle_locs['target_goal'] == goal_to_analyze)&(all_cycle_locs['missed_shot'] == 1)).value_counts()
    all_locs_data['scored_by_loc'] = all_cycle_locs['loc'].where((all_cycle_locs['target_goal'] == goal_to_analyze)&(all_cycle_locs['missed_shot'] == 0)).value_counts()/len(all_cycle_locs['target_goal'])
    all_locs_data['missed_by_loc'] = all_cycle_locs['loc'].where((all_cycle_locs['target_goal'] == goal_to_analyze)&(all_cycle_locs['missed_shot'] == 1)).value_counts()/len(all_cycle_locs['target_goal'])

    all_locs_data.fillna(0, inplace=True)                                         #fill with 0 so our plotting can handle it
    all_locs_data['scored_pct_by_loc'] = all_locs_data['scored_by_loc_raw']/(all_locs_data['scored_by_loc_raw']+all_locs_data['missed_by_loc_raw'])
    return all_locs_data

def main():
    # ---- Data Ingest and Mergeing
    # --- Connect to our database and extract info ---
    con = sqlite3.connect("/home/skye/Documents/Scripts/FRC/2022/scouting/cblite/wildrank.cblite2/db.sqlite3")
    cursor = con.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    
    # TBA Retrieved Data
    cursor.execute("SELECT json FROM revs where doc_type = 'match' AND doc_id=1")
    col_names = list(map(lambda x: x[0], cursor.description))
    #print(col_names)
    #print(cursor.fetchall())

    df_tba_results = pd.read_sql_query("SELECT json FROM revs where doc_type = 'match'", con)
    df_tba_results_e = df_tba_results.join(df_tba_results['json'].apply(json.loads).apply(pd.Series))
    df_tba_results_e = df_tba_results_e.sort_values(by=['time'],ignore_index=True).reset_index(drop=True)
    df_tba_results_alliances = pd.concat({k: pd.DataFrame(v).T for k, v in df_tba_results_e['alliances'].items()}, axis=0)
    df_tba_results_alliances = pd.DataFrame(df_tba_results_alliances['team_keys'])

    df_tba_results_alliances[['team1','team2','team3']] = pd.DataFrame(df_tba_results_alliances['team_keys'].tolist(), index= df_tba_results_alliances.index)
    all_teams = pd.unique(df_tba_results_alliances[['team1','team2','team3']].values.ravel('K'))
    team_matches = {}
    for team in all_teams:
        select_indices = list(np.where((df_tba_results_alliances["team1"] == team) | \
                                       (df_tba_results_alliances["team2"] == team) | \
                                       (df_tba_results_alliances["team3"] == team))[0])
        team_matches[team] = select_indices
    # we do this list comprehension thing to handle the differnt number of matches a team has in the schedule if they're a surrogate
    df_alliance_color = pd.DataFrame({ key:pd.Series(value) for key, value in team_matches.items() }).T
    
    # Get our collected data
    cursor.execute("SELECT json FROM revs where doc_type = 'match_result'")
    df_user_results = pd.read_sql_query("SELECT json FROM revs where doc_type = 'match_result'", con)
    col_names = list(map(lambda x: x[0], cursor.description))
    results = cursor.fetchall()
    # convert our column of json data to a df with all variables from that json
    df_user_results_e = df_user_results.join(df_user_results['json'].apply(json.loads).apply(pd.Series))


    # --- Begin Analysis  ---
    x_polygons = [[-46.637,-98.936,-98.936,-29.580,-17.909],
            [-93.280,-40.981,32.982,4.254,-23.924,],
            [93.280,40.981,-32.982,-4.254,23.924],
            [46.637,98.936,98.936,29.580,17.909],
            [-73.457,-163.686,-163.686,-113.686,-113.686,-52.746],
            [-113.686,-163.686,-163.686,-67.801,59.801,39.091,-47.090,-113.686],
            [73.457,163.686,163.686,113.686,113.686,52.746],
            [113.686,163.686,163.686,67.801,-59.801,-39.091,47.090,113.686]]
    y_polygons = [[93.280,40.981,-32.982,-4.254,23.924],
            [-46.637,-98.936,-98.936,-29.580,-17.909],
            [46.637,98.936,98.936,29.580,17.909],
            [-93.280,-40.981,32.982,4.254,-23.924],
            [158.030,67.801,7.375,7.375,47.090,108.030],
            [-7.375,-7.375,-67.801,-163.686,-163.686,-113.686,-113.686,-47.090],
            [-158.030,-67.801,-7.375,-7.375,-47.090,-108.030],
            [7.375,7.375,67.801,163.686,163.686,113.686,113.686,47.090]]

    x_text = [-58.400,-24.190,24.19,58.400,-100.000,-100.000,100.000,100.000,0.000]
    y_text = [24.190,-58.400,58.340,-24.190,100.000,-100.000,100.000,-100.000,0.000]


    goal_dict = {"Unknown":0, "Upper":1, "Lower":2}
    aggregation_dict = {"Match Normalized Percentages":0, "All Recorded Shots":1}

    #Plot the overall the key to these bivariate shooting location miss vs score data
    x_size,y_size = 200,200
    xx, yy = np.mgrid[-x_size/2:x_size/2,-y_size/2:y_size/2]
    C_map, zplot = colorFromBivariateData(xx,yy,xx,yy)     #the underlying colors are the bivariate key for the loc plot, all locs are independ to eachother

    # ---- App Definitions ----
    # Layout 

    app = Dash(__name__)

    app.layout = html.Div([
        html.H4('Interactive plot for team shooting locations'),
        dcc.Graph(id="scatter-plot"),
        
        #html.H4('Range Clipping'),
        #dcc.RangeSlider(
        #id='range-slider',
        #min=0, max=6, step=0.1,
        #marks={0: '0', 6: '2.5'},
        #value=[0.5, 5.5]),
        html.Div([
            html.Div([    
                html.H4('Goal to analyze'),
                dcc.Dropdown(
                id='goal-to-analyze',
                options=[{'label': i, 'value': goal_dict.get(i)} for i in list(goal_dict.keys())],
                value=goal_dict.get("Upper"))
                ], style={'width':'30%'}),
            html.Div([
                html.H4('Aggregation Method'),
                dcc.Dropdown(
                id='aggregation-method',
                options=[{'label': i, 'value': aggregation_dict.get(i)} for i in list(aggregation_dict.keys())],
                value=aggregation_dict.get("All Recorded Shots"))
                ], style={'width':'30%'})
            ], style = {'display':'flex', 'flex-direction':'row'}),

        html.H4('Teams to use as a baseline'),
        dcc.Dropdown(
        id='teams-as-baseline',
        options=[{'label': i[3:], 'value': i} for i in np.sort(all_teams)],
        multi=True,
        value=None),

        html.H4('Teams to exclude from baseline'),
        dcc.Dropdown(
        id='teams-exclude-from-baseline',
        options=[{'label': i[3:], 'value': i} for i in np.sort(all_teams)],
        multi=True,
        value=None),

        html.H4('Teams to analyze'),
        dcc.Dropdown(
        id='teams-to-analyze',
        options=[{'label': i[3:], 'value': i} for i in np.sort(all_teams)],
        multi=True,
        value=np.sort(all_teams)[0])])

    @app.callback(
        Output("scatter-plot", "figure"),
        Input("goal-to-analyze", "value"),
        Input("aggregation-method", "value"), 
        Input("teams-to-analyze", "value"),
        Input("teams-as-baseline", "value"),
        Input("teams-exclude-from-baseline", "value"))
    def update_chart(goal_to_analyze, aggregation_method, teams_to_analyze, teams_as_baseline, teams_to_exclude):
        if not isinstance(teams_to_analyze, list): teams_to_analyze = [teams_to_analyze]    # take care of the annoying loose typing from dash
        if not isinstance(teams_to_exclude, list): teams_to_exclude = [teams_to_exclude]    # take care of the annoying loose typing from dash
        locs_data = filter_data(df_user_results_e, df_tba_results_alliances,
                                teams_to_include=teams_to_analyze, goal_to_analyze=goal_to_analyze)
        all_locs_data = filter_data(df_user_results_e, df_tba_results_alliances,
                                teams_to_include=teams_as_baseline, teams_to_exclude=teams_to_exclude,
                                goal_to_analyze=goal_to_analyze)
        if aggregation_method == 0:
            bi_colors, _bi_z_plot = colorFromBivariateData(locs_data['missed_by_loc'].to_numpy(),
                                    locs_data['scored_by_loc'].to_numpy(),
                                    all_locs_data['missed_by_loc'].to_numpy(),
                                    all_locs_data['scored_by_loc'].to_numpy())
        else if aggregation_method == 1:
                        bi_colors, _bi_z_plot = colorFromBivariateData(locs_data['missed_by_loc_raw'].to_numpy(),
                                    locs_data['scored_by_loc_raw'].to_numpy(),
                                    all_locs_data['missed_by_loc_raw'].to_numpy(),
                                    all_locs_data['scored_by_loc_raw'].to_numpy())

        for idx in range(len(x_polygons)):
            attempts = locs_data.iloc[idx, locs_data.columns.get_loc('scored_by_loc')] +  \
                locs_data.iloc[idx, locs_data.columns.get_loc('missed_by_loc')]
            if(attempts == 0.0):
                bi_colors[idx][3] = 0

        
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.6, 0.4],
            row_heights=[1],
            specs=[[{"type": "scatter"}, {"type": "image"}]])
        fig.update_yaxes(scaleanchor = "x",scaleratio = 1, range=[-180, 180], showticklabels=False)
        fig.update_xaxes(showticklabels=False)
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            template='plotly_white')

        # "boundry lines" are the edge lines in plotly
        for (x, y, c) in zip(x_polygons, y_polygons, bi_colors):
            #print(f'rgb{tuple(list(map(int,c[:3]*255)))}')
            fig.add_trace( go.Scatter(x=x, y=y, fill="toself", 
                                      mode = 'none',
                                      fillcolor = f'rgba{tuple(list(map(int,c*255)))}',                         
                                      name = f"c ={c}"),
                                      row=1, col=1)
        
        #fig_img = px.imshow(C_map, origin='lower')
        #fig.add_trace(fig_img, row=1, col=2)   
        #img = Image.open('/home/skye/Documents/Scripts/FRC/2022/scouting/value_key.png')
        #fig.add_trace(go.Image(z=img), row=1, col=2)

        return fig


    app.run_server(debug=True)

    ''' def update_bar_chart(slider_range):
            values = [1.1 , 2.2, 3.2, 1.2, 3, 4.2, 2.8, 4.9]
            vmin = 0 #values.min()
            vmax = 10 #values.max()
            vmin, vmax = slider_range
            pl_colors = plotly.colors.sequential.Viridis

            fig = go.Figure()
            fig.update_yaxes(scaleanchor = "x",scaleratio = 1)

            # "boundry lines" are the edge lines in plotly
            for (x, y, v) in zip(x_polygons, y_polygons, values):    
                fig.add_trace( go.Scatter(x=x, y=y, fill="toself", 
                                          mode = 'none',
                                          fillcolor = str(get_colors_for_vals(np.array([v]), vmin, vmax, pl_colors)[0]),                            
                                          name = f"v ={v}",                               
                                         )
                              
                             )
            return fig'''

if __name__ == '__main__':
    main()
