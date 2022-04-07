import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate


import re
import numpy as np
import pandas as pd
import sqlite3
import json

import flask
import glob
import os

extension = '.jpg'
image_directory = '/home/skye/Documents/Scripts/FRC/2022/scouting/2022_fim_lakeview/'
list_of_images = [os.path.basename(x) for x in glob.glob(f'{image_directory}*{extension}')]
teams = [x[:-6] for x in list_of_images]
#print(teams)
static_image_route = '/2022_fim_lakeview/'
db_directory = "/home/skye/Documents/Scripts/FRC/2022/scouting/cblite_lakeview_m52/wildrank.cblite2/db.sqlite3"


def main():
    con = sqlite3.connect(db_directory)
    cursor = con.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    cursor.execute("SELECT json FROM revs where doc_type = 'pit'")
    df_user_results = pd.read_sql_query("SELECT json FROM revs where doc_type = 'pit'", con)
    col_names = list(map(lambda x: x[0], cursor.description))
    results = cursor.fetchall()
    df_user_results_e = df_user_results.join(df_user_results['json'].apply(json.loads).apply(pd.Series))
    df_pit_results = df_user_results_e.join(pd.DataFrame(list(df_user_results_e['data'])))
    cols = ['team_key', 'length', 'width', 'height', 'weight', 'drivetrain', 'external_intake',
        'retract_intake', 'cargo_scorer', 'vision', 'auto_location', 'auto_drives', 'auto_shots_high',
        'auto_shots_low', 'climb_height', 'climb_time']
    df_pit_results_e = df_pit_results[cols]
    
    app = dash.Dash()

    app.layout = html.Div([
        html.H4('Team to Analyze'),
        dcc.Dropdown(
            id='image-dropdown',
            options=[{'label': i, 'value': i} for i in sorted(list(set(teams)))],
            value=teams[0]),
        html.Br(),

        html.Section(id="slideshow", children=[
            html.Div(id="slideshow-container", children=[
                html.Div(id="image"),
                dcc.Interval(id='interval', interval=3000),
            ],  style={'width': '50%', 'height': '100%'}
            ),
            html.Div([
                html.Br(),
            ]),
            html.Div([
                dcc.Textarea(id='pit_data',
                            value='Textarea content initialized\nwith multiple lines of text',
                            style={'width': '300%', 'height': '100%',  'fontSize': '24px'},
                ),
            ]),

        ], style = {'display':'flex', 'flex-direction':'row'}),
    ])

    @app.callback(Output('pit_data','value'),
                  Input('image-dropdown','value'))
    def update_data(value):
        print(f'frc{value}')
        pd.options.display.max_colwidth =10000
        out = df_pit_results_e[df_pit_results_e['team_key']==f'frc{value}']
        #out = out.replace(',', '\n')
        print(out.T.to_string())
        return out.T.to_string()

    @app.callback(Output('image', 'children'),
                  [Input('interval', 'n_intervals'),
                   Input('image-dropdown', 'value')])
    def display_image(n, value):
        divisor = teams.count(value)            # the number of images the team has in the folder
        for x in range(divisor):
            if n == None or n % divisor == x:
                return html.Img(src = image_directory + value + f'_{x+1}'+ extension, 
                                style={'height':'90%', 'width':'90%'})

    # Add a static image route that serves images from desktop
    # Be *very* careful here - you don't want to serve arbitrary files
    # from your computer or server
    @app.server.route(f'{image_directory}<image_path>{extension}')
    def serve_image(image_path):
        image_name = f'{image_path}{extension}'
        if image_name not in list_of_images:
            raise Exception(f'"{image_path}" is excluded from the allowed static files')
        return flask.send_from_directory(image_directory, image_name)

    app.run_server(debug=True)

if __name__ == '__main__':
    main()