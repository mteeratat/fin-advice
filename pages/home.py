from dash import Dash, html, dcc, Input, Output, State, ctx, register_page

home = register_page(__name__, path='/')

layout = html.Div(children=[
    html.H1(children='Home', style={'display':'flex', 'justify-content':'center'}),

    html.H1(children='TUTORIAL', style={'display':'flex', 'justify-content':'center'}),
])