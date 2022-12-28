from dash import Dash, html, dcc, Input, Output, State, ctx, register_page

home = register_page(__name__, path='/')

layout = html.Div(children=[
    html.H1(children='Home'),

    html.Div(children='''
        Dash: A web prelication framework for your data.
    '''),
])