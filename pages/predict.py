from dash import Dash, html, dcc, Input, Output, State, ctx, register_page

pre = register_page(__name__)

layout = html.Div(children=[
    html.H1(children='Predict'),

    html.Div(children='ticker'),
    html.Div(children='graph'),
    html.Div(children='model'),
])