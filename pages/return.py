import yfinance as yf
from dash import Dash, html, dcc, Input, Output, State, ctx, register_page, callback
import pandas as pd
import plotly.express as px
from indicator.ma import SMA, EMA
import numpy as np

ret = register_page(__name__)

ptt = yf.Ticker('ptt.bk')
data = ptt.history(interval='1wk', period='1y')
close = data[['Close']]
port = pd.DataFrame(data=[100 for i in range(close.shape[0])], index=close.index)
print(port)

fig = px.line(close, title='PTT')
fig2 = px.line(port, title='PTT')

layout = html.Div(children=[
    html.H1(children='Return'),

    html.P(children='Initial money = 100 Baht', ),

    dcc.Graph(
        id='example-graph',
        figure=fig
    ),

    dcc.Graph(
        id='example-graph2',
        figure=fig2
    ),

    html.Button('SMA', id='btn1', n_clicks=0),
    html.Button('EMA', id='btn2', n_clicks=0),
    # html.Div(id='change'),
])

@callback(
    # Output('change', 'children'),
    Output('example-graph', 'figure'),
    Output('example-graph2', 'figure'),
    Input('btn1', 'n_clicks'),
    Input('btn2', 'n_clicks'),
)
def change_indicator(btn1, btn2):
    global close, port
    price = close
    msg = 'Hello'
    fig = px.line(price, title='PTT')
    fig2 = px.line(port, title='PTT')
    if 'btn1' == ctx.triggered_id:
        msg = 'Button 1 pressed'
        price = price.assign(sma=SMA(data[['Close']]))
        fig = px.line(price, title='PTT')
        fig2 = px.line(cross_return(price), title='PTT')

    if 'btn2' == ctx.triggered_id:
        msg = 'Button 2 pressed'
        price = price.assign(ema=EMA(data[['Close']]))
        fig = px.line(price, title='PTT')
        fig2 = px.line(cross_return(price), title='PTT')

    print()
    return fig, fig2

def cross_return(price):

    price['signal'] = np.where(price.iloc[:, 0] < price.iloc[:, 1], 1.0, 0.0)
    price['position'] = price['signal'].diff()
    price['spend'] = price['Close'] * price['position']
    price['cash'] = 100 - price['spend'].cumsum()
    # print(price)

    return price[['cash']]
