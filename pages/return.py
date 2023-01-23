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
# print(port)

fig = px.line(close, title='PTT', markers=True)
fig2 = px.line(port, title='PTT', markers=True)

layout = html.Div(children=[
    html.H1(children='Return'),

    dcc.Graph(
        id='example-graph',
        figure=fig
    ),

    html.Button('SMA', id='btn1', n_clicks=0),
    html.Button('EMA', id='btn2', n_clicks=0),
    
    html.P(children='Initial money = 100 Baht', ),

    dcc.Graph(
        id='example-graph2',
        figure=fig2
    ),

    html.P(id='final', children='Final money = '),

])

@callback(
    Output('example-graph', 'figure'),
    Output('example-graph2', 'figure'),
    Output('final', 'children'),
    Input('btn1', 'n_clicks'),
    Input('btn2', 'n_clicks'),
)
def change_indicator(btn1, btn2):
    global close, port
    price = close
    # msg = 'Hello'
    fig = px.line(price, title='PTT', markers=True)
    fig2 = px.line(port, title='PTT', markers=True)
    if 'btn1' == ctx.triggered_id:
        price = price.assign(sma=SMA(data[['Close']]))
        price = cross_bs(price)
        fig = px.line(price[['Close', 'sma']], title='PTT', markers=True)
        fig['data'][0].line.color = '#636efa'
        fig['data'][1].line.color = '#ab63fa'

        buy = price.loc[price['b/s'] == 'buy']
        for xx in buy.index : fig.add_vline(x = xx, line_color="#00cc96")
        sell = price.loc[price['b/s'] == 'sell']
        for xx in sell.index : fig.add_vline(x = xx, line_color="#ef553b")

        ret = cross_return(price)
        fig2 = px.line(ret, title='PTT', markers=True)
        for xx in buy.index : fig2.add_vline(x = xx, line_color="#00cc96")
        for xx in sell.index : fig2.add_vline(x = xx, line_color="#ef553b")

    if 'btn2' == ctx.triggered_id:
        price = price.assign(ema=EMA(data[['Close']]))
        price = cross_bs(price)
        fig = px.line(price[['Close', 'ema']], title='PTT', markers=True)
        fig['data'][0].line.color = '#636efa'
        fig['data'][1].line.color = '#ab63fa'

        buy = price.loc[price['b/s'] == 'buy']
        for xx in buy.index : fig.add_vline(x = xx, line_color="#00cc96")
        sell = price.loc[price['b/s'] == 'sell']
        for xx in sell.index : fig.add_vline(x = xx, line_color="#ef553b")

        ret = cross_return(price)
        fig2 = px.line(ret, title='PTT', markers=True)
        for xx in buy.index : fig2.add_vline(x = xx, line_color="#00cc96")
        for xx in sell.index : fig2.add_vline(x = xx, line_color="#ef553b")
    
    # print(ret.iloc[-1][0])
    # print(price)
    return fig, fig2, f"Final money = {ret.iloc[-1][0]:.2f} Baht"

def cross_return(price):

    price = cross_pos(price)
    price['spend'] = price['Close'] * price['position']
    price['cash'] = 100 - price['spend'].cumsum()
    res = price[['cash']]

    return res

def cross_pos(price):
    price['signal'] = np.where(price.iloc[:, 0] < price.iloc[:, 1], 1.0, 0.0)
    price['position'] = price['signal'].diff()

    return price

def cross_bs(price):
    price = cross_pos(price)
    price['b/s'] = pd.Series('-', index=price.index)
    price['b/s'] = price['position'].apply(lambda x: 'buy' if x == 1 else ('sell' if x == -1 else '-'))

    return price