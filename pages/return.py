import yfinance as yf
from dash import Dash, html, dcc, Input, Output, State, ctx, register_page, callback
import pandas as pd
import plotly.express as px
from indicator.ma import SMA, EMA, bolband
import numpy as np

ret = register_page(__name__)

ptt = yf.Ticker('ptt.bk')
data = ptt.history(interval='1wk', period='1y')
close = data[['Close']]
port = pd.DataFrame(data=[100 for i in range(close.shape[0])], index=close.index)

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
    html.Button('Boll', id='btn3', n_clicks=0),
    
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
    Input('btn3', 'n_clicks'),
)
def change_indicator(btn1, btn2, btn3):
    global close, port
    price = close.copy()
    fig = px.line(price, title='PTT', markers=True)
    fig2 = px.line(port, title='PTT', markers=True)
    if 'btn1' == ctx.triggered_id:
        price = price.assign(sma=SMA(data[['Close']]))
        fig = px.line(price, title='PTT', markers=True)
        fig['data'][0].line.color = '#636efa'
        fig['data'][1].line.color = '#ab63fa'

        ret = cross_return(price)
        fig2 = px.line(ret, title='PTT', markers=True)

        # price = price.assign(bs=cross_bs(price))
        cross_bs(price)
        buy = price.loc[price['b/s'] == 'buy']
        for xx in buy.index : fig.add_vline(x = xx, line_color="#00cc96")
        sell = price.loc[price['b/s'] == 'sell']
        for xx in sell.index : fig.add_vline(x = xx, line_color="#ef553b")
        for xx in buy.index : fig2.add_vline(x = xx, line_color="#00cc96")
        for xx in sell.index : fig2.add_vline(x = xx, line_color="#ef553b")

    if 'btn2' == ctx.triggered_id:
        price = price.assign(ema=EMA(data[['Close']]))
        fig = px.line(price[['Close', 'ema']], title='PTT', markers=True)
        fig['data'][0].line.color = '#636efa'
        fig['data'][1].line.color = '#ab63fa'

        ret = cross_return(price)
        fig2 = px.line(ret, title='PTT', markers=True)

        # price = price.assign(bs=cross_bs(price))
        cross_bs(price)
        buy = price.loc[price['b/s'] == 'buy']
        for xx in buy.index : fig.add_vline(x = xx, line_color="#00cc96")
        sell = price.loc[price['b/s'] == 'sell']
        for xx in sell.index : fig.add_vline(x = xx, line_color="#ef553b")
        for xx in buy.index : fig2.add_vline(x = xx, line_color="#00cc96")
        for xx in sell.index : fig2.add_vline(x = xx, line_color="#ef553b")
    
    if 'btn3' == ctx.triggered_id:
        bolband(price)
        # price = price.assign(uband=bol['uband'], lband=bol['lband'])
        fig = px.line(price, title='PTT', markers=True)
        fig['data'][0].line.color = '#636efa'
        fig['data'][1].line.color = '#ab63fa'
        fig['data'][2].line.color = '#ab63fa'

        ret = bol_return(price)
        fig2 = px.line(ret, title='PTT', markers=True)

        # price = price.assign(bs=bol_pos_bs(price))
        bol_bs(price)
        buy = price.loc[price['b/s'] == 'buy']
        for xx in buy.index : fig.add_vline(x = xx, line_color="#00cc96")
        sell = price.loc[price['b/s'] == 'sell']
        for xx in sell.index : fig.add_vline(x = xx, line_color="#ef553b")
        for xx in buy.index : fig2.add_vline(x = xx, line_color="#00cc96")
        for xx in sell.index : fig2.add_vline(x = xx, line_color="#ef553b")

    return fig, fig2, f"Final money = {ret.iloc[-1][0]:.2f} Baht"


def cross_pos(price):
    price['signal'] = np.where(price.iloc[:, 0] < price.iloc[:, 1], 1.0, 0.0)
    price['position'] = price['signal'].diff()

    return price

def cross_return(price):
    price = cross_pos(price)
    price['spend'] = price['Close'] * price['position']
    price['cash'] = 100 - price['spend'].cumsum()

    return price[['cash']]

def cross_bs(price):
    price = cross_pos(price)
    price['b/s'] = pd.Series('-', index=price.index)
    price['b/s'] = price['position'].apply(lambda x: 'buy' if x == 1 else ('sell' if x == -1 else '-'))

    return price[['b/s']]

def bol_pos(price):
    # signal -> 0-middle, 1-below lband, 2-above uband
    # position -> 0->1=1=buy, 1->0=-1=neutral, 0->2=2=sell, 2->0=-2=neutral
    price['signal'] = np.where(price.iloc[:, 0] < price.iloc[:, 2], 1.0, np.where(price.iloc[:, 0] > price.iloc[:, 1], 2.0, 0.0))
    price['position'] = price['signal'].diff()

    return price

def bol_return(price):
    price = bol_pos(price)
    bs = price[['position']]
    bs.loc[bs['position']==-1] = 0
    bs.loc[bs['position']==-2] = 0
    bs.loc[bs['position']==2] = -1
    price['spend'] = price['Close'] * bs['position']
    price['cash'] = 100 - price['spend'].cumsum()

    return price[['cash']]

def bol_bs(price):
    price = bol_pos(price)
    price['b/s'] = pd.Series('-', index=price.index)
    price['b/s'] = price['position'].apply(lambda x: 'buy' if x == 1 else ('sell' if x == 2 else '-'))

    return price[['b/s']]