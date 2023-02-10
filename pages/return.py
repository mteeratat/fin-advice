import yfinance as yf
from dash import Dash, html, dcc, Input, Output, State, ctx, register_page, callback
import pandas as pd
import plotly.express as px
from indicator.ma_graph import SMA, EMA, bolband
from indicator.ma_ret import cross_return, cross_bs, bol_return, bol_bs
from indicator.sup_res import supres
import numpy as np

ret = register_page(__name__)

close = pd.DataFrame()
port = pd.DataFrame(data=[100 for i in range(100)])

fig = px.line(port, title='ticker', markers=True)
fig2 = px.line(port, title='ticker', markers=True)

layout = html.Div(children=[
    html.H1(children='Return'),

    dcc.Input(id='ticker', debounce=True),
    html.Button('Search', id='search', n_clicks=0),
    dcc.Link('Find tickers from yfinance', href='https://finance.yahoo.com/lookup',style={'textAlign': 'center','font-size':'0.7vw'}),

    dcc.Graph(
        id='example-graph',
        figure=fig
    ),

    html.Button('SMA', id='btn1', n_clicks=0),
    html.Button('EMA', id='btn2', n_clicks=0),
    html.Button('Boll', id='btn3', n_clicks=0),
    html.Button('sup-res', id='btn4', n_clicks=0),
    
    html.P(id='init', title='Initial money', children=''),

    dcc.Graph(
        id='example-graph2',
        figure=fig2
    ),

    html.P(id='final', title='Final money', children=''),

])

@callback(
    Output('example-graph', 'figure'),
    Output('example-graph2', 'figure'),
    Output('init', 'children'),
    Output('final', 'children'),
    Input('btn1', 'n_clicks'),
    Input('btn2', 'n_clicks'),
    Input('btn3', 'n_clicks'),
    Input('btn4', 'n_clicks'),
    Input('ticker', 'value'), 
    Input('search', 'n_clicks'),
)
def change_indicator(btn1, btn2, btn3, btn4, ticker, search):
    global close, port
    price = close.copy()

    fig = px.line(port, title='ticker', markers=True)
    fig2 = px.line(port, title='ticker', markers=True)
    last = "Final money : cash = 0, stocks values = 0, total = 0"
    first = 'Initial money : 0'

    if 'search' == ctx.triggered_id:
        tick = yf.Ticker(ticker)
        data = tick.history(interval='1wk', period='1y')
        close = data[['Close']]

        fig = px.line(close, title=ticker, markers=True)

    if 'btn1' == ctx.triggered_id:
        price = price.assign(sma=SMA(price))
        fig = px.line(price, title=ticker, markers=True)
        fig['data'][0].line.color = '#636efa'
        fig['data'][1].line.color = '#ab63fa'

        ret = cross_return(price)
        fig2 = px.line(ret, title=ticker, markers=True)

        # price = price.assign(bs=cross_bs(price))
        cross_bs(price)
        buy = price.loc[price['b/s'] == 'buy']
        for xx in buy.index : fig.add_vline(x = xx, line_color="#00cc96")
        sell = price.loc[price['b/s'] == 'sell']
        for xx in sell.index : fig.add_vline(x = xx, line_color="#ef553b")
        for xx in buy.index : fig2.add_vline(x = xx, line_color="#00cc96")
        for xx in sell.index : fig2.add_vline(x = xx, line_color="#ef553b")

        if buy.shape[0] == sell.shape[0]:
            last = f"Final money : cash = {ret.iloc[-1][0]:.2f}, stocks values = 0, total = {ret.iloc[-1][0]:.2f}"    
        elif buy.shape[0]-sell.shape[0] == 1:
            last = f"Final money : cash = {ret.iloc[-1][0]:.2f}, stocks values = {close.iloc[-1][0]:.2f}, total = {ret.iloc[-1][0]+close.iloc[-1][0]:.2f}"
        first = f"Initial money : {close['Close'].mean()*10:.2f}"

    if 'btn2' == ctx.triggered_id:
        price = price.assign(ema=EMA(price))
        fig = px.line(price[['Close', 'ema']], title=ticker, markers=True)
        fig['data'][0].line.color = '#636efa'
        fig['data'][1].line.color = '#ab63fa'

        ret = cross_return(price)
        fig2 = px.line(ret, title=ticker, markers=True)

        # price = price.assign(bs=cross_bs(price))
        cross_bs(price)
        buy = price.loc[price['b/s'] == 'buy']
        for xx in buy.index : fig.add_vline(x = xx, line_color="#00cc96")
        sell = price.loc[price['b/s'] == 'sell']
        for xx in sell.index : fig.add_vline(x = xx, line_color="#ef553b")
        for xx in buy.index : fig2.add_vline(x = xx, line_color="#00cc96")
        for xx in sell.index : fig2.add_vline(x = xx, line_color="#ef553b")

        if buy.shape[0] == sell.shape[0]:
            last = f"Final money : cash = {ret.iloc[-1][0]:.2f}, stocks values = 0, total = {ret.iloc[-1][0]:.2f}"    
        elif buy.shape[0]-sell.shape[0] == 1:
            last = f"Final money : cash = {ret.iloc[-1][0]:.2f}, stocks values = {close.iloc[-1][0]:.2f}, total = {ret.iloc[-1][0]+close.iloc[-1][0]:.2f}"
        first = f"Initial money : {close['Close'].mean()*10:.2f}"

    if 'btn3' == ctx.triggered_id:
        bolband(price)
        # price = price.assign(uband=bol['uband'], lband=bol['lband'])
        fig = px.line(price, title=ticker, markers=True)
        fig['data'][0].line.color = '#636efa'
        fig['data'][1].line.color = '#ab63fa'
        fig['data'][2].line.color = '#ab63fa'

        ret = bol_return(price)
        fig2 = px.line(ret, title=ticker, markers=True)

        # price = price.assign(bs=bol_pos_bs(price))
        bol_bs(price)
        buy = price.loc[price['b/s'] == 'buy']
        for xx in buy.index : fig.add_vline(x = xx, line_color="#00cc96")
        sell = price.loc[price['b/s'] == 'sell']
        for xx in sell.index : fig.add_vline(x = xx, line_color="#ef553b")
        for xx in buy.index : fig2.add_vline(x = xx, line_color="#00cc96")
        for xx in sell.index : fig2.add_vline(x = xx, line_color="#ef553b")

        if buy.shape[0] == sell.shape[0]:
            last = f"Final money : cash = {ret.iloc[-1][0]:.2f}, stocks values = 0, total = {ret.iloc[-1][0]:.2f}"    
        elif buy.shape[0]-sell.shape[0] == 1:
            last = f"Final money : cash = {ret.iloc[-1][0]:.2f}, stocks values = {close.iloc[-1][0]:.2f}, total = {ret.iloc[-1][0]+close.iloc[-1][0]:.2f}"
        first = f"Initial money : {close['Close'].mean()*10:.2f}"

    if 'btn4' == ctx.triggered_id:
        #sort_values() ไม่ได้ inplace
        price = supres(price)
        # print(price)
        fig = px.line(price['Close'], title=ticker, markers=True)
        fig['data'][0].line.color = '#636efa'
        crit = price.loc[price['crit'] == 1]
        # print(crit.shape)
        for yy in crit['Close'] : fig.add_hline(y = yy, line_color="#ab63fa")

    return fig, fig2, first, last
    # return fig