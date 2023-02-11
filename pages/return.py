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

#color: https://plotly.com/python/discrete-color/
#display: https://developer.mozilla.org/en-US/docs/Web/CSS/display

layout = html.Div(children=[
    html.H1(children='Return'),

    html.Div(
        children=[
            dcc.Input(id='ticker', debounce=True, placeholder='Ticker', required=True,),
            dcc.Input(id='interval', debounce=True, placeholder='Interval:1m,1h,1d,1wk,1mo', required=True,),
        ],
    ),
    
    dcc.Link('Find tickers from yfinance', href='https://finance.yahoo.com/lookup',style={'textAlign': 'center','font-size':'0.7vw'}),

    html.P(),

    html.Div(
        style={'display' : 'flex'}, 
        children=[
            dcc.Input(id='period', debounce=True, placeholder='Period:1d,1mo,1y,ytd,max'),
            html.Button('Get Data', id='get_data1', n_clicks=0),
        ],
    ),

    html.P('or'),
    
    html.Div(
        style={'display' : 'flex'}, 
        children=[
            dcc.Input(id='start', debounce=True, placeholder='start:YYYY-MM-DD'),
            dcc.Input(id='end', debounce=True, placeholder='end:YYYY-MM-DD'),
            html.Button('Get Data', id='get_data2', n_clicks=0),
        ],
    ),

    dcc.Graph(
        id='example-graph',
        figure=fig
    ),

    html.Div(
        id='indicator',
        style={'display' : 'block'}, 
        children=[
            html.Div(
                style={'display' : 'block'}, 
                children=[
                    dcc.Link('Simple Moving Average', href='https://www.investopedia.com/terms/s/sma.asp',style={'textAlign': 'center','font-size':'vw'}),
                    dcc.Input(id='sma_input', debounce=True, placeholder='SMA days'),
                    html.Button('SMA', id='btn1', n_clicks=0),
                ],
            ),
            html.Div(
                style={'display' : 'block'}, 
                children=[
                    dcc.Link('Exponential Moving Average', href='https://www.investopedia.com/terms/e/ema.asp',style={'textAlign': 'center','font-size':'vw'}),
                    dcc.Input(id='ema_input', debounce=True, placeholder='EMA days'),
                    html.Button('EMA', id='btn2', n_clicks=0),
                ],
            ),
            html.Div(
                style={'display' : 'block'}, 
                children=[
                    dcc.Link('Absolute Price Oscillator', href='https://www.marketvolume.com/technicalanalysis/apo.asp',style={'textAlign': 'center','font-size':'vw'}),
                    dcc.Input(id='apo_input1', debounce=True, placeholder='shorter SMA days'),
                    dcc.Input(id='apo_input2', debounce=True, placeholder='longer SMA days'),
                    html.Button('APO', id='btn5', n_clicks=0),
                ],
            ),
            html.Div(
                style={'display' : 'block'}, 
                children=[
                    dcc.Link('Bollinger Band', href='https://www.investopedia.com/terms/b/bollingerbands.asp',style={'textAlign': 'center','font-size':'vw'}),
                    dcc.Input(id='boll_input1', debounce=True, placeholder='SMA days'),
                    dcc.Input(id='boll_input2', debounce=True, placeholder='std factor'),
                    html.Button('Boll', id='btn3', n_clicks=0),
                ],
            ),
            html.Button('sup-res', id='btn4', n_clicks=0, style={'display' : 'inline-block'}, ),
        ]
    ),
    
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
    Input('btn5', 'n_clicks'),
    Input('ticker', 'value'), 
    Input('interval', 'value'), 
    Input('period', 'value'), 
    Input('get_data1', 'n_clicks'),
    Input('start', 'value'), 
    Input('end', 'value'), 
    Input('get_data2', 'n_clicks'),
    Input('sma_input', 'value'),
    Input('ema_input', 'value'),
    Input('boll_input1', 'value'),
    Input('boll_input2', 'value'),
    Input('apo_input1', 'value'),
    Input('apo_input2', 'value'),
)
def change_indicator(btn1, btn2, btn3, btn4, btn5, ticker, interval, period, get_data1, start, end, get_data2, sma_input, ema_input, boll_input1, 
boll_input2, apo_input1, apo_input2):
    global close, port
    price = close.copy()

    fig = px.line(port, title='ticker', markers=True)
    fig2 = px.line(port, title='ticker', markers=True)
    last = "Final money : cash = 0, stocks values = 0, total = 0"
    first = 'Initial money : 0'

    if 'get_data1' == ctx.triggered_id:
        tick = yf.Ticker(ticker)
        data = tick.history(interval=interval, period=period)
        close = data[['Close']]

        fig = px.line(close, title=ticker, markers=True)
    
    if 'get_data2' == ctx.triggered_id:
        tick = yf.Ticker(ticker)
        data = tick.history(interval=interval, start=start, end=end)
        close = data[['Close']]

        fig = px.line(close, title=ticker, markers=True)

    if 'btn1' == ctx.triggered_id:
        price = price.assign(sma=SMA(price,sma_input))
        fig = px.line(price, title=ticker, markers=True)
        fig['data'][0].line.color = '#636efa'
        fig['data'][1].line.color = '#ab63fa'

        ret = cross_return(price,0,1)
        fig2 = px.line(ret, title=ticker, markers=True)

        # price = price.assign(bs=cross_bs(price))
        cross_bs(price,0,1)
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
        price = price.assign(ema=EMA(price,ema_input))
        fig = px.line(price[['Close', 'ema']], title=ticker, markers=True)
        fig['data'][0].line.color = '#636efa'
        fig['data'][1].line.color = '#ab63fa'

        ret = cross_return(price,0,1)
        fig2 = px.line(ret, title=ticker, markers=True)

        # price = price.assign(bs=cross_bs(price))
        cross_bs(price,0,1)
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

    if 'btn5' == ctx.triggered_id:
        price = price.assign(sma1=SMA(price,apo_input1))
        price = price.assign(sma2=SMA(price['Close'],apo_input2))
        fig = px.line(price, title=ticker, markers=True)
        fig['data'][0].line.color = '#636efa'
        fig['data'][1].line.color = '#ab63fa'
        fig['data'][2].line.color = '#ff97ff'

        ret = cross_return(price,1,2)
        fig2 = px.line(ret, title=ticker, markers=True)

        # price = price.assign(bs=cross_bs(price))
        cross_bs(price,1,2)
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
        bolband(price,boll_input1,boll_input2)
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