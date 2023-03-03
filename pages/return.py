import yfinance as yf
from dash import Dash, html, dcc, Input, Output, State, ctx, register_page, callback
import pandas as pd
import plotly.express as px
from indicator.ma_graph import SMA, EMA
from indicator.ma_ret import cross_return, cross_bs
from indicator.bolband_graph import bolband
from indicator.bolband_ret import bol_return, bol_bs
from indicator.sup_res import supres
from indicator.rsi_graph import rsi
from indicator.rsi_ret import rsi_return, rsi_bs
import numpy as np
import base64
import io
import dash_uploader as du

ret = register_page(__name__)

close = pd.DataFrame()
port = pd.DataFrame(data=[100 for i in range(100)])

fig = px.line(port, title='ticker', markers=True)
fig2 = px.line(port, title='ticker', markers=True)
fig4 = px.line(port, title='ticker', markers=True)

#color: https://plotly.com/python/discrete-color/
#display: https://developer.mozilla.org/en-US/docs/Web/CSS/display

name = 'ticker'

layout = html.Div(children=[
    html.H1(children='Return'),

    html.Div(
        children=[
            dcc.Input(id='ticker', debounce=True, placeholder='Ticker', required=True, value='',),
            dcc.Input(id='interval', debounce=True, placeholder='Interval:1m,1h,1d,1wk,1mo', required=True, value='',),
        ],
    ),
    
    dcc.Link('Find tickers from yfinance', href='https://finance.yahoo.com/lookup',style={'textAlign': 'center','font-size':'0.7vw'}),

    html.P(),

    html.Div(
        style={'display' : 'flex'}, 
        children=[
            dcc.Input(id='period', debounce=True, placeholder='Period:1d,1mo,1y,ytd,max', value='',),
            html.Button('Get Data', id='get_data1', n_clicks=0),
        ],
    ),

    html.P('or'),
    
    html.Div(
        style={'display' : 'flex'}, 
        children=[
            dcc.Input(id='start', debounce=True, placeholder='start:YYYY-MM-DD', value='',),
            dcc.Input(id='end', debounce=True, placeholder='end:YYYY-MM-DD', value='',),
            html.Button('Get Data', id='get_data2', n_clicks=0),
        ],
    ),

    html.P('or'),

    # dcc.Upload(id='upload', children=html.Button('Upload File (CSV)', id='upload_btn')),
    html.Button(id='upload_btn', children=dcc.Upload(id='upload', children='Upload File (CSV)')),
    # html.Div(id='upload_output', children=''),

    html.P('or'),

    html.Button('Reset Data', id='reset_data', n_clicks=0),

    dcc.Graph(
        id='example-graph',
        figure=fig
    ),

    dcc.Graph(
        id='example-graph4',
        figure=fig4,
        style={'display' : 'none'},
    ),

    html.Button('Download File (CSV)', id='download_btn'),
    dcc.Download(id='download'),

    html.Div(
        id='indicator',
        style={'display' : 'block'}, 
        children=[
            html.Div(
                style={'display' : 'block'}, 
                children=[
                    dcc.Link('Simple Moving Average', href='https://www.investopedia.com/terms/s/sma.asp',style={'textAlign': 'center','font-size':'vw'}),
                    dcc.Input(id='sma_input', debounce=True, placeholder='SMA days', value='',),
                    html.Button('SMA', id='btn1', n_clicks=0),
                ],
            ),
            html.Div(
                style={'display' : 'block'}, 
                children=[
                    dcc.Link('Exponential Moving Average', href='https://www.investopedia.com/terms/e/ema.asp',style={'textAlign': 'center','font-size':'vw'}),
                    dcc.Input(id='ema_input', debounce=True, placeholder='EMA days', value='',),
                    html.Button('EMA', id='btn2', n_clicks=0),
                ],
            ),
            html.Div(
                style={'display' : 'block'}, 
                children=[
                    dcc.Link('Absolute Price Oscillator', href='https://www.marketvolume.com/technicalanalysis/apo.asp',style={'textAlign': 'center','font-size':'vw'}),
                    dcc.Input(id='apo_input1', debounce=True, placeholder='shorter SMA days', value='',),
                    dcc.Input(id='apo_input2', debounce=True, placeholder='longer SMA days', value='',),
                    html.Button('APO', id='btn5', n_clicks=0),
                ],
            ),
            html.Div(
                style={'display' : 'block'}, 
                children=[
                    dcc.Link('Bollinger Band', href='https://www.investopedia.com/terms/b/bollingerbands.asp',style={'textAlign': 'center','font-size':'vw'}),
                    dcc.Input(id='boll_input1', debounce=True, placeholder='SMA days', value='',),
                    dcc.Input(id='boll_input2', debounce=True, placeholder='std factor', value='',),
                    html.Button('Boll', id='btn3', n_clicks=0),
                ],
            ),
            html.Button('RSI', id='btn6', n_clicks=0,),
            # html.Button('sup-res', id='btn4', n_clicks=0, style={'display' : 'inline-block'}, ),
            html.Button('sup-res', id='btn4', n_clicks=0, style={'display' : 'none'}, ),
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
    Output('example-graph4', 'style'),
    Output('example-graph4', 'figure'),
    Input('btn1', 'n_clicks'),
    Input('btn2', 'n_clicks'),
    Input('btn3', 'n_clicks'),
    Input('btn4', 'n_clicks'),
    Input('btn5', 'n_clicks'),
    Input('btn6', 'n_clicks'),
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
    Input('upload', 'contents'),
    State('upload', 'filename'),
    Input('upload_btn', 'n_clicks'),
    Input('reset_data', 'n_clicks'),
)
def change_indicator(btn1, btn2, btn3, btn4, btn5, btn6, ticker, interval, period, get_data1, start, end, get_data2, sma_input, ema_input, boll_input1, 
boll_input2, apo_input1, apo_input2, contents, filename, upload_btn, reset_data):
    global close, port, name
    price = close.copy()

    # print(ctx.triggered_id)
    fig = px.line(port, title=name, markers=True)
    fig2 = px.line(port, title=name, markers=True)
    last = "Final money : cash = 0, stocks values = 0, total = 0"
    first = 'Initial money : 0'
    st = style={'display' : 'none'}
    fig4 = px.line(port, title=name, markers=True)

    if 'reset_data' == ctx.triggered_id:
        close = port.copy()
        fig = px.line(port, title='ticker', markers=True)
        fig2 = px.line(port, title='ticker', markers=True)
        name = 'ticker'

    if 'get_data1' == ctx.triggered_id:
        tick = yf.Ticker(ticker)
        data = tick.history(interval=interval, period=period)
        close = data[['Close']]

        name = ticker
        fig = px.line(close, title=name, markers=True)
    
    if 'get_data2' == ctx.triggered_id:
        tick = yf.Ticker(ticker)
        data = tick.history(interval=interval, start=start, end=end)
        close = data[['Close']]

        name = ticker
        fig = px.line(close, title=name, markers=True)

    if 'upload' == ctx.triggered_id:
        # content_type, content_string = contents.split(',')
        # decoded = base64.b64decode(content_string)
        # filenames = io.StringIO(decoded.decode('utf-8'))
        # print(filename)
        try:
            if 'csv' in filename:
                # Assume that the user uploaded a CSV file
                content_type, content_string = contents.split(',')
                decoded = base64.b64decode(content_string)
                filenames = io.StringIO(decoded.decode('utf-8'))

                data = pd.read_csv(filenames)
                data.index = data['Date']
                close = data[['Close']]

                name = filename
                fig = px.line(close, title=name, markers=True)
            else:
                print('Error: Only CSV files are supported')
                
        except Exception as e:
            print(e)
            print('There was an error processing this file.')
        
    if 'btn1' == ctx.triggered_id:
        price = price.assign(sma=SMA(price,sma_input))
        fig = px.line(price, title=name, markers=True)
        fig['data'][0].line.color = '#636efa'
        fig['data'][1].line.color = '#ab63fa'

        ret = cross_return(price,0,1)
        fig2 = px.line(ret, title=name, markers=True)

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
        fig = px.line(price[['Close', 'ema']], title=name, markers=True)
        fig['data'][0].line.color = '#636efa'
        fig['data'][1].line.color = '#ab63fa'

        ret = cross_return(price,0,1)
        fig2 = px.line(ret, title=name, markers=True)

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
        fig = px.line(price, title=name, markers=True)
        fig['data'][0].line.color = '#636efa'
        fig['data'][1].line.color = '#ab63fa'
        fig['data'][2].line.color = '#ff97ff'

        ret = cross_return(price,1,2)
        fig2 = px.line(ret, title=name, markers=True)

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
        fig = px.line(price, title=name, markers=True)
        fig['data'][0].line.color = '#636efa'
        fig['data'][1].line.color = '#ab63fa'
        fig['data'][2].line.color = '#ab63fa'

        ret = bol_return(price)
        fig2 = px.line(ret, title=name, markers=True)

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

    if 'btn6' == ctx.triggered_id:
        price['rsi'] = rsi(price)
        rsi_res = rsi(price)
        fig4 = px.line(price['rsi'], title='RSI', markers=True)
        fig4['data'][0].line.color = '#ab63fa'
        fig4.add_hline(y = 70, line_color="#ffa15a")
        fig4.add_hline(y = 30, line_color="#ffa15a")
        fig4.update_yaxes(range = [0,100])

        fig = px.line(price['Close'], title=name, markers=True)
        st = style={}

        # print(price)
        ret = rsi_return(price)
        fig2 = px.line(ret, title=name, markers=True)

        print(price)
        rsi_bs(price)
        print(price)
        buy = price.loc[price['b/s'] == 'buy']
        for xx in buy.index : fig.add_vline(x = xx, line_color="#00cc96")
        sell = price.loc[price['b/s'] == 'sell']
        for xx in sell.index : fig.add_vline(x = xx, line_color="#ef553b")
        for xx in buy.index : fig2.add_vline(x = xx, line_color="#00cc96")
        for xx in sell.index : fig2.add_vline(x = xx, line_color="#ef553b")
        for xx in buy.index : fig4.add_vline(x = xx, line_color="#00cc96")
        for xx in sell.index : fig4.add_vline(x = xx, line_color="#ef553b")

        if buy.shape[0] == sell.shape[0]:
            last = f"Final money : cash = {ret.iloc[-1][0]:.2f}, stocks values = 0, total = {ret.iloc[-1][0]:.2f}"    
        elif buy.shape[0]-sell.shape[0] == 1:
            last = f"Final money : cash = {ret.iloc[-1][0]:.2f}, stocks values = {close.iloc[-1][0]:.2f}, total = {ret.iloc[-1][0]+close.iloc[-1][0]:.2f}"
        first = f"Initial money : {close['Close'].mean()*10:.2f}"


    if 'btn4' == ctx.triggered_id:
        #sort_values() ไม่ได้ inplace
        price = supres(price)
        # print(price)
        fig = px.line(price['Close'], title=name, markers=True)
        fig['data'][0].line.color = '#636efa'
        crit = price.loc[price['crit'] == 1]
        # print(crit.shape)
        for yy in crit['Close'] : fig.add_hline(y = yy, line_color="#ab63fa")


    return fig, fig2, first, last, st, fig4
    # return fig

@callback(
    Output('download', 'data'),
    Input('download_btn', 'n_clicks'),
)
def download_csv(download_btn):
    if 'download_btn' == ctx.triggered_id:
        return dcc.send_data_frame(close.to_csv, f"{name}.csv")

# @callback(
#     Output('upload_output', 'children'),
#     Input('upload_btn', 'contents'),
# )
# def upload(list_of_contents, list_of_names, list_of_dates):
    # if list_of_contents is not None:
    #     children = [
    #         parse_contents(c, n, d) for c, n, d in
    #         zip(list_of_contents, list_of_names, list_of_dates)]
    # return children
    # print(list_of_contents)