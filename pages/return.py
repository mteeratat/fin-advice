import yfinance as yf
from dash import Dash, html, dcc, Input, Output, State, ctx
import pandas as pd
import plotly.express as px
from indicator.ma import SMA, EMA

app = Dash(__name__)

ptt = yf.Ticker('ptt.bk')
data = ptt.history(interval='1d', period='1y')
close = data[['Close']]

# close = close.assign(sma=SMA(data[['Close']]))
# close = close.assign(ema=EMA(data[['Close']]))
# print(close)
fig = px.line(close, title='PTT')

app.layout = html.Div(children=[
    html.H1(children='Hello Dashyyyyyy'),

    html.Div(children='''
        Dash: A web application framework for your data.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    ),
    # html.Div(id='example-graph'),

    html.Button('SMA', id='btn1', n_clicks=0),
    html.Button('EMA', id='btn2', n_clicks=0),
    html.Div(id='change'),
])

@app.callback(
    # Output('change', 'children'),
    Output('example-graph', 'figure'),
    Input('btn1', 'n_clicks'),
    Input('btn2', 'n_clicks'),
)
def change_indicator(btn1, btn2):
    global close
    price = close
    msg = 'Hello'
    fig = px.line(price, title='PTT')
    if 'btn1' == ctx.triggered_id:
        msg = 'Button 1 pressed'
        price = price.assign(sma=SMA(data[['Close']]))
        fig = px.line(price, title='PTT')
    if 'btn2' == ctx.triggered_id:
        msg = 'Button 2 pressed'
        price = price.assign(ema=EMA(data[['Close']]))
        fig = px.line(price, title='PTT')
    # print(price)
    # print(msg)
    print(fig)
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)