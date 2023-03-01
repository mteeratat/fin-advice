from dash import Dash, html, dcc, Input, Output, State, ctx, register_page, callback
import yfinance as yf
import plotly.express as px
import pmdarima as pmd
import pandas as pd
import base64
import io
import pickle
from sklearn.preprocessing import StandardScaler

pre = register_page(__name__)

close = pd.DataFrame()
port = pd.DataFrame(data=[100 for i in range(100)])

fig = px.line(port, title='ticker', markers=True)

name = 'ticker'

layout = html.Div(children=[
    html.H1(children='Predict'),

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

    html.P('or'),

    html.Button('Reset Data', id='reset_data', n_clicks=0),

    dcc.Graph(
        id='example-graph3',
        figure=fig
    ),

    html.Button('Download File (CSV)', id='download_btn'),
    dcc.Download(id='download'),

    html.Div(children='model'),

    html.Button('ARIMA_backtest', id='btn1', n_clicks=0),
    html.Button('ARIMA_forecast', id='btn2', n_clicks=0),
    html.Button('model 3', id='btn3', n_clicks=0),
])

@callback(
    Output('example-graph3', 'figure'),
    Input('btn1', 'n_clicks'),
    Input('btn2', 'n_clicks'),
    Input('ticker', 'value'), 
    Input('interval', 'value'), 
    Input('period', 'value'), 
    Input('get_data1', 'n_clicks'),
    Input('start', 'value'), 
    Input('end', 'value'),
    Input('get_data2', 'n_clicks'),
    Input('upload', 'contents'),
    Input('upload', 'filename'),
    Input('reset_data', 'n_clicks'),
)
def change_model(btn1, btn2, ticker, interval, period, start, end, contents, filename, get_data1, get_data2, reset_data):
    global close, port, name
    price = close.copy()

    # print(ctx.triggered_id)
    fig = px.line(port, title=name, markers=True)

    if 'reset_data' == ctx.triggered_id:
        fig = px.line(port, title='ticker', markers=True)
        fig2 = px.line(port, title='ticker', markers=True)

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
        try:
            if 'csv' in filename:
                # Assume that the user uploaded a CSV file
                content_type, content_string = contents.split(',')
                decoded = base64.b64decode(content_string)
                filenames = io.StringIO(decoded.decode('utf-8'))

                data = pd.read_csv(filenames)
                close = data[['Close']]

                name = filename
                fig = px.line(close, title=name, markers=True)
            else:
                print('Error: Only CSV files are supported')
                
        except Exception as e:
            print(e)
            print('There was an error processing this file.')

    if 'btn1' == ctx.triggered_id:
        # index = int(close.shape[0]*9/10)
        # train = close.iloc[:][:index]
        # test = close.iloc[:][index:]
        # arima_model = arimamodel(train.dropna())
        # print(arima_model.summary())

        # forecastt = arima_model.predict(len(test), alpha=0.05)

        pickled_model = pickle.load(open('model\\arima_trained_backtest.pkl', 'rb'))
        txt = pickle.load(open('model\\arima_data.pkl', 'rb'))

        # print(txt)

        #### backtest #####
        temp = close[['Close']]
        temp['diff'] = temp['Close'] - temp['Close'].shift(1)
        # temp['forecast'] = pickled_model.forecast(5)
        print(len(close))
        temp['forecast'] = pickled_model.predict(close.index[0],close.index[-1])
        index = len(close) - len(temp['forecast'].dropna())

        scaler = StandardScaler()
        # f = temp[['diff']].dropna()
        # print(f)
        scaler.fit(temp[['diff']].dropna())

        temp['compute'] = temp['diff']
        abc = temp[['forecast']].dropna()
        abc.columns = ['diff']
        norm = scaler.transform(abc)
        # print(norm.T[0])
        normm = pd.Series(norm.T[0])
        # temp['compute'].iloc[index:] = temp['forecast'].dropna()
        temp['compute'].iloc[index:] = normm
        temp['cumsum'] = temp['compute'].cumsum()

        print(temp)

        forecast = temp[['forecast']].copy()
        forecast.iloc[index:] = close['Close'].iloc[0] + temp[['cumsum']].iloc[index:]
        # forecast = close['Close'].iloc[0] + temp['cumsum']
        pred = close.assign(predict=forecast)

        print(pred)
        fig = px.line(pred, title='arima backtest', markers=True)

    if 'btn2' == ctx.triggered_id:

        pickled_model = pickle.load(open('model\\arima_trained_forecast.pkl', 'rb'))

        forecast = pickled_model.forecast(5)
        # print(forecast)
        pred = pd.concat([close,forecast])

        # print(pred)
        fig = px.line(pred, title='arima forecast', markers=True)
    
    return fig

def arimamodel(timeseriesarray):
    autoarima_model = pmd.auto_arima(timeseriesarray, 
                              start_p=1, 
                              start_q=2,
                              start_Q=2,
                              max_Q=5,
                              test="adf",
                              trace=True)
    return autoarima_model