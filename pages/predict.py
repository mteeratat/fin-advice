from dash import html, dcc, Input, Output, State, ctx, register_page, callback
import yfinance as yf
import plotly.express as px
import pmdarima as pmd
import pandas as pd
import base64
import io
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np
from datetime import timedelta

predict_page = register_page(__name__)

close = pd.DataFrame()
port = pd.DataFrame(data=[100 for i in range(100)])
pred = pd.DataFrame(data=[100 for i in range(100)])

fig = px.line(port, title='Ticker', markers=True)

name = '{Ticker}'

state = 0

layout = html.Div( children=[
    html.H1(children='Predict', style={'display': 'flex', 'justify-content': 'center'}),

    html.Div(
        children=[
            # dcc.Input(id='ticker', debounce=True, placeholder='Ticker', required=True, value='', style={'flex': 'initial'},),
            dcc.Dropdown(options=['ADVANC','CPN','PTT'], id='ticker', placeholder='Ticker', style={'flex': 0.1},),
            # dcc.Input(id='interval', debounce=True, placeholder='Interval:1m,1h,1d,1wk,1mo', required=True, value='',),
            dcc.Link('Find tickers from yfinance', href='https://finance.yahoo.com/lookup', style={'textAlign': 'center','font-size':'0.7vw'}),
        ],
        style={'display': 'flex', 'justify-content': 'center'},
    ),

    html.P(),
    
    html.Div(
        children=[
            dcc.Dropdown(options=['1d', '1wk', '1mo'], id='interval', placeholder='Interval: time between each period', style={'flex': 0.25},),
        ],
        style={'display': 'flex', 'justify-content': 'center'},
    ),

    html.P(),

    html.Div(
        style={'display' : 'flex', 'justify-content': 'center'}, 
        children=[
            # dcc.Input(id='period', debounce=True, placeholder='Period:1d,1mo,1y,ytd,max', value='',),
            dcc.Dropdown(options=['6mo', '1y', '2y', '3y'], id='period', placeholder='Period: when to get data backward from today', style={'flex': 0.25},),
            html.Button('Get Data', id='get_data1', n_clicks=0),
        ],
    ),

    html.P('or', style={'display': 'flex', 'justify-content': 'center'}),
    
    html.Div(
        style={'display' : 'flex', 'justify-content': 'center'}, 
        children=[
            dcc.Input(id='start', debounce=True, placeholder='start:YYYY-MM-DD', value='',),
            dcc.Input(id='end', debounce=True, placeholder='end:YYYY-MM-DD', value='',),
            html.Button('Get Data', id='get_data2', n_clicks=0),
        ],
    ),

    # html.P('or'),

    # dcc.Upload(id='upload', children=html.Button('Upload File (CSV)', id='upload_btn')),
    html.Button(id='upload_btn', children=dcc.Upload(id='upload', children='Upload File (CSV)'), style={'display' : 'none'}),

    html.P('or', style={'display': 'flex', 'justify-content': 'center'}),

    html.Div(style={'display': 'flex', 'justify-content': 'center'},
        children=[html.Button('Reset Data', id='reset_data', n_clicks=0,),],
    ),
    
    dcc.Graph(
        id='example-graph3',
        figure=fig
    ),

    html.Div(style={'display': 'flex', 'justify-content': 'center'},
        children=[
            html.P(id='eval'),
        ]
    ),

    html.Div(style={'display': 'flex', 'justify-content': 'center'},
        children=[
            html.Button('Download File (CSV)', id='download_btn2', style={'display': 'flex', 'justify-content': 'center'}),
            dcc.Download(id='download2'),
        ],
    ),
    
    html.H2('model', style={'display': 'flex', 'justify-content': 'center'}),
    html.Div(style={'display': 'flex', 'justify-content': 'center'},
        children=[
            html.Button('ARIMA_backtest', id='arima_bt_btn', n_clicks=0),
            html.Button('ARIMA_forecast', id='arima_fc_btn', n_clicks=0),
            html.Button('RandomForest_forecast', id='rf_fc_btn', n_clicks=0),
        ]
    ),

    
])

@callback(
    Output('example-graph3', 'figure'),
    Output('eval', 'children'),
    Input('ticker', 'value'), 
    Input('interval', 'value'), 
    Input('period', 'value'), 
    Input('get_data1', 'n_clicks'),
    Input('start', 'value'), 
    Input('end', 'value'),
    Input('get_data2', 'n_clicks'),
    Input('upload', 'contents'),
    State('upload', 'filename'),
    Input('reset_data', 'n_clicks'),
    Input('arima_bt_btn', 'n_clicks'),
    Input('arima_fc_btn', 'n_clicks'),
    Input('rf_fc_btn', 'n_clicks'),
)
def change_model(ticker, interval, period, start, end, contents, filename, get_data1, get_data2, reset_data, arima_bt_btn, arima_fc_btn, rf_fc_btn):
    global close, port, name, state, pred

    evall = ''

    # print(ctx.triggered_id)
    fig = px.line(port, title=name, markers=True)

    if 'reset_data' == ctx.triggered_id:
        close = port.copy()
        fig = px.line(port, title='ticker', markers=True)
        name = 'ticker'
        state = 0

    if 'get_data1' == ctx.triggered_id:
        tick = yf.Ticker(ticker+'.bk')

        if interval == '1mo':
            data = tick.history(interval='1d', period='3y')
            close = data[['Close']]
            close = close.asfreq('M').bfill()
        else:
            data = tick.history(interval=interval, period=period)
            close = data[['Close']]

        name = ticker
        # print(close)
        fig = px.line(close, title=name, markers=True)
    
    if 'get_data2' == ctx.triggered_id:
        tick = yf.Ticker(ticker)
        data = tick.history(interval=interval, start=start, end=end)
        close = data[['Close']]

        if interval == '1mo':
            close = close.asfreq('M').bfill()
        elif interval == '1wk':
            close = close.asfreq('5b').ffill()
        elif interval == '1d':
            close = close.asfreq('b').ffill()

        name = ticker
        fig = px.line(close, title=name, markers=True)

    if 'upload' == ctx.triggered_id:
        # content_type, content_string = contents.split(',')
        # decoded = base64.b64decode(content_string)
        # filenames = io.StringIO(decoded.decode('utf-8'))
        # print(filename)
        # print(contents)
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

    # if 'arima_bt_btn' == ctx.triggered_id:
    #     # index = int(close.shape[0]*9/10)
    #     # train = close.iloc[:][:index]
    #     # test = close.iloc[:][index:]
    #     # arima_model = arimamodel(train.dropna())
    #     # print(arima_model.summary())

    #     # forecastt = arima_model.predict(len(test), alpha=0.05)

    #     pickled_model = pickle.load(open('model\\arima_trained_backtest.pkl', 'rb'))
    #     txt = pickle.load(open('model\\arima_data.pkl', 'rb'))

    #     # print(txt)

    #     #### backtest #####
    #     temp = close[['Close']]
    #     # temp = close[['Close']].asfreq('d').ffill()
    #     temp['diff'] = temp['Close'] - temp['Close'].shift(1)
    #     # temp['forecast'] = pickled_model.forecast(5)
    #     # print(type(close.index[0]))
    #     # print(close.index[0])
    #     temp['forecast'] = pickled_model.predict(start=close.index[0], end=close.index[-1])
    #     index = len(close) - len(temp['forecast'].dropna())
    #     # print(temp)

    #     scaler = StandardScaler()
    #     # f = temp[['diff']].dropna()
    #     # print(f)
    #     scaler.fit(temp[['diff']].dropna())

    #     temp['compute'] = temp['diff']
    #     abc = temp[['forecast']].dropna()
    #     abc.columns = ['diff']
    #     norm = scaler.transform(abc)
    #     # print(norm.T[0])
    #     normm = pd.Series(norm.T[0])
    #     # temp['compute'].iloc[index:] = temp['forecast'].dropna()
    #     temp['compute'].iloc[index:] = normm
    #     temp['cumsum'] = temp['compute'].cumsum()

    #     # print(temp)

    #     forecast = temp[['forecast']].copy()
    #     forecast.iloc[index:] = close['Close'].iloc[0] + temp[['cumsum']].iloc[index:]
    #     # forecast = close['Close'].iloc[0] + temp['cumsum']
    #     pred = close.assign(predict=forecast)
    #     pred.columns = ['Close','backtest']
    #     # print(pred)

    #     fig = px.line(pred, title='arima backtest', markers=True)

    #### backtest #####
    if 'arima_bt_btn' == ctx.triggered_id:

        pickled_model = pickle.load(open(f'model\\{interval}_{ticker}_arima.pkl', 'rb'))

        temp = close[['Close']]
        temp['forecast'] = pickled_model.predict(start=close.index[0], end=close.index[-1])

        fig = px.line(temp, title='ARIMA backtest', markers=True)

        mape = mean_absolute_percentage_error(temp['Close'], temp['forecast'])
        mse = mean_squared_error(temp['Close'], temp['forecast'])
        rmse = np.sqrt(mse)
        corr = temp['Close'].corr(temp['forecast'])
        
        print(f'rmse = {rmse}')
        print(f'mape = {mape}')
        print(f'corr = {corr}')

        evall = f'RMSE = {rmse}, MAPE = {mape*100}%, Correlation = {corr}'

    #### forecast #####
    if 'arima_fc_btn' == ctx.triggered_id:
        state = 1

        pickled_model = pickle.load(open(f'model\\{interval}_{ticker}_arima.pkl', 'rb'))
        if interval == '1mo':
            forecast = pickled_model.forecast(6)
        elif interval == '1wk':
            forecast = pickled_model.forecast(8)
        elif interval == '1d':
            forecast = pickled_model.forecast(10)
        # print(forecast)
        pred = pd.concat([close,forecast])
        pred.columns = ['Close','predict']

        # print(pred)
        fig = px.line(pred, title='ARIMA forecast', markers=True)
    
        pred['Date'] = pred.index
        pred.index = pred['Date']
        pred = pred.drop(['Date'], axis=1)
        # print(pred)
    
    if 'rf_fc_btn' == ctx.triggered_id:

        X_test = close.iloc[-5:]

        pickled_model = pickle.load(open(f'model\\{ticker}_random_forest.pkl', 'rb'))

        y_pred = pickled_model.predict(X_test)
        forecast = pd.Series(y_pred)
        fut = []
        for i in range(5):
            tmr = close.index[-1] + timedelta(days=3+i)
            fut.append(tmr)
        forecast.index = fut
        print(forecast)
        pred = pd.concat([close,forecast])
        pred.columns = ['Close','predict']
        print(pred)

        fig = px.line(pred, title='Random Forest forecast', markers=True)

        pred['Date'] = pred.index
        pred.index = pred['Date']
        pred = pred.drop(['Date'], axis=1)
    

    return fig, evall

@callback(
    Output('download2', 'data'),
    Input('download_btn2', 'n_clicks'),
)
def download_csv(download_btn):
    global state
    if 'download_btn2' == ctx.triggered_id:
        if state == 0 :
            return dcc.send_data_frame(close.to_csv, f"{name}.csv")
        elif state == 1:
            state == 0
            return dcc.send_data_frame(pred.to_csv, f"{name}.csv")

# def arimamodel(timeseriesarray):
#     autoarima_model = pmd.auto_arima(timeseriesarray, 
#                               start_p=1, 
#                               start_q=2,
#                               start_Q=2,
#                               max_Q=5,
#                               test="adf",
#                               trace=True)
#     return autoarima_model