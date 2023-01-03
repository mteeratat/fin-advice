from dash import Dash, html, dcc, Input, Output, State, ctx, register_page, callback
import yfinance as yf
import plotly.express as px
import pmdarima as pmd

pre = register_page(__name__)

ptt = yf.Ticker('ptt.bk')
data = ptt.history(interval='1wk', period='1y')
close = data[['Close']]

fig = px.line(close, title='PTT')

layout = html.Div(children=[
    html.H1(children='Predict'),

    html.Div(children='ticker'),

    dcc.Graph(
        id='example-graph3',
        figure=fig
    ),

    html.Div(children='model'),

    html.Button('ARIMA', id='btn1', n_clicks=0),
])

@callback(
    Output('example-graph3', 'figure'),
    Input('btn1', 'n_clicks'),
)
def change_model(btn1):
    global close
    if 'btn1' == ctx.triggered_id:
        # print(close)
        train = close.iloc[:][:40]
        test = close.iloc[:][40:]
        arima_model = arimamodel(train.dropna())
        # print(arima_model.summary())

        forecastt = arima_model.predict(len(test), alpha=0.05)
        pred = close.assign(predict=forecastt)
        print(pred)
        fig = px.line(pred, title='T')

    else:
        fig = px.line(close, title='PTT')
    
    return fig

# def prepare():

def arimamodel(timeseriesarray):
    autoarima_model = pmd.auto_arima(timeseriesarray, 
                              start_p=1, 
                              start_q=2,
                              start_Q=2,
                              max_Q=5,
                              test="adf",
                              trace=True)
    return autoarima_model