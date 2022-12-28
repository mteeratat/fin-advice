import yfinance as yf
from dash import Dash, html, dcc, Input, Output, State, ctx, page_registry, page_container
import pandas as pd
import plotly.express as px
from indicator.ma import SMA, EMA

app = Dash(__name__, use_pages=True)

app.layout = html.Div(children=[
    html.H1(children='Hello Dashyyyyyy'),
    html.Div(
        [
            html.Div(
                dcc.Link(
                    f"{page['name']} - {page['path']}", href=page['relative_path']
                )
            )
            for page in page_registry.values()
        ]
    ),
    page_container
])

if __name__ == '__main__':
    app.run_server(debug=True)