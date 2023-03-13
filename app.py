from dash import Dash, html, dcc, Input, Output, State, ctx, page_registry, page_container

app = Dash(__name__, use_pages=True)

app.layout = html.Div(
    children=[
        html.H1(
            className='center',
            children=[
                dcc.Link(
                    className='pure-text',
                    children='Fin-Advice', 
                    href=page_registry['pages.home']['relative_path'],
                )
            ],
        ),
        html.H3(className='center', children='Financial Advisor',),
        html.Div(className='div-header-btn',
            children=[
                dcc.Link(className='pure-text header-link',
                    children=[
                        html.Button(className='header-btn button-30',
                            children=[
                                f"{page['name']}"
                            ],
                        ),
                    ],
                    href=page['relative_path'],
                )
                for page in page_registry.values()
            ]
        ),
        page_container,
    ],
    style={
        # 'display': 'grid',
        # 'horizontal-alignment': 'center',
        # 'margin': 'auto',
    }
)

if __name__ == '__main__':
    app.run_server(debug=True)