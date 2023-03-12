from dash import Dash, html, dcc, Input, Output, State, ctx, page_registry, page_container

app = Dash(__name__, use_pages=True)

app.layout = html.Div(
    children=[
        html.H1(
            children=[
                dcc.Link(style={'text-decoration': 'none', 'color': 'black'},
                        children='Fin-Advice', href=page_registry['pages.home']['relative_path'],
                )
            ], 
            style={'display':'flex', 'justify-content':'center'},
        ),
        html.H2(children='financial advising', style={'display':'flex', 'justify-content':'center'}),
        html.Div(style={'display':'flex', 'justify-content':'space-around'},
            children=[
                html.Div(children=
                    dcc.Link(
                        f"{page['name']} - {page['path']}", href=page['relative_path']
                    )
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