# Dash Modules
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_flexbox_grid as dfx
from dash.dependencies import Input, Output, State
import dash_table
import dash_dangerously_set_inner_html
from dash.exceptions import PreventUpdate

#Python Generic & Local Modules
import base64
from config import *
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

def encodedImage(imageFile):
    """
    Encode image for Dash Application
    Args:
        full_filepath: location of image
    returns:
        encode: decoded image
    """
    imageFile = "".join([METRICS_PATH, imageFile])
    encoded = base64.b64encode(open(imageFile, 'rb').read())
    print(imageFile)
    return 'data:image/jpg;base64,{}'.format(encoded.decode())

app.layout = html.Div([
    dcc.Tabs([
        html.Div([html.H2("Performance Metrics: ")], style={'marginBottom': 50, 'marginTop': 25}),
        dbc.Row(
            [
                dbc.Col(html.Div([html.Img(src = encodedImage('ROCAUC.png'), id = 'rocauc_image', height=700, width=720)])),
                dbc.Col(html.Div([html.Img(src = encodedImage('ClassificationReport.png'), id = 'classificationreport_image', height=700, width=720)])),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(html.Div([html.Img(src = encodedImage('PrecisionRecallCurve.png'), id = 'precisionrecall_image', height=700, width=720)])),
                dbc.Col(html.Div([html.Img(src = encodedImage('ConfusionMatrix.png'), id = 'confusion_matrix_image', height=700, width=720)])),
            ]
        )

    ])

])

if __name__ == '__main__':
    app.run_server(debug=False)
