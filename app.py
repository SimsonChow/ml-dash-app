# Dash & Plotly Modules
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_flexbox_grid as dfx
from dash.dependencies import Input, Output, State
import dash_table
import dash_dangerously_set_inner_html
from dash.exceptions import PreventUpdate
import plotly.express as px
import dash_dangerously_set_inner_html

#Python Generic & Local Modules
import base64
import shap, eli5
from config import *
from ml_utils_xai import *
import pandas as pd

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
 
df = pd.read_csv('Data/Output/test.csv')
app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Metrics', children=[
            html.Div([html.H2("Performance Metrics: ")], style={'marginLeft': 10, 'marginBottom': 50, 'marginTop': 25}),
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
        ]),

        dcc.Tab(label='Explainable AI - SHAP', children=[
            html.Div([html.H2(" Feature Importance With SHAP: ")], style={'marginLeft': 10, 'marginBottom': 50, 'marginTop': 25}),
            html.Div([html.H4("Please Choose a Death Event Label: ")], style={'marginLeft': 10, 'marginTop': 25}),
            dcc.Dropdown(id = 'shapLabelOption',  options=[
                    {'label':'Benign', 'value': 0},
                    {'label':'Death', 'value':1}
                ] 
                , value = 1
                , style={'width':300,'paddingLeft':10 }),

            dbc.Row(
            [
                dbc.Col(html.Div([dcc.Graph(id='shapBarPlot')], style = {'marginTop':55}))
            ]
            ),
        ]),
        dcc.Tab(label='Explainable AI - ELI5', children=[
            html.Div([html.H2(" Explain Datapoint Prediction with ELI5: ")], style={'marginLeft': 10, 'marginBottom': 50, 'marginTop': 25}),
            
            dbc.Row(
                [
                    dbc.Col([
                            html.Div([html.H4("Test Dataset: ")], style={'marginLeft': 10, 'marginBottom': 50, 'marginTop': 25}),
                            html.Div([         
                            dash_table.DataTable(
                                    data=df.to_dict('records'),
                                    columns=[{'id': c, 'name': c} for c in df.columns],
                                    style_table={'height': '400px', 'overflowY': 'auto'}
                                )
                            ], style={'marginLeft': 10, 'marginTop': 150, 'marginTop': 25}),
                            html.Div([html.H4("Enter an index row for analysis: ")], style={'marginTop':10,'marginLeft': 10, 'marginBottom': 15, 'marginTop': 25}),
                            html.Div(dbc.Input(id='rowIndex', type='number', value = 0 , style={'marginLeft': 10,'width':125})), 
                            html.Div(dbc.Button('Submit', id='eli5Button', n_clicks=0, style={'marginLeft': 10, 'marginBottom': 50, 'marginTop': 10})),
                            html.Div(id='eli5Table', style = {'marginLeft': 10, 'marginBottom': 50,'marginTop':10})]),
                ],
            ),
        ]),
    ]),
    
])

@app.callback(Output('shapBarPlot', 'figure'), 
              [Input('shapLabelOption','value')])
def shapCallback(shapLabelOption):
    """
    SHAP Global Interpretability Callback - Bar chart.
    Purpose: Visualize feature importance from SHAP package 
    Steps:
        1) Convert shap_values into a df
        2) Get the columns of the training dataset
        3) Assign the same column of the training dataset to the shap values dataframe
        4) Copy the train data df and reset the index so that it is compatible with shap value df
        5) Get  the correlation coefficient and append it to a new df (corr_df) - This will be use to plot the bar chart
    """
    # Load SHAP correlation Dataframe 
    shapObjectName = 'RandomForestClassifier_SHAP_LABEL_{}'.format(shapLabelOption)
    outputShapCorrelationDataframe = loadObject(shapObjectName, OUTPUT_PATH)

    # Generate Shap figure
    shapFigure = px.bar(outputShapCorrelationDataframe, x="Features", y="Correlation",title="<b>Feature Importance (Postive/Negative) Correlation <br>",
            height=650, width = 1200,color = np.where(outputShapCorrelationDataframe['Correlation']>0,'Positive Correlation','Negative Correlation'))
    return shapFigure

@app.callback(Output('eli5Table', 'children'),
    [Input('eli5Button','n_clicks')],
    [State('rowIndex', 'value')])
def eli5Callback(clicks, rowIndex):
    """
    ELI5 Local Interpretability Callback - Output results table.
    Purpose: Visualize ELI5 explanation table  
    """
    modelName = "RandomForestClassifier"
    trainedModel = loadObject(modelName, OUTPUT_PATH) 
    testDataset = pd.read_csv(OUTPUT_PATH + 'test.csv')

    #Calls EL5 explain prediction API - Format as HTML
    explainPredictionObject = eli5.explain_prediction(trainedModel, testDataset.iloc[rowIndex], top=10, target_names= {0: ' Benign', 1:' Death'})
    HTMLFormat = eli5.format_as_html(explainPredictionObject, show=('targets','decision_tree','transition_features'))

    #Construct ELI5 table
    tableHeader = [html.Thead(html.Tr([html.Th("Row #"+ str(rowIndex))]), style={"fontSize":14})]
    tableBody = [html.Tbody(dash_dangerously_set_inner_html.DangerouslySetInnerHTML(HTMLFormat), style={"fontSize":14})]
    outputELI5Table = dbc.Table(tableHeader + tableBody, hover=True, responsive=True, striped=True)
    
    return outputELI5Table
if __name__ == '__main__':
    app.run_server(debug=False)