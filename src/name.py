import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import pickle
import requests
import pandas as pd
from urllib.request import urlopen
import json
import plotly.express as px
import plotly.graph_objects as go

import pickle
import requests
import pandas as pd
from urllib.request import urlopen
import json
import os 

with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import pandas as pd
import plotly.express as px
import xarray as xr
    
with open('master2.pkl', 'rb') as f:
    master = pickle.load(f)

with open('commodity_uom.pkl', 'rb') as f:
    uom = pickle.load(f)

import geopandas as gpd

# Load the 'naturalearth_lowres' dataset that's bundled with Geopandas
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Filter the data for USA and then load US counties
usa = world[world['name'] == "United States of America"]
counties2 = gpd.read_file('https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_us_county_20m.zip')

# Create a dictionary to store the polygons
county_polygons = {}

for _, row in counties2.iterrows():
    rma_code = row["STATEFP"] + row["COUNTYFP"]
    county_polygons[rma_code] = row["geometry"]

mydir = os.getcwd() # would be the MAIN folder
mydir_tmp = mydir + "/datasets" 

my_token = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjI1MzQwMjMwMDc5OSwiaWF0IjoxNjMwNTMwMTgwLCJzdWIiOiJhMDIzYjUwYi0wOGQ2LTQwY2QtODNiMS1iMTExZDA2Mzk1MmEifQ.qHy4B0GK22CkYOTO8gsxh0YzE8oLMMa6My8TvhwhxMk'

my_url = 'https://api.dclimate.net/apiv4/rma-code-lookups/valid_states'
head = {"Authorization": my_token}
r = requests.get(my_url, headers=head)
state_codes = r.json()

my_url = 'https://api.dclimate.net/apiv4/rma-code-lookups/valid_counties/'
head = {"Authorization": my_token}
state_county = {}
for key in state_codes.keys():
    r = requests.get(my_url + str(key), headers=head)
    county_codes = r.json()
    state_county[key] = county_codes

month_map = {
    1: 'January',
    2: 'February',
    3: 'March',
    4: 'April',
    5: 'May',
    6: 'June',
    7: 'July',
    8: 'August',
    9: 'September',
    10:'October',
    11:'November',
    12:'December'
}

commodity_mapping = {
    '0054': 'Apples', 
    '0012': 'Blueberries',
    '0041': 'Corn',
    '0021': 'Cotton',
    #'0014': 'Buckwheat',
    #'0132': 'Cucumbers',
    #'0058': 'Cranberries',
    '0051': 'Grain Sorghum',
    '0053': 'Grapes', 
    '0034': 'Peaches',
    '0081': 'Soybeans',
    '0011': 'Wheat',
    #'0038': 'Sugarcane',
    #'0094': 'Rye',
    #'0042': 'Sweet Corn',
    #'0201': 'Grapefruit', 
    #'0202': 'Lemons', 
    #'0227': 'Oranges', 
    #'0087': 'Tomatoes',
}

commodities = commodity_mapping.keys()

state_map = {'01': 'Alabama', '02': 'Alaska', '04': 'Arizona', '05': 'Arkansas',
             '06': 'California', '08': 'Colorado', '09': 'Connecticut',
             '10': 'Delaware', '12': 'Florida', '13': 'Georgia', '15': 'Hawaii',
             '16': 'Idaho', '17': 'Illinois', '18': 'Indiana', '19': 'Iowa',
             '20': 'Kansas', '21': 'Kentucky', '22': 'Louisiana', '23': 'Maine',
             '24': 'Maryland', '25': 'Massachusetts', '26': 'Michigan',
             '27': 'Minnesota', '28': 'Mississippi', '29': 'Missouri', '30': 'Montana',
             '31': 'Nebraska', '32': 'Nevada', '33': 'New Hampshire', '34': 'New Jersey',
             '35': 'New Mexico', '36': 'New York', '37': 'North Carolina', '38': 'North Dakota',
             '39': 'Ohio', '40': 'Oklahoma', '41': 'Oregon', '42': 'Pennsylvania',
             '44': 'Rhode Island', '45': 'South Carolina', '46': 'South Dakota',
             '47': 'Tennessee', '48': 'Texas', '49': 'Utah', '50': 'Vermont',
             '51': 'Virginia', '53': 'Washington', '54': 'West Virginia',
             '55': 'Wisconsin', '56': 'Wyoming'}


def fips_map(fips):
    state_num = fips[:2]
    county_num = fips[2:]
    state_name = state_map[state_num]
    my_url = 'https://api.dclimate.net/apiv4/rma-code-lookups/valid_counties/'
    head = {"Authorization": my_token}
    r = requests.get(my_url + state_num, headers=head)
    county_codes = r.json()
    county_name = county_codes[county_num]
    return f"{county_name}, {state_name}"

def calculate_difference(val, start_year, end_year, month, comparison_year):
    # Convert the input years and month to integers
    start_year = int(start_year)
    end_year = int(end_year)
    month = int(month)
    comparison_year = int(comparison_year)

    # Calculate the index for the comparison_year and given month
    comparison_index = (comparison_year - 2000) * 12 + month - 1

    # Calculate the indices for each year from start_year to end_year for the given month
    monthly_indices = [(year - 2000) * 12 + month - 1 for year in range(start_year, end_year + 1)]

    # Calculate the average value for the given month over the specified years
    avg_value = sum(val[index] for index in monthly_indices) / len(monthly_indices)

    # Calculate the difference between the value in the comparison_year and the average value
    difference = val[comparison_index] - avg_value

    return difference

def return_peril_figures(month, start_year, end_year, comparison_year):
    # Initialize the master dictionary
    master = {}

    # Loop through all .nc files in the directory
    for filename in os.listdir(mydir_tmp):
        if filename.endswith(".nc"):
            # Construct the full path to the file
            filepath = os.path.join(mydir_tmp, filename)
            
            # Open the dataset
            ds = xr.open_dataset(filepath)
            
            # Extract the values of 't2m'
            t2m_values = ds['t2m'].values
            
            # Store the filename (without extension) as a key and the t2m values as its value in the master dictionary
            key_name = os.path.splitext(filename)[0]  # get filename without extension
            master[key_name] = t2m_values

    prop_master = {}
    for county_code in list(county_polygons.keys()):
        if f"dataset_{list(county_polygons.keys()).index(county_code)}" in master.keys():
            prop_master[county_code] = master[f"dataset_{list(county_polygons.keys()).index(county_code)}"]

    # 1. Extract the most recent value from each array
    recent_values = {key: calculate_difference(val, start_year, end_year, month, comparison_year) for key, val in prop_master.items()}

    # 2. Create a dataframe
    df = pd.DataFrame(list(recent_values.items()), columns=['FIPS', 'Temperature'])
    df['FIPS'] = df['FIPS'].astype(str)
    print(df)
    # 3. Plot the choropleth
    min_unemp = df['Temperature'].min()
    max_unemp = df['Temperature'].max()

    try:
        # Calculate the ratio where 0 falls within the range
        white_ratio = (- min_unemp) / (max_unemp - min_unemp)
        # Create a custom color scale
        colorscale = [
            [0, "blue"],
            [white_ratio, "white"],
            [1, "red"]
        ]
    except Exception as e:
        print(e)
        # Create a custom color scale
        colorscale = [
            [0, "red"],
            [1, "green"]
        ] 
    fig = px.choropleth(
        df, 
        geojson=counties,
        locations='FIPS',
        color='Temperature',
        color_continuous_scale=colorscale,
        scope="usa",
        title=f'Monthly County-Wide Minimum Temperature Difference From {comparison_year} vs {start_year}-{end_year} Average', 
        hover_name=df.index
    )
    fig.update_geos(fitbounds="locations", visible=True)

    # Update colorbar ticktext and tickvals to include measure
    num_ticks = 5  # For example, choose how many tick points you want
    tickvals = [min_unemp + i * (max_unemp - min_unemp) / (num_ticks - 1) for i in range(num_ticks)]
    ticktext = [f"{tick_val:.2f} °C" for tick_val in tickvals]  # The .2f formats the float to 2 decimal places

    fig.update_layout(
        coloraxis_colorbar=dict(tickvals=tickvals, ticktext=ticktext),
        margin={"r":0,"l":0,"b":0}
    )

    fig.update_layout(
        autosize=False,
        width=1600,
        height=800,
    )

    return fig



def return_figures(selected_commodity, start_year, end_year, comparison_year, yield_type, result_type):
    master_diff = {}
    master_percent = {}

    commodities = [selected_commodity]

    if result_type == 'pcnt':
        measure = '%'
    else:
        try:
            measure = uom[selected_commodity]
        except Exception as e:
            print(f"No measure found for {selected_commodity}. Defaulted to no unit of measure. Error code: {e}")
            measure = ''

    for commodity in commodities:
        yield_diff = {}
        yield_percent = {}
        for state in state_county.keys():
            for county in state_county[state].keys():
                try:
                    df = master[commodity][state][county]

                    # Filter to Irrigation Practice Code = 2
                    df_filtered = df[df['Irrigation Practice Code'] == 2]
                    if len(df_filtered) == 0:
                        df_filtered = df[df['Irrigation Practice Code'] == 997]
                    # Calculate 10-year average Yield Amount for 2012-2021
                    average_yield = df_filtered.loc[(df_filtered['Yield Year'] >= start_year) & (df_filtered['Yield Year'] <= end_year), yield_type].mean()

                    # Subtract this average from the 2022 Yield Amount
                    df_2022 = df_filtered[df_filtered['Yield Year'] == comparison_year]
                    print(df_2022[yield_type])
                    print(average_yield)
                    difference = df_2022[yield_type] - average_yield
                    
                    # Calculate the percentage difference
                    percent_diff = (df_2022[yield_type] / average_yield) * 100

                    yield_diff[f"{state}{county}"] = difference
                    yield_percent[f"{state}{county}"] = percent_diff

                except Exception as e:
                    print(e)
                    pass
                
        for key in yield_diff.keys():
            try:
                temp_diff = yield_diff[key][(yield_diff[key].index[0])]
                yield_diff[key] = temp_diff
                
                temp_percent = yield_percent[key][(yield_percent[key].index[0])]
                yield_percent[key] = temp_percent
            except Exception as e:
                print(e)
                yield_diff[key] = None
                yield_percent[key] = None
            
        master_diff[commodity] = yield_diff
        master_percent[commodity] = yield_percent


    master_mapping = {
        'pcnt': master_percent,
        'amnt': master_diff
    }

    # Ensure the /csvs/ subfolder exists or create it
    output_folder = 'csvs'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for method in master_mapping.keys():
        for commodity in commodities:
            # Convert the dictionary to a DataFrame
            df = pd.DataFrame(list(master_mapping[method][commodity].items()), columns=['fips', 'unemp'])

            # Write the DataFrame to a CSV file inside the /csvs/ subfolder
            df.to_csv(os.path.join(output_folder, f'{method}_{commodity}_output.csv'), index=False)

    legend_mapping = {
        'pcnt': 'Percentage Yield',
        'amnt': 'Yield Difference'
    }

    for method in [result_type]:
        for commodity in commodities:
            # Read the DataFrame from a CSV file inside the /csvs/ subfolder
            df = pd.read_csv(os.path.join(output_folder, f"{method}_{commodity}_output.csv"),
                            dtype={"fips": str})

            
            #df['fips_name'] = df['fips'].apply(fips_map)



            min_unemp = df['unemp'].min()
            max_unemp = df['unemp'].max()

            try:
                # Calculate the ratio where 100 falls within the range
                white_ratio = (100 - min_unemp) / (max_unemp - min_unemp)
                if method == 'amnt':
                    total_range = max_unemp - min_unemp
                    distance_from_negative = 0 - min_unemp
                    white_ratio = distance_from_negative / total_range
                # Create a custom color scale
                colorscale = [
                    [0, "red"],
                    [white_ratio, "white"],
                    [1, "green"]
                ]
            except Exception as e:
                print(e)
                # Create a custom color scale
                colorscale = [
                    [0, "red"],
                    [1, "green"]
                ] 
            try:
                fig = px.choropleth(df, geojson=counties, locations='fips', color='unemp',
                                    color_continuous_scale=colorscale,
                                    range_color=(min_unemp, max_unemp),
                                    scope="usa",
                                    title=f'{comparison_year} vs Average {start_year}-{end_year} {commodity_mapping[commodity]} {yield_type}',        
                                    labels={'unemp':f'{legend_mapping[method]}', 'fips': 'Location'}
                                )



                # Update colorbar ticktext and tickvals to include measure
                num_ticks = 5  # For example, choose how many tick points you want
                tickvals = [min_unemp + i * (max_unemp - min_unemp) / (num_ticks - 1) for i in range(num_ticks)]
                ticktext = [f"{tick_val:.2f} {measure}" for tick_val in tickvals]  # The .2f formats the float to 2 decimal places

                fig.update_layout(
                    coloraxis_colorbar=dict(tickvals=tickvals, ticktext=ticktext),
                    margin={"r":0,"l":0,"b":0}
                )
                                # Update the size of the figure
                fig.update_layout(
                    autosize=False,
                    width=1600,
                    height=800,
    )

                return fig 
            except Exception as e:
                print(e)
                return None

# initialize the Dash app
app = dash.Dash(__name__,title="csvtest", suppress_callback_exceptions=True)

# Define the app layout
app.layout = html.Div([
    html.H1("Yield and Peril Analysis Dashboard"),
    
    dcc.Tabs(id="tabs", value='yield_view', children=[
        dcc.Tab(label='Yield View', value='yield_view'),
        dcc.Tab(label='Peril View', value='peril_view'),
    ]),
    html.Div(id='tab-content')
    
])

# Yield View Layout
yield_layout = html.Div([
    # For commodity_dropdown
    html.Div([
        html.Label('Select Commodity:'),
        dcc.Dropdown(
            id='commodity_dropdown', 
            options=[{'label': value, 'value': key} for key, value in commodity_mapping.items()],
            value=list(commodity_mapping.keys())[0]
        )
    ], style={'marginBottom': '10px'}),
    
    # For yield_type
    html.Div([
        html.Label('Select Yield Type:'),
        dcc.Dropdown(
            id='yield_type', 
            options=[
                {'label': 'Yield Amount', 'value': 'Yield Amount'},
                {'label': 'Detrended Yield Amount', 'value': 'Detrended Yield Amount'},
                #{'label': 'Trended Yield Amount', 'value': 'Trended Yield Amount'}
            ],
            value='Yield Amount'
        )
    ], style={'marginBottom': '10px'}),

    # For result_type
    html.Div([
        html.Label('Select Result Type:'),
        dcc.Dropdown(id='result_type', 
                    options = [{'label': 'Relative Percentage Yield', 'value': 'pcnt'}, {'label': 'Relative Yield Difference', 'value': 'amnt'}],
                    value = 'pcnt'),

        ], style={'marginBottom': '10px'}),

    # For year_range_slider
    html.Div([
        html.Label('Select Benchmark Years:'),
        dcc.RangeSlider(
            id='year_range_slider',
            min=2000,
            max=2022,
            value=[2012, 2021],
            marks={i: str(i) for i in range(2000, 2022, 1)},
            step=None
        )
    ], style={'marginBottom': '10px'}),
    
    # For comparison_year_slider
    html.Div([
        html.Label('Select Comparison Year:'),
        dcc.Slider(
            id='comparison_year_slider',
            min=2000,
            max=2022,
            value=2022,
            marks={i: str(i) for i in range(2000, 2023, 1)},
            step=None,
            included=False
        )
    ], style={'marginBottom': '20px'}),
    
    # For Run button, with increased size using CSS
    html.Button('Run', id='update_button', style={'fontSize': '14px'}),


    dcc.Loading(
        id="loading",
        type="circle",
        children=[
            dcc.Graph(id='yield_graph')
        ]
    ),
    html.Div(id='error_message'),
])

# Peril View Layout
peril_layout = html.Div([

    # For peril_dropdown
    html.Div([
        html.Label('Select Peril:'),
        dcc.Dropdown(
            id='peril_dropdown', 
            options=[{'label': 'Minimum Temperature', 'value': 'min_temp'}],
            value='min_temp'
        )
    ], style={'marginBottom': '10px'}),

    # For year_range_slider2
    html.Div([
        html.Label('Select Benchmark Years:'),
        dcc.RangeSlider(
            id='year_range_slider2',
            min=2000,
            max=2022,
            value=[2012, 2021],
            marks={i: str(i) for i in range(2000, 2022, 1)},
            step=None
        )
    ], style={'marginBottom': '10px'}),
    
    # For comparison_year_slider2
    html.Div([
        html.Label('Select Comparison Year:'),
        dcc.Slider(
            id='comparison_year_slider2',
            min=2000,
            max=2022,
            value=2022,
            marks={i: str(i) for i in range(2000, 2023, 1)},
            step=None,
            included=False
        )
    ], style={'marginBottom': '10px'}),

    # For month slider
    html.Div([
        html.Label('Select Month:'),
        dcc.Slider(
            id='month',
            min=1,
            max=12,
            value=1,
            marks={i: month_map[i] for i in range(1, 13, 1)},
            step=None,
            included=False
        )
    ], style={'marginBottom': '20px'}),
    
    # For Run button, with increased size using CSS
    html.Button('Run', id='update_button_peril', style={'fontSize': '14px'}),
    
    # For the loading circle and the graph
    dcc.Loading(
        id="loading2",
        type="circle",
        children=[
            dcc.Graph(id='peril_graph')
        ]
    )
])

# Callback for Tab Content
@app.callback(Output('tab-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'yield_view':
        return yield_layout
    elif tab == 'peril_view':
        return peril_layout

# Yield Figure Update Callback
# define the app callback
@app.callback(
    [Output('yield_graph', 'figure'), Output('error_message', 'children')],
    [Input('update_button', 'n_clicks')],
    [State('commodity_dropdown', 'value'),
     State('year_range_slider', 'value'),
     State('comparison_year_slider', 'value'),
     State('yield_type', 'value'),
     State('result_type', 'value')]
)
def update_figure(n_clicks, selected_commodity, year_range, comparison_year, yield_type, result_type):
    # if the button has not been clicked, there is no need to update the figure
    if n_clicks is None:
        return dash.no_update, ""
    
    try:
        # Extract start_year and end_year from year_range
        start_year, end_year = year_range
        fig = return_figures(selected_commodity, start_year, end_year, comparison_year, yield_type, result_type)
        # Insert your existing code here for fetching the data and creating the figure,
        # with changes to use the selected_commodity, start_year, end_year, and comparison_year values
        # ...
        # Return the figure
        return fig, ""
    except Exception as e:
        # Return an error message if something goes wrong
        return go.figure(), f"An error occurred with your input: {str(e)}"

# Peril Figure Update Callback
@app.callback(
    Output('peril_graph', 'figure'),
    [Input('update_button_peril', 'n_clicks')],
    [State('month', 'value'),
     State('year_range_slider2', 'value'),
     State('comparison_year_slider2', 'value')]
)
def update_peril_figure(n_clicks, month, year_range, comparison_year):
    if n_clicks is None:
        return dash.no_update

    try:
        # Extract start_year and end_year from year_range
        start_year, end_year = year_range
        fig = return_peril_figures(month, start_year, end_year, comparison_year)
        # ... [insert your code for fetching peril data and creating the figure]
        return fig
    except Exception as e:
        return go.Figure(), f"An error occurred with your input: {str(e)}"

if __name__ == '__main__':
    app.run_server(debug=True)
