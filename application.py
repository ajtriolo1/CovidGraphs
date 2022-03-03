import dash
import dash_core_components as dcc
import dash_html_components as html
import datetime as dt
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
from pytz import timezone
from urllib.request import Request, urlopen
import numpy as np

us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

abbrev_us_state = dict(map(reversed, us_state_abbrev.items()))
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', 'dropdown.css']
app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets
)
application = app.server
app.title = 'Covid Graphs'

us_pop = pd.read_csv('csvData.csv')
global_pop = pd.read_csv('countryPop.csv')

def update_data():
    global df_us
    global df_global
    global df_votes
    global df_state_votes

    df_global = pd.read_csv('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv')
    df_global.loc[df_global['new_cases'] < 0, ['new_cases']] = '0'
    df_global.loc[df_global['new_deaths'] < 0, ['new_deaths']] = '0'
    df_global.fillna(0, inplace=True)

    df_us = pd.read_csv('https://api.covidactnow.org/v2/states.timeseries.csv?apiKey=024d1309069f4e0a88b5afa505e9f470')
    df_us.fillna(0, inplace=True)
    df_us = df_us[(df_us.state != 'VI') & (df_us.state != 'MP') & (df_us.state != 'GU') & (df_us.state != 'AS')]
    df_us.loc[df_us['actuals.newCases'] < 0, ['actuals.newCases']] = '0'
    df_us.loc[df_us['actuals.newDeaths'] < 0, ['actuals.newDeaths']] = '0'
    df_us.loc[(df_us['state'] == 'NJ') & (df_us['date'] == '2020-06-25'), ['actuals.newDeaths']] = '26'
    df_us.loc[(df_us['state'] == 'NY') & (df_us['date'] == '2020-05-07'), ['actuals.newDeaths']] = '220'
    df_us.loc[(df_us['state'] == 'MA') & (df_us['date'] == '2020-04-25'), ['actuals.newDeaths']] = '190'
    df_us.loc[(df_us['state'] == 'MA') & (df_us['date'] == '2020-04-26'), ['actuals.newDeaths']] = '197'
    df_us.loc[(df_us['state'] == 'MA') & (df_us['date'] == '2020-06-01'), ['actuals.newCases']] = '700'
    df_us.loc[(df_us['state'] == 'IN') & (df_us['date'] == '2020-04-29'), ['actuals.newDeaths']] = '49'
    df_us.loc[(df_us['state'] == 'WY') & (df_us['date'] == '2020-04-29'), ['actuals.newCases']] = '11'
    df_us.loc[(df_us['state'] == 'NJ') & (df_us['date'] == '2020-07-08'), ['actuals.newDeaths']] = '53'
    df_us.loc[(df_us['state'] == 'NJ') & (df_us['date'] == '2020-07-16'), ['actuals.newDeaths']] = '31'
    df_us.loc[(df_us['state'] == 'NJ') & (df_us['date'] == '2020-07-22'), ['actuals.newDeaths']] = '24'
    df_us.loc[(df_us['state'] == 'NJ') & (df_us['date'] == '2020-08-12'), ['actuals.newDeaths']] = '9'
    df_us.loc[(df_us['state'] == 'NJ') & (df_us['date'] == '2021-01-04'), ['actuals.newCases']] = '5000'
    
    df_votes = pd.read_csv('https://raw.githubusercontent.com/tonmcg/US_County_Level_Election_Results_08-20/master/2020_US_County_Level_Presidential_Results.csv')
    df_votes['winner'] = df_votes.apply(lambda row: label_votes(row), axis=1)
    df_votes['color'] = df_votes.apply(lambda row: label_color(row), axis=1)
    df_votes.rename(columns={'state_name':'state', 'county_name':'county'}, inplace=True)
    df_state_votes = df_votes.groupby(['state'])['votes_gop'].agg('sum').reset_index()
    df_state_votes['votes_dem'] = df_votes.groupby(['state'])['votes_dem'].agg('sum').reset_index()['votes_dem']
    df_state_votes['total_votes'] = df_votes.groupby(['state'])['total_votes'].agg('sum').reset_index()['total_votes']
    df_state_votes['per_gop'] = df_state_votes.apply(lambda row: row['votes_gop']/row['total_votes'], axis=1)
    df_state_votes['per_dem'] = df_state_votes.apply(lambda row: row['votes_dem']/row['total_votes'], axis=1)
    df_state_votes['winner'] = df_state_votes.apply(lambda row: label_votes(row), axis=1)
    df_state_votes['color'] = df_state_votes.apply(lambda row: label_color(row), axis=1)
    
def label_votes(row):
    if row['per_gop'] > 0.55:
        return 'gop'
    elif row['per_dem'] > 0.55:
        return 'dem'
    else:
        return 'split'
        
def label_color(row):
    if row['winner'] == 'gop':
        return 'red'
    elif row['winner'] == 'dem':
        return 'blue'
    else:
        return 'purple'

update_data()

def get_county_data(state):
    state=us_state_abbrev[state]
    df = pd.read_csv('https://api.covidactnow.org/v2/county/{state}.timeseries.csv?apiKey=024d1309069f4e0a88b5afa505e9f470'.format(state = state))
    tod = dt.datetime.now()
    d = dt.timedelta(weeks=2)
    week_ago = tod - d
    date = week_ago.strftime('%Y-%m-%d')
    df = df[df['date'] >= date]
    df_state = pd.read_csv('https://api.covidactnow.org/v2/county/{state}.csv?apiKey=024d1309069f4e0a88b5afa505e9f470'.format(state = state))
    df_state = pd.merge(df_state, df_votes.loc[df_votes['state']==abbrev_us_state[state]][['county', 'winner', 'color']], on='county', how='left')
    df_state['color'].fillna('purple', inplace=True)
    df = pd.merge(df, df_state[['county', 'population']], on='county', how='left')
    
    df_deaths = df.groupby('county').agg({'actuals.newDeaths':'sum'})
    df_deaths.reset_index(inplace=True)
    df_deaths.rename(columns={'actuals.newDeaths':'deaths'}, inplace=True)
    df_deaths = pd.merge(df_deaths,df_state[['county', 'population', 'color']], on='county', how='left')
    
    df_vaxPct = df.groupby('county').agg({'metrics.vaccinationsCompletedRatio':max}).reset_index().rename(columns={'metrics.vaccinationsCompletedRatio':'VaxPct'})
    df_new = df_deaths.merge(df_vaxPct, how='left', on='county')
    df_new['deaths_per_100k'] = df_new['deaths']/(df_new['population']/100000)
    df_new = df_new[df_new['population'] >= 50000]
    df_new.sort_values(by='VaxPct', inplace=True)
    df_new.dropna(inplace=True)
    
    if df_new['VaxPct'].size == 0:
        df_new['VaxPct'] = 0
        df_new['deaths_per_100k'] = 0
        df_new['curve_fit'] = 0
        return df_new
    
    model = np.poly1d(np.polyfit(df_new['VaxPct'], df_new['deaths_per_100k'], 3))
    df_new['curve_fit'] = model(df_new['VaxPct'])
    
    return df_new

def get_state_data():
    df = df_us.copy()
    df['state'].replace(abbrev_us_state, inplace=True)
    tod = dt.datetime.now()
    d = dt.timedelta(weeks=2)
    week_ago = tod - d
    date = week_ago.strftime('%Y-%m-%d')
    df = df[df['date'] >= date]
    df_state = pd.read_csv('https://api.covidactnow.org/v2/states.csv?apiKey=024d1309069f4e0a88b5afa505e9f470')
    df_state['state'].replace(abbrev_us_state, inplace=True)
    df_state = pd.merge(df_state, df_state_votes[['state', 'winner', 'color']], on='state', how='left')
    df_state['color'].fillna('purple', inplace=True)

    df_deaths = df.groupby('state').agg({'actuals.newDeaths':'sum'})
    df_deaths.reset_index(inplace=True)
    df_deaths.rename(columns={'actuals.newDeaths':'deaths'}, inplace=True)
    df_deaths = pd.merge(df_deaths,df_state[['state', 'population', 'color']], on='state', how='left')

    df_vaxPct = df.groupby('state').agg({'metrics.vaccinationsCompletedRatio':max}).reset_index().rename(columns={'metrics.vaccinationsCompletedRatio':'VaxPct'})
    df_new = df_deaths.merge(df_vaxPct, how='left', on='state')
    df_new['deaths_per_100k'] = df_new['deaths']/(df_new['population']/100000)
    df_new = df_new[df_new['population'] >= 50000]
    df_new.sort_values(by='VaxPct', inplace=True)
    df_new.dropna(inplace=True)
    
    model = np.poly1d(np.polyfit(df_new['VaxPct'], df_new['deaths_per_100k'], 3))
    df_new['curve_fit'] = model(df_new['VaxPct'])

    return df_new

def unique_sorted_values(array):
    unique = array.unique().tolist()
    unique.sort()
    return unique



sorted_states = unique_sorted_values(df_us['state'])
sorted_states = [abbrev_us_state[state] for state in sorted_states]
sorted_countries = unique_sorted_values(df_global['location'])

app.layout = html.Div([
    dcc.Interval(
      id='interval-component',
      interval=60*60*1000,
      n_intervals=0
    ),
    html.Div([
        html.Label('Country', style={'color':'#fff'}),
        dcc.Dropdown(
            id='country-selector',
            options=[{'label': i, 'value': i} for i in sorted_countries],
            value = 'United States'
        )
    ], className="six columns", style={'width': '48%', 'margin-bottom': '15px'}),
    html.Div([
        html.Label('State', style={'color':'#fff'}),
        dcc.Dropdown(
            id='state-selector',
            options=[{'label': i, 'value': i} for i in sorted_states],
            placeholder = "Select a State"
        )
    ], className="six columns", style={'width': '48%', 'margin-bottom': '15px'}),
    html.Div([
        html.Div([
            dcc.Loading(
                id="loading-cases",
                type="graph",
                children=dcc.Graph(id='cases-vs-days-lin')
            )
        ], className="six columns"),
        html.Div([
            dcc.Loading(
                id="loading-deaths",
                type="graph",
                children=dcc.Graph(id='deaths-vs-days'),
            )
        ], className="six columns")
    ], className="row"),

    html.Div([
        html.Div([
            dcc.Loading(
                id="loading-vax",
                type="graph",
                children=dcc.Graph(id='vax-vs-deaths'),
            )
        ], className="six columns"),
        html.Div([
            dcc.Loading(
                id="loading-report",
                type="dot",
                children = html.Div(id='state-report', style={'text-align':'center', 'width':'auto'}),
            )
        ], className="six columns", style={'display':'flex', 'justify-content':'center'}),
    ], className="row", style={'margin-top': '35px', 'margin-bottom':'10px'}),
    html.I("Note: Color of markers on \"Deaths vs. Vaccination Rate\" graph is associated with how each state/county voted in the 2020 presidential election.", style={'color':'#9b9b9b'}),
    html.Div(id='time-value', style={'color':'#9b9b9b'})
])

@app.callback(
    Output('time-value', 'children'),
    [Input('interval-component', 'n_intervals')])
def update_time(n):
    update_data()
    return html.I("Last Updated: " + str(dt.datetime.now(timezone('US/Eastern')).strftime("%b %d %Y %I:%M:%S %p")) + " (EST)")

@app.callback(
    Output('state-selector', 'disabled'),
    [Input('country-selector', 'value')])
def enable_state(country_selected):
    if(country_selected == 'United States'):
        return None
    else:
        return 'true'

@app.callback(
    [Output('cases-vs-days-lin', 'figure'), Output('deaths-vs-days', 'figure'), Output('vax-vs-deaths', 'figure'), Output('state-report', 'children')],
    [Input('state-selector', 'value'), Input('country-selector', 'value'), Input('interval-component', 'n_intervals')])
def update_plots(state_selected, country_selected, n):
    if((state_selected != None) & (country_selected == 'United States')):
        df = df_us[df_us['state'] == us_state_abbrev[state_selected]]
        df['newCases'] = df['actuals.newCases']
        df['newDeaths'] = df['actuals.newDeaths']

        df['cases_smoothed'] = df['actuals.newCases'].rolling(7).mean()
        df['deaths_smoothed'] = df['actuals.newDeaths'].rolling(7).mean()
        
        df_county = get_county_data(state_selected)
        df_county = df_county.loc[df_county['VaxPct'] > 0]

        if(df['newCases'].iloc[-1] == 0.0):
            df = df.iloc[:-1]

        trace1=go.Bar(
            x=df['date'],
            y=df['actuals.newCases'],
            name='Cases'
        )
        trace2=go.Scatter(
            x=df['date'],
            y=df['cases_smoothed'],
            name = "Cases smoothed"
        )
        trace3=go.Bar(
            x=df['date'],
            y=df['actuals.newDeaths'],
            name='Deaths'
        )
        trace4=go.Scatter(
            x=df['date'],
            y=df['deaths_smoothed'],
            name = "Deaths smoothed"
        )
        trace5=go.Scatter(
            x=df_county['VaxPct'],
            y=df_county['deaths_per_100k'],
            mode='markers',
            marker_color=df_county['color'],
            hovertemplate=
            '<b>County</b>: %{text}<extra></extra>'+
            '<br><b>Vaccination Rate</b>: %{x:.3f}</br>'+
            '<b>Deaths</b>: %{y:.2f}',
            text=df_county['county']
        )
        # trace6=go.Bar(
        #     x=df['date'],
        #     y=pd.to_numeric(df['actuals.newCases'])/(pop/100000),
        #     name='Cases per 100k'
        # )
        trace7=go.Scatter(
            x=df_county['VaxPct'],
            y=df_county['curve_fit'],
            mode='lines'
        )
        state_url = "https://covidactnow.org/embed/us/" + us_state_abbrev[state_selected].lower()
        state_report = html.Iframe(src=state_url, height="370", width="350", style={'frameBorder':'0'})
    else:
        df = df_global.loc[df_global['location']==country_selected].reset_index()

        pop = int(df['population'].iloc[-1])

        full_vax_col = df['people_fully_vaccinated']
        full_vax = 0
        full_vax_pct = 0
        if full_vax_col[full_vax_col != 0].shape[0] != 0:
            full_vax = int(full_vax_col[full_vax_col != 0].iloc[-1])
            full_vax_pct = round((full_vax/pop)*100, 2)

        total_vax_col = df['people_vaccinated']
        total_vax = 0
        total_vax_pct = 0
        if total_vax_col[total_vax_col != 0].shape[0] != 0:
            total_vax = int(total_vax_col[total_vax_col != 0].iloc[-1])
            total_vax_pct = round((total_vax/pop)*100, 2)

        boosted_col = df['total_boosters']
        boosted = 0
        boosted_pct = 0
        if boosted_col[boosted_col != 0].shape[0] != 0:
            boosted = int(boosted_col[boosted_col != 0].iloc[-1])
            boosted_pct = round((boosted/pop)*100, 2)

        df['cases_smoothed'] = df.new_cases.rolling(7).mean()
        df['deaths_smoothed'] = df.new_deaths.rolling(7).mean()
        
        df_state = get_state_data()
        df_state = df_state.loc[df_state['VaxPct'] > 0]

        trace1=go.Bar(
            x=df['date'],
            y=df['new_cases'],
            name='Cases'
        )
        trace2=go.Scatter(
            x=df['date'],
            y=df['cases_smoothed'],
            name = "Cases smoothed"
        )
        trace3=go.Bar(
            x=df['date'],
            y=df['new_deaths'],
            name='Deaths'
        )
        trace4=go.Scatter(
            x=df['date'],
            y=df['deaths_smoothed'],
            name = "Deaths smoothed"
        )
        trace5=go.Scatter(
            x=df_state['VaxPct'],
            y=df_state['deaths_per_100k'],
            mode='markers',
            marker_color=df_state['color'],
            hovertemplate=
            '<b>State</b>: %{text}<extra></extra>'+
            '<br><b>Vaccination Rate</b>: %{x:.3f}</br>'+
            '<b>Deaths</b>: %{y:.2f}',
            text=df_state['state']
        )
        # trace6=go.Bar(
        #     x=df['date'],
        #     y=pd.to_numeric(df['new_cases'])/(pop/100000),
        #     name='Cases per 100k'
        # )
        trace7=go.Scatter(
            x=df_state['VaxPct'],
            y=df_state['curve_fit'],
            mode='lines'
        )
        if(country_selected == "United States"):
            state_report = html.Iframe(src="https://covidactnow.org/embed/risk/us/", style={'height': '450px', 'width':'460px','scrolling': 'no'})
        else:
            trace5=go.Scatter(
                x=df['total_cases'],
                y=df['new_cases'],
                mode='markers'
            )
            state_report = html.Div([
                html.H1(
                    children=[
                        html.Strong("Vaccinated (1+ Dose): ", style={'color':'#fff'}), 
                        html.Span("{pct}%".format(pct=total_vax_pct if total_vax_pct > 0.0 else '--'), style={'color':'#fff'})
                    ], 
                    style={'margin':'10px'}
                ),
                html.H1(
                    children=[
                        html.Strong("Fully Vaccinated: ", style={'color':'#fff'}), 
                        html.Span("{pct}%".format(pct=full_vax_pct if full_vax_pct > 0.0 else '--'), style={'color':'#fff'})
                    ], 
                    style={'margin':'10px'}
                ),
                html.H1(
                    children=[
                        html.Strong("Boosted: ", style={'color':'#fff'}), 
                        html.Span("{pct}%".format(pct=boosted_pct if boosted_pct>0.0 else "--"), style={'color':'#fff'})
                    ], 
                    style={'margin-bottom':'0px'}
                )
            ], style={'border':'solid white 1px', 'border-radius':'25px', 'text-align':'center', 'top':'50%', 'transform':'translateY(50%)'})

    plot1={
        'data':[trace1, trace2],
        'layout': dict(
            title={'text':'Change in Daily Cases (Linear)','font':{'color':'#fff'}},
            xaxis={'title': 'Date', 'color':'#fff'},
            yaxis={'title': {'text':'New Cases', 'standoff':10}, 'automargin':True, 'color':'#fff'},
            margin={'l': 40, 'b': 40, 't': 40, 'r': 10},
            legend={'x': 0, 'y': 1, 'font':{'color':'#fff'}},
            plot_bgcolor='#121212',
            paper_bgcolor='#121212',
            hoverlabel={'bordercolor':'white','font':{'color':'white'}, 'namelength':-1},
            hovermode='closest'
        )

    }

    plot2={
        'data':[trace3, trace4],
        'layout': dict(
            title={'text':'Change in Daily Deaths (Linear)', 'font':{'color':'#fff'}},
            xaxis={'title': 'Date', 'color':'#fff'},
            yaxis={'title': {'text':'New Deaths', 'standoff':10}, 'automargin':True,'color':'#fff'},
            margin={'l': 40, 'b': 40, 't': 40, 'r': 10},
            legend={'x': 0, 'y': 1,'font':{'color':'#fff'}},
            plot_bgcolor='#121212',
            paper_bgcolor='#121212',
            hoverlabel={'bordercolor':'white','font':{'color':'white'}, 'namelength':-1},
            hovermode='closest'
        )

    }

    plot3={
        'data':[trace5, trace7],
        'layout': dict(
            title={'text':'Deaths vs. Vaccination Rate', 'font':{'color':'#fff'}},
            xaxis={'title':'Vaccination Rate', 'color':'#fff'},
            yaxis={'title':{'text':'Deaths per 100k (Over Last Two Weeks)', 'standoff':10}, 'automargin':True, 'color':'#fff'},
            margin={'l': 40, 'b': 40, 't': 40, 'r': 10},
            plot_bgcolor='#121212',
            paper_bgcolor='#121212',
            showlegend=False,
            hoverlabel={'bordercolor':'white','font':{'color':'white'}, 'namelength':-1},
            hovermode='closest'
        )

    }
        
    plot4={
        'data':[trace5],
        'layout': dict(
            title={'text':'Cumulative Cases vs Change in Cases (loglog)', 'font':{'color':'#fff'}},
            xaxis={'title': 'Cumulative Cases (log)', 'type':'log', 'color':'#fff'},
            yaxis={'title': {'text':'Change in Cases (log)','standoff':6}, 'type':'log', 'automargin':True, 'color':'#fff'},
            margin={'l': 40, 'b': 40, 't': 40, 'r': 10},
            legend={'x': 0, 'y': 1, 'font':{'color':'#fff'}},
            plot_bgcolor='#121212',
            paper_bgcolor='#121212',
            hoverlabel={'bordercolor':'white','font':{'color':'white'}, 'namelength':-1},
            hovermode='closest'
        )

    }

    if (country_selected == "United States"):
        if plot3['data'][0].y.size > 0:
            plot3['layout']['yaxis'].update(range=[0,1.1*max(plot3['data'][0].y)])
        return plot1, plot2, plot3, state_report
    else:
        return plot1, plot2, plot4, state_report


if __name__ == '__main__':
    application.run(debug=True, port=8080)










