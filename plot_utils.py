import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import json

def create_plot(df, time_range='max'):
    
    if time_range == '1m':

        filtered_df = df.tail(21)
    elif time_range == '3m':

        filtered_df = df.tail(63)
    elif time_range == '1y':

        filtered_df = df.tail(252)
    elif time_range == '5y':

        filtered_df = df.tail(1260)
    else:
        filtered_df = df

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered_df['Date'].dt.strftime('%Y-%m-%d').tolist(),
        y=filtered_df['Close'].astype(float).tolist(),
        mode='lines',
        name='Close Price'
        ))

    fig.update_layout(
        title=f'Asian Paints Close Price ({time_range.upper()})',
        xaxis_title='Date',
        yaxis_title='Close Price',
        template='plotly_dark'
    )

    return pio.to_json(fig)
