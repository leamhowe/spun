import pandas as pd
import altair as alt

# Load the CSV file created by the previous script
file_name = "20251113_hyperparameter_comparison.csv"
try:
    df = pd.read_csv(file_name)
    
    # Fill any NaN values (like in snow_weight) with a string 'N/A' 
    # This makes tooltips cleaner.
    df = df.fillna('N/A')

    # Create the interactive scatter plot
    chart = alt.Chart(df).mark_circle(size=100, opacity=0.7).encode(
        # X-axis: RMSE_snow. Lower is better, so we want points on the left.
        x=alt.X('RMSE_snow', 
                scale=alt.Scale(zero=False),
                title='RMSE Snow (Lower is better)'),
        
        # Y-axis: Dice. Higher is better, so we want points at the top.
        y=alt.Y('Dice', 
                scale=alt.Scale(zero=False),
                title='Dice Score (Higher is better)'),
        
        # Color: Use the 'weights' column to compare groups.
        color=alt.Color('weights', title="Weights"),
        
        # Tooltip: Show all relevant info on hover.
        tooltip=[
            'weights',
            'trial',
            'RMSE_snow',
            'Dice',
            'loss_weighting',
            'loss_fn',
            'lr',
            'batch_size',
            'snow_thresh',
            'snow_weight',
            'dice_loss_weight'
        ]
    ).properties(
        title='Hyperparameter Trade-off: Dice vs. RMSE',
        width=700,
        height=500
    ).interactive() # This makes the plot zoomable and pannable

    # Save the chart as a JSON file
    output_chart_json = "20251113_hyperparameter_tradeoff_plot.json"
    chart.save(output_chart_json)
    
    print(f"Chart saved to {output_chart_json}")
    print("This plot shows the 'Pareto front' of your optimal models.")
    print("Look for points in the TOP-LEFT corner.")

except FileNotFoundError:
    print(f"Error: The file '{file_name}' was not found.")
    print("Please make sure the CSV file is in the same directory.")
except Exception as e:
    print(f"An error occurred: {e}")