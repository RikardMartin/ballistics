#%%
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import streamlit as st


# Create a Streamlit app
st.set_page_config(page_title="Ballistics simulator")
st.title("Bullet ballistics simulator")

# Initialize session state to track whether the simulation has been run
if 'simulation_run' not in st.session_state:
    st.session_state.simulation_run = False
    st.session_state.results_df = None  # Store the simulation result dataframe


##% Parameters

# Environmental constants
rho = 1.293
g = 9.82
Cd = 0.5

# Initial conditions
st.write("Specify initial conditions:")
x0 = st.number_input("Enter value for x0:", value=0.0, step=0.1)
y0 = st.number_input("Enter value for y0:", value=0.0, step=0.1)
theta0_deg = st.slider("Select angle θ₀ (degrees):", min_value=-90, max_value=90, value=0)

# Gear specifications 
# .22 hornet v-max
calibers = {
    ".22 hornet - Hornady v-max": {
        'L': 5.6e-3,
        'm_gr': 35,
        'v0': 945
    },
    ".30-06 - Federal premium": {
        'L': 0.0762,
        'm_gr': 180,
        'v0': 823
    }

}

# Add a "Custom" option to the list of calibers
caliber_options = list(calibers.keys()) + ['Custom']

# Let the user select a caliber or choose to input custom values
selected_caliber = st.selectbox("Select a caliber or choose 'Custom' to input your own:", caliber_options)

# Initialize the caliber_data dictionary
caliber_data = {}

# If the user selects "Custom", show input fields for the custom caliber
if selected_caliber == 'Custom':
    st.write("Input custom caliber specifications:")
    L = st.number_input("Enter bullet diameter in meters:", value=0.0, step=0.0001, format='%.4f')
    m_gr = st.number_input("Enter the mass (m_gr) in grains:", value=0, step=1)
    v0 = st.number_input("Enter the initial velocity (v0) in m/s:", value=0, step=1)
    
    # Save custom input into a dictionary
    caliber_data = {
        'L': L,
        'm_gr': m_gr,
        'v0': v0
    }

else:
    # If a pre-defined caliber is selected, retrieve the corresponding data
    caliber_data = calibers[selected_caliber]


#%% Functions

def setup(x0, y0, theta0_deg, caliber):

    # Calculated constants
    A = np.pi*caliber['L']**2/4
    theta0 = theta0_deg*np.pi/180
    vx0 = caliber['v0']*np.cos(theta0)
    vy0 = caliber['v0']*np.sin(theta0)
    m = caliber['m_gr']*6.479891e-5
    C = 0.5*Cd*rho*A

    st.write(f"Configuration: x0={x0}, y0={y0}, θ₀={theta0_deg}°, caliber={selected_caliber}, caliber values={caliber_data}")

    return vx0, vy0, m, C


def calc_trajectory(x0, y0, vx0, vy0, m, C):

    max_timesteps = 500
    dt = 0.001

    timesteps = np.arange(0, max_timesteps)
    r = {'x': np.zeros(max_timesteps), 'y': np.zeros(max_timesteps)}
    v = {'x': np.zeros(max_timesteps), 'y': np.zeros(max_timesteps), 'v': np.zeros(max_timesteps)}
    a = {'x': np.zeros(max_timesteps), 'y': np.zeros(max_timesteps)}

    r['x'][0] = x0
    r['y'][0] = y0
    v['x'][0] = vx0
    v['y'][0] = vy0

    for t in timesteps[:-1]:
        v['v'][t] = np.sqrt(v['x'][t]**2 + v['y'][t]**2)
        # print('v:', v['v'][t])

        a['x'][t] = -C/m*v['x'][t]*v['v'][t]
        a['y'][t] = -C/m*v['y'][t]*v['v'][t] - g
        # print('ax:', a['x'][t], 'ay:', a['y'][t])

        v['x'][t+1] = v['x'][t] + a['x'][t]*dt
        v['y'][t+1] = v['y'][t] + a['y'][t]*dt

        r['x'][t+1] = r['x'][t] + v['x'][t]*dt + 0.5*a['x'][t]*dt**2
        r['y'][t+1] = r['y'][t] + v['y'][t]*dt + 0.5*a['y'][t]*dt**2

    timesteps = timesteps*dt
    data = {
        'x': r['x'],
        'y': r['y'],
        'v': v['v'],
        't': timesteps
    }
    
    return pd.DataFrame(data)



# Button to run the simulation
if st.button("Calculate ballistics"):

    # Set up the simulation
    vx0, vy0, m, C = setup(x0, y0, theta0_deg, caliber_data)

    # Run the simulation and store the results in session state
    st.session_state.results_df = calc_trajectory(x0, y0, vx0, vy0, m, C)

    st.session_state.simulation_run = True  # Mark that simulation has been run

    
if st.session_state.simulation_run:

    # Display the dataframe
    st.markdown("## Simulation Results:")
    st.write("x = horisontal distance, y = vertical distance, v = speed, t = time on seconds")

    # Available dimensions from the results dataframe
    results_df = st.session_state.results_df
    dimensions = results_df.columns.tolist()

    # User selection for x-axis and y-axis
    x_axis = st.selectbox("Select X-axis dimension:", options=dimensions, key='x_axis')
    y_axis = st.selectbox("Select Y-axis dimension:", options=dimensions, key='y_axis')

    # Ensure both axes are selected before plotting
    if st.button('Plot'):
        if x_axis and y_axis:
            # Create a plot using Plotly Express
            fig = px.line(results_df, x=x_axis, y=y_axis,
                            title=f"Plotting {y_axis} against {x_axis}",
                            labels={x_axis: f'{x_axis}', y_axis: f'{y_axis}'})
            
            # Customize plot size and background colors
            fig.update_layout(
                width=800,  # Set width of the plot
                height=600,  # Set height of the plot
                paper_bgcolor='black',  # Background color of the entire plot
                plot_bgcolor='lightblue',  # Background color of the plotting area
                title_font_size=20  # Increase the font size of the title
            )
            
            # Display the plot in Streamlit
            st.plotly_chart(fig)
        else:
            st.write("Please select both X-axis and Y-axis dimensions.")
