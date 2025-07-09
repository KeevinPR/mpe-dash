import dash
import dash_bootstrap_components as dbc

# Initialize the Dash app with a Bootstrap theme for consistent styling
# Global variables
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        'https://bayes-interpret.com/Evidence/MPEDash/assets/liquid-glass.css'  # Apple Liquid Glass CSS
    ],
    requests_pathname_prefix='/Evidence/MPEDash/',
    suppress_callback_exceptions=True
)
app.title = "MPE Dash App"  # Title shown in the browser tab

# Import the layout and callback registrations
from layout_mpe import layout
from callbacks_mpe import register_callbacks

# Set the application layout
app.layout = layout

# Register all the callbacks with the app
register_callbacks(app)

# Expose the server for WSGI (e.g., Gunicorn)
server = app.server

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8057)
