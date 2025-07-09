import dash
import dash_bootstrap_components as dbc

# Initialize the Dash app with a Bootstrap theme for consistent styling
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
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

# Run the app for local testing (will not execute in production when using Gunicorn)
if __name__ == "__main__":
    app.run_server(debug=True)
