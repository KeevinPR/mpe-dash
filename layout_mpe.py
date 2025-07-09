from dash import dcc, html
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto

# Prepare an initial default network (Asia network) for testing
# The file "models/asia.bif" is expected to exist in the models folder.
# We will load its content as base64 to set as default data for the app.
import base64, os
default_data = None
default_filename = "asia.bif"
default_path = os.path.join("models", default_filename)
if os.path.exists(default_path):
    with open(default_path, "rb") as f:
        default_content = f.read()
        # Encode file content to base64 for storing in dcc.Store
        default_base64 = base64.b64encode(default_content).decode("utf-8")
        default_data = {
            "filename": default_filename,
            "content": default_base64
            # (We will parse this content in the callbacks; state names will be determined later)
        }

# Define the layout of the Dash application
# Using a Bootstrap Container for consistent padding/responsive design
layout = dbc.Container(fluid=True, children=[
    # Title
    html.H3("Most Probable Explanation (MPE) Inference", className="text-center mt-4 mb-4"),
    
    # Hidden data storage for the network and evidence (for inter-callback communication)
    dcc.Store(id="stored-network", data=default_data),
    dcc.Store(id="evidence-store", data=[]),
    
    # Upload and Run controls
    dbc.Row([
        # File upload component (for .bif or .json files)
        dbc.Col([
            dcc.Upload(
                id="upload-network",
                children=html.Div([
                    "Drag and Drop or ",
                    html.A("Select a Bayesian Network File", href="#")
                ]),
                style={
                    "width": "100%", "height": "60px", "lineHeight": "60px",
                    "borderWidth": "1px", "borderStyle": "dashed",
                    "borderRadius": "5px", "textAlign": "center",
                    "margin-bottom": "10px"
                },
                # Allow only one file and expect .bif, .xml, or .json format
                multiple=False
            )
        ], width=6),
        # Inference algorithm selection (only one option for now, but scalable)
        dbc.Col([
            dbc.Label("Inference Algorithm:", html_for="algorithm-select"),
            dcc.Dropdown(
                id="algorithm-select",
                options=[{"label": "Exact (Variable Elimination)", "value": "exact"}],
                value="exact",  # default selection
                clearable=False
            )
        ], width=3),
        # Run MPE button
        dbc.Col([
            dbc.Label(" ", style={"visibility": "hidden"}),  # spacer label for alignment
            html.Button("Run MPE", id="run-button", n_clicks=0, className="btn btn-primary")
        ], width=3)
    ], className="mb-4"),
    
    # Evidence selection controls
    dbc.Row([
        dbc.Col([
            dbc.Label("Evidence Variable:"),
            dcc.Dropdown(
                id="evidence-variable",
                options=[],  # filled dynamically after network load
                value=None,
                placeholder="Select variable"
            )
        ], width=4),
        dbc.Col([
            dbc.Label("Evidence Value:"),
            dcc.Dropdown(
                id="evidence-value",
                options=[],
                value=None,
                placeholder="Select value",
                disabled=True  # disabled until a variable is selected
            )
        ], width=4),
        dbc.Col([
            dbc.Label(" ", style={"visibility": "hidden"}),  # spacer for alignment
            html.Button("Add Evidence", id="add-evidence", n_clicks=0, className="btn btn-secondary")
        ], width=2),
        dbc.Col([
            dbc.Label(" ", style={"visibility": "hidden"}),  # spacer
            html.Button("Clear Evidence", id="clear-evidence", n_clicks=0, className="btn btn-light")
        ], width=2)
    ], className="mb-2"),
    dbc.Row([
        dbc.Col([
            html.Div(id="evidence-list", children="No evidence selected.", className="text-muted")
        ], width=12)
    ], className="mb-4"),
    
    # Output: Cytoscape network graph and MPE result text
    dbc.Row([
        dbc.Col([
            cyto.Cytoscape(
                id="cytoscape-network",
                elements=[],  # populated dynamically with network nodes/edges
                layout={"name": "cose"},  # use force-directed layout for initial display
                style={"width": "100%", "height": "400px"},
                stylesheet=[
                    # Style for all nodes: display label
                    {"selector": "node", "style": {"label": "data(label)"}},
                    # Highlight evidence nodes with a distinct color and border
                    {"selector": "[evidence = 'True']", 
                     "style": {"background-color": "#97e6ff", "border-color": "#3b99fc", "border-width": "2px"}},
                    # Style for edges: curved lines with arrows
                    {"selector": "edge", 
                     "style": {"curve-style": "bezier", "target-arrow-color": "#ccc",
                               "target-arrow-shape": "triangle", "arrow-scale": 1.2,
                               "line-color": "#ccc"}}
                ]
            )
        ], width=8),
        dbc.Col([
            html.Div(id="mpe-output", children=[
                # This will contain the output text (assignment and probability) after running MPE
            ], style={"whiteSpace": "pre-line"})  # preserve whitespace/newlines if any
        ], width=4)
    ])
])
