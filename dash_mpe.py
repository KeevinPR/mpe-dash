import dash
from dash import dcc, html, Input, Output, State, ALL
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import pandas as pd
import time
import json
import logging
import base64
import sys
import os
from dash.exceptions import PreventUpdate
from pgmpy.readwrite import BIFReader
from pgmpy.inference import VariableElimination
import warnings

# Add parent directory to sys.path to resolve imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import session management components
try:
    from dash_session_manager import start_session_manager, get_session_manager
    from dash_session_components import create_session_components, setup_session_callbacks, register_long_running_process
    SESSION_MANAGEMENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Session management not available: {e}")
    SESSION_MANAGEMENT_AVAILABLE = False
    def start_session_manager(): pass
    def get_session_manager(): return None
    def create_session_components(): return None, html.Div()
    def setup_session_callbacks(app): pass
    def register_long_running_process(session_id): pass

# Import MPE solver functions
from mpe_solver import load_model_from_bytes, calculate_joint_probability

warnings.filterwarnings("ignore")

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

print("🚀 MPE DASHBOARD STARTING...")
print(f"Python: {sys.version}")
print(f"Dash version: {dash.__version__}")
print(f"Session management: {SESSION_MANAGEMENT_AVAILABLE}")

# Start the global session manager
if SESSION_MANAGEMENT_AVAILABLE:
    start_session_manager()

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
app.title = "MPE Dash App"
server = app.server
print("✅ Dash app created")

# Safari Compatibility CSS Fix for Liquid Glass Effects
SAFARI_FIX_CSS = """
<style>
/* === SAFARI LIQUID GLASS COMPATIBILITY FIXES === */
@media not all and (min-resolution:.001dpcm) {
    @supports (-webkit-appearance:none) {
        .card {
            background: transparent !important;
        }
        .card::before {
            background: rgba(255, 255, 255, 0.12) !important;
            -webkit-backdrop-filter: blur(15px) saturate(180%) !important;
            backdrop-filter: blur(15px) saturate(180%) !important;
        }
        .btn {
            background: transparent !important;
            -webkit-backdrop-filter: blur(15px) !important;
            backdrop-filter: blur(15px) !important;
        }
        .btn::before {
            background: rgba(255, 255, 255, 0.12) !important;
        }
        .form-control {
            background: rgba(255, 255, 255, 0.15) !important;
            -webkit-backdrop-filter: blur(10px) !important;
            backdrop-filter: blur(10px) !important;
        }
    }
}
@supports not (backdrop-filter: blur(1px)) {
    .card {
        background: rgba(255, 255, 255, 0.85) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
    }
    .btn {
        background: rgba(255, 255, 255, 0.2) !important;
    }
}
</style>
"""

# Global model cache
cached_model = None

# Create session components
if SESSION_MANAGEMENT_AVAILABLE:
    session_components = html.Div([
        dcc.Store(id='session-id-store', data=None),
        dcc.Store(id='heartbeat-counter', data=0),
        dcc.Interval(
            id='heartbeat-interval',
            interval=5*1000,
            n_intervals=0,
            disabled=False
        ),
        dcc.Interval(
            id='cleanup-interval', 
            interval=30*1000,
            n_intervals=0,
            disabled=False
        ),
        html.Div(id='session-status', style={'display': 'none'}),
        html.Script("""
            if (!window.dashSessionId) {
                window.dashSessionId = 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
            }
            document.addEventListener('click', function() {
                if (window.dashHeartbeat) window.dashHeartbeat();
            });
            document.addEventListener('keypress', function() {
                if (window.dashHeartbeat) window.dashHeartbeat();
            });
            window.addEventListener('beforeunload', function() {
                if (navigator.sendBeacon) {
                    navigator.sendBeacon('/dash/_disconnect', JSON.stringify({
                        session_id: window.dashSessionId
                    }));
                }
            });
            if (window.parent !== window) {
                try {
                    window.parent.addEventListener('beforeunload', function() {
                        if (navigator.sendBeacon) {
                            navigator.sendBeacon('/dash/_disconnect', JSON.stringify({
                                session_id: window.dashSessionId
                            }));
                        }
                    });
                } catch(e) {
                    console.log('Cross-origin iframe detected');
                }
            }
        """),
    ], style={'display': 'none'})
    session_id = None
else:
    session_id = None
    session_components = html.Div()

# App layout
print("🎨 Creating layout...")
app.layout = html.Div([
    # Safari Compatibility Fix
    html.Div([
        dcc.Markdown(SAFARI_FIX_CSS, dangerously_allow_html=True)
    ], style={'display': 'none'}),
    
    # SESSION MANAGEMENT COMPONENTS
    session_components,
    
    dcc.Store(id='notification-store'),
    html.Div(id='notification-container', style={
        'position': 'fixed',
        'bottom': '20px',
        'right': '20px',
        'zIndex': '1000',
        'width': '300px',
        'transition': 'all 0.3s ease-in-out',
        'transform': 'translateY(100%)',
        'opacity': '0'
    }),
    
    dcc.Loading(
        id="global-spinner",
        type="default",
        fullscreen=False,
        color="#00A2E1",
        style={
            "position": "fixed",
            "top": "50%",
            "left": "50%",
            "transform": "translate(-50%, -50%)",
            "zIndex": "999999"
        },
        children=html.Div([
            html.H1("Most Probable Explanation (MPE)", style={'textAlign': 'center'}),

            html.Div(
                className="link-bar",
                style={"textAlign": "center", "marginBottom": "20px"},
                children=[
                    html.A(
                        children=[
                            html.Img(
                                src="https://cig.fi.upm.es/wp-content/uploads/github.png",
                                style={"height": "24px", "marginRight": "8px"}
                            ),
                            "MPE GitHub"
                        ],
                        href="https://github.com/KeevinPR/mpe-dash",
                        target="_blank",
                        className="btn btn-outline-info me-2"
                    ),
                    html.A(
                        children=[
                            html.Img(
                                src="https://cig.fi.upm.es/wp-content/uploads/2023/11/cropped-logo_CIG.png",
                                style={"height": "24px", "marginRight": "8px"}
                            ),
                            "Documentation"
                        ],
                        href="https://pgmpy.org/",
                        target="_blank",
                        className="btn btn-outline-primary me-2"
                    ),
                ]
            ),

            html.Div([
                html.P(
                    "Find the Most Probable Explanation (MPE) for your Bayesian Network given evidence.",
                    style={"textAlign": "center", "maxWidth": "800px", "margin": "0 auto"}
                )
            ], style={"marginBottom": "20px"}),

            # (1) BIF Upload
            html.Div(className="card", children=[
                html.Div([
                    html.H3("1. Load Bayesian Network (.bif)", style={'display': 'inline-block', 'marginRight': '10px', 'textAlign': 'center'}),
                    dbc.Button(
                        html.I(className="fa fa-question-circle"),
                        id="help-button-bif-upload",
                        color="link",
                        style={"display": "inline-block", "verticalAlign": "middle", "padding": "0", "marginLeft": "5px"}
                    ),
                ], style={"textAlign": "center", "position": "relative"}),
                html.Div([
                    html.Div([
                        html.Img(
                            src="https://img.icons8.com/ios-glyphs/40/cloud--v1.png",
                            className="upload-icon"
                        ),
                        html.Div("Drag and drop or select a .bif file", className="upload-text")
                    ]),
                    dcc.Upload(
                        id='upload-bif',
                        children=html.Div([], style={'display': 'none'}),
                        className="upload-dropzone",
                        multiple=False
                    ),
                ], className="upload-card"),

                html.Div([
                    dcc.Checklist(
                        id='use-default-network',
                        options=[{'label': 'Use default Asia network (asia.bif)', 'value': 'default'}],
                        value=[],
                        style={'display': 'inline-block', 'marginTop': '10px'}
                    ),
                    dbc.Button(
                        html.I(className="fa fa-question-circle"),
                        id="help-button-default-dataset",
                        color="link",
                        style={"display": "inline-block", "marginLeft": "8px"}
                    ),
                html.Div(id='upload-status', style={'textAlign': 'center', 'color': 'green'}),
                ], style={'textAlign': 'center'}),
            ]),

            # (2) Evidence Selection
            html.Div(className="card", children=[
                html.Div([
                    html.H3("2. Select Evidence", style={'display': 'inline-block', 'marginRight': '10px', 'textAlign': 'center'}),
                    dbc.Button(
                        html.I(className="fa fa-question-circle"),
                        id="help-button-evidence",
                        color="link",
                        style={"display": "inline-block", "verticalAlign": "middle", "padding": "0", "marginLeft": "5px"}
                    ),
                ], style={"textAlign": "center", "position": "relative"}),
                
                # Buttons for bulk selection
                html.Div([
                    dbc.Button(
                        "Select All",
                        id="select-all-evidence",
                        color="outline-primary",
                        size="sm",
                        style={'marginRight': '10px'}
                    ),
                    dbc.Button(
                        "Clear All",
                        id="clear-evidence",
                        color="outline-secondary",
                        size="sm"
                    )
                ], style={'textAlign': 'center', 'marginBottom': '15px'}),
                
                # Checkbox container for evidence variables
                html.Div(
                    id='evidence-checkbox-container',
                    style={
                        'maxHeight': '200px',
                        'overflowY': 'auto',
                        'border': '1px solid #ddd',
                        'borderRadius': '5px',
                        'padding': '10px',
                        'margin': '0 auto',
                        'width': '80%',
                        'backgroundColor': '#f8f9fa'
                    }
                ),
                
                html.Div(id='evidence-values-container')
            ]),

            # (3) Run MPE
            html.Div([
                html.Div([
                    dbc.Button(
                        [
                            html.I(className="fas fa-play-circle me-2"),
                            "Run MPE"
                        ],
                        id='run-mpe-button',
                        n_clicks=0,
                        color="info",
                        className="btn-lg",
                        style={
                            'fontSize': '1.1rem',
                            'padding': '0.75rem 2rem',
                            'borderRadius': '8px',
                            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                            'transition': 'all 0.3s ease',
                            'backgroundColor': '#00A2E1',
                            'border': 'none',
                            'margin': '1rem 0',
                            'color': 'white',
                            'fontWeight': '500'
                        }
                    )
                ], style={'textAlign': 'center'}),
            ], style={'textAlign': 'center'}),

            html.Br(),
            html.Div(id='mpe-results'),

            # Network Visualization
            html.Div(className="card", id='network-visualization', style={'display': 'none'}, children=[
                html.H4("Network Visualization", style={'textAlign': 'center', 'marginBottom': '20px'}),
                cyto.Cytoscape(
                    id="cytoscape-network",
                    elements=[],
                    layout={"name": "cose"},
                    style={"width": "100%", "height": "400px"},
                    stylesheet=[
                        {"selector": "node", "style": {"label": "data(label)", "text-valign": "center", "text-halign": "center"}},
                        {"selector": "[evidence = 'True']", 
                         "style": {"background-color": "#97e6ff", "border-color": "#3b99fc", "border-width": "2px"}},
                        {"selector": "edge", 
                         "style": {"curve-style": "bezier", "target-arrow-color": "#ccc",
                                   "target-arrow-shape": "triangle", "arrow-scale": 1.2,
                                   "line-color": "#ccc"}}
                    ]
                )
            ]),

            # Hidden stores
            dcc.Store(id='stored-network'),
        ])
    ),
    
    # Popovers
    dbc.Popover(
        [
            dbc.PopoverHeader(
                [
                    "Bayesian Network Requirements",
                    html.I(className="fa fa-check-circle ms-2", style={"color": "#198754"})
                ],
                style={"backgroundColor": "#f8f9fa", "fontWeight": "bold"}
            ),
            dbc.PopoverBody(
                [
                    html.Ul([
                        html.Li([html.Strong("Format: "), "BIF (Bayesian Interchange Format) file"]),
                        html.Li([html.Strong("Structure: "), "Must be a valid Bayesian network"]),
                        html.Li([html.Strong("Default: "), "You can use the default Asia network for testing"]),
                    ]),
                ],
                style={"backgroundColor": "#ffffff", "borderRadius": "0 0 0.25rem 0.25rem", "maxWidth": "300px"}
            ),
        ],
        id="help-popover-bif-upload",
        target="help-button-bif-upload",
        placement="right",
        is_open=False,
        trigger="hover",
    ),

    dbc.Popover(
        [
            dbc.PopoverHeader("Help", style={"backgroundColor": "#f8f9fa", "fontWeight": "bold"}),
            dbc.PopoverBody([
                html.P([
                    "For details about the default dataset, check out: ",
                    html.A("asia.bif", href="https://github.com/pgmpy/pgmpy/tree/dev/examples", target="_blank"),
                ]),
                html.P("Feel free to upload your own dataset at any time.")
            ]),
        ],
        id="help-popover-default-dataset",
        target="help-button-default-dataset",
        placement="right",
        is_open=False,
        trigger="hover"
    ),

    dbc.Popover(
        [
            dbc.PopoverHeader("Evidence Selection", style={"backgroundColor": "#f8f9fa", "fontWeight": "bold"}),
            dbc.PopoverBody([
                html.P("Evidence variables are the observed states in your network."),
                html.P("Select variables and their values to condition the MPE computation."),
                html.P("MPE finds the most probable assignment for all other variables."),
            ]),
        ],
        id="help-popover-evidence",
        target="help-button-evidence",
        placement="right",
        is_open=False,
        trigger="hover",
    ),
])
print("✅ Layout complete!")

def get_model(stored_network):
    """Use cached model if available, otherwise parse from stored_network"""
    global cached_model
    if cached_model is not None:
        return cached_model
    if not stored_network:
        return None

    try:
        if stored_network['network_type'] == 'string':
            content_bytes = base64.b64decode(stored_network['content'])
            model = load_model_from_bytes(content_bytes, stored_network['network_name'])
            cached_model = model
            return model
        elif stored_network['network_type'] == 'path':
            with open(stored_network['content'], 'rb') as f:
                content_bytes = f.read()
            model = load_model_from_bytes(content_bytes, stored_network['network_name'])
            cached_model = model
            return model
        else:
            return None
    except Exception as e:
        print(f"❌ Model parsing error: {e}")
        return None

# Callbacks start here...
@app.callback(
    Output('stored-network', 'data'),
    Output('upload-status', 'children'),
    Output('use-default-network', 'value'),
    Output('cytoscape-network', 'elements'),
    Output('network-visualization', 'style'),
    Input('upload-bif', 'contents'),
    State('upload-bif', 'filename'),
    Input('use-default-network', 'value')
)
def load_network(contents, filename, use_default_value):
    """Load network and update all dependent components"""
    print(f"📞 CALLBACK: use_default={use_default_value}, has_file={contents is not None}")
    
    global cached_model
    cached_model = None
    
    # Default network visualization style (hidden)
    viz_style = {'display': 'none'}
    
    # PRIORITY 1: Handle file upload
    if contents is not None:
        content_type, content_string = contents.split(',')
        try:
            content_bytes = base64.b64decode(content_string)
            model = load_model_from_bytes(content_bytes, filename)
            
            # Prepare network graph elements
            elements = []
            for var in model.nodes():
                elements.append({
                    "data": {"id": var, "label": var, "evidence": "False"}
                })
            for u, v in model.edges():
                elements.append({
                    "data": {"source": u, "target": v}
                })
            
            viz_style = {'display': 'block'}
            msg = f"Successfully loaded network from {filename}."
            print(f"✅ File loaded: {filename}")
            
            return (
                {
                    'network_name': filename,
                    'network_type': 'string',
                    'content': content_string
                },
                msg,
                [],  # Uncheck default
                elements,
                viz_style
            )
        except Exception as e:
            print(f"❌ File error: {e}")
            return None, f"Error loading {filename}: {e}", use_default_value, [], viz_style
    
    # PRIORITY 2: Handle default checkbox
    if 'default' in use_default_value:
        print("✅ DEFAULT CHECKBOX CHECKED!")
        try:
            # Use relative path from current directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            default_path = os.path.join(current_dir, 'models', 'asia.bif')
            
            if not os.path.exists(default_path):
                default_path = 'models/asia.bif'
            
            print(f"📁 Loading: {default_path}")
            
            with open(default_path, 'rb') as f:
                content_bytes = f.read()
            
            model = load_model_from_bytes(content_bytes, 'asia.bif')
            print(f"✅ Model loaded: {len(model.nodes())} nodes")
            
            elements = []
            for var in model.nodes():
                elements.append({
                    "data": {"id": var, "label": var, "evidence": "False"}
                })
            for u, v in model.edges():
                elements.append({
                    "data": {"source": u, "target": v}
                })
            
            viz_style = {'display': 'block'}
            msg = "Using default network: asia.bif"
            
            print("🔄 Returning successful result")
            return (
                {
                    'network_name': 'asia.bif',
                    'network_type': 'path',
                    'content': default_path
                },
                msg,
                use_default_value,
                elements,
                viz_style
            )
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
            return None, f"Error reading default network: {e}", use_default_value, [], viz_style
    else:
        print("❌ Default checkbox NOT checked")

    return None, "No network selected.", use_default_value, [], viz_style

# Populate the evidence dropdown only if a model is available
@app.callback(
    Output('evidence-checkbox-container', 'children'),
    Input('stored-network', 'data')
)
def update_evidence_variables(stored_network):
    m = get_model(stored_network)
    if not m:
        return html.Div("No network loaded", style={'textAlign': 'center', 'color': '#666'})
    
    variables = sorted(list(m.nodes()))
    if not variables:
        return html.Div("No variables found", style={'textAlign': 'center', 'color': '#666'})
    
    # Create checkboxes in a grid layout
    checkboxes = []
    for i, var in enumerate(variables):
        checkboxes.append(
            html.Div([
                dcc.Checklist(
                    id={'type': 'evidence-checkbox', 'index': var},
                    options=[{'label': f' {var}', 'value': var}],
                    value=[],
                    style={'margin': '0'}
                )
            ], style={'display': 'inline-block', 'width': '50%', 'marginBottom': '5px'})
        )
    
    return html.Div(checkboxes, style={'columnCount': '2', 'columnGap': '20px'})

# Build the dynamic evidence-value dropdowns
@app.callback(
    Output('evidence-values-container', 'children'),
    Input({'type': 'evidence-checkbox', 'index': ALL}, 'value'),
    State('stored-network', 'data')
)
def update_evidence_values(checkbox_values, stored_network):
    # Get selected evidence variables from checkboxes
    ctx = dash.callback_context
    if not ctx.inputs:
        return []
    
    # Extract selected variables
    evidence_vars = []
    for input_info in ctx.inputs_list[0]:
        if input_info['value']:  # If checkbox is checked
            var_name = input_info['id']['index']
            evidence_vars.append(var_name)
    
    if not evidence_vars:
        return []

    m = get_model(stored_network)
    if m is None:
        return []

    children = []
    for var in evidence_vars:
        cpd = m.get_cpds(var)
        if hasattr(cpd, "state_names") and cpd.state_names and var in cpd.state_names:
            states = cpd.state_names[var]
        else:
            card = m.get_cardinality(var)
            states = [str(i) for i in range(card)]
        
        children.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Label(
                                f"Select value for {var}",
                                style={'width': '40%', 'textAlign': 'right', 'paddingRight': '10px'}
                            ),
                            dbc.Select(
                                id={'type': 'evidence-value-dropdown', 'index': var},
                                options=[{'label': s, 'value': s} for s in states],
                                value=states[0] if states else None,
                                style={'width': '60%'}
                            )
                        ],
                        style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}
                    )
                ],
                style={'marginBottom': '10px', 'width': '50%', 'margin': '0 auto'}
            )
        )
    return children

# Callbacks for evidence selection buttons
@app.callback(
    Output({'type': 'evidence-checkbox', 'index': ALL}, 'value'),
    [Input('select-all-evidence', 'n_clicks'),
     Input('clear-evidence', 'n_clicks')],
    [State({'type': 'evidence-checkbox', 'index': ALL}, 'id')],
    prevent_initial_call=True
)
def update_evidence_selection(select_all_clicks, clear_clicks, checkbox_ids):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'select-all-evidence':
        # Select all checkboxes
        return [[checkbox_id['index']] for checkbox_id in checkbox_ids]
    elif button_id == 'clear-evidence':
        # Clear all checkboxes
        return [[] for _ in checkbox_ids]
    
    raise PreventUpdate

@app.callback(
    Output('mpe-results', 'children'),
    Output('cytoscape-network', 'elements', allow_duplicate=True),
    Output('notification-store', 'data'),
    Input('run-mpe-button', 'n_clicks'),
    State('stored-network', 'data'),
    State({'type': 'evidence-value-dropdown', 'index': ALL}, 'value'),
    State({'type': 'evidence-value-dropdown', 'index': ALL}, 'id'),
    State('session-id-store', 'data'),
    prevent_initial_call=True
)
def run_mpe(n_clicks, stored_network, evidence_values, evidence_ids, session_id):
    """Run MPE computation"""
    if not n_clicks:
        raise PreventUpdate

    # Register process with session manager
    if SESSION_MANAGEMENT_AVAILABLE and session_id:
        register_long_running_process(session_id)
        logger.info(f"Registered MPE process for session {session_id}")

    model = get_model(stored_network)
    if model is None:
        return (
            html.Div("No network loaded. Please upload or select the default option.", 
                     style={'color': 'red', 'textAlign': 'center'}),
            [],
            create_error_notification("No network loaded", "Error")
        )

    # Build evidence dictionary from dropdowns
    evidence_dict = {}
    if evidence_values and evidence_ids:
        for ev_id, val in zip(evidence_ids, evidence_values):
            if val is not None:  # ignore if none
                var = ev_id['index']
                cpd = model.get_cpds(var)
                if cpd and hasattr(cpd, "state_names") and cpd.state_names and var in cpd.state_names:
                    state_names = cpd.state_names[var]
                    try:
                        state_idx = state_names.index(val)
                    except ValueError:
                        state_idx = None
                else:
                    try:
                        state_idx = int(val)
                    except:
                        state_idx = None
                if state_idx is not None:
                    evidence_dict[var] = state_idx

    try:
        # Perform MPE inference
        infer = VariableElimination(model)
        all_vars = set(model.nodes())
        evidence_vars = set(evidence_dict.keys())
        query_vars = list(all_vars - evidence_vars)

        if len(query_vars) > 0:
            mpe_assignment = infer.map_query(variables=query_vars, evidence=evidence_dict)
        else:
            mpe_assignment = {}

        # Convert MPE assignment to state indices (pgmpy map_query returns state names, not indices)
        full_assignment = {}
        
        # Add evidence variables with their indices
        for var, state_idx in evidence_dict.items():
            full_assignment[var] = int(state_idx)
        
        # Add MPE variables, converting state names to indices if necessary
        for var, state_value in mpe_assignment.items():
            cpd = model.get_cpds(var)
            if cpd and hasattr(cpd, "state_names") and cpd.state_names and var in cpd.state_names:
                # Convert state name to index
                try:
                    if isinstance(state_value, str):
                        state_idx = cpd.state_names[var].index(state_value)
                    else:
                        state_idx = int(state_value)
                except (ValueError, TypeError):
                    # If conversion fails, assume it's already an index
                    state_idx = int(state_value)
            else:
                # No state names, assume it's already an index
                state_idx = int(state_value)
            full_assignment[var] = state_idx

        # Calculate joint probability
        joint_prob = calculate_joint_probability(model, full_assignment)

        # Format results
        assignment_items = []
        for var, state_idx in sorted(full_assignment.items()):
            label = str(state_idx)
            cpd = model.get_cpds(var)
            if cpd and hasattr(cpd, "state_names") and cpd.state_names and var in cpd.state_names:
                try:
                    label = str(cpd.state_names[var][state_idx])
                except:
                    label = str(state_idx)
            assignment_items.append(f"{var} = {label}")

        # Create results table
        df = pd.DataFrame({
            'Variable': [item.split(' = ')[0] for item in assignment_items],
            'MPE Value': [item.split(' = ')[1] for item in assignment_items]
        })

        table = dbc.Table.from_dataframe(
            df,
            bordered=True,
            striped=True,
            hover=True,
            responsive=True,
            className="mt-2"
        )

        result_card = dbc.Card(
            dbc.CardBody([
                html.H4("MPE Results", className="card-title", style={'textAlign': 'center'}),
                html.P(f"Joint Probability: {joint_prob:.6g}", style={'textAlign': 'center', 'fontSize': '16px'}),
                table
            ])
        )

        # Update cytoscape elements
        new_elements = []
        for var in model.nodes():
            state_idx = full_assignment.get(var)
            label = str(state_idx)
            cpd = model.get_cpds(var)
            if cpd and hasattr(cpd, "state_names") and cpd.state_names and var in cpd.state_names:
                try:
                    label = str(cpd.state_names[var][state_idx])
                except:
                    label = str(state_idx)
            
            new_elements.append({
                "data": {
                    "id": var,
                    "label": f"{var}: {label}",
                    "evidence": "True" if var in evidence_dict else "False"
                }
            })
        
        for u, v in model.edges():
            new_elements.append({
                "data": {"source": u, "target": v}
            })

        return result_card, new_elements, None

    except Exception as e:
        error_msg = f"Error computing MPE: {str(e)}"
        logger.error(error_msg)
        return (
            html.Div(error_msg, style={'color': 'red', 'textAlign': 'center'}),
            [],
            create_error_notification(error_msg, "MPE Error")
        )

# Helper functions for notifications
def create_error_notification(message, header="Error"):
    return {'message': message, 'header': header, 'icon': 'danger'}

def create_info_notification(message, header="Info"):
    return {'message': message, 'header': header, 'icon': 'info'}

# Notification callback
@app.callback(
    [Output('notification-container', 'children'),
     Output('notification-container', 'style')],
    Input('notification-store', 'data')
)
def show_notification(data):
    if data is None:
        return None, {
            'position': 'fixed', 'bottom': '20px', 'right': '20px', 'zIndex': '1000',
            'width': '300px', 'transition': 'all 0.3s ease-in-out',
            'transform': 'translateY(100%)', 'opacity': '0'
        }
    
    toast = dbc.Toast(
        data['message'], header=data['header'], icon=data['icon'],
        is_open=True, dismissable=True,
        style={'width': '100%', 'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
               'borderRadius': '8px', 'marginBottom': '10px'}
    )
    
    container_style = {
        'position': 'fixed', 'bottom': '20px', 'right': '20px', 'zIndex': '1000',
        'width': '300px', 'transition': 'all 0.3s ease-in-out',
        'transform': 'translateY(0)', 'opacity': '1'
    }
    
    return toast, container_style

# Popover callbacks
@app.callback(
    Output("help-popover-bif-upload", "is_open"),
    Input("help-button-bif-upload", "n_clicks"),
    State("help-popover-bif-upload", "is_open")
)
def toggle_bif_upload_popover(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("help-popover-default-dataset", "is_open"),
    Input("help-button-default-dataset", "n_clicks"),
    State("help-popover-default-dataset", "is_open")
)
def toggle_default_dataset_popover(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("help-popover-evidence", "is_open"),
    Input("help-button-evidence", "n_clicks"),
    State("help-popover-evidence", "is_open")
)
def toggle_evidence_popover(n, is_open):
    if n:
        return not is_open
    return is_open

# Session management callbacks
if SESSION_MANAGEMENT_AVAILABLE:
    @app.callback(
        Output('session-id-store', 'data'),
        Input('heartbeat-interval', 'n_intervals'),
        State('session-id-store', 'data'),
        prevent_initial_call=False
    )
    def initialize_session(n_intervals, stored_session_id):
        if stored_session_id is None:
            session_manager = get_session_manager()
            new_session_id = session_manager.register_session()
            session_manager.register_process(new_session_id, os.getpid())
            logger.info(f"New MPE session created: {new_session_id}")
            return new_session_id
        return stored_session_id
    
    @app.callback(
        Output('session-status', 'children'),
        Input('heartbeat-interval', 'n_intervals'),
        State('session-id-store', 'data'),
        prevent_initial_call=True
    )
    def send_heartbeat(n_intervals, session_id):
        if session_id:
            session_manager = get_session_manager()
            session_manager.heartbeat(session_id)
            if n_intervals % 12 == 0:
                logger.info(f"MPE heartbeat sent for session: {session_id}")
            return f"Heartbeat sent: {n_intervals}"
        return "No session"
    
    @app.callback(
        Output('heartbeat-counter', 'data'),
        Input('cleanup-interval', 'n_intervals'),
        State('session-id-store', 'data'),
        prevent_initial_call=True
    )
    def periodic_cleanup_check(n_intervals, session_id):
        if session_id:
            session_manager = get_session_manager()
            active_sessions = session_manager.get_active_sessions()
            if session_id not in active_sessions:
                logger.warning(f"MPE session {session_id} expired")
                return n_intervals
        return n_intervals

if __name__ == '__main__':
    logger.info("=== STARTING MPE DASHBOARD APPLICATION ===")
    logger.info("Running in standalone mode")
    try:
        app.run(debug=True, host='0.0.0.0', port=8057)
    except Exception as e:
        logger.error(f"❌ Error starting application: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise
else:
    logger.info("=== MPE DASHBOARD LOADED AS MODULE ===")
    logger.info("Ready to serve requests")
