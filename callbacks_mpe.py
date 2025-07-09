from dash import Input, Output, State, callback, no_update
import base64, json
from mpe_solver import load_model_from_bytes, calculate_joint_probability

# Use a global variable to cache the current model for quick access in callbacks
current_model = None

def register_callbacks(app):
    """
    Register all Dash callbacks for the MPE app. This function is called from dash_mpe.py.
    """
    @app.callback(
        Output("stored-network", "data"),
        Output("evidence-variable", "options"),
        Output("evidence-store", "data"),
        Output("cytoscape-network", "elements"),
        Input("upload-network", "contents"),
        State("upload-network", "filename"),
        prevent_initial_call=True
    )
    def update_network(upload_contents, upload_filename):
        """
        Callback to handle a new network file upload. Parses the uploaded file and updates the stored network data,
        evidence dropdown options, clears any existing evidence, and updates the network graph visualization.
        """
        if upload_contents is None:
            return no_update, no_update, no_update, no_update
        content_type, content_string = upload_contents.split(",")
        file_bytes = base64.b64decode(content_string)
        filename = upload_filename or "uploaded_model"
        # Parse the uploaded network file to build the model
        try:
            model = load_model_from_bytes(file_bytes, filename)
        except Exception as e:
            # If parsing fails, do not update (in a real app, one might show an error message)
            print(f"Error loading model: {e}")
            return no_update, no_update, no_update, no_update
        # Update global current_model cache
        global current_model
        current_model = model
        # Prepare evidence variable options (list of variables in the model)
        variables = list(model.nodes())
        variables.sort()  # sort alphabetically for convenience
        var_options = [{"label": var, "value": var} for var in variables]
        # Prepare network graph elements for Cytoscape (nodes and edges)
        elements = []
        for var in model.nodes():
            elements.append({
                "data": {"id": var, "label": var, "evidence": "False"}
            })
        for u, v in model.edges():
            elements.append({
                "data": {"source": u, "target": v}
            })
        # Create data to store: include file content and state names if available
        store_data = {
            "filename": filename,
            "content": content_string
            # We do not store state_names here to avoid recursion; we will use current_model for state info
        }
        # Clear any existing evidence selections
        empty_evidence = []
        return store_data, var_options, empty_evidence, elements

    @app.callback(
        Output("evidence-value", "options"),
        Output("evidence-value", "disabled"),
        Input("evidence-variable", "value")
    )
    def update_evidence_value_options(selected_var):
        """
        When the user selects an evidence variable, populate the possible value options for that variable.
        Enables the evidence value dropdown once options are available.
        """
        if selected_var is None or selected_var == "":
            # No variable selected, disable the value dropdown
            return [], True
        # Ensure a model is loaded (should be, if variable list is populated)
        global current_model
        if current_model is None or selected_var not in current_model.nodes():
            return [], True
        # Get state names for the selected variable, if available
        cpd = current_model.get_cpds(selected_var)
        options = []
        if cpd:
            # If state names are defined in the CPD, use them; otherwise use index numbers
            if hasattr(cpd, "state_names") and cpd.state_names and selected_var in cpd.state_names:
                state_names = cpd.state_names[selected_var]
                options = [{"label": str(name), "value": str(name)} for name in state_names]
            else:
                # Fallback: use numeric indices as string labels
                card = current_model.get_cardinality(selected_var)
                options = [{"label": str(i), "value": str(i)} for i in range(card)]
        else:
            # If no CPD (should not happen in a valid model), disable selection
            return [], True
        return options, False

    @app.callback(
        Output("evidence-store", "data"),
        Input("add-evidence", "n_clicks"),
        State("evidence-variable", "value"),
        State("evidence-value", "value"),
        State("evidence-store", "data"),
        prevent_initial_call=True
    )
    def add_evidence(n_clicks, var, val, evidence_list):
        """
        Add the selected evidence (variable and its value) to the evidence list.
        Updates the evidence-store which holds the list of evidence dicts.
        """
        if n_clicks is None or n_clicks == 0:
            return evidence_list  # no change
        # Ensure both variable and value are selected
        if not var or val is None:
            return evidence_list
        # Avoid duplicate evidence entries for the same variable (update existing or ignore)
        evidence_list = evidence_list or []
        for ev in evidence_list:
            if ev["var"] == var:
                ev["value"] = val
                break
        else:
            evidence_list.append({"var": var, "value": val})
        return evidence_list

    @app.callback(
        Output("evidence-store", "data", allow_duplicate=True),
        Input("clear-evidence", "n_clicks"),
        prevent_initial_call=True
    )
    def clear_evidence(n_clicks):
        """
        Clear all evidence selections.
        """
        if n_clicks is None or n_clicks == 0:
            return no_update
        # Return an empty list to reset evidence-store
        return []

    @app.callback(
        Output("evidence-list", "children"),
        Input("evidence-store", "data"),
        prevent_initial_call=True
    )
    def display_evidence_list(evidence_list):
        """
        Display the current evidence selections in a human-readable format.
        """
        if evidence_list is None or len(evidence_list) == 0:
            return html.Span("No evidence selected.", className="text-muted")
        # Create a list of "Var = Value" strings
        items = []
        for ev in evidence_list:
            items.append(f"{ev['var']} = {ev['value']}")
        return html.Div([
            html.Strong("Evidence: "),
            html.Span("; ".join(items))
        ])

    @app.callback(
        Output("mpe-output", "children"),
        Output("cytoscape-network", "elements"),
        Input("run-button", "n_clicks"),
        State("algorithm-select", "value"),
        State("evidence-store", "data"),
        prevent_initial_call=True
    )
    def run_mpe(n_clicks, algorithm, evidence_list):
        """
        Perform the MPE computation when the "Run MPE" button is clicked.
        Outputs the most probable full assignment and its joint probability, 
        and updates the network graph with node labels reflecting the MPE result.
        """
        if n_clicks is None or n_clicks == 0:
            return no_update, no_update
        global current_model
        if current_model is None:
            return html.Div("No model loaded.", className="text-danger"), no_update
        # Prepare evidence for inference: convert evidence values to state indices
        evidence_dict = {}
        if evidence_list:
            for ev in evidence_list:
                var = ev["var"]
                val = ev["value"]
                # Determine the state index corresponding to the evidence value
                cpd = current_model.get_cpds(var)
                if cpd and hasattr(cpd, "state_names") and cpd.state_names and var in cpd.state_names:
                    # If state names are defined, find the index of the matching name
                    state_names = cpd.state_names[var]
                    try:
                        state_idx = state_names.index(val)
                    except ValueError:
                        # If value not found (should not happen if data is consistent), skip
                        state_idx = None
                else:
                    # If no named states, assume the value is numeric (as string)
                    try:
                        state_idx = int(val)
                    except:
                        state_idx = None
                if state_idx is not None:
                    evidence_dict[var] = state_idx
        # Select inference algorithm (currently only exact Variable Elimination)
        if algorithm == "exact":
            infer = VariableElimination(current_model)
        else:
            infer = VariableElimination(current_model)
        # Determine variables to maximize over (exclude evidence variables)
        all_vars = set(current_model.nodes())
        evidence_vars = set(evidence_dict.keys())
        query_vars = list(all_vars - evidence_vars)
        # Compute MPE assignment: most likely values for query_vars given evidence
        if len(query_vars) > 0:
            mpe_assignment = infer.map_query(variables=query_vars, evidence=evidence_dict)
        else:
            mpe_assignment = {}
        # Combine with evidence assignment
        full_assignment = mpe_assignment.copy()
        for var, state_idx in evidence_dict.items():
            full_assignment[var] = state_idx
        # Compute joint probability of the full assignment
        joint_prob = calculate_joint_probability(current_model, full_assignment)
        # Format the MPE assignment for display (convert indices to state names if possible)
        assignment_items = []
        for var, state_idx in sorted(full_assignment.items()):
            # Convert state index to name if available
            label = str(state_idx)
            cpd = current_model.get_cpds(var)
            if cpd and hasattr(cpd, "state_names") and cpd.state_names and var in cpd.state_names:
                try:
                    label = str(cpd.state_names[var][state_idx])
                except Exception:
                    label = str(state_idx)
            assignment_items.append(f"{var} = {label}")
        assignment_text = ",  ".join(assignment_items)
        # Format the joint probability with a reasonable precision
        prob_text = f"{joint_prob:.6g}"
        result_text = [
            html.P(f"MPE Assignment: {assignment_text}"),
            html.P(f"Joint Probability: {prob_text}")
        ]
        # Update cytoscape node labels to include MPE values
        new_elements = []
        for var in current_model.nodes():
            # Determine label with assignment value
            state_idx = full_assignment.get(var)
            label = str(state_idx)
            cpd = current_model.get_cpds(var)
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
        for u, v in current_model.edges():
            new_elements.append({
                "data": {"source": u, "target": v}
            })
        return result_text, new_elements
