import base64, json
from io import BytesIO
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.readwrite import BIFReader, XMLBIFReader

def load_model_from_bytes(content_bytes: bytes, filename: str) -> BayesianNetwork:
    """
    Load a Bayesian network model from file content bytes, given the filename to infer format.
    Supports .bif (Bayesian Interchange Format), .xml (XMLBIF), or .json (custom JSON structure).
    Returns a pgmpy BayesianNetwork object.
    """
    fname = filename.lower()
    if fname.endswith(".bif") or fname.endswith(".bifxml") or fname.endswith(".xml"):
        # If it's .bif or .xml, use pgmpy readers. 
        # Note: .bifxml is an XML variant sometimes used; treat it as xml.
        # Decode bytes to string and use appropriate reader.
        content_str = content_bytes.decode("utf-8")
        if fname.endswith(".bif"):
            # Use BIFReader for .bif files
            with open("temp_network.bif", "w") as tempf:
                tempf.write(content_str)
            model = BIFReader("temp_network.bif").get_model()
        else:
            # Use XMLBIFReader for .xml files
            with open("temp_network.xml", "w") as tempf:
                tempf.write(content_str)
            model = XMLBIFReader("temp_network.xml").get_model()
        return model
    elif fname.endswith(".json"):
        # Load from a JSON representation of the network
        content_str = content_bytes.decode("utf-8")
        data = json.loads(content_str)
        # Build BayesianNetwork from JSON spec
        model = BayesianNetwork()
        # Add nodes and edges
        if "edges" in data:
            model.add_edges_from(data["edges"])
        if "nodes" in data:
            # Ensure all listed nodes are added (especially if isolated or no parents/children)
            for node_info in data["nodes"]:
                if isinstance(node_info, dict):
                    node_name = node_info.get("name")
                else:
                    node_name = node_info
                if node_name not in model.nodes():
                    model.add_node(node_name)
        # Add CPDs
        if "cpds" in data:
            for cpd_info in data["cpds"]:
                var = cpd_info["variable"]
                # Determine variable's state count
                if "states" in cpd_info:
                    var_card = len(cpd_info["states"])
                else:
                    # If state names not explicitly given, infer from provided CPT
                    # (number of rows in values = variable cardinality)
                    var_card = len(cpd_info["values"])
                parents = cpd_info.get("evidence", []) or cpd_info.get("parents", [])
                # Prepare values for TabularCPD
                values = cpd_info["values"]
                # Ensure values is a 2D list for TabularCPD
                if not any(isinstance(row, list) for row in values):
                    # If values is a flat list (for a single column), convert to list of lists
                    values = [[val] for val in values]
                # Determine parent cardinalities
                evidence_cards = []
                for parent in parents:
                    # Find parent info in nodes list to get number of states
                    parent_card = None
                    if "nodes" in data:
                        for node_info in data["nodes"]:
                            if isinstance(node_info, dict) and node_info.get("name") == parent:
                                if "states" in node_info:
                                    parent_card = len(node_info["states"])
                                break
                    # If not found in nodes or states not given, try to infer from CPD values length
                    evidence_cards.append(parent_card)
                if parents:
                    # Infer any missing parent cardinalities by dividing total columns
                    total_cols = len(values[0])
                    known_prod = 1
                    unknown_count = evidence_cards.count(None)
                    for card in evidence_cards:
                        if card is not None:
                            known_prod *= card
                    if unknown_count > 0 and known_prod > 0:
                        inferred_val = int(total_cols // known_prod) if unknown_count == 1 else None
                        evidence_cards = [
                            inferred_val if c is None else c for c in evidence_cards
                        ]
                # Create TabularCPD
                if not parents:
                    cpd = TabularCPD(variable=var, variable_card=var_card, values=values)
                else:
                    cpd = TabularCPD(variable=var, variable_card=var_card, 
                                     values=values, evidence=parents, evidence_card=evidence_cards)
                model.add_cpds(cpd)
        # Validate the model structure and parameters
        model.check_model()
        return model
    else:
        raise ValueError(f"Unsupported file format: {filename}")

def calculate_joint_probability(model: BayesianNetwork, assignment: dict) -> float:
    """
    Calculate the joint probability of a full variable assignment in the given model.
    `assignment` should be a dict of {variable: state_index}, including all variables in the model.
    Assumes the model's CPDs are fully specified.
    """
    prob = 1.0
    for node in model.nodes():
        cpd = model.get_cpds(node)
        if cpd is None:
            # If a CPD is missing, skip (or one could assume uniform)
            continue
        node_state = assignment[node]
        # Determine parent states for this assignment
        parent_states = []
        if cpd.evidence:  # list of parent variables if any
            for parent in cpd.evidence:
                parent_state = assignment[parent]
                parent_states.append(parent_state)
        # Retrieve the probability value for this node's state given parent states
        # TabularCPD stores values in an array accessible via state indices.
        if not cpd.evidence:
            # No parents: probability is just the single-column value for the state
            prob_val = cpd.values[node_state]
        else:
            # With parents: need to index into cpd.values using parent states
            # cpd.values shape: (node_cardinality x (prod of parents' cardinalities))
            # Compute column index from parent_states list (assuming order matches cpd.evidence order)
            col_index = 0
            if parent_states:
                # Calculate the index as a mixed radix number with parent state indices
                # e.g., for evidence [P1, P2] with card [c1, c2]: index = P1_index * c2 + P2_index
                # Generalized for multiple parents:
                parent_cards = [model.get_cardinality(p) for p in cpd.evidence]
                # Multiply out indices: for each parent state, multiply by product of cards of subsequent parents
                for idx, parent_state in enumerate(parent_states):
                    # product of cardinals for parents after this index
                    remaining_cards = 1
                    for card in parent_cards[idx+1:]:
                        remaining_cards *= card
                    col_index += parent_state * remaining_cards
            prob_val = cpd.values[node_state][col_index]
        prob *= prob_val
    return prob
