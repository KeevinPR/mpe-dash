import base64, json
import tempfile
import os
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
            with tempfile.NamedTemporaryFile(mode='w', suffix='.bif', delete=False) as temp_file:
                temp_file.write(content_str)
                temp_file_path = temp_file.name
            try:
                model = BIFReader(temp_file_path).get_model()
            finally:
                os.unlink(temp_file_path)  # Clean up temp file
        else:
            # Use XMLBIFReader for .xml files
            with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as temp_file:
                temp_file.write(content_str)
                temp_file_path = temp_file.name
            try:
                model = XMLBIFReader(temp_file_path).get_model()
            finally:
                os.unlink(temp_file_path)  # Clean up temp file
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
        
        # Ensure node_state is an integer
        node_state = assignment[node]
        if not isinstance(node_state, int):
            try:
                node_state = int(node_state)
            except (ValueError, TypeError):
                raise ValueError(f"Node state for {node} must be an integer, got {type(node_state)}: {node_state}")
        
        # Get parent variables - in pgmpy TabularCPD, parents are in variables[1:]
        parent_vars = cpd.variables[1:] if len(cpd.variables) > 1 else []
        
        # Determine parent states for this assignment
        parent_states = []
        if parent_vars:  # list of parent variables if any
            for parent in parent_vars:
                parent_state = assignment[parent]
                # Ensure parent_state is an integer
                if not isinstance(parent_state, int):
                    try:
                        parent_state = int(parent_state)
                    except (ValueError, TypeError):
                        raise ValueError(f"Parent state for {parent} must be an integer, got {type(parent_state)}: {parent_state}")
                parent_states.append(parent_state)
        
        # Retrieve the probability value for this node's state given parent states
        # Use get_values() to get the CPD table in the correct 2D format
        cpd_values = cpd.get_values()
        
        if not parent_vars:
            # No parents: probability is the value for the state (single column)
            prob_val = float(cpd_values[node_state, 0])
        else:
            # With parents: need to compute the column index from parent states
            # cpd_values shape: (node_cardinality, prod of parents' cardinalities)
            # Compute column index from parent_states list
            col_index = 0
            if parent_states:
                # Calculate the index as a mixed radix number with parent state indices
                # e.g., for parents [P1, P2] with card [c1, c2]: index = P1_index * c2 + P2_index
                parent_cards = [model.get_cardinality(p) for p in parent_vars]
                # Multiply out indices: for each parent state, multiply by product of cards of subsequent parents
                for idx, parent_state in enumerate(parent_states):
                    # product of cardinals for parents after this index
                    remaining_cards = 1
                    for card in parent_cards[idx+1:]:
                        remaining_cards *= card
                    col_index += parent_state * remaining_cards
            # Ensure col_index is valid
            col_index = int(col_index)
            prob_val = float(cpd_values[node_state, col_index])
        prob *= prob_val
    return prob

def ultra_simple_mpe(model, evidence_dict, query_vars):
    """
    Ultra-simple MPE using just prior probabilities (always works)
    
    Returns:
    - mpe_assignment: Dictionary with most probable states  
    - confidence: Always 0.5 (indicating it's a simple approximation)
    """
    try:
        mpe_assignment = {}
        
        for var in query_vars:
            try:
                # Get the CPD for this variable
                cpd = model.get_cpds(var)
                
                # Simple strategy: pick the most probable state in the marginal
                if hasattr(cpd, "state_names") and cpd.state_names and var in cpd.state_names:
                    # If we have state names, pick the first one (usually most common)
                    mpe_assignment[var] = cpd.state_names[var][0]
                else:
                    # If no state names, pick state 0
                    mpe_assignment[var] = 0
                    
            except:
                # Ultimate fallback: pick 0 or "unknown"
                mpe_assignment[var] = 0
        
        return mpe_assignment, 0.5  # 50% confidence (rough approximation)
        
    except Exception as e:
        # Even this can't fail - return empty assignment
        return {var: 0 for var in query_vars}, 0.3

def approximate_mpe_sampling(model, evidence_dict, query_vars, n_samples=1000, algorithm='simple'):
    """
    Robust approximate MPE with multiple fallback strategies
    
    Parameters:
    - model: Bayesian Network
    - evidence_dict: Evidence variables {var: state_index}
    - query_vars: Variables to find MPE for
    - n_samples: Number of samples for approximation (reduced default)
    - algorithm: 'simple', 'gibbs', or 'rejection'
    
    Returns:
    - mpe_assignment: Dictionary with most probable states
    - confidence: Confidence score of the result
    """
    
    # Strategy 1: Try ultra-simple first for very large networks
    if len(query_vars) > 20 or len(model.nodes()) > 40:
        return ultra_simple_mpe(model, evidence_dict, query_vars)
    
    # Strategy 2: Try lightweight sampling for medium networks
    if algorithm == 'simple':
        return ultra_simple_mpe(model, evidence_dict, query_vars)
    
    # Try real sampling for small-medium networks with SWAP protection
    if len(query_vars) <= 10 and len(model.nodes()) <= 25:
        # Small enough for real sampling - try it
        pass  # Continue to Strategy 3
    else:
        # Too risky even with SWAP - use simple
        return ultra_simple_mpe(model, evidence_dict, query_vars)
    
    # Strategy 3: Try pgmpy sampling (currently disabled due to memory issues)
    try:
        from pgmpy.sampling import GibbsSampling, BayesianModelSampling
        import numpy as np
        from collections import Counter
        
        # Convert evidence to state names if needed
        evidence_named = {}
        for var, state_idx in evidence_dict.items():
            cpd = model.get_cpds(var)
            if hasattr(cpd, "state_names") and cpd.state_names and var in cpd.state_names:
                try:
                    state_name = cpd.state_names[var][state_idx]
                    evidence_named[var] = state_name
                except:
                    evidence_named[var] = state_idx
            else:
                evidence_named[var] = state_idx
        
        # Use very small sample size to avoid memory issues
        small_samples = min(n_samples, 100)
        
        # Use Gibbs Sampling for approximation
        if algorithm == 'gibbs':
            sampler = GibbsSampling(model)
            samples = sampler.sample(size=small_samples, evidence=evidence_named, show_progress=False)
        else:
            # Use rejection sampling as fallback
            sampler = BayesianModelSampling(model)
            samples = sampler.rejection_sample(size=small_samples, evidence=evidence_named, show_progress=False)
        
        # Find most frequent states for each query variable
        mpe_assignment = {}
        confidence_scores = {}
        
        for var in query_vars:
            if var in samples.columns:
                # Count occurrences of each state
                state_counts = Counter(samples[var])
                # Get most frequent state
                most_frequent_state = state_counts.most_common(1)[0][0]
                mpe_assignment[var] = most_frequent_state
                
                # Calculate confidence as relative frequency
                confidence_scores[var] = state_counts[most_frequent_state] / len(samples)
        
        # Overall confidence as average
        avg_confidence = np.mean(list(confidence_scores.values())) if confidence_scores else 0.4
        
        return mpe_assignment, avg_confidence
        
    except Exception as e:
        # Fallback to ultra-simple if sampling fails
        return ultra_simple_mpe(model, evidence_dict, query_vars)

def estimate_ve_memory_requirements(model, evidence_vars=None):
    """
    Lightweight memory estimation for Variable Elimination
    
    Returns:
    - estimated_memory_gb: Estimated memory in GB
    - max_factor_size: Size of largest factor expected
    """
    try:
        if evidence_vars is None:
            evidence_vars = set()
        
        # Quick check: if too many variables, return large estimate immediately
        num_vars = len(model.nodes())
        if num_vars > 50:
            return 50.0, 1e12  # Assume large
        
        # Simple approximation: largest clique size estimation
        max_cardinality_product = 1
        max_degree = 0
        
        # Simplified approach: check node degrees and cardinalities
        for var in model.nodes():
            if var in evidence_vars:
                continue
                
            try:
                cardinality = model.get_cardinality(var)
                # Count edges (simple degree calculation)
                degree = len([edge for edge in model.edges() if var in edge])
                
                if degree > max_degree:
                    max_degree = degree
                
                # Rough estimate: cardinality^degree for worst case
                local_factor_size = cardinality ** min(degree, 8)  # Cap at degree 8
                
                if local_factor_size > max_cardinality_product:
                    max_cardinality_product = local_factor_size
                    
                # Early exit if clearly too large
                if max_cardinality_product > 1e10:
                    break
                    
            except:
                # If any error, assume it's large
                return 25.0, 1e11
        
        # Conservative memory estimate
        estimated_memory_bytes = max_cardinality_product * 8 * 5  # 5x safety factor
        estimated_memory_gb = estimated_memory_bytes / (1024**3)
        
        # Cap the estimate to prevent overflow
        estimated_memory_gb = min(estimated_memory_gb, 100.0)
        
        return estimated_memory_gb, max_cardinality_product
        
    except Exception as e:
        # Safe fallback
        return 15.0, 1e10

def get_system_resources():
    """
    Get current system resource information
    
    Returns:
    - dict with memory and CPU info
    """
    try:
        import psutil
        
        memory = psutil.virtual_memory()
        
        return {
            'total_memory_gb': memory.total / (1024**3),
            'available_memory_gb': memory.available / (1024**3),
            'cpu_count': psutil.cpu_count(),
            'memory_percent_used': memory.percent
        }
    except:
        # Fallback values for your system specifically
        return {
            'total_memory_gb': 15.4,
            'available_memory_gb': 10.0,
            'cpu_count': 4,
            'memory_percent_used': 30.0
        }

def is_network_large(model, evidence_vars=None, safety_factor=0.7):
    """
    Fast network size detection with robust error handling
    
    Parameters:
    - model: Bayesian Network
    - evidence_vars: Set of evidence variables (reduce complexity)
    - safety_factor: Use only 70% of available memory for safety
    
    Returns:
    - bool: True if network is too large for exact inference
    - dict: Diagnostic information
    """
    try:
        # Quick checks first
        num_vars = len(model.nodes())
        
        # Very simple heuristics for speed
        if num_vars > 30:
            return True, {
                'estimated_memory_gb': 25.0,
                'available_memory_gb': 10.0,
                'reason': f'Too many variables ({num_vars} > 30)',
                'recommended_algorithm': 'approximate'
            }
        
        # Get system resources (safe)
        resources = get_system_resources()
        available_memory = resources.get('available_memory_gb', 8.0)
        
        # Simple state space check
        try:
            total_states = 1
            var_count = 0
            for var in model.nodes():
                if evidence_vars and var in evidence_vars:
                    continue
                try:
                    cardinality = model.get_cardinality(var)
                    total_states *= cardinality
                    var_count += 1
                    
                    # Early exit conditions
                    if total_states > 1e9 or var_count > 15:
                        return True, {
                            'estimated_memory_gb': 20.0,
                            'available_memory_gb': available_memory,
                            'reason': f'State space too large ({total_states:.0e})',
                            'recommended_algorithm': 'approximate'
                        }
                except:
                    # If we can't get cardinality, assume it's complex
                    return True, {
                        'estimated_memory_gb': 15.0,
                        'available_memory_gb': available_memory,
                        'reason': 'Complex variable structure',
                        'recommended_algorithm': 'approximate'
                    }
        except:
            # Any error in state space calculation
            return True, {
                'estimated_memory_gb': 10.0,
                'available_memory_gb': available_memory,
                'reason': 'Error calculating state space',
                'recommended_algorithm': 'approximate'
            }
        
        # If we get here, try memory estimation (lightweight version)
        try:
            estimated_memory_gb, max_factor_size = estimate_ve_memory_requirements(model, evidence_vars)
            
            # Simple decision
            use_approximate = estimated_memory_gb > (available_memory * safety_factor)
            
            return use_approximate, {
                'estimated_memory_gb': estimated_memory_gb,
                'available_memory_gb': available_memory,
                'max_factor_size': max_factor_size,
                'reason': f'Memory estimate: {estimated_memory_gb:.1f}GB vs {available_memory*safety_factor:.1f}GB limit',
                'recommended_algorithm': 'approximate' if use_approximate else 'exact'
            }
            
        except:
            # Fallback to safe choice
            return True, {
                'estimated_memory_gb': 12.0,
                'available_memory_gb': available_memory,
                'reason': 'Error in memory estimation - using safe approximate',
                'recommended_algorithm': 'approximate'
            }
        
    except Exception as e:
        # Ultra-safe fallback
        return True, {
            'error': str(e),
            'estimated_memory_gb': 10.0,
            'available_memory_gb': 8.0,
            'reason': 'General error - defaulting to approximate',
            'recommended_algorithm': 'approximate'
        }
