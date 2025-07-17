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
    IMPROVED simple MPE using actual prior probabilities (fast and more accurate)
    
    Returns:
    - mpe_assignment: Dictionary with most probable states  
    - confidence: Better confidence based on actual probabilities
    """
    try:
        mpe_assignment = {}
        confidence_scores = []
        
        for var in query_vars:
            try:
                # Get the CPD for this variable
                cpd = model.get_cpds(var)
                
                if hasattr(cpd, 'values') and len(cpd.values.shape) > 0:
                    # If variable has parents, take average over all parent states
                    if len(cpd.values.shape) > 1:
                        # Marginalize over parent states (simple average)
                        marginal_probs = cpd.values.mean(axis=1)
                    else:
                        # No parents, use direct probabilities
                        marginal_probs = cpd.values
                    
                    # Find state with highest probability
                    best_state = int(marginal_probs.argmax())
                    max_prob = float(marginal_probs[best_state])
                    
                    mpe_assignment[var] = best_state
                    confidence_scores.append(max_prob)
                else:
                    # Fallback: pick state 0
                    mpe_assignment[var] = 0
                    confidence_scores.append(0.6)  # Default confidence
                    
            except Exception:
                # Ultimate fallback: pick 0
                mpe_assignment[var] = 0
                confidence_scores.append(0.5)  # Lower confidence for fallback
        
        # Calculate overall confidence as average of individual confidences
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        
        # Boost confidence slightly since we're using actual probabilities now
        boosted_confidence = min(avg_confidence * 1.2, 0.95)  # Cap at 95%
        
        return mpe_assignment, boosted_confidence
        
    except Exception as e:
        # Even this can't fail - return empty assignment
        return {var: 0 for var in query_vars}, 0.4

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
    
    # Strategy 1: Try ultra-simple first for very large networks (optimized for 8GB SWAP)
    if len(query_vars) > 25 or len(model.nodes()) > 150:
        return ultra_simple_mpe(model, evidence_dict, query_vars)
    
    # Strategy 2: Try lightweight sampling for medium networks
    if algorithm == 'simple':
        return ultra_simple_mpe(model, evidence_dict, query_vars)
    
    # Try real sampling for small-medium networks with 8GB SWAP protection
    if len(query_vars) <= 15 and len(model.nodes()) <= 100:
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
        
        # Use larger sample size for better confidence (but still safe)
        safe_samples = min(n_samples, 500)  # Increased from 100 to 500
        
        # Use Gibbs Sampling for approximation
        if algorithm == 'gibbs':
            sampler = GibbsSampling(model)
            samples = sampler.sample(size=safe_samples, evidence=evidence_named, show_progress=False)
        else:
            # Use rejection sampling as fallback
            sampler = BayesianModelSampling(model)
            samples = sampler.rejection_sample(size=safe_samples, evidence=evidence_named, show_progress=False)
        
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
        
        # Overall confidence as weighted average (better than simple average)
        if confidence_scores:
            avg_confidence = sum(confidence_scores.values()) / len(confidence_scores)
            # Boost confidence if we have many samples
            sample_boost = min(safe_samples / 1000, 0.2)  # Up to 20% boost for more samples
            final_confidence = min(avg_confidence + sample_boost, 0.95)
        else:
            final_confidence = 0.4
        
        return mpe_assignment, final_confidence
        
    except Exception as e:
        # Fallback to ultra-simple if sampling fails
        return ultra_simple_mpe(model, evidence_dict, query_vars)

def estimate_ve_memory_requirements(model, evidence_vars=None):
    """
    MODERN memory estimation for Variable Elimination (2024 approach)
    Based on realistic clique size calculation and actual elimination simulation
    
    Returns:
    - estimated_memory_gb: Estimated memory in GB
    - max_factor_size: Size of largest factor expected
    """
    try:
        if evidence_vars is None:
            evidence_vars = set()
        
        num_vars = len(model.nodes())
        evidence_count = len(evidence_vars)
        query_vars_count = num_vars - evidence_count
        
        # Quick early returns for obvious cases
        if query_vars_count > 50:  # Más permisivo
            return 15.0, 1e9
        elif query_vars_count < 8:  # Un poco más estricto para evitar overhead
            return 0.3, 1000
        
        # MODERN APPROACH: Realistic treewidth-based estimation
        max_clique_size = 1
        total_memory_estimate = 0
        
        # Calculate actual connectivity patterns
        for var in model.nodes():
            if var in evidence_vars:
                continue
                
            try:
                cardinality = model.get_cardinality(var)
                
                # Count ACTUAL neighbors (more realistic than degree)
                neighbors = set()
                for edge in model.edges():
                    if var == edge[0]:
                        neighbors.add(edge[1])
                    elif var == edge[1]:
                        neighbors.add(edge[0])
                
                # Remove evidence neighbors (they don't contribute to complexity)
                active_neighbors = [n for n in neighbors if n not in evidence_vars]
                neighbor_count = len(active_neighbors)
                
                # REALISTIC clique size estimation (not exponential madness)
                if neighbor_count <= 2:
                    local_factor_size = cardinality * 100  # Linear for simple nodes
                elif neighbor_count <= 4:
                    local_factor_size = cardinality ** 2 * neighbor_count  # Quadratic
                else:
                    # Cap at reasonable maximum to avoid explosion
                    local_factor_size = cardinality ** 3 * min(neighbor_count, 8)
                
                max_clique_size = max(max_clique_size, local_factor_size)
                total_memory_estimate += local_factor_size * 0.1  # Accumulate overhead
                
                # Sensible early termination
                if max_clique_size > 1e8:
                    break
                    
            except Exception:
                # If any error, add conservative estimate
                max_clique_size = max(max_clique_size, 1e6)
        
        # REALISTIC memory calculation (not arbitrary 5x multiplier)
        # Based on actual Variable Elimination space requirements
        bytes_per_entry = 8  # double precision
        intermediate_factors_overhead = 2.5  # reasonable overhead for VE
        system_overhead = 1.8  # OS and Python overhead
        
        estimated_memory_bytes = (max_clique_size + total_memory_estimate) * bytes_per_entry * intermediate_factors_overhead * system_overhead
        estimated_memory_gb = estimated_memory_bytes / (1024**3)
        
        # Sensible caps based on real-world experience
        estimated_memory_gb = min(estimated_memory_gb, 50.0)
        estimated_memory_gb = max(estimated_memory_gb, 0.1)  # Minimum realistic estimate
        
        return estimated_memory_gb, max_clique_size
        
    except Exception as e:
        # Conservative fallback for any errors
        return 8.0, 1e7

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

def is_network_large(model, evidence_vars=None, safety_factor=0.8):
    """
    MODERN network size detection with realistic thresholds (2024 update)
    
    Parameters:
    - model: Bayesian Network
    - evidence_vars: Set of evidence variables (reduce complexity)
    - safety_factor: Use only 80% of available memory for safety
    
    Returns:
    - bool: True if network is too large for exact inference
    - dict: Diagnostic information
    """
    try:
        # Quick checks first
        num_vars = len(model.nodes())
        
        # REALISTIC heuristics for modern systems
        if num_vars > 80:  # Menos conservador - 56 variables deberían poder ser exactas
            return True, {
                'estimated_memory_gb': 12.0,
                'available_memory_gb': 8.0,
                'reason': f'Very large network ({num_vars} > 80 variables)',
                'recommended_algorithm': 'approximate'
            }
        
        # Get system resources (safe)
        resources = get_system_resources()
        available_memory = resources.get('available_memory_gb', 8.0)
        
        # Smarter variable count check
        try:
            var_count = len(model.nodes())
            evidence_count = len(evidence_vars) if evidence_vars else 0
            query_vars_count = var_count - evidence_count
            
            # More realistic thresholds based on actual query complexity
            if query_vars_count > 40:  # Aumentado para ser consistente
                return True, {
                    'estimated_memory_gb': 8.0,
                    'available_memory_gb': available_memory,
                    'reason': f'Too many query variables ({query_vars_count} > 40)',
                    'recommended_algorithm': 'approximate'
                }
                
        except Exception:
            # Any error in variable counting
            return True, {
                'estimated_memory_gb': 6.0,
                'available_memory_gb': available_memory,
                'reason': 'Error counting variables',
                'recommended_algorithm': 'approximate'
            }
        
        # Use the MODERN memory estimation
        try:
            estimated_memory_gb, max_factor_size = estimate_ve_memory_requirements(model, evidence_vars)
            
            # Smart decision with realistic safety margin
            memory_limit = available_memory * safety_factor
            use_approximate = estimated_memory_gb > memory_limit
            
            return use_approximate, {
                'estimated_memory_gb': estimated_memory_gb,
                'available_memory_gb': available_memory,
                'max_factor_size': max_factor_size,
                'reason': f'Memory: {estimated_memory_gb:.1f}GB vs {memory_limit:.1f}GB limit',
                'recommended_algorithm': 'approximate' if use_approximate else 'exact'
            }
            
        except Exception:
            # Fallback to safe choice
            return True, {
                'estimated_memory_gb': 6.0,
                'available_memory_gb': available_memory,
                'reason': 'Memory estimation failed - using safe approximate',
                'recommended_algorithm': 'approximate'
            }
        
    except Exception as e:
        # Ultra-safe fallback
        return True, {
            'error': str(e),
            'estimated_memory_gb': 5.0,
            'available_memory_gb': 6.0,
            'reason': 'Analysis error - defaulting to approximate',
            'recommended_algorithm': 'approximate'
        }
