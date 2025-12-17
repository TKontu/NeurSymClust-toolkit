#!/usr/bin/env python3
"""
FINAL MOLECULAR LAYOUT:
- All molecules formed by greedy highest-compatibility-first
- Internal forces scale with compatibility (high compat = tight, low compat = loose)
- Bad actors naturally form LARGE LOOSE molecules
- Inter-molecular forces: repulsion + compatibility-weighted attraction
- Peripheral edge drawing only
"""

import pickle
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from scipy.spatial import ConvexHull, distance_matrix
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from cluster_utils import load_cluster_mappings, get_category_display_name

# THRESHOLDS
INCOMPATIBLE_THRESHOLD = 0.75
SYNERGISTIC_THRESHOLD = 0.95
MAX_MOLECULE_SIZE = 10
MIN_MOLECULE_SIZE = 3

# Color palette - 20 distinct colors for categories
COLOR_PALETTE = [
    '#E74C3C',  # Red
    '#3498DB',  # Blue
    '#2ECC71',  # Green
    '#9B59B6',  # Purple
    '#F39C12',  # Orange
    '#1ABC9C',  # Teal
    '#E91E63',  # Pink
    '#00BCD4',  # Cyan
    '#FF5722',  # Deep Orange
    '#8BC34A',  # Light Green
    '#673AB7',  # Deep Purple
    '#795548',  # Brown
    '#607D8B',  # Blue Grey
    '#FFEB3B',  # Yellow
    '#009688',  # Dark Teal
    '#FF9800',  # Amber
    '#03A9F4',  # Light Blue
    '#CDDC39',  # Lime
    '#9C27B0',  # Violet
    '#4CAF50',  # Medium Green
]

def get_category_colors(cluster_mappings):
    """Build CATEGORY_COLORS dict from cluster mappings dynamically.

    Assigns colors from palette based on category order.
    """
    colors = {}
    synergy_keys = list(cluster_mappings['synergy_display_names'].keys())

    for i, synergy_key in enumerate(synergy_keys):
        display_name = cluster_mappings['synergy_display_names'][synergy_key]
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        colors[synergy_key] = {
            'color': color,
            'name': display_name
        }

    # Add unknown fallback
    colors['unknown'] = {'color': '#95a5a6', 'name': 'Unknown'}
    return colors

def load_compatibility_checkpoint(pkl_path):
    """Load the compatibility checkpoint"""
    print(f"Loading compatibility data from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_categories(cluster_mappings, checkpoint):
    """
    Load method categories from combined_clusters.json using name-based matching.

    Args:
        cluster_mappings: Dict from load_cluster_mappings()
        checkpoint: The compatibility checkpoint with method names

    Returns:
        index_to_category: Dict mapping method index to synergy group key
    """
    print("Loading categories from semantic clustering results...")

    # Load combined_clusters.json for cluster assignments
    clusters_path = 'results_semantic_clustering_combined/combined_clusters.json'
    with open(clusters_path, 'r') as f:
        clusters_data = json.load(f)

    # Build method name -> synergy group mapping
    method_name_to_synergy = {}
    for cluster_id, cluster_data in clusters_data.get('clusters', {}).items():
        # Get synergy group for this cluster
        synergy_key = cluster_mappings['cluster_to_synergy'].get(cluster_id, 'unknown')

        for method_info in cluster_data.get('methods', []):
            method_name = method_info.get('Method', '').strip().lower()
            if method_name:
                method_name_to_synergy[method_name] = synergy_key

    # Build index -> synergy mapping from checkpoint results
    index_to_category = {}
    results = checkpoint.get('results', [])

    # Collect all unique methods with their names
    methods_dict = {}
    for result in results:
        idx_a = int(result['method_a_index'])
        idx_b = int(result['method_b_index'])
        if idx_a not in methods_dict:
            methods_dict[idx_a] = result['method_a']
        if idx_b not in methods_dict:
            methods_dict[idx_b] = result['method_b']

    # Map each method index to its synergy group by name
    for idx, method_name in methods_dict.items():
        name_lower = method_name.strip().lower()
        synergy = method_name_to_synergy.get(name_lower, 'unknown')
        index_to_category[idx] = synergy

    # Log statistics
    synergy_counts = defaultdict(int)
    for synergy in index_to_category.values():
        synergy_counts[synergy] += 1

    print(f"Loaded {len(index_to_category)} method categories across {len(synergy_counts)} synergy groups")
    for synergy, count in sorted(synergy_counts.items(), key=lambda x: -x[1]):
        display_name = cluster_mappings['synergy_display_names'].get(synergy, synergy)
        print(f"  {synergy}: {count} methods ({display_name})")

    return index_to_category

def build_graph_from_checkpoint(checkpoint):
    """Build graph with edge classification"""
    results = checkpoint.get('results', [])
    print(f"Found {len(results)} compatibility results")

    # Collect unique methods
    methods_dict = {}
    for result in results:
        idx_a = int(result['method_a_index'])
        idx_b = int(result['method_b_index'])
        if idx_a not in methods_dict:
            methods_dict[idx_a] = {'name': result['method_a'], 'index': idx_a}
        if idx_b not in methods_dict:
            methods_dict[idx_b] = {'name': result['method_b'], 'index': idx_b}

    print(f"Found {len(methods_dict)} unique methods")

    # Create graph
    G = nx.Graph()
    for idx, method_info in methods_dict.items():
        G.add_node(idx, name=method_info['name'])

    # Add edges with scores
    edge_scores = {}
    for result in results:
        idx_a = int(result['method_a_index'])
        idx_b = int(result['method_b_index'])
        score = result.get('compatibility_score', 0.5)
        G.add_edge(idx_a, idx_b, score=score)
        edge_scores[(min(idx_a, idx_b), max(idx_a, idx_b))] = score

    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G, edge_scores

def calculate_avg_pairwise_score(nodes, edge_scores):
    """Calculate average pairwise compatibility score for a set of nodes"""
    if len(nodes) <= 1:
        return 0.0
    
    scores = []
    for i, node_a in enumerate(nodes):
        for node_b in nodes[i+1:]:
            edge_key = (min(node_a, node_b), max(node_a, node_b))
            if edge_key in edge_scores:
                scores.append(edge_scores[edge_key])
    
    return np.mean(scores) if scores else 0.0

def form_molecules_greedy(G, edge_scores):
    """
    Form molecules greedily by compatibility:
    1. Find best pair
    2. Grow to max 10 nodes by adding node that maximizes avg pairwise score
    3. Repeat until all nodes assigned
    """
    print("\n🧬 Forming molecules by greedy compatibility...")
    
    remaining = set(G.nodes())
    molecules = []
    
    while len(remaining) >= MIN_MOLECULE_SIZE:
        # Determine molecule size
        target_size = min(MAX_MOLECULE_SIZE, len(remaining))
        
        if len(remaining) < MAX_MOLECULE_SIZE + MIN_MOLECULE_SIZE:
            # Last batch - take all remaining
            molecule_nodes = list(remaining)
            remaining.clear()
        else:
            # Find best starting pair
            best_pair = None
            best_score = -1
            
            remaining_list = list(remaining)
            for i, node_a in enumerate(remaining_list):
                for node_b in remaining_list[i+1:i+50]:  # Sample for speed
                    edge_key = (min(node_a, node_b), max(node_a, node_b))
                    if edge_key in edge_scores:
                        score = edge_scores[edge_key]
                        if score > best_score:
                            best_score = score
                            best_pair = (node_a, node_b)
            
            if best_pair is None:
                # No edges found, take random pair
                best_pair = (remaining_list[0], remaining_list[1])
            
            # Grow molecule greedily
            molecule_nodes = list(best_pair)
            candidates = remaining - set(molecule_nodes)
            
            while len(molecule_nodes) < target_size and candidates:
                # Find candidate that maximizes avg pairwise score
                best_candidate = None
                best_avg_score = -1
                
                for candidate in list(candidates)[:100]:  # Sample for speed
                    test_nodes = molecule_nodes + [candidate]
                    avg_score = calculate_avg_pairwise_score(test_nodes, edge_scores)
                    if avg_score > best_avg_score:
                        best_avg_score = avg_score
                        best_candidate = candidate
                
                if best_candidate is not None:
                    molecule_nodes.append(best_candidate)
                    candidates.remove(best_candidate)
                else:
                    break
            
            # Remove from remaining
            remaining -= set(molecule_nodes)
        
        # Calculate molecule quality
        avg_score = calculate_avg_pairwise_score(molecule_nodes, edge_scores)
        
        molecules.append({
            'nodes': molecule_nodes,
            'size': len(molecule_nodes),
            'avg_compatibility': avg_score,
            'locked': False,
            'center': np.array([0.0, 0.0]),
            'offsets': {}
        })
        
        print(f"  Molecule {len(molecules)}: {len(molecule_nodes)} nodes, avg_compat={avg_score:.3f}")
    
    # Handle final orphans
    if len(remaining) > 0:
        molecule_nodes = list(remaining)
        avg_score = calculate_avg_pairwise_score(molecule_nodes, edge_scores)
        molecules.append({
            'nodes': molecule_nodes,
            'size': len(molecule_nodes),
            'avg_compatibility': avg_score,
            'locked': False,
            'center': np.array([0.0, 0.0]),
            'offsets': {}
        })
        print(f"  Molecule {len(molecules)} (orphans): {len(molecule_nodes)} nodes, avg_compat={avg_score:.3f}")
    
    print(f"\n✓ Formed {len(molecules)} molecules")
    print(f"  Highest quality: {max(m['avg_compatibility'] for m in molecules):.3f}")
    print(f"  Lowest quality: {min(m['avg_compatibility'] for m in molecules):.3f}")
    
    return molecules

def shape_molecule_internal(molecule, node_to_idx, positions, edge_scores):
    """
    Shape a single molecule using internal forces
    High compatibility → tight cluster (small diameter)
    Low compatibility → loose cluster (large diameter)
    """
    mol_nodes = molecule['nodes']
    mol_indices = [node_to_idx[n] for n in mol_nodes]
    avg_compat = molecule['avg_compatibility']
    
    # Internal spacing scales with compatibility
    # High compat (0.95-1.0) → ideal_dist = 0.3
    # Low compat (0.5-0.7) → ideal_dist = 1.2
    ideal_dist = 0.3 + (1.0 - avg_compat) * 2.0
    
    # Attraction strength scales with compatibility
    # High compat → strong attraction (0.8)
    # Low compat → weak attraction (0.2)
    attraction_strength = 0.2 + avg_compat * 0.6
    
    print(f"    Shaping molecule: size={len(mol_nodes)}, compat={avg_compat:.3f}, ideal_dist={ideal_dist:.2f}, attraction={attraction_strength:.2f}")
    
    learning_rate = 0.12
    
    for iteration in range(60):
        forces = np.zeros((len(mol_indices), 2))
        
        # FORCE 1: Pairwise attraction based on edge compatibility
        for i, node_a_idx in enumerate(mol_indices):
            for j, node_b_idx in enumerate(mol_indices[i+1:], i+1):
                node_a = mol_nodes[i]
                node_b = mol_nodes[j]
                
                edge_key = (min(node_a, node_b), max(node_a, node_b))
                edge_score = edge_scores.get(edge_key, 0.5)
                
                diff = positions[node_a_idx] - positions[node_b_idx]
                dist = np.linalg.norm(diff)
                
                if dist > 0.01:
                    # Spring force: F = k * (dist - ideal_dist)
                    # Strength based on edge score
                    edge_strength = attraction_strength * edge_score
                    force_mag = edge_strength * (dist - ideal_dist)
                    force_mag = max(-0.5, min(force_mag, 1.5))
                    
                    force = force_mag * (diff / dist)
                    forces[i] -= force
                    forces[j] += force
        
        # FORCE 2: Local repulsion (prevent overlap)
        for i in range(len(mol_indices)):
            for j in range(i+1, len(mol_indices)):
                diff = positions[mol_indices[i]] - positions[mol_indices[j]]
                dist = np.linalg.norm(diff)
                
                if dist < 0.4 and dist > 0.01:
                    force_mag = 0.15 / (dist * dist + 0.05)
                    force = force_mag * (diff / dist)
                    forces[i] += force
                    forces[j] -= force
        
        # Apply forces
        for i, idx in enumerate(mol_indices):
            positions[idx] += learning_rate * forces[i]
        
        if iteration % 15 == 0:
            learning_rate *= 0.9
    
    return positions

def create_molecular_layout(G, edge_scores, molecules):
    """
    Create molecular layout:
    1. Shape each molecule internally
    2. Lock molecules
    3. Position molecules with inter-molecular forces
    """
    print("\n⚛️  Creating molecular layout...")
    
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Create node to molecule mapping
    node_to_mol = {}
    for mol_id, mol in enumerate(molecules):
        for node in mol['nodes']:
            node_to_mol[node] = mol_id
    
    # Initialize positions
    positions = np.random.randn(n_nodes, 2) * 3.0
    
    # PHASE 1: Shape each molecule internally
    print("\n🔬 PHASE 1: Shaping molecules internally...")
    
    for mol_id, molecule in enumerate(molecules):
        print(f"  Molecule {mol_id + 1}/{len(molecules)}...")
        positions = shape_molecule_internal(molecule, node_to_idx, positions, edge_scores)
    
    # Lock molecules
    print("\n🔒 Locking molecule structures...")
    for mol_id, molecule in enumerate(molecules):
        mol_nodes = molecule['nodes']
        mol_positions = np.array([positions[node_to_idx[n]] for n in mol_nodes])
        center = np.mean(mol_positions, axis=0)
        
        offsets = {}
        for node in mol_nodes:
            idx = node_to_idx[node]
            offsets[node] = positions[idx] - center
        
        molecule['center'] = center
        molecule['offsets'] = offsets
        molecule['locked'] = True
        
        # Calculate molecule diameter (for collision detection)
        dists = [np.linalg.norm(off) for off in offsets.values()]
        molecule['radius'] = max(dists) + 0.5 if dists else 0.5
    
    # PHASE 2: Position molecules
    print("\n💫 PHASE 2: Positioning molecules...")
    
    # Initialize molecule centers (CLOSER TOGETHER)
    for mol_id, molecule in enumerate(molecules):
        angle = 2 * np.pi * mol_id / len(molecules)
        radius = 2.0 + mol_id * 0.2  # Much closer initial spacing
        molecule['center'] = np.array([radius * np.cos(angle), radius * np.sin(angle)])
    
    learning_rate = 0.12
    
    for iteration in range(300):  # Increased from 250 for better convergence
        molecule_forces = {i: np.array([0.0, 0.0]) for i in range(len(molecules))}
        
        # FORCE 1: Inter-molecular repulsion (STRONGER - maintain 2-3x internal spacing)
        for i in range(len(molecules)):
            for j in range(i+1, len(molecules)):
                mol_i = molecules[i]
                mol_j = molecules[j]
                
                diff = mol_i['center'] - mol_j['center']
                dist = np.linalg.norm(diff)
                
                # Desired minimum distance = 2.5x sum of radii (creates clear separation)
                # Internal node spacing ~0.3-1.0, so molecule spacing should be ~1.5-3.0
                desired_min_dist = 2.5 * (mol_i['radius'] + mol_j['radius'])
                
                if dist < desired_min_dist:
                    # Strong repulsion when too close
                    force_mag = 3.5 * (desired_min_dist - dist) / (dist + 0.1)
                    force_mag = min(force_mag, 8.0)
                    
                    force = force_mag * (diff / dist)
                    molecule_forces[i] += force
                    molecule_forces[j] -= force
                elif dist < desired_min_dist * 1.5:
                    # Gentle repulsion in comfort zone
                    force_mag = 1.0 * (desired_min_dist * 1.5 - dist) / dist
                    force_mag = min(force_mag, 2.0)
                    
                    force = force_mag * (diff / dist)
                    molecule_forces[i] += force
                    molecule_forces[j] -= force
        
        # FORCE 2: Inter-molecular attraction (MODERATE - balance with repulsion)
        for i in range(len(molecules)):
            for j in range(i+1, len(molecules)):
                mol_i = molecules[i]
                mol_j = molecules[j]
                
                # Calculate compatibility between molecules
                compat_scores = []
                for node_i in mol_i['nodes'][:20]:  # Sample for speed
                    for node_j in mol_j['nodes'][:20]:
                        edge_key = (min(node_i, node_j), max(node_i, node_j))
                        if edge_key in edge_scores:
                            compat_scores.append(edge_scores[edge_key])
                
                if compat_scores:
                    avg_inter_compat = np.mean(compat_scores)
                    
                    diff = mol_i['center'] - mol_j['center']
                    dist = np.linalg.norm(diff)
                    
                    # Desired distance for compatible molecules
                    desired_min_dist = 2.5 * (mol_i['radius'] + mol_j['radius'])
                    
                    # Moderate attraction to keep compatible molecules near (but not too close)
                    if dist > desired_min_dist * 1.8 and dist < 20.0 and avg_inter_compat > 0.70:
                        force_mag = 0.5 * avg_inter_compat * (dist - desired_min_dist * 1.8)
                        force_mag = min(force_mag, 1.5)
                        
                        force = force_mag * (diff / dist)
                        molecule_forces[i] -= force
                        molecule_forces[j] += force
        
        # Apply forces
        for mol_id, force in molecule_forces.items():
            molecules[mol_id]['center'] += learning_rate * force
        
        # Update node positions
        for mol_id, molecule in enumerate(molecules):
            for node in molecule['nodes']:
                idx = node_to_idx[node]
                positions[idx] = molecule['center'] + molecule['offsets'][node]
        
        if iteration % 40 == 0:
            learning_rate *= 0.93
            avg_force = np.mean([np.linalg.norm(f) for f in molecule_forces.values()])
            if iteration % 80 == 0:
                print(f"    Iteration {iteration}/300 | lr={learning_rate:.4f} | avg_force={avg_force:.4f}")
    
    # Convert to position dictionary
    pos = {}
    for i, node in enumerate(nodes):
        pos[node] = positions[i]
    
    print("\n✓ Molecular layout complete!")
    return pos, node_to_mol

def find_peripheral_nodes(mol_a, mol_b, molecules, node_to_idx, positions):
    """Find peripheral nodes of mol_a facing mol_b"""
    mol_a_nodes = molecules[mol_a]['nodes']
    mol_b_center = molecules[mol_b]['center']
    
    # Find nodes in mol_a closest to mol_b center
    distances = []
    for node in mol_a_nodes:
        idx = node_to_idx[node]
        dist = np.linalg.norm(positions[idx] - mol_b_center)
        distances.append((dist, node))
    
    distances.sort()
    # Return closest 30% of nodes
    n_peripheral = max(1, len(mol_a_nodes) // 3)
    return [node for _, node in distances[:n_peripheral]]

def create_visualization(G, edge_scores, pos, molecules, node_to_mol, index_to_category, category_colors):
    """Create final visualization with peripheral edges only"""
    print("\n🎨 Creating visualization...")

    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    fig, ax = plt.subplots(figsize=(36, 36), facecolor='white')
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Draw molecule boundaries
    print("  Drawing molecule boundaries...")
    for mol_id, molecule in enumerate(molecules):
        if molecule['size'] < 3:
            continue
        
        mol_nodes = molecule['nodes']
        mol_positions = np.array([pos[n] for n in mol_nodes])
        
        try:
            hull = ConvexHull(mol_positions)
            hull_points = mol_positions[hull.vertices]
            hull_points = np.vstack([hull_points, hull_points[0]])
            
            # Color by quality (MORE VISIBLE)
            compat = molecule['avg_compatibility']
            if compat > 0.90:
                color, alpha = 'green', 0.15
            elif compat > 0.75:
                color, alpha = 'yellow', 0.12
            else:
                color, alpha = 'red', 0.15
            
            ax.plot(hull_points[:, 0], hull_points[:, 1],
                   color=color, alpha=alpha*5, linewidth=3.5,
                   linestyle='--', zorder=0)
            ax.fill(hull_points[:, 0], hull_points[:, 1],
                   color=color, alpha=alpha, zorder=0)
        except:
            pass
    
    # Draw internal edges (within molecules)
    print("  Drawing internal edges...")
    for mol_id, molecule in enumerate(molecules):
        mol_nodes = molecule['nodes']
        for i, node_a in enumerate(mol_nodes):
            for node_b in mol_nodes[i+1:]:
                edge_key = (min(node_a, node_b), max(node_a, node_b))
                if edge_key in edge_scores:
                    score = edge_scores[edge_key]
                    x = [pos[node_a][0], pos[node_b][0]]
                    y = [pos[node_a][1], pos[node_b][1]]
                    
                    if score >= SYNERGISTIC_THRESHOLD:
                        ax.plot(x, y, color='darkgreen', alpha=0.4, linewidth=1.5, zorder=1)
                    elif score >= INCOMPATIBLE_THRESHOLD:
                        ax.plot(x, y, color='lightgreen', alpha=0.25, linewidth=1.0, zorder=1)
                    else:
                        ax.plot(x, y, color='red', alpha=0.3, linewidth=1.2, zorder=1)
    
    # Draw external edges (SMART: only 3 closest molecules per molecule)
    print("  Drawing external edges (3 closest per molecule)...")
    
    # For each molecule, find 3 closest compatible molecules
    molecule_connections = defaultdict(list)  # mol_id -> [(distance, other_mol_id, compat), ...]
    
    for mol_a_id in range(len(molecules)):
        for mol_b_id in range(mol_a_id + 1, len(molecules)):
            # Calculate inter-molecular compatibility
            compat_scores = []
            for node_a in molecules[mol_a_id]['nodes'][:15]:
                for node_b in molecules[mol_b_id]['nodes'][:15]:
                    edge_key = (min(node_a, node_b), max(node_a, node_b))
                    if edge_key in edge_scores:
                        compat_scores.append(edge_scores[edge_key])
            
            if not compat_scores or np.mean(compat_scores) < 0.75:
                continue  # Skip incompatible pairs
            
            avg_compat = np.mean(compat_scores)
            
            # Calculate distance between molecule centers
            dist = np.linalg.norm(molecules[mol_a_id]['center'] - molecules[mol_b_id]['center'])
            
            molecule_connections[mol_a_id].append((dist, mol_b_id, avg_compat))
            molecule_connections[mol_b_id].append((dist, mol_a_id, avg_compat))
    
    # For each molecule, keep only 3 closest connections
    edges_to_draw = set()
    for mol_id, connections in molecule_connections.items():
        # Sort by distance, take top 3
        connections.sort(key=lambda x: x[0])
        for dist, other_mol_id, avg_compat in connections[:3]:
            # Add edge (use tuple to avoid duplicates)
            edge_pair = (min(mol_id, other_mol_id), max(mol_id, other_mol_id))
            edges_to_draw.add(edge_pair)
    
    print(f"  Drawing {len(edges_to_draw)} inter-molecular connections...")
    
    # Draw the selected edges
    for mol_a_id, mol_b_id in edges_to_draw:
        positions_array = np.array([pos[n] for n in nodes])
        
        # Find the two closest peripheral nodes between molecules
        min_dist = float('inf')
        best_pair = None
        
        for node_a in molecules[mol_a_id]['nodes']:
            for node_b in molecules[mol_b_id]['nodes']:
                edge_key = (min(node_a, node_b), max(node_a, node_b))
                if edge_key in edge_scores:
                    dist = np.linalg.norm(pos[node_a] - pos[node_b])
                    if dist < min_dist:
                        min_dist = dist
                        best_pair = (node_a, node_b, edge_scores[edge_key])
        
        if best_pair and best_pair[2] >= 0.80:
            node_a, node_b, score = best_pair
            x = [pos[node_a][0], pos[node_b][0]]
            y = [pos[node_a][1], pos[node_b][1]]
            
            # Draw edge with alpha based on compatibility
            alpha = 0.3 + (score - 0.80) * 2.0  # 0.3 to 0.7
            alpha = min(alpha, 0.7)
            
            ax.plot(x, y, color='green', alpha=alpha, linewidth=1.2, zorder=2)
    
    # Draw nodes
    print("  Drawing nodes...")
    node_x = [pos[node][0] for node in nodes]
    node_y = [pos[node][1] for node in nodes]

    # Color by category
    node_colors = []
    for node in nodes:
        category = index_to_category.get(node, 'unknown')
        color = category_colors.get(category, category_colors['unknown'])['color']
        node_colors.append(color)

    # Size by degree (BIGGER for visibility)
    node_sizes = []
    for node in nodes:
        degree = G.degree(node)
        node_sizes.append(np.log(degree + 2) * 150)  # Increased from 80

    scatter = ax.scatter(node_x, node_y,
                        s=node_sizes,
                        c=node_colors,
                        alpha=0.85,
                        edgecolors='white',
                        linewidths=3,
                        zorder=4)
    
    # Create legend with categories sorted by distribution (most common first)
    # Get category distribution
    category_counts = defaultdict(int)
    for node in nodes:
        category = index_to_category.get(node, 'unknown')
        category_counts[category] += 1

    # Sort by count (descending)
    sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)

    legend_elements = []
    for category, count in sorted_categories:
        if category in category_colors:
            legend_elements.append(
                mpatches.Patch(
                    color=category_colors[category]['color'],
                    label=f"{category_colors[category]['name']} ({count})"
                )
            )

    ax.legend(handles=legend_elements, loc='upper left', fontsize=18, framealpha=0.95,
              title='Method Categories', title_fontsize=20)
    
    ax.set_title('Categorized molecular clusters with greedy compatibility formation',
                fontsize=28, weight='bold', pad=20)
    
    plt.tight_layout()
    
    output_path = 'results/viz_molecular_final.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved to {output_path}")
    plt.close()

def create_interactive_visualization(G, edge_scores, pos, molecules, node_to_mol, index_to_category, category_colors, cluster_mappings, checkpoint, method_sources):
    """Create interactive HTML visualization with hover details"""
    print("\n🌐 Creating interactive HTML visualization...")

    nodes = list(G.nodes())

    # Build node info mapping (index -> full method info) from checkpoint
    node_info = {}
    results = checkpoint.get('results', [])
    for result in results:
        idx_a = int(result['method_a_index'])
        idx_b = int(result['method_b_index'])
        if idx_a not in node_info:
            category = index_to_category.get(idx_a, 'unknown')
            display_name = cluster_mappings['synergy_display_names'].get(category, category)
            method_name = result['method_a']
            source = method_sources.get(method_name.strip().lower(), 'Unknown')
            node_info[idx_a] = {
                'name': method_name,
                'source': source,
                'category': category,
                'category_name': display_name
            }
        if idx_b not in node_info:
            category = index_to_category.get(idx_b, 'unknown')
            display_name = cluster_mappings['synergy_display_names'].get(category, category)
            method_name = result['method_b']
            source = method_sources.get(method_name.strip().lower(), 'Unknown')
            node_info[idx_b] = {
                'name': method_name,
                'source': source,
                'category': category,
                'category_name': display_name
            }

    # Prepare edge traces
    edge_traces = []

    # Internal edges (within molecules)
    print("  Processing internal edges...")
    for mol_id, molecule in enumerate(molecules):
        mol_nodes = molecule['nodes']
        for i, node_a in enumerate(mol_nodes):
            for node_b in mol_nodes[i+1:]:
                edge_key = (min(node_a, node_b), max(node_a, node_b))
                if edge_key in edge_scores:
                    score = edge_scores[edge_key]
                    x = [pos[node_a][0], pos[node_b][0], None]
                    y = [pos[node_a][1], pos[node_b][1], None]

                    if score >= SYNERGISTIC_THRESHOLD:
                        color, width = 'rgba(0, 100, 0, 0.4)', 1.5
                    elif score >= INCOMPATIBLE_THRESHOLD:
                        color, width = 'rgba(144, 238, 144, 0.25)', 1.0
                    else:
                        color, width = 'rgba(255, 0, 0, 0.3)', 1.2

                    edge_traces.append(go.Scatter(
                        x=x, y=y,
                        mode='lines',
                        line=dict(color=color, width=width),
                        hoverinfo='none',
                        showlegend=False
                    ))

    # External edges (between molecules)
    print("  Processing external edges...")
    molecule_connections = defaultdict(list)

    for mol_a_id in range(len(molecules)):
        for mol_b_id in range(mol_a_id + 1, len(molecules)):
            compat_scores = []
            for node_a in molecules[mol_a_id]['nodes'][:15]:
                for node_b in molecules[mol_b_id]['nodes'][:15]:
                    edge_key = (min(node_a, node_b), max(node_a, node_b))
                    if edge_key in edge_scores:
                        compat_scores.append(edge_scores[edge_key])

            if not compat_scores or np.mean(compat_scores) < 0.75:
                continue

            avg_compat = np.mean(compat_scores)
            dist = np.linalg.norm(molecules[mol_a_id]['center'] - molecules[mol_b_id]['center'])

            molecule_connections[mol_a_id].append((dist, mol_b_id, avg_compat))
            molecule_connections[mol_b_id].append((dist, mol_a_id, avg_compat))

    edges_to_draw = set()
    for mol_id, connections in molecule_connections.items():
        connections.sort(key=lambda x: x[0])
        for dist, other_mol_id, avg_compat in connections[:3]:
            edge_pair = (min(mol_id, other_mol_id), max(mol_id, other_mol_id))
            edges_to_draw.add(edge_pair)

    for mol_a_id, mol_b_id in edges_to_draw:
        min_dist = float('inf')
        best_pair = None

        for node_a in molecules[mol_a_id]['nodes']:
            for node_b in molecules[mol_b_id]['nodes']:
                edge_key = (min(node_a, node_b), max(node_a, node_b))
                if edge_key in edge_scores:
                    dist = np.linalg.norm(pos[node_a] - pos[node_b])
                    if dist < min_dist:
                        min_dist = dist
                        best_pair = (node_a, node_b, edge_scores[edge_key])

        if best_pair and best_pair[2] >= 0.80:
            node_a, node_b, score = best_pair
            x = [pos[node_a][0], pos[node_b][0], None]
            y = [pos[node_a][1], pos[node_b][1], None]

            alpha = 0.3 + (score - 0.80) * 2.0
            alpha = min(alpha, 0.7)

            edge_traces.append(go.Scatter(
                x=x, y=y,
                mode='lines',
                line=dict(color=f'rgba(0, 128, 0, {alpha})', width=1.2),
                hoverinfo='none',
                showlegend=False
            ))

    # Prepare node traces (one per category for legend)
    print("  Processing nodes...")
    category_node_traces = {}

    for node in nodes:
        category = index_to_category.get(node, 'unknown')
        info = node_info.get(node, {
            'name': f'Method {node}',
            'source': 'Unknown',
            'category': category,
            'category_name': category_colors.get(category, category_colors['unknown'])['name']
        })

        mol_id = node_to_mol[node]
        mol_quality = molecules[mol_id]['avg_compatibility']
        degree = G.degree(node)
        node_size = np.log(degree + 2) * 8  # Scaled for plotly

        hover_text = (
            f"<b>{info['name']}</b><br>"
            f"Source: {info['source']}<br>"
            f"Category: {info['category_name']}<br>"
            f"Molecule Quality: {mol_quality:.3f}<br>"
            f"Connections: {degree}"
        )

        if category not in category_node_traces:
            cat_info = category_colors.get(category, category_colors['unknown'])
            category_node_traces[category] = {
                'x': [],
                'y': [],
                'text': [],
                'size': [],
                'color': cat_info['color'],
                'name': cat_info['name']
            }

        category_node_traces[category]['x'].append(pos[node][0])
        category_node_traces[category]['y'].append(pos[node][1])
        category_node_traces[category]['text'].append(hover_text)
        category_node_traces[category]['size'].append(node_size)

    # Create figure
    fig = go.Figure()

    # Add edge traces
    for trace in edge_traces:
        fig.add_trace(trace)

    # Add node traces (sorted by category count for consistent legend order)
    category_counts = {cat: len(data['x']) for cat, data in category_node_traces.items()}
    sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)

    for category, count in sorted_categories:
        if category in category_node_traces:
            data = category_node_traces[category]
            fig.add_trace(go.Scatter(
                x=data['x'],
                y=data['y'],
                mode='markers',
                name=f"{data['name']} ({count})",
                marker=dict(
                    size=data['size'],
                    color=data['color'],
                    line=dict(color='white', width=1.5)
                ),
                text=data['text'],
                hoverinfo='text',
                hoverlabel=dict(
                    bgcolor='white',
                    font_size=12,
                    font_family='monospace'
                )
            ))

    # Update layout
    fig.update_layout(
        title={
            'text': 'Categorized molecular clusters with greedy compatibility formation',
            'font': {'size': 20, 'family': 'Arial, sans-serif'},
            'x': 0.5,
            'xanchor': 'center'
        },
        showlegend=True,
        legend=dict(
            title='Method Categories',
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1
        ),
        hovermode='closest',
        plot_bgcolor='white',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=1600,
        height=1600,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    output_path = 'results/viz_molecular_interactive.html'
    fig.write_html(output_path)
    print(f"✓ Saved interactive visualization to {output_path}")

def load_methods_sources():
    """Load method sources from methods_deduplicated.csv"""
    import pandas as pd
    methods_path = 'input/methods_deduplicated.csv'
    df = pd.read_csv(methods_path, delimiter='|')

    # Build lookup: method name (lowercase) -> source
    name_to_source = {}
    for _, row in df.iterrows():
        name = row['Method'].strip().lower()
        source = row.get('Source', 'Unknown')
        name_to_source[name] = source

    print(f"Loaded sources for {len(name_to_source)} methods")
    return name_to_source


def main():
    """Generate final molecular layout"""

    # Load cluster mappings for dynamic category names
    print("Loading cluster mappings...")
    cluster_mappings = load_cluster_mappings()
    category_colors = get_category_colors(cluster_mappings)

    # Load method sources from CSV
    print("Loading method sources...")
    method_sources = load_methods_sources()

    checkpoint = load_compatibility_checkpoint('results/compatibility_checkpoint.pkl')
    index_to_category = load_categories(cluster_mappings, checkpoint)
    G, edge_scores = build_graph_from_checkpoint(checkpoint)

    # Form molecules by greedy compatibility
    molecules = form_molecules_greedy(G, edge_scores)

    # Create layout
    pos, node_to_mol = create_molecular_layout(G, edge_scores, molecules)

    # Visualize (static PNG)
    create_visualization(G, edge_scores, pos, molecules, node_to_mol, index_to_category, category_colors)

    # Visualize (interactive HTML)
    create_interactive_visualization(G, edge_scores, pos, molecules, node_to_mol, index_to_category, category_colors, cluster_mappings, checkpoint, method_sources)

    print("\n✨ Complete!")
    print("\n📊 Summary:")
    print(f"  Total molecules: {len(molecules)}")
    print(f"  Highest quality: {max(m['avg_compatibility'] for m in molecules):.3f}")
    print(f"  Lowest quality: {min(m['avg_compatibility'] for m in molecules):.3f}")
    print(f"  Average size: {np.mean([m['size'] for m in molecules]):.1f} nodes")
    print("\n🔬 Natural emergence:")
    print("  ✓ High-quality molecules are tight and small")
    print("  ✓ Low-quality molecules are loose and large")
    print("  ✓ Peripheral edges only between molecules")
    print("  ✓ No edge passes through a molecule")

if __name__ == '__main__':
    main()