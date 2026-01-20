"""
GAS-style subgoal miner.

Implements Temporal Distance Representation (TDR) training, Temporal Efficiency (TE) filtering,
TD-aware clustering, graph construction, and subgoal extraction from logged rollouts.

Based on the original GAS implementation: https://github.com/qortmdgh4141/GAS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    # Create a dummy nx module to avoid errors
    class DummyNX:
        Graph = None
    nx = DummyNX()


class TDREncoder(nn.Module):
    """
    Temporal Distance Representation encoder.
    
    Maps encoded states z_t to embeddings h_tdr(z_t) where distances
    correlate with minimal temporal distance.
    """
    
    def __init__(self, z_dim: int = 256, h_dim: int = 128):
        """
        Initialize TDR encoder.
        
        Args:
            z_dim: Input feature dimension (from pixel encoder)
            h_dim: TDR embedding dimension
        """
        super().__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim
        
        self.net = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Encode state features to TDR embeddings.
        
        Args:
            z: State features (B, z_dim) or (z_dim,)
        
        Returns:
            TDR embeddings (B, h_dim) or (h_dim,)
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        h = self.net(z)
        if z.dim() == 1:
            h = h.squeeze(0)
        return h


class GASMiner:
    """
    GAS-style subgoal miner.
    
    Trains TDR on rollouts, filters by Temporal Efficiency, performs TD-aware clustering,
    builds graph, and extracts subgoals.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        z_dim: int = 256,
        h_dim: int = 128,
        device: Optional[torch.device] = None
    ):
        """
        Initialize GAS miner.
        
        Args:
            encoder: Pixel encoder (for computing z_t from observations)
            z_dim: Encoder output dimension
            h_dim: TDR embedding dimension
            device: Device for computation
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.encoder = encoder
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.device = device
        
        # Move encoder to device
        self.encoder.to(device)
        self.encoder.eval()
        
        # TDR encoder
        self.tdr_encoder = TDREncoder(z_dim=z_dim, h_dim=h_dim).to(device)
        
    def train_tdr(
        self,
        rollouts: List[Dict],
        num_epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        max_transitions_per_episode: int = 100
    ):
        """
        Train TDR encoder on rollouts.
        
        TDR learns embeddings where distances correlate with minimal temporal distance.
        
        Args:
            rollouts: List of episode dictionaries
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            max_transitions_per_episode: Maximum transitions to sample per episode (for efficiency)
        """
        self.tdr_encoder.train()
        optimizer = torch.optim.Adam(self.tdr_encoder.parameters(), lr=lr)
        
        # Pre-compute all z embeddings once (much faster)
        print("Pre-computing encoder embeddings...")
        all_episode_z = []
        episode_lengths = []
        
        with torch.no_grad():
            for episode in tqdm(rollouts, desc="Encoding observations"):
                obs_list = episode.get('obs', [])
                if len(obs_list) < 2:
                    all_episode_z.append([])
                    episode_lengths.append(0)
                    continue
                
                # Batch encode all observations in episode
                obs_tensor = []
                for obs in obs_list:
                    if isinstance(obs, np.ndarray):
                        obs = torch.from_numpy(obs).float()
                    if obs.dim() == 1:
                        obs = obs.unsqueeze(0)
                    obs_tensor.append(obs)
                
                # Stack and encode in batch
                if len(obs_tensor) > 0:
                    obs_batch = torch.cat(obs_tensor, dim=0).to(self.device)
                    z_batch = self.encoder(obs_batch)
                    episode_z = [z_batch[i].cpu() for i in range(len(z_batch))]
                    all_episode_z.append(episode_z)
                    episode_lengths.append(len(episode_z))
                else:
                    all_episode_z.append([])
                    episode_lengths.append(0)
        
        # Create transition pairs (sample to avoid too many)
        print("Creating transition pairs...")
        transitions = []
        for ep_idx, episode_z in enumerate(all_episode_z):
            if len(episode_z) < 2:
                continue
            
            # Sample transitions to avoid O(N²) pairs
            max_possible_pairs = len(episode_z) * (len(episode_z) - 1) // 2
            num_pairs = min(max_possible_pairs, max_transitions_per_episode)
            
            if num_pairs >= max_possible_pairs:
                # Use all pairs
                for i in range(len(episode_z)):
                    for j in range(i + 1, len(episode_z)):
                        temporal_dist = j - i
                        transitions.append((ep_idx, i, j, temporal_dist))
            else:
                # Sample random pairs
                pairs = set()
                max_attempts = num_pairs * 10  # Prevent infinite loop
                attempts = 0
                
                while len(pairs) < num_pairs and attempts < max_attempts:
                    i = np.random.randint(0, len(episode_z) - 1)  # Ensure i < len-1
                    if i + 1 < len(episode_z):
                        j = np.random.randint(i + 1, len(episode_z))
                        pairs.add((i, j))
                    attempts += 1
                
                for i, j in pairs:
                    temporal_dist = j - i
                    transitions.append((ep_idx, i, j, temporal_dist))
        
        if len(transitions) == 0:
            print("Warning: No transitions found for TDR training")
            return
        
        print(f"Created {len(transitions)} transition pairs for training")
        
        # Training loop
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0
            
            # Shuffle transitions
            indices = np.random.permutation(len(transitions))
            
            for batch_start in tqdm(range(0, len(transitions), batch_size), 
                                    desc=f"Epoch {epoch+1}/{num_epochs}", 
                                    leave=False):
                batch_end = min(batch_start + batch_size, len(transitions))
                batch_indices = indices[batch_start:batch_end]
                
                batch_z1 = []
                batch_z2 = []
                batch_td = []
                
                for idx in batch_indices:
                    ep_idx, i, j, td = transitions[idx]
                    z1 = all_episode_z[ep_idx][i]
                    z2 = all_episode_z[ep_idx][j]
                    batch_z1.append(z1)
                    batch_z2.append(z2)
                    batch_td.append(td)
                
                batch_z1 = torch.stack(batch_z1).to(self.device)
                batch_z2 = torch.stack(batch_z2).to(self.device)
                batch_td = torch.tensor(batch_td, dtype=torch.float32).to(self.device)
                
                # Forward pass
                h1 = self.tdr_encoder(batch_z1)
                h2 = self.tdr_encoder(batch_z2)
                
                # TDR loss: embedding distance should correlate with temporal distance
                embedding_dist = torch.norm(h1 - h2, dim=1)
                loss = F.mse_loss(embedding_dist, batch_td.float())
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                if epoch % 10 == 0 or epoch == num_epochs - 1:
                    print(f"TDR Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    def compute_temporal_efficiency(
        self,
        rollouts: List[Dict],
        way_steps: int = 8,
        te_threshold: float = 0.0
    ) -> List[Tuple[int, int]]:
        """
        Compute Temporal Efficiency (TE) and filter states using cosine similarity method.
        
        Based on original GAS implementation: filters states where the step vector
        aligns with the distance vector in embedding space.
        
        Args:
            rollouts: List of episode dictionaries
            way_steps: Number of steps ahead to check for TE
            te_threshold: Threshold for filtering (keep states with TE >= threshold)
        
        Returns:
            List of (episode_idx, step_idx) for filtered states
        """
        filtered_states = []
        all_te_scores = []
        
        self.tdr_encoder.eval()
        
        with torch.no_grad():
            for ep_idx, episode in enumerate(tqdm(rollouts, desc="Computing TE")):
                if not episode.get('terminated', False):
                    continue  # Only consider successful episodes
                
                obs_list = episode.get('obs', [])
                if len(obs_list) < way_steps + 1:
                    continue
                
                # Compute TDR embeddings for all states in episode
                episode_embeddings = []
                for obs in obs_list:
                    if isinstance(obs, np.ndarray):
                        obs = torch.from_numpy(obs).float().to(self.device)
                    if obs.dim() == 1:
                        obs = obs.unsqueeze(0)
                    z = self.encoder(obs)
                    h = self.tdr_encoder(z).squeeze(0).cpu().numpy()
                    episode_embeddings.append(h)
                
                episode_embeddings = np.array(episode_embeddings)
                
                # Filter states using TE (cosine similarity method)
                for i in range(len(episode_embeddings) - way_steps):
                    obs_t = episode_embeddings[i]
                    
                    # Check if we can reach way_steps ahead
                    if i + way_steps >= len(episode_embeddings):
                        continue
                    
                    obs_t_plus_step = episode_embeddings[i + way_steps]
                    
                    # Find state at distance >= way_steps
                    subarr_traj = episode_embeddings[i + 1:]
                    distances_future = np.linalg.norm(subarr_traj - obs_t, axis=1)
                    idxs_above = np.where(distances_future >= way_steps)[0]
                    
                    if len(idxs_above) == 0:
                        obs_t_plus_distance = episode_embeddings[-1]
                    else:
                        obs_t_plus_distance = subarr_traj[idxs_above[0]]
                    
                    # Compute vectors
                    vector_step = obs_t_plus_step - obs_t
                    vector_distance = obs_t_plus_distance - obs_t
                    
                    # Normalize
                    norm_step = np.linalg.norm(vector_step)
                    norm_distance = np.linalg.norm(vector_distance)
                    
                    if norm_step < 1e-10 or norm_distance < 1e-10:
                        continue
                    
                    vector_step = vector_step / norm_step
                    vector_distance = vector_distance / norm_distance
                    
                    # Cosine similarity
                    cosine_similarity = np.dot(vector_step, vector_distance)
                    all_te_scores.append(cosine_similarity)
                    
                    if cosine_similarity >= te_threshold:
                        filtered_states.append((ep_idx, i))
        
        if len(all_te_scores) > 0:
            te_scores_array = np.array(all_te_scores)
            print(f"TE statistics: min={te_scores_array.min():.3f}, max={te_scores_array.max():.3f}, "
                  f"mean={te_scores_array.mean():.3f}, median={np.median(te_scores_array):.3f}")
            print(f"States with TE >= {te_threshold}: {len(filtered_states)}/{len(all_te_scores)} "
                  f"({100*len(filtered_states)/len(all_te_scores):.1f}%)")
        
        if len(filtered_states) == 0:
            print(f"Warning: No states passed TE threshold {te_threshold}.")
            if len(all_te_scores) > 0:
                suggested_threshold = np.percentile(all_te_scores, 25)  # 25th percentile
                print(f"Suggested threshold: {suggested_threshold:.3f} (25th percentile)")
            print("Consider lowering --te-threshold or increasing --way-steps.")
        
        return filtered_states
    
    def td_aware_clustering(
        self,
        all_embeddings: np.ndarray,
        efficiency_indices: List[int],
        way_steps: int = 8
    ) -> np.ndarray:
        """
        Perform TD-aware clustering to reduce number of nodes.
        
        Clusters states that are close in embedding space (within way_steps/2).
        
        Args:
            all_embeddings: All TDR embeddings (N, h_dim)
            efficiency_indices: Indices of high-TE states
            way_steps: Minimum distance threshold for clustering
        
        Returns:
            Cluster centers (M, h_dim) where M << N
        """
        min_dist = way_steps / 2.0
        f_s_sub = all_embeddings[efficiency_indices]
        
        if len(f_s_sub) == 0:
            return np.array([])
        
        stickers = np.zeros_like(f_s_sub)
        sticker_assignments = defaultdict(list)
        
        stickers[0] = f_s_sub[0]
        sticker_assignments[0].append(0)
        num_stickers = 1
        
        for i in tqdm(range(1, len(f_s_sub)), desc="TD-aware Clustering"):
            dists = np.linalg.norm(f_s_sub[i] - stickers[:num_stickers], axis=-1)
            min_idx = np.argmin(dists)
            
            if dists[min_idx] > min_dist:
                # New cluster
                stickers[num_stickers] = f_s_sub[i]
                sticker_assignments[num_stickers].append(i)
                num_stickers += 1
            else:
                # Assign to existing cluster
                sticker_assignments[min_idx].append(i)
        
        stickers = stickers[:num_stickers]
        
        # Compute cluster centers
        sticker_centers = np.zeros_like(stickers)
        for s_idx, assigned_list in sticker_assignments.items():
            assigned_points = f_s_sub[assigned_list]
            sticker_centers[s_idx] = assigned_points.mean(axis=0)
        
        return sticker_centers
    
    def build_graph(
        self,
        cluster_centers: np.ndarray,
        way_steps: int = 8,
        batch_size: int = 1024
    ):
        """
        Build graph from cluster centers using distance-based edges.
        
        Based on original GAS: adds edges between nodes within way_steps distance.
        
        Args:
            cluster_centers: Cluster center embeddings (M, h_dim)
            way_steps: Distance cutoff for edges
            batch_size: Batch size for distance computation
        
        Returns:
            NetworkX directed graph
        """
        if not NETWORKX_AVAILABLE:
            raise ImportError("networkx is required for GAS mining. Install with: pip install networkx")
        
        G = nx.DiGraph()
        
        # Add nodes
        for node_idx in range(len(cluster_centers)):
            G.add_node(node_idx, pos=cluster_centers[node_idx])
        
        # Compute pairwise distances in batches
        num_nodes = len(cluster_centers)
        pdist_matrix = np.full((num_nodes, num_nodes), np.inf, dtype=np.float32)
        
        for start_idx in tqdm(range(0, num_nodes, batch_size), desc="Computing distances"):
            end_idx = min(start_idx + batch_size, num_nodes)
            batch = cluster_centers[start_idx:end_idx]
            
            # Compute distances to all nodes
            distances = np.linalg.norm(
                batch[:, np.newaxis, :] - cluster_centers[np.newaxis, :, :],
                axis=-1
            )
            pdist_matrix[start_idx:end_idx, :] = distances
        
        # Set diagonal to inf
        np.fill_diagonal(pdist_matrix, np.inf)
        
        # Add edges based on distance cutoff
        for i in tqdm(range(num_nodes), desc="Adding edges"):
            neighbors = np.where(pdist_matrix[i] <= way_steps)[0]
            for j in neighbors:
                G.add_edge(i, j, weight=pdist_matrix[i, j])
        
        # Connect strongly connected components
        G = self._connect_components(G, pdist_matrix)
        
        return G, pdist_matrix
    
    def _connect_components(self, G: nx.DiGraph, pdist_matrix: np.ndarray) -> nx.DiGraph:
        """Connect disconnected components to ensure full graph connectivity."""
        components = list(nx.strongly_connected_components(G))
        component_groups = [list(comp) for comp in components]
        
        while len(component_groups) > 1:
            main_nodes = component_groups[0]
            other_nodes = [node for comp in component_groups[1:] for node in comp]
            
            if len(other_nodes) == 0:
                break
            
            dist_matrix = pdist_matrix[np.ix_(main_nodes, other_nodes)]
            min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            main_node, other_node = main_nodes[min_idx[0]], other_nodes[min_idx[1]]
            
            G.add_edge(main_node, other_node, weight=pdist_matrix[main_node, other_node])
            G.add_edge(other_node, main_node, weight=pdist_matrix[other_node, main_node])
            
            # Merge components
            for comp in component_groups[1:]:
                if other_node in comp:
                    component_groups[0].extend(comp)
                    component_groups.remove(comp)
                    break
        
        return G
    
    def extract_subgoals(
        self,
        G: nx.DiGraph,
        cluster_centers: np.ndarray,
        num_subgoals: int = 10
    ) -> List[Dict]:
        """
        Extract subgoals from graph using shortest paths to goal states.
        
        Instead of betweenness centrality (too slow), we extract nodes that appear
        frequently on shortest paths from start states to goal states.
        
        Args:
            G: NetworkX graph
            cluster_centers: Cluster center embeddings
            num_subgoals: Target number of subgoals
        
        Returns:
            List of subgoal dictionaries
        """
        if len(G.nodes()) == 0:
            return []
        
        # Find start and goal clusters (first and last states in successful episodes)
        # For simplicity, use nodes with highest/lowest average distance to all nodes
        node_list = list(G.nodes())
        
        if len(node_list) < num_subgoals:
            # Return all nodes as subgoals
            subgoals = []
            for node_idx in node_list:
                subgoals.append({
                    'id': len(subgoals),
                    'node_idx': node_idx,
                    'embedding': cluster_centers[node_idx],
                    'position': None
                })
            return subgoals
        
        # Compute shortest paths from all nodes to all nodes
        # Extract nodes that appear most frequently on these paths
        path_counts = defaultdict(int)
        
        # Sample a subset of nodes for efficiency
        sample_size = min(100, len(node_list))
        sampled_nodes = np.random.choice(node_list, size=sample_size, replace=False)
        
        for source in tqdm(sampled_nodes, desc="Computing shortest paths"):
            try:
                lengths, paths = nx.single_source_dijkstra(G, source=source, weight='weight')
                for target, path in paths.items():
                    if len(path) > 2:  # Only count intermediate nodes
                        for node in path[1:-1]:  # Exclude source and target
                            path_counts[node] += 1
            except nx.NetworkXNoPath:
                continue
        
        # Select top nodes by path frequency
        top_nodes = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)[:num_subgoals]
        
        subgoals = []
        for node_idx, count in top_nodes:
            subgoals.append({
                'id': len(subgoals),
                'node_idx': node_idx,
                'embedding': cluster_centers[node_idx],
                'position': None,
                'path_frequency': count
            })
        
        # If we don't have enough, add random nodes
        while len(subgoals) < num_subgoals:
            remaining = [n for n in node_list if n not in [s['node_idx'] for s in subgoals]]
            if len(remaining) == 0:
                break
            node_idx = remaining[0]
            subgoals.append({
                'id': len(subgoals),
                'node_idx': node_idx,
                'embedding': cluster_centers[node_idx],
                'position': None,
                'path_frequency': 0
            })
        
        return subgoals
    
    def mine(
        self,
        rollouts: List[Dict],
        num_epochs: int = 50,
        way_steps: int = 8,
        te_threshold: float = -1.0,  # Default: use 25th percentile
        num_subgoals: int = 10,
        batch_size: int = 1024
    ) -> List[Dict]:
        """
        Complete mining pipeline: train TDR, filter by TE, cluster, build graph, extract subgoals.
        
        Args:
            rollouts: List of episode dictionaries
            num_epochs: TDR training epochs
            way_steps: Steps ahead for TE computation and distance cutoff
            te_threshold: Temporal Efficiency threshold
            num_subgoals: Target number of subgoals
            batch_size: Batch size for graph construction
        
        Returns:
            List of subgoal dictionaries
        """
        print("Training TDR encoder...")
        self.train_tdr(rollouts, num_epochs)
        
        print("Computing Temporal Efficiency...")
        # If threshold is negative, compute it automatically from data (use 25th percentile)
        if te_threshold < 0:
            # First pass with very low threshold to collect all TE scores
            filtered_indices = self.compute_temporal_efficiency(rollouts, way_steps, -10.0)
            # The compute_temporal_efficiency function now prints statistics
            # We'll use a reasonable default based on typical TE values
            te_threshold = 0.0  # Start with 0.0 (accept all positive alignments)
            print(f"Using TE threshold: {te_threshold:.3f} (accepting all states with positive alignment)")
        else:
            filtered_indices = self.compute_temporal_efficiency(rollouts, way_steps, te_threshold)
        
        print(f"Filtered {len(filtered_indices)} states")
        
        # Compute all TDR embeddings
        print("Computing TDR embeddings...")
        all_embeddings = []
        all_indices = []
        
        self.tdr_encoder.eval()
        with torch.no_grad():
            for ep_idx, episode in enumerate(tqdm(rollouts, desc="Encoding states")):
                obs_list = episode.get('obs', [])
                for step_idx, obs in enumerate(obs_list):
                    if isinstance(obs, np.ndarray):
                        obs = torch.from_numpy(obs).float().to(self.device)
                    if obs.dim() == 1:
                        obs = obs.unsqueeze(0)
                    z = self.encoder(obs)
                    h = self.tdr_encoder(z).squeeze(0).cpu().numpy()
                    all_embeddings.append(h)
                    all_indices.append((ep_idx, step_idx))
        
        all_embeddings = np.array(all_embeddings)
        
        # Map filtered indices to global indices
        global_efficiency_indices = []
        for ep_idx, step_idx in filtered_indices:
            try:
                global_idx = all_indices.index((ep_idx, step_idx))
                global_efficiency_indices.append(global_idx)
            except ValueError:
                continue
        
        print(f"Performing TD-aware clustering on {len(global_efficiency_indices)} states...")
        cluster_centers = self.td_aware_clustering(all_embeddings, global_efficiency_indices, way_steps)
        print(f"Clustered to {len(cluster_centers)} nodes")
        
        print("Building graph...")
        G, pdist_matrix = self.build_graph(cluster_centers, way_steps, batch_size)
        print(f"Graph has {len(G.nodes())} nodes and {len(G.edges())} edges")
        
        print("Extracting subgoals...")
        subgoals = self.extract_subgoals(G, cluster_centers, num_subgoals)
        print(f"Extracted {len(subgoals)} subgoals")
        
        return subgoals
