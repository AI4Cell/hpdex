"""
Synthetic Data Generators for HPDEX Testing

Generates realistic test datasets for various testing scenarios.
"""

import numpy as np
import pandas as pd
import anndata as ad
from typing import Tuple, Optional
from scipy.sparse import csr_matrix


class SyntheticDataGenerator:
    """Generator for synthetic test data matching single-cell characteristics."""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def generate_kernel_test_data(
        self,
        n_genes: int,
        n_ref: int,
        n_tar: int,
        data_type: str = 'float',
        add_ties: bool = False,
        effect_size: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate test data for kernel testing.
        
        Args:
            n_genes: Number of genes
            n_ref: Number of reference samples  
            n_tar: Number of target samples
            data_type: 'float', 'int', or 'mixed'
            add_ties: Whether to add tied values
            effect_size: Effect size for differential expression
            
        Returns:
            (ref_data, tar_data) as (n_genes, n_samples) arrays
        """
        np.random.seed(self.random_seed)
        
        if data_type == 'float':
            # Generate normal data
            ref_data = np.random.normal(0, 1, (n_genes, n_ref))
            tar_data = np.random.normal(effect_size, 1, (n_genes, n_tar))
            
        elif data_type == 'int':
            # Generate count data (Poisson-like)
            ref_data = np.random.poisson(5, (n_genes, n_ref))
            tar_data = np.random.poisson(5 + effect_size, (n_genes, n_tar))
            
        else:  # mixed
            # Mix of continuous and discrete patterns
            ref_data = np.random.normal(0, 1, (n_genes, n_ref))
            tar_data = np.random.normal(effect_size, 1, (n_genes, n_tar))
            
            # Make some genes discrete
            discrete_genes = np.random.choice(n_genes, n_genes // 3, replace=False)
            ref_data[discrete_genes] = np.round(ref_data[discrete_genes])
            tar_data[discrete_genes] = np.round(tar_data[discrete_genes])
        
        # Add ties if requested
        if add_ties:
            self._add_ties(ref_data, tar_data, tie_fraction=0.3)
        
        return ref_data.astype(np.float64), tar_data.astype(np.float64)
    
    def generate_sc_dataset(
        self,
        n_cells: int,
        n_genes: int,
        n_groups: int,
        differential_fraction: float = 0.2,
        effect_size: float = 1.5,
        sparsity: float = 0.0,
        add_batch_effect: bool = False
    ) -> ad.AnnData:
        """
        Generate realistic single-cell RNA-seq dataset.
        
        Args:
            n_cells: Number of cells
            n_genes: Number of genes  
            n_groups: Number of cell groups/types
            differential_fraction: Fraction of genes that are differential
            effect_size: Effect size for differential genes
            sparsity: Fraction of zero values (0.0 = dense)
            add_batch_effect: Whether to add batch effects
            
        Returns:
            AnnData object with synthetic single-cell data
        """
        np.random.seed(self.random_seed)
        
        # Create group labels
        cells_per_group = n_cells // n_groups
        group_labels = []
        for i in range(n_groups):
            group_name = f"group_{i}" if i > 0 else "control"
            count = cells_per_group if i < n_groups - 1 else n_cells - i * cells_per_group
            group_labels.extend([group_name] * count)
        
        group_labels = np.array(group_labels[:n_cells])
        
        # Generate base expression with realistic single-cell characteristics
        X = self._generate_realistic_expression(n_cells, n_genes)
        
        # Add differential expression
        n_diff_genes = int(n_genes * differential_fraction)
        diff_genes = np.random.choice(n_genes, n_diff_genes, replace=False)
        
        for i, group in enumerate(np.unique(group_labels)):
            if group == "control":
                continue
                
            group_mask = group_labels == group
            
            # Add differential expression to subset of genes
            group_diff_genes = np.random.choice(
                diff_genes, 
                min(len(diff_genes), max(1, n_diff_genes // 2)), 
                replace=False
            )
            
            # Some genes up, some down
            n_up = len(group_diff_genes) // 2
            up_genes = group_diff_genes[:n_up]
            down_genes = group_diff_genes[n_up:]
            
            # Apply effects
            X[np.ix_(group_mask, up_genes)] *= (1 + effect_size)
            X[np.ix_(group_mask, down_genes)] *= (1 - effect_size * 0.5)
        
        # Add sparsity
        if sparsity > 0:
            zero_mask = np.random.random((n_cells, n_genes)) < sparsity
            X[zero_mask] = 0
        
        # Add batch effects if requested
        if add_batch_effect:
            batch_labels = np.random.choice(['batch1', 'batch2'], n_cells)
            batch2_mask = batch_labels == 'batch2'
            batch_genes = np.random.choice(n_genes, n_genes // 4, replace=False)
            X[np.ix_(batch2_mask, batch_genes)] *= 1.2
        else:
            batch_labels = ['batch1'] * n_cells
        
        # Create obs DataFrame
        obs = pd.DataFrame({
            'group': group_labels,
            'batch': batch_labels,
            'n_genes': np.sum(X > 0, axis=1),  # Number of detected genes
            'total_counts': np.sum(X, axis=1)  # Total UMI counts
        })
        obs.index = [f"cell_{i}" for i in range(n_cells)]
        
        # Create var DataFrame
        var = pd.DataFrame({
            'gene_type': ['protein_coding'] * n_genes,
            'highly_variable': np.random.choice([True, False], n_genes),
            'mean_expression': np.mean(X, axis=0),
            'dispersion': np.var(X, axis=0) / (np.mean(X, axis=0) + 1e-8)
        })
        var.index = [f"GENE_{i:04d}" for i in range(n_genes)]
        
        # Convert to sparse if high sparsity
        if sparsity > 0.5:
            X = csr_matrix(X)
        
        # Create AnnData object
        adata = ad.AnnData(X=X, obs=obs, var=var)
        
        # Add some metadata
        adata.uns['dataset_info'] = {
            'n_cells': n_cells,
            'n_genes': n_genes,
            'n_groups': n_groups,
            'differential_fraction': differential_fraction,
            'effect_size': effect_size,
            'sparsity': sparsity,
            'synthetic': True
        }
        
        return adata
    
    def _generate_realistic_expression(self, n_cells: int, n_genes: int) -> np.ndarray:
        """Generate realistic single-cell expression matrix."""
        # Base expression follows log-normal distribution
        # (typical for log-normalized sc-RNA-seq data)
        
        # Gene-specific parameters
        gene_mean = np.random.gamma(2, 0.5, n_genes)  # Average expression per gene
        gene_dispersion = np.random.gamma(1, 0.3, n_genes)  # Gene-specific noise
        
        # Cell-specific parameters  
        cell_size_factor = np.random.gamma(3, 0.3, n_cells)  # Cell size variation
        
        # Generate expression
        X = np.zeros((n_cells, n_genes))
        
        for i in range(n_genes):
            # Negative binomial-like generation
            mu = gene_mean[i] * cell_size_factor
            sigma = mu + gene_dispersion[i] * mu**2
            
            # Use normal approximation for speed
            X[:, i] = np.random.normal(mu, np.sqrt(sigma))
            X[:, i] = np.maximum(X[:, i], 0)  # Ensure non-negative
        
        # Add some zeros for realism
        zero_prob = np.random.beta(1, 3, n_genes)  # Gene-specific dropout
        for i in range(n_genes):
            zero_mask = np.random.random(n_cells) < zero_prob[i]
            X[zero_mask, i] = 0
        
        return X
    
    def _add_ties(self, ref_data: np.ndarray, tar_data: np.ndarray, tie_fraction: float = 0.3):
        """Add tied values to test data."""
        n_genes, n_ref = ref_data.shape
        n_tar = tar_data.shape[1]
        
        for i in range(n_genes):
            if np.random.random() < tie_fraction:
                # Pick a value to tie
                tie_value = ref_data[i, 0]
                
                # Add ties in reference group
                n_ties_ref = np.random.randint(2, min(5, n_ref))
                tie_indices = np.random.choice(n_ref, n_ties_ref, replace=False)
                ref_data[i, tie_indices] = tie_value
                
                # Add ties in target group
                n_ties_tar = np.random.randint(1, min(4, n_tar))
                tie_indices = np.random.choice(n_tar, n_ties_tar, replace=False)
                tar_data[i, tie_indices] = tie_value
    
    def generate_edge_case_data(self, case_type: str, **kwargs) -> ad.AnnData:
        """Generate data for specific edge case scenarios."""
        if case_type == "tiny_groups":
            return self._generate_tiny_groups(**kwargs)
        elif case_type == "constant_genes":
            return self._generate_constant_genes(**kwargs)
        elif case_type == "sparse_data":
            return self._generate_sparse_data(**kwargs)
        elif case_type == "extreme_values":
            return self._generate_extreme_values(**kwargs)
        else:
            raise ValueError(f"Unknown edge case type: {case_type}")
    
    def _generate_tiny_groups(self, n_cells: int = 20, n_genes: int = 50, 
                             min_group_size: int = 3) -> ad.AnnData:
        """Generate dataset with very small groups."""
        # Create groups with minimum size
        n_groups = n_cells // min_group_size
        group_labels = []
        
        for i in range(n_groups):
            group_name = f"group_{i}" if i > 0 else "control"
            group_labels.extend([group_name] * min_group_size)
        
        # Fill remaining cells
        remaining = n_cells - len(group_labels)
        if remaining > 0:
            group_labels.extend(["control"] * remaining)
        
        group_labels = np.array(group_labels[:n_cells])
        
        # Generate simple expression data
        X = np.random.normal(5, 2, (n_cells, n_genes))
        X = np.maximum(X, 0)
        
        obs = pd.DataFrame({'group': group_labels})
        obs.index = [f"cell_{i}" for i in range(n_cells)]
        
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
        
        return ad.AnnData(X=X, obs=obs, var=var)
    
    def _generate_constant_genes(self, n_cells: int = 500, n_genes: int = 100,
                                constant_gene_fraction: float = 0.3) -> ad.AnnData:
        """Generate dataset with some constant genes."""
        X = np.random.normal(5, 2, (n_cells, n_genes))
        X = np.maximum(X, 0)
        
        # Make some genes constant
        n_constant = int(n_genes * constant_gene_fraction)
        constant_genes = np.random.choice(n_genes, n_constant, replace=False)
        
        for gene_idx in constant_genes:
            constant_value = np.random.uniform(0, 10)
            X[:, gene_idx] = constant_value
        
        # Create groups
        group_labels = ['control'] * (n_cells // 2) + ['treatment'] * (n_cells // 2)
        if len(group_labels) < n_cells:
            group_labels.extend(['control'] * (n_cells - len(group_labels)))
        
        obs = pd.DataFrame({'group': group_labels[:n_cells]})
        obs.index = [f"cell_{i}" for i in range(n_cells)]
        
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
        
        return ad.AnnData(X=X, obs=obs, var=var)
    
    def _generate_sparse_data(self, n_cells: int = 1000, n_genes: int = 500,
                             sparsity: float = 0.8) -> ad.AnnData:
        """Generate highly sparse dataset."""
        return self.generate_sc_dataset(
            n_cells=n_cells,
            n_genes=n_genes,
            n_groups=3,
            sparsity=sparsity
        )
    
    def _generate_extreme_values(self, n_cells: int = 500, n_genes: int = 100,
                                include_outliers: bool = True,
                                outlier_magnitude: float = 10.0) -> ad.AnnData:
        """Generate dataset with extreme values and outliers."""
        X = np.random.normal(5, 2, (n_cells, n_genes))
        X = np.maximum(X, 0)
        
        if include_outliers:
            # Add extreme outliers
            n_outliers = n_cells // 20  # 5% outlier cells
            outlier_cells = np.random.choice(n_cells, n_outliers, replace=False)
            outlier_genes = np.random.choice(n_genes, n_genes // 5, replace=False)
            
            for cell in outlier_cells:
                for gene in outlier_genes:
                    if np.random.random() < 0.1:  # 10% chance per gene
                        X[cell, gene] *= outlier_magnitude
        
        # Create groups
        group_labels = ['control'] * (n_cells // 2) + ['treatment'] * (n_cells // 2)
        if len(group_labels) < n_cells:
            group_labels.extend(['control'] * (n_cells - len(group_labels)))
        
        obs = pd.DataFrame({'group': group_labels[:n_cells]})
        obs.index = [f"cell_{i}" for i in range(n_cells)]
        
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
        
        return ad.AnnData(X=X, obs=obs, var=var)
