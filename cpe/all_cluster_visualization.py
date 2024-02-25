from typing import Dict, List

import matplotlib.pyplot as plt
import numpy.typing as npt
import pandas as pd

from Bio import SeqUtils
from Bio.SeqUtils.ProtParam import ProteinAnalysis




def gc_content(seqs: List[str]) -> List[float]:
    """Given a list of DNA sequences, return each sequence's GC content.

    Parameters
    ----------
    seqs : List[str]
        A list of DNA sequences.

    Returns
    -------
    List
        GC content of each DNA sequence.
    """
    return [SeqUtils.gc_fraction(seq) for seq in seqs]


CODON_TO_AA = {
    'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
    'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
    'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
    'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
    'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
    'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
    'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
    'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
    'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
    'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
    'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
    'TAC': 'Y', 'TAT': 'Y', 'TAA': '_', 'TAG': '_',
    'TGC': 'C', 'TGT': 'C', 'TGA': '_', 'TGG': 'W',
    'XXX':'X'
}

def translate(sequence: str): 
    amino_acids = []
    for i in range(0, len(sequence), 3):
        codon = sequence[i:i+3]
        if len(codon) == 3:
            amino_acid = CODON_TO_AA.get(codon, 'X')
            amino_acids.append(amino_acid)
    return ''.join(amino_acids)


class PlotClustersData:

    """

    This class has functions that help plot hidden state data


    Parameters
    ---------------
    hidden_states: npt.ArrayLike
      A numpy array with the tsne embeddings of the model

    labels: npt.ArrayLike
      A numpy array with the labels (binary classification)

    gc_content: npt.ArrayLike
      A numpy array with the gc_content associated with each sequence and label


    Returns
    ----------
    plot_df: pandas.core.frame.DataFrame
      A dataframe with the (x,y) coordinates extracted from your embeddings that are needed to plot the clusters

    hue: List[float]
      A list with the GC content associated with each sequence

    plt_title: str
      A string with the plot of the title describing the plot


    You must run this with the plotting code shown below as plot_df(Pandas dataframe) and hue(hue)
    """

    def __init__(
        self,
        sequences: List[str],
        tsne_hidden_states: npt.ArrayLike,
        labels: npt.ArrayLike,
        label_dict: Dict,
        tokenizer_type: str,
        model_type_with_params: str = None
    ):
        self.sequences = sequences
        self.tsne_hidden_states = np.array(tsne_hidden_states)
        self.labels = labels
        self.label_dict = label_dict
        self.amino_acids = [translate(sequence) for sequence in self.sequences]
        self.tokenizer_type = tokenizer_type
        self.model_type_with_params = model_type_with_params

        # label_dict is a dictionary with each key being the str value of the label
        # and the value being its corresponding 1-n labels' label num
        # lables have label numbers

    def separate_clusters_gc_content(self, label_mask: int):
        """Mask is the cluster we are isolating"""
        mask = self.labels == label_mask
        mask_adj_tsne = self.tsne_hidden_states[mask]
        new_gc = list(self.gc_content[mask])
        plot_df = pd.DataFrame({"x": mask_adj_tsne[:, 0], "y": mask_adj_tsne[:, 1]})
        hue = new_gc
        type_of_seq = list(self.label_dict.keys())[
            list(self.label_dict.values()).index(label_mask)
        ]
        plt_title = f"Only {type_of_seq} colored w/ GC"
        
        cbar_label = 'GC Content'
        
        return plot_df, hue, plt_title, cbar_label

    def plot_gc_content(self, gc_threshold_low=0, gc_threshold_high=100, include_thresholds_in_title: bool = True):
        
        gc_content_of_seqs = np.array(gc_content(self.sequences))
        
        # Filter sequences based on both low and high GC content thresholds
        gc_indices = np.where((gc_content_of_seqs > gc_threshold_low) & (gc_content_of_seqs < gc_threshold_high))[0]

        # Create separate DataFrames for low and high GC content
        plot_df = pd.DataFrame({"x": self.tsne_hidden_states[gc_indices, 0], "y": self.tsne_hidden_states[gc_indices, 1]})

        # Create hue values for both low and high GC content
        hue = gc_content_of_seqs[gc_indices]


        plt_title = f"GC Coloring Using {self.tokenizer_type} for a {self.model_type_with_params} model"
        if include_thresholds_in_title:
            plt_title += f" (Min GC = {gc_threshold_high}% , Max GC = {gc_threshold_high}%"
        
        cbar_label = 'GC Content'
        
        return plot_df, hue, plt_title, cbar_label
    
    def plot_seq_len(self, min_len=0, max_len=99999999999, include_thresholds_in_title: bool = True):
        seq_lens = np.array([len(seq) for seq in self.sequences])
        
        # Filter sequences based on both min and max sequence length thresholds
        seq_len_indices = np.where((seq_lens > min_len) & (seq_lens < max_len))[0]

        # Create separate DataFrames for low and high GC content
        plot_df = pd.DataFrame({"x": self.tsne_hidden_states[seq_len_indices, 0], "y": self.tsne_hidden_states[seq_len_indices, 1]})
        #high_gc_df = pd.DataFrame({"x": self.tsne_hidden_states[high_gc_indices, 0], "y": self.tsne_hidden_states[high_gc_indices, 1]})
        
        # Create hue values for both low and high GC content
        hue = seq_lens[seq_len_indices]

        plt_title = f"Sequence Lengths Coloring Using {self.tokenizer_type} for a {self.model_type_with_params} model"
        if include_thresholds_in_title:
            plt_title += f" (Min len = {min_len} bpgc_threshold_high , Max len = {max_len} bp"
        
        cbar_label = 'Sequence Length'
        
        return plot_df, hue, plt_title, cbar_label
    
    def plot_molecular_weight(self):
        molecular_weights = [ProteinAnalysis(protein_aa).molecular_weight() for protein_aa in self.amino_acids]
        
        plot_df = pd.DataFrame(
            {"x": self.tsne_hidden_states[:, 0], "y": self.tsne_hidden_states[:, 1]}
        )
        hue = molecular_weights
        plt_title = f"Molecular Weights Coloring Using {self.tokenizer_type} for a {self.model_type_with_params} model"
        
        cbar_label = 'Molecular Weights'
        
        return plot_df, hue, plt_title, cbar_label
    
    def plot_isoelectric_point(self):
        isoelectric_point = [ProteinAnalysis(protein_aa).isoelectric_point() for protein_aa in self.amino_acids]
        
        plot_df = pd.DataFrame(
            {"x": self.tsne_hidden_states[:, 0], "y": self.tsne_hidden_states[:, 1]}
        )
        hue = isoelectric_point
        plt_title = f"Isoelectric Point Coloring Using {self.tokenizer_type} for a {self.model_type_with_params} model"
        
        cbar_label = 'Isoelectric Point'
        
        return plot_df, hue, plt_title, cbar_label
    
    def plot_aromaticity(self):
        aromaticity = [ProteinAnalysis(protein_aa).aromaticity() for protein_aa in self.amino_acids]
        
        plot_df = pd.DataFrame(
            {"x": self.tsne_hidden_states[:, 0], "y": self.tsne_hidden_states[:, 1]}
        )
        hue = aromaticity
        plt_title = f"Aromaticity Coloring Using {self.tokenizer_type} for a {self.model_type_with_params} model"
        
        cbar_label = 'Aromaticity'
        
        return plot_df, hue, plt_title, cbar_label
    
    def plot_instability_index(self):
        instability_index = [ProteinAnalysis(protein_aa).instability_index() for protein_aa in self.amino_acids]
        
        plot_df = pd.DataFrame(
            {"x": self.tsne_hidden_states[:, 0], "y": self.tsne_hidden_states[:, 1]}
        )
        hue = instability_index
        plt_title = f"Instability Index Coloring Using {self.tokenizer_type} for a {self.model_type_with_params} model"
        
        cbar_label = 'Instability'
        
        return plot_df, hue, plt_title, cbar_label
    
    def plot_flexibility(self):
        flexibility = [ProteinAnalysis(protein_aa).flexibility() for protein_aa in self.amino_acids]
        
        plot_df = pd.DataFrame(
            {"x": self.tsne_hidden_states[:, 0], "y": self.tsne_hidden_states[:, 1]}
        )
        hue = flexibility
        plt_title = f"Flexibility Coloring Using {self.tokenizer_type} for a {self.model_type_with_params} model"
        
        cbar_label = 'Flexibility'
        
        return plot_df, hue, plt_title, cbar_label
    
    def plot_molar_extinction_coefficient(self):
        molar_extinction_coeff = [ProteinAnalysis(protein_aa).molar_extinction_coefficient() for protein_aa in self.amino_acids]
        
        plot_df = pd.DataFrame(
            {"x": self.tsne_hidden_states[:, 0], "y": self.tsne_hidden_states[:, 1]}
        )
        hue = molar_extinction_coeff
        plt_title = f"Molar Extinction Coefficient Coloring Using {self.tokenizer_type} for a {self.model_type_with_params} model"
        
        cbar_label = 'Molar Extinction Coefficient'
        
        return plot_df, hue, plt_title, cbar_label

    def separate_clusters_labels(self):
        """This will plot both clusters, and shade them according to their labels (two clusters)"""
        plot_df = pd.DataFrame(
            {"x": self.tsne_hidden_states[:, 0], "y": self.tsne_hidden_states[:, 1]}
        )
        hue = list(self.labels)
        plt_title = "All types colored with label"
        
        cbar_label = 'Label'
        
        return plot_df, hue, plt_title, cbar_label

    def plot_clusters(self, plot_df, hue, plt_title, color_label):
        """
        Create a 2D scatter plot with colors and a color bar.
        Parameters:
            x: x-axis data
            y: y-axis data
            colors: color data for each point
            color_label: label for the color bar
        """
        
        sc = plt.scatter(plot_df['x'], plot_df['y'], c=hue, cmap='viridis', s=10)

        # Add color bar
        cbar = plt.colorbar(sc)
        cbar.set_label(color_label)
        # Set axis labels
        plt.xlabel(r"$z_1$", fontsize=12)
        plt.ylabel(r"$z_2$", fontsize=12)

        # Set plot title if needed
        plt.title(plt_title)

        # Show the plot
        plt.show(block=False)



# TODO: Classes are best used when there is internal state to keep track of
#       (e.g., think of how a list might be implemented). A function will
#       suffice. See my example above.

# TODO: plot_df and hue are not defined until one of the class methods are run