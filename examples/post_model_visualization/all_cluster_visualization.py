from typing import Dict, List

import matplotlib.pyplot as plt
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from Bio import SeqUtils


def plot_2d_scatter(points: npt.ArrayLike, hue: npt.ArrayLike) -> None:
    """Show a 2d scatter plot of `points` colored acording to `hue`.

    Parameters
    ----------
    points: npt.ArrayLike
      A (N, 2) array of (x,y) coordinates to plot.

    hue: npt.ArrayLike
      A (N,) array of hue values associated with each point (e.g., class labels,
      scalar descriptors)
    """
    # Represent the points array with a dataframe to give labels to x,y axes
    plot_df = pd.DataFrame({"x": points[:, 0], "y": points[:, 1]})

    # Plot the scatter plot using seaborn style
    with plt.style.context("seaborn-poster"):
        fig = plt.figure(figsize=(10, 10))
        ax = sns.scatterplot(
            data=plot_df,
            x="x",
            y="y",
            palette="viridis",
            hue=hue,
            legend=True,
            edgecolor="white",
            linewidth=0,
            s=25,
        )
        plt.xlabel(r"$z_1$", fontsize=22)
        plt.ylabel(r"$z_2$", fontsize=22)


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




    !!!NOTE: This Class only provides the Pandas dataframe and the hue for your data!!!

    You must run this with the plotting code shown below as plot_df(Pandas dataframe) and hue(hue)
    """

    def __init__(
        self,
        tsne_hidden_states: npt.ArrayLike,
        labels: npt.ArrayLike,
        gc_content: npt.ArrayLike,
        label_dict: Dict,
    ):
        self.tsne_hidden_states = tsne_hidden_states
        self.labels = labels
        self.gc_content = gc_content
        self.label_dict = label_dict

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
        return plot_df, hue, plt_title

    def plot_both_clusters_gc_content(self):
        """This will plot both clusters, and shade them according to their gc content"""
        plot_df = pd.DataFrame(
            {"x": self.tsne_hidden_states[:, 0], "y": self.tsne_hidden_states[:, 1]}
        )
        hue = list(self.gc_content)
        plt_title = "All types colored w/ GC"
        return plot_df, hue, plt_title

    def separate_clusters_labels(self):
        """This will plot both clusters, and shade them according to their labels (two clusters)"""
        plot_df = pd.DataFrame(
            {"x": self.tsne_hidden_states[:, 0], "y": self.tsne_hidden_states[:, 1]}
        )
        hue = list(self.labels)
        plt_title = "All types colored w/ label"
        return plot_df, hue, plt_title

    def plot_clusters(self, plot_df, hue, plot_title):
        # PLOTTING CODE:
        with plt.style.context("seaborn-poster"):
            fig = plt.figure(figsize=(10, 10))  # noqa
            ax = sns.scatterplot(
                data=plot_df,
                x="x",
                y="y",
                palette="viridis",
                hue=hue,
                legend=True,
                edgecolor="white",
                linewidth=0,
                s=25,
            )
            plt.xlabel(r"$z_1$", fontsize=22)
            plt.ylabel(r"$z_2$", fontsize=22)
            plt.title(plot_title)
            # plt.suptitle(f"{label_paint_name} {label_note}")
            plt.rc("legend", fontsize=22)
            ax.tick_params(axis="both", which="major", labelsize=22)
            ax.tick_params(axis="both", which="minor", labelsize=22)
            sns.move_legend(
                ax,
                "lower center",
                bbox_to_anchor=(0.5, -0.25),
                ncol=4,
                title=None,
                frameon=False,
            )


# TODO: Classes are best used when there is internal state to keep track of
#       (e.g., think of how a list might be implemented). A function will
#       suffice. See my example above.

# TODO: plot_df and hue are not defined until one of the class methods are run
