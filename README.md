# Representation learning for biological sequences using genome-scale language models
We experimentally show that our proposed CPE tokenization algorithm outpeforms current k-mer state of the art tokenization in speed and performance on biological downstream tasks.
<p align="center">
<img src=examples/imgs/visual_abstract_white_background.png />
</p>

Our approach begins with a diverse genome corpus, then translates every 3 characters into an associated ASCII
character. This helps retain a genome-level representation while also compressing the nucleotides by 3-letter codons. Then,
training the codon-pair encoding (CPE) tokenizer using the byte-pair encoding algorithm, on this CPE corpus, we extract a
trained CPE tokenizer. Then we use this tokenizer to tokenize an input genome corpus, from which we extract local sequence
motifs; this is the step that most of the current models miss which is why they suffer performance issues. We then input the
motifs through a genome-scale language model, which we utilize for downstream biological tasks.

## Usage
Note: these models are from a previous task - [GenSLMs for SARS-CoV-2](https://github.com/ramanathanlab/genslm/tree/main).
We will we publishing the latest version soon :)

Using the notebook requires that you have the model weights downloaded from this [Globus Endpoint](https://app.globus.org/file-manager?origin_id=25918ad0-2a4e-4f37-bcfc-8183b19c3150&origin_path=%2F&two_pane=true).

Use GenSLMs to compute embeddings for downsteam tasks, generate synthetic sequences, or easily utilize them for custom applications.

In the following example, we present 2 applications of GenSLMs: generation and sequence similarity. [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_aIG54ifHjMVJmTMYPczIhvXDS8iiPvQ?usp=sharing#scrollTo=BqMbZLu2Ox05)





