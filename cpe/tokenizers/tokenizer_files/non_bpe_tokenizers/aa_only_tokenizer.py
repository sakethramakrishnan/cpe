from Bio.Seq import translate


def aa_only_tokenizer(dna_seq, k=1):
    aa_seq = translate(dna_seq)
    tokens = list(aa_seq)
    grouped_tokens = []
    for i in range(0, len(tokens), k):
        grouped_tokens.append("".join(tokens[i : i + k]))
    return grouped_tokens
