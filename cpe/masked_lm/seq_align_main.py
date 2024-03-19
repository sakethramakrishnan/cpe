from sequence_alignment import GlobalAlignment, LocalAlignment, SequenceAlignment


def get_aligned_seqs():
    # Get input from CLI program
    seqs = [
        'ATGAGCAATCAATCAGCAATCAATCTCACCATGAGCAATCTCACCATGAGCAATCTCACCATGAGCAATCTCACCATGAGCAAGGTCACCACCTGGAA',
        'ATGGCAAACGATGGCATGAGCAAGGTCACCATGAGCAAGGTCACCATGAGCAAGGTCACCATGAGCAAGGTCACCATGAGCAAGGTCACCCGC'
    ]


    


    # Save results to database and print them
    
    ga_align1, ga_align2, ga_alignment_score, la_align1, la_align2, la_alignment_score = get_results(seqs[0], seqs[1])

    return ga_align1, ga_align2, ga_alignment_score, la_align1, la_align2, la_alignment_score




# Save alignment results from all combinations of sequences to database
# (And print them if program is being run as CLI tool)
def get_results(seq1, seq2):
    
    #for seq1, seq2 in combinations(seqs, 2):
        
        ga = GlobalAlignment((seq1, seq2))
        la = LocalAlignment((seq1, seq2))

        ga_align1, ga_align2 = ga.align
        la_align1, la_align2 = la.align

        return ga_align1, ga_align2, ga.alignment_score, la_align1, la_align2, la.alignment_score
        
        

