"""Process datasets to transition matrices"""
import numpy as np
import pandas as pd


def extract_seqs(df, value_id, sort_id, group_id):
    """Extract sequences from dataframe

    Args:
        df: Data to extract sequences from.
        value_id: ID of the value column.
        sort_id: ID of the column to sort by.
        group_id: ID of the column to group by.

    Returns:
        seqs, vocab: list, dict.
    """
    # Extract vocabulary
    unique_vals = df[value_id].unique()
    vocab = dict(zip(unique_vals, range(len(unique_vals))))
    # Remap values
    df[value_id].replace(vocab, inplace=True)
    # Extract sequences
    ordered = df.sort_values([group_id, sort_id], ascending=True)
    grouped = ordered.groupby(group_id)
    seqs = grouped[value_id].apply(list).values
    return seqs, vocab


def transition_matrix(seqs, vocab, k=0):
    """Learn global Markov transition matrix from sequences

    Args:
        seqs: Contains sequences to learn from.
        vocab: Words in sequences.
        k: Smoothing parameter from Dirchlet prior.

    Returns:
        T: Transition matrix in the form of an np.array.
    """
    n = len(vocab)
    alpha = np.zeros((n, n+1)) # Note: +1 for end_token
    gamma = np.zeros(n)
    # Fill with counts
    for seq in seqs:
        if len(seq) > 1:
            for i, j in zip(seq[:-1], seq[1:]):
                alpha[i, j] = alpha[i, j] + 1
            alpha[j, n] = alpha[j, n] + 1
        else:
            alpha[seq[0], n] = alpha[seq[0], n] + 1
        gamma[seq[0]] = gamma[seq[0]] + 1
    # Normalize
    z = np.sum(alpha, axis=1).reshape((n, 1))
    alpha = (alpha + k) / (z + n*k)
    gamma = gamma / np.sum(gamma)
    return alpha, gamma


if __name__ == '__main__':
    seqs = [[0, 1], [1, 0]]
    vocab = {0: 0, 1: 1}
    alpha, gamma = transition_matrix(seqs, vocab)
    print alpha

