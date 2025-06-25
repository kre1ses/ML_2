from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import xml.etree.ElementTree as ET
from collections import Counter


@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()

    sentence_pairs = []
    alignments = []

    # необходимая замена
    content = content.replace('&', '&amp;')

    for sentence in ET.ElementTree(ET.fromstring(content)).getroot().findall("s"):
        source_text_english = sentence.find("english").text.strip().split()
        source_text_czech = sentence.find("czech").text.strip().split()

        sentence_pair = SentencePair(source_text_english, source_text_czech)
        sentence_pairs.append(sentence_pair)

        sure_align = []
        possible_align = []

        if sentence.find("sure") is not None and sentence.find("sure").text:
            for pair_sure in sentence.find("sure").text.strip().split():
                sure_align.append(tuple(map(int, pair_sure.split("-"))))

        if sentence.find("possible") is not None and sentence.find("possible").text:
            for pair_possible in sentence.find("possible").text.strip().split():
                possible_align.append(tuple(map(int, pair_possible.split("-"))))

        alignments.append(LabeledAlignment(sure=sure_align, possible=possible_align))
    return sentence_pairs, alignments

def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language

    """
    source_counter = Counter()
    target_counter = Counter()
    
    for pair in sentence_pairs:
        source_counter.update(pair.source)
        target_counter.update(pair.target)
    
    if freq_cutoff is not None:
        source_vocab = [token for token, _ in source_counter.most_common(freq_cutoff)]
        target_vocab = [token for token, _ in target_counter.most_common(freq_cutoff)]
    else:
        source_vocab = list(source_counter.keys())
        target_vocab = list(target_counter.keys())
    
    # словарь token->index
    source_dict = {token: idx for idx, token in enumerate(source_vocab, start=0)}
    target_dict = {token: idx for idx, token in enumerate(target_vocab, start=0)}
    
    return source_dict, target_dict


def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.
    
    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    tokenized_sentence_pairs = []
    
    for pair in sentence_pairs:
        source_tokens = [source_dict[token] for token in pair.source if token in source_dict]
        target_tokens = [target_dict[token] for token in pair.target if token in target_dict]

        if source_tokens and target_tokens:
            tokenized_sentence_pairs.append(TokenizedSentencePair(
                source_tokens=np.array(source_tokens, dtype=np.int32),
                target_tokens=np.array(target_tokens, dtype=np.int32)
            ))
    
    return tokenized_sentence_pairs
