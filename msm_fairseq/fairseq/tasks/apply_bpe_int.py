#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Rico Sennrich

"""Use operations learned with learn_bpe.py to encode a new text.
The text will not be smaller, but use only a fixed vocabulary, with rare words
encoded as variable-length sequences of subword units.

Reference:
Rico Sennrich, Barry Haddow and Alexandra Birch (2015). Neural Machine Translation of Rare Words with Subword Units.
Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.
"""

from __future__ import unicode_literals, division

import sys
import os
import inspect
import codecs
import io
import argparse
import re
import warnings
import random
import copy
import heapq

# hack for python2/3 compatibility
from io import open

word_merge = 0
phrase_merge = 0
cross_merge = 0
cross_dic = {}


class Dictionary(object):
    """A mapping from symbols to consecutive integers"""

    def __init__(
        self,
        pad="<pad>",
        eos="</s>",
        unk="<unk>",
        bos="<s>",
        extra_special_symbols=None,
    ):
        self.unk_word, self.pad_word, self.eos_word = unk, pad, eos
        self.symbols = []
        self.count = []
        self.indices = {}
        self.bos_index = self.add_symbol(bos)
        self.pad_index = self.add_symbol(pad)
        self.eos_index = self.add_symbol(eos)
        self.unk_index = self.add_symbol(unk)
        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s)
        self.nspecial = len(self.symbols)

    def __eq__(self, other):
        return self.indices == other.indices

    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def __contains__(self, sym):
        return sym in self.indices

    def index(self, sym):
        """Returns the index of the specified symbol"""
        assert isinstance(sym, str)
        if sym in self.indices:
            return self.indices[sym]
        return self.unk_index

    def unk_string(self, escape=False):
        """Return unknown string, optionally escaped as: <<unk>>"""
        if escape:
            return "<{}>".format(self.unk_word)
        else:
            return self.unk_word

    def add_symbol(self, word, n=1):
        """Adds a word to the dictionary"""
        if word in self.indices:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx

    def update(self, new_dict):
        """Updates counts from new dictionary."""
        for word in new_dict.symbols:
            idx2 = new_dict.indices[word]
            if word in self.indices:
                idx = self.indices[word]
                self.count[idx] = self.count[idx] + new_dict.count[idx2]
            else:
                idx = len(self.symbols)
                self.indices[word] = idx
                self.symbols.append(word)
                self.count.append(new_dict.count[idx2])

    @classmethod
    def load(cls, f):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        d = cls()
        d.add_from_file(f)
        return d

    def add_from_file(self, f):
        """
        Loads a pre-existing dictionary from a text file and adds its symbols
        to this instance.
        """
        if isinstance(f, str):
            try:
                with open(f, "r", encoding="utf-8") as fd:
                    self.add_from_file(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception(
                    "Incorrect encoding detected in {}, please "
                    "rebuild the dataset".format(f)
                )
            return

        lines = f.readlines()
        indices_start_line = 0
        for line in lines[indices_start_line:]:
            idx = line.rfind(" ")
            if idx == -1:
                raise ValueError(
                    "Incorrect dictionary format, expected '<token> <cnt>'"
                )
            word = line[:idx]
            count = int(line[idx + 1 :])
            self.indices[word] = len(self.symbols)
            self.symbols.append(word)
            self.count.append(count)


class BPE(object):

    def __init__(self, codes, merges=-1, separator='@@', vocab=None, glossaries=None, dictionary=None, one_word_merges=-1):
        assert dictionary is not None
        dictionary = copy.deepcopy(dictionary)
        self.last_word_map = {}
        self.unk_id = dictionary.index('<unk>')

        codes.seek(0)
        offset = 1

        # check version information
        firstline = codes.readline()
        if firstline.startswith('#version:'):
            self.version = tuple([int(x) for x in re.sub(r'(\.0+)*$', '', firstline.split()[-1]).split(".")])
            offset += 1
        else:
            self.version = (0, 1)
            codes.seek(0)

        self.bpe_codes = []
        self.merge_list = []

        self.unk_index = dictionary.unk_index
        self.eos_index = dictionary.index("</w>")

        def update_last_word_map(word):
            if not word.endswith("</w>"):
                return
            new_id = dictionary.index(word)
            old_id = dictionary.index(word[:-4])
            # if self.dictionary[old_id] == word[:-4]:
            self.last_word_map[old_id] = new_id
        word2merge = {}
        pair_set = set()
        for n, item in enumerate(codes):
            if n < merges or merges == -1:
                item = tuple(item.strip('\r\n ').split(' '))
                if len(item) != 2:
                    sys.stderr.write(
                        'Error: invalid line {0} in BPE codes file: {1}\n'.format(i + offset, ' '.join(item)))
                    sys.stderr.write('The line should exist of exactly two subword units, separated by whitespace\n')
                    sys.exit(1)

                word_a, word_b = item
                word_new = word_a + word_b
                word_a_id = dictionary.add_symbol(word_a, 1)
                word_b_id = dictionary.add_symbol(word_b, 1)
                word_new_id = dictionary.add_symbol(word_new, 1)
                if word_a_id not in word2merge:
                    word2merge[word_a_id] = 1
                if word_b_id not in word2merge:
                    word2merge[word_b_id] = 1
                if word_new_id not in word2merge:
                    word2merge[word_new_id] = word2merge[word_a_id] + word2merge[word_b_id]

                if one_word_merges > 0 and word2merge[word_new_id] > one_word_merges:
                    # print(word_a, word_b, word_new)
                    continue

                update_last_word_map(word_a)
                update_last_word_map(word_b)

                if (word_a_id, word_b_id) in pair_set:
                    continue
                pair_set.add((word_a_id, word_b_id))
                self.bpe_codes.append((word_a_id, word_b_id))
                self.merge_list.append(word_new_id)

        # some hacking to deal with duplicates (only consider first instance)
        self.bpe_codes = dict([(code, i) for (i, code) in enumerate(self.bpe_codes)])

        self.separator = separator
        self.vocab = vocab

        self.glossaries = glossaries if glossaries else []
        self.glossaries_regex = re.compile('^({})$'.format('|'.join(glossaries))) if glossaries else None

        print('last word map size = {0}'.format(len(self.last_word_map)))
        print('bpe code size = {0}'.format(len(self.bpe_codes)))


    def process_line(self, line, dropout=0):
        """segment line, dealing with leading and trailing whitespace"""

        out = ""

        leading_whitespace = len(line)-len(line.lstrip('\r\n '))
        if leading_whitespace:
            out += line[:leading_whitespace]

        out += self.segment(line)

        trailing_whitespace = len(line)-len(line.rstrip('\r\n '))
        if trailing_whitespace and trailing_whitespace != len(line):
            out += line[-trailing_whitespace:]
        return out

    def segment(self, sentence):
        """segment single sentence (whitespace-tokenized string) with BPE encoding"""
        sentence = sentence.strip('\r\n ')
        words = sentence.split(' ')
        ids = [self.dictionary.index(word) for word in words]
        begin_of_word = self.encode_int(ids, version=self.version)
        assert len(words) == len(begin_of_word), "len(words)={0}, len(mask)={1}".format(len(words), len(begin_of_word))
        segments = []
        word = ""
        for i in range(len(words)):
            if begin_of_word[i] == 1 and i != 0:
                segments.append(word)
                word = ""
            word += words[i]
        if word != "":
            segments.append(word)
        s = ' '.join(segments)
        return ' '.join(segments)

    def _isolate_glossaries(self, word):
        word_segments = [word]
        for gloss in self.glossaries:
            word_segments = [out_segments for segment in word_segments
                                 for out_segments in isolate_glossary(segment, gloss)]
        return word_segments

    def get_begin_of_word(self, words, dict):
        ids = [dict.index(word) for word in words]
        begin_of_word = self.encode_int(ids, version=self.version, return_length=False)
        assert len(words) == len(begin_of_word)
        return begin_of_word

    def encode_int(self, orig, version=(0, 2), verbose=True, return_length=True):
        """Encode word based on list of BPE merge operations, which are applied consecutively
        """
        if len(self.merge_list) == 0:
            return [1] * len(orig)
        if len(orig) == 0:
            return []
        if version == (0, 2):  # more consistent handling of word-final segments
            # word = orig[:-1] + [self.dictionary.index(self.dictionary[orig[-1]] + '</w>')]
            # std_id = word[-1]
            new_id = self.last_word_map[orig[-1]] if orig[-1] in self.last_word_map else self.unk_index
            word = orig[:-1] + [new_id]
        else:
            raise NotImplementedError

        if verbose:
            global phrase_merge, word_merge, cross_merge, cross_dic
        left = list(range(-1, len(word)-1))
        right = list(range(1, len(word)+1))
        right[-1] = -1
        merge_word = list(word)

        heap = [(self.bpe_codes[(merge_word[i], merge_word[i + 1])], i, i + 1, merge_word[i], merge_word[i + 1])
                for i in range(len(merge_word) - 1)
                if (merge_word[i], merge_word[i + 1]) in self.bpe_codes]
        heapq.heapify(heap)

        while len(heap) > 0:
            _, word_a_id, word_b_id, word_a, word_b = heapq.heappop(heap)
            if word_a != merge_word[word_a_id] or word_b != merge_word[word_b_id]:
                continue
            bigram = merge_word[word_a_id], merge_word[word_b_id]

            bigram = self.merge_list[self.bpe_codes[bigram]]
            merge_word[word_a_id] = bigram
            # remove word b
            right[word_a_id] = right[word_b_id]
            if right[word_b_id] != -1:
                left[right[word_b_id]] = word_a_id
            left[word_b_id] = -1
            right[word_b_id] = -1
            merge_word[word_b_id] = ""

            if left[word_a_id] != -1:
                ia, ib = left[word_a_id], word_a_id
                pair = merge_word[ia], merge_word[ib]
                if pair in self.bpe_codes:
                    heapq.heappush(heap, (self.bpe_codes[pair], ia, ib, merge_word[ia], merge_word[ib]))
            if right[word_a_id] != -1:
                ia, ib = word_a_id, right[word_a_id]
                pair = merge_word[ia], merge_word[ib]
                if pair in self.bpe_codes:
                    heapq.heappush(heap, (self.bpe_codes[pair], ia, ib, merge_word[ia], merge_word[ib]))

        word_len = []
        p = 0
        while p != -1:
            if right[p] != -1:
                word_len.append(right[p] - p)
            else:
                word_len.append(len(right)-p)
            p = right[p]

        if return_length:
            return word_len
        begin_of_word = []
        for l in word_len:
            begin_of_word.append(1)
            begin_of_word.extend([0] * (l - 1))

        # don't print end-of-word symbols
        # if word[-1] == self.eos_index:
        #     begin_of_word = begin_of_word[:-1]

        return begin_of_word

    # def encode_int(self, orig, version=(0, 2)):
    #     """Encode word based on list of BPE merge operations, which are applied consecutively
    #     """
    #     if version == (0, 1):
    #         word = orig + [self.dictionary.index('</w>')]
    #     elif version == (0, 2):  # more consistent handling of word-final segments
    #         word = orig[:-1] + [self.dictionary.index(self.dictionary[orig[-1]] + '</w>')]
    #     else:
    #         raise NotImplementedError
    #     index = [(i, i) for i in range(len(word))]
    #
    #     global phrase_merge, word_merge, cross_merge, cross_dic
    #     while len(word) > 1:
    #
    #         # get list of symbol pairs; optionally apply dropout
    #         # pairs = [(self.bpe_codes[pair], i, pair) for (i, pair) in enumerate(zip(word, word[1:])) if
    #         #          pair in self.bpe_codes]
    #
    #         min_merge_index = -1
    #         min_index = -1
    #         bigram = None
    #         for i in range(len(word)-1):
    #             pair = word[i], word[i+1]
    #             if pair not in self.bpe_codes:
    #                 continue
    #             if min_merge_index == -1 or self.bpe_codes[pair] < min_merge_index:
    #                 bigram = pair
    #                 min_merge_index = self.bpe_codes[pair]
    #                 min_index = i
    #
    #         if bigram is None:
    #             break
    #
    #         # get first merge operation in list of BPE codes
    #         # bigram = min(pairs)[2]
    #         text = self.dictionary[bigram[0]], self.dictionary[bigram[1]]
    #         if text[0][0] in ["▁", ',', '.', ':', ';', "'", ')', '"']:
    #             if text[1].startswith("▁"):
    #                 phrase_merge += 1
    #             else:
    #                 word_merge += 1
    #         else:
    #             if text[1].startswith("▁"):
    #                 cross_merge += 1
    #                 if text[0] not in cross_dic:
    #                     cross_dic[text[0]] = 0
    #                 cross_dic[text[0]] += 1
    #             else:
    #                 word_merge += 1
    #
    #         bigram = self.merge_map[bigram]
    #         word = word[:min_index] + [bigram] + word[min_index + 2:]
    #         index = index[:min_index] + [(index[min_index][0], index[min_index+1][1])] + index[min_index + 2:]
    #
    #     # don't print end-of-word symbols
    #     if self.dictionary[word[-1]] == '</w>':
    #         index = index[:-1]
    #
    #     begin_of_word = []
    #     for (s, t) in index:
    #         begin_of_word.extend([1] + [0] * (t-s))
    #     return begin_of_word

def create_parser(subparsers=None):

    if subparsers:
        parser = subparsers.add_parser('apply-bpe',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="learn BPE-based word segmentation")
    else:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="learn BPE-based word segmentation")

    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Input file (default: standard input).")
    parser.add_argument(
        '--codes', '-c', type=argparse.FileType('r'), metavar='PATH',
        required=True,
        help="File with BPE codes (created by learn_bpe.py).")
    parser.add_argument(
        '--merges', '-m', type=int, default=-1,
        metavar='INT',
        help="Use this many BPE operations (<= number of learned symbols)"+
             "default: Apply all the learned merge operations")
    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w'), default=sys.stdout,
        metavar='PATH',
        help="Output file (default: standard output)")
    parser.add_argument(
        '--separator', '-s', type=str, default='@@', metavar='STR',
        help="Separator between non-final subword units (default: '%(default)s'))")
    parser.add_argument(
        '--vocabulary', type=argparse.FileType('r'), default=None,
        metavar="PATH",
        help="Vocabulary file (built with get_vocab.py). If provided, this script reverts any merge operations that produce an OOV.")
    parser.add_argument(
        '--dictionary', type=argparse.FileType('r'), default=None,
        metavar="PATH",
        help="dictionary file")
    parser.add_argument(
        '--vocabulary-threshold', type=int, default=None,
        metavar="INT",
        help="Vocabulary threshold. If vocabulary is provided, any word with frequency < threshold will be treated as OOV")
    parser.add_argument(
        '--dropout', type=float, default=0,
        metavar="P",
        help="Dropout BPE merge operations with probability P (Provilkov et al., 2019). Use this on training data only.")
    parser.add_argument(
        '--glossaries', type=str, nargs='+', default=None,
        metavar="STR",
        help="Glossaries. Words matching any of the words/regex provided in glossaries will not be affected "+
             "by the BPE (i.e. they will neither be broken into subwords, nor concatenated with other subwords. "+
             "Can be provided as a list of words/regex after the --glossaries argument. Enclose each regex in quotes.")

    return parser


if __name__ == '__main__':

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    newdir = os.path.join(currentdir, 'subword_nmt')
    if os.path.isdir(newdir):
        warnings.simplefilter('default')
        warnings.warn(
            "this script's location has moved to {0}. This symbolic link will be removed in a future version. Please point to the new location, or install the package and use the command 'subword-nmt'".format(newdir),
            DeprecationWarning
        )

    # python 2/3 compatibility
    if sys.version_info < (3, 0):
        sys.stderr = codecs.getwritedr('UTF-8')(sys.stderr)
        sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
        sys.stdin = codecs.getreader('UTF-8')(sys.stdin)
    else:
        sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', write_through=True, line_buffering=True)

    parser = create_parser()
    args = parser.parse_args()

    # read/write files as UTF-8
    args.codes = codecs.open(args.codes.name, encoding='utf-8')
    if args.input.name != '<stdin>':
        args.input = codecs.open(args.input.name, encoding='utf-8')
    if args.output.name != '<stdout>':
        args.output = codecs.open(args.output.name, 'w', encoding='utf-8')
    if args.vocabulary:
        args.vocabulary = codecs.open(args.vocabulary.name, encoding='utf-8')

    if args.vocabulary:
        vocabulary = read_vocabulary(args.vocabulary, args.vocabulary_threshold)
    else:
        vocabulary = None

    dictionary = Dictionary.load(args.dictionary)
    if sys.version_info < (3, 0):
        args.separator = args.separator.decode('UTF-8')
        if args.glossaries:
            args.glossaries = [g.decode('UTF-8') for g in args.glossaries]


    bpe = BPE(args.codes, args.merges, args.separator, vocabulary, args.glossaries, dictionary=dictionary)

    for line in args.input:
        args.output.write(bpe.process_line(line, args.dropout))

    print("word merge:", word_merge)
    print("phrase merge:", phrase_merge)
    print("cross merge:", cross_merge)
    print("cross dic:", sorted([(value, key) for key, value in cross_dic.items()], reverse=True))
