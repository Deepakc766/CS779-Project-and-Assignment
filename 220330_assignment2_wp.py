
# import argparse
# import collections
# import math
# import sys
# pad_list = ["<pad>", "<unk>", "<s>", "</s>"]
# class wordPiece_tokenizer:
#     def __init__(self, vocab_size):
#         self.vocab_size = vocab_size
#         self.special_token = pad_list
#         self.vocabulary = []
#         self.vocabulary_set = set()
#         self.trained = False

#     def initialse_vocab(self, corpus_tokens):
#         unique_chars=set("".join(corpus_tokens))
#         all_chars=sorted(unique_chars)
#         vocab=self.special_token+all_chars

#         for char in all_chars:
#             vocab.append("##"+char)
#         self.vocabulary=vocab
#         self.vocabulary_set=set(vocab)
#         tokenised_corpus = []
#         for word in corpus_tokens:
#             tokens = []
#             for i, ch in enumerate(word):
#                 if i == 0:
#                     tokens.append(ch)
#                 else:
#                     tokens.append("##" + ch)
#             tokenised_corpus.append(tokens)

#         return tokenised_corpus

#     def get_count_pairs(self, tokenised_corpus):
#         pair_freq_count = collections.Counter()
#         for token in tokenised_corpus:
#             for i in range(len(token) - 1):
#                 pair = (token[i], token[i + 1])  
#                 pair_freq_count[pair] += 1
#         return pair_freq_count

#     def compute_likelihood(self, f_new, N):
#         if f_new <= 0 or N <= 0:
#             return -float("inf")
#         return -f_new * math.log(f_new / N)

#     def merge_pairs(self, tokenised_corpus, best_pair):
#         token1, token2 = best_pair
#         if token1.startswith("##"):
#             token = token1[2:]
#             merged_token = "##" + token + token2.replace("##", "")
#         else:
#             merged_token = token1 + token2.replace("##", "")

#         new_tokenised_corpus = []
#         for token in tokenised_corpus:
#             i = 0
#             new_token = []
#             while i < len(token):
#                 if i < len(token) - 1 and (token[i], token[i + 1]) == best_pair:
#                     new_token.append(merged_token)
#                     i += 2 
#                 else:
#                     new_token.append(token[i])
#                     i += 1
#             new_tokenised_corpus.append(new_token)

#         return new_tokenised_corpus, merged_token

#     def training(self, corpus_tokens):
#         """
#         Training merges until vocab_size (including 4 special tokens) reached.
#         """
#         tokenised_corpus = self.initialse_vocab(corpus_tokens)
#         N = sum(len(token) for token in tokenised_corpus)
#         while len(self.vocabulary) < self.vocab_size:
#             pair_counts = self.get_count_pairs(tokenised_corpus)
#             if not pair_counts:
#                 break

#             scored_pairs = []
#             for pair, freq in pair_counts.items():
#                 f_new = freq
#                 delta_L = self.compute_likelihood(f_new, N)

#                 token1, token2 = pair
#                 if token1.startswith("##"):
#                     merge_token_name = "##" + token1[2:] + token2.replace("##", "")
#                 else:
#                     merge_token_name = token1 + token2.replace("##", "")

#                 scored_pairs.append((delta_L, merge_token_name, pair, freq))
#             scored_pairs.sort(key=lambda x: (-x[0], x[1]))
#             best_delta, best_token_name, best_pair, best_freq = scored_pairs[0]

#             tokenised_corpus, merged_token = self.merge_pairs(tokenised_corpus, best_pair)

#             if merged_token not in self.vocabulary_set:
#                 self.vocabulary.append(merged_token)
#                 self.vocabulary_set.add(merged_token)

#             N -= best_freq
#             if N <= 0:
#                 break

#             # safety check
#             if len(self.vocabulary) >= self.vocab_size:
#                 break

#         self.trained = True
#         return self.vocabulary

#     def tokenise(self, texts):
#         if not self.trained:
#             print("Tokenizer not trained, run training first.")
#             return []

#         words = texts.strip().split()
#         output_tokens = []
#         for word in words:
#             start = 0
#             word_tokens = []
#             L = len(word)
#             while start < L:
#                 end = L
#                 found = None
#                 while end > start:
#                     sub = word[start:end]
#                     if start == 0:
#                         # try plain then '##' variant
#                         if sub in self.vocabulary_set:
#                             found = sub
#                             break
#                         if ("##" + sub) in self.vocabulary_set:
#                             found = "##" + sub
#                             break
#                     else:
#                         # try continuation first
#                         if ("##" + sub) in self.vocabulary_set:
#                             found = "##" + sub
#                             break
#                         if sub in self.vocabulary_set:  # fallback
#                             found = sub
#                             break
#                     end -= 1

#                 if found:
#                     word_tokens.append(found)
#                     start = end
#                 else:
#                     word_tokens.append("<unk>")
#                     start += 1

#             output_tokens.extend(word_tokens)

#         return output_tokens

#     def detokenise(self, tokens):
#         words = []
#         current_word = ""
#         for token in tokens:
#             # ignore some special tokens but keep <unk>
#             if token in {"<pad>", "<s>", "</s>"}:
#                 continue
#             if token == "<unk>":
#                 if current_word:
#                     words.append(current_word)
#                     current_word = ""
#                 words.append("<unk>")
#                 continue
#             if token.startswith("##"):
#                 current_word += token[2:]
#             else:
#                 if current_word:
#                     words.append(current_word)
#                 current_word = token
#         if current_word:
#             words.append(current_word)
#         return " ".join(words)

#     @staticmethod
#     def read_corpus(path=""):
#         with open(path, "r", encoding="utf-8") as file:
#             text = file.read().strip()
#         words = text.split()
#         return words


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--train", type=str, required=True,
#                         help="Path to input text file to train your tokenizer")
#     parser.add_argument("--input", type=str, required=True,
#                         help="Path to input sample-text.txt for tokenization and detokenization")
#     parser.add_argument("--vocab_size", type=int, required=True,
#                         help="Target vocab size (incl. specials)")
#     args = parser.parse_args()

#     print("reading training corpus...")
#     train_words = wordPiece_tokenizer.read_corpus(args.train)

#     print(f"initializing tokenizer with vocab_size={args.vocab_size}...")
#     tokenizer = wordPiece_tokenizer(args.vocab_size)

#     print("training tokenizer...")
#     vocabulary = tokenizer.training(train_words)
#     print(f"Training complete. Final vocab size = {len(vocabulary)}")


#     vocab_file = f"220330_assignment2_wp_vocab_{args.vocab_size}.txt"
#     with open(vocab_file, "w", encoding="utf-8") as file:
#         for token in vocabulary:
#             file.write(token + "\n")
#     print(f"Vocabulary saved to {vocab_file}")


#     print("Reading input file for tokenization...")
#     with open(args.input, "r", encoding="utf-8") as file:
#         text = file.read()

#     print("Tokenizing input text...")
#     tokenised_text = tokenizer.tokenise(text)

#     token_file = f"220330_assignment2_wp_tokens.txt"
#     with open(token_file, "w", encoding="utf-8") as file:
#         for token in tokenised_text:
#             file.write(token + "\n")
#     print(f"Tokenized text saved to {token_file}")

#     # Detokenize
#     print("Detokenizing...")
#     detoken_text = tokenizer.detokenise(tokenised_text)

#     detoken_file = f"220330_assignment2_wp_detokenized.txt"
#     with open(detoken_file, "w", encoding="utf-8") as file:
#         file.write(detoken_text)
#     print(f"Detokenized text saved to {detoken_file}")

#     print("Done")


# if __name__ == "__main__":
#     main()




"""above is my brute force of WordPiece tokenizer implementation using simple lists and dictionaries where we are recomputing the pair frequencies and delta-likelihood scores from scratch after each merge. This is inefficient for large corpora and vocab sizes.
because we taking too much  time due to recomputing pair frequencies and delta-likelihood scores from scratch after each merge..
so in above time complexity is :
O(V^2 * N) where V is vocab size and N is corpus size.
# Below is optimized implementation using linked-list structure for efficient updates and lazy heap management and trie for fast tokenization.
in my optimized approach I have used :
linkedlist for efficient updates of token sequences after merges so our time complexity is reduced to O(V * N log V) where V is vocab size and N is corpus size.
Trie for fast tokenization of input text and time comlexity is O(M) where M is length of input text.
lazy heap management to avoid recomputing all pair scores after each merge
This reduces the number of score recomputations and speeds up the merging process.
so our final time complexity is O(V * N log V + M) where V is vocab size, N is corpus size and M is length of input text.
so out time complexity got reduced from O(V^2 * N) to O(V * N log V + M)
which is much more efficient for large corpora and vocab sizes like 25 mb txt file and Millions of tokens.



My optimized ALgo is :
1.Make a Trie data structure for fast tokenization.
2.Initialize the vocabulary with special tokens=["<pad>", "<unk>", "<s>", "</s>"], single characters, and '##' prefixed characters.
3.Create a linked-list structure to represent each distinct tokenized word in the corpus, with nodes for each token and pointers to previous and next nodes.
4.Maintain a mapp of token pairs to their occurrences in the linked-list nodes and their frequency counts.
5.used  max-heap (priority queue) to efficiently retrieve the token pair with the highest delta-likelihood score for merging.
6.When merging a token pair, update the linked-list structure to replace occurrences of the pair with a new merged token node, and adjust pointers accordingly.
7.Update the frequency counts of affected token pairs and push their updated scores to the heap (lazy update).
8.Repeat steps 5-7 until the vocabulary size reaches the desired limit.
Create 3 output files:
- Vocabulary file: Contains the final vocabulary of tokens after training.
- Tokenized file: Contains the tokenized version of the input text using the trained tokenizer.
- Detokenized file: Contains the detokenized text reconstructed from the tokenized version.

My sample output for my extracted tex file on 26 mb is(directly taken from my terminal):
initializing from corpus
[init] nodes=2550196, tokens=4568, N=21801968, time=1.88s
Building initial heap (delta-likelihood scores) 
Planned merges: 432 (current vocab 4568)
Merging (delta-L): 100%|█████████████████████████████████████████████████████████████████████████| 432/432 [00:05<00:00, 75.79it/s]
Training finished. merges performed 432. vocab size 5000. time 7.64s
Saved vocab to 220330_assignment2_wp_vocab_5000.txt
Saved tokens to 220330_assignment2_wp_tokens.txt
Saved detokenized to 220330_assignment2_wp_detokenized.txt
Timings: train 7.64s, tokenize+IO 10.05s, total 17.69s

file name : 220330_assignment2_wp.py
Run script :python 220330_assignment2_wp.py --train extracted.txt --input extracted.txt --vocab_size 5000
"""


import argparse
import math
import time
import heapq
from collections import Counter, defaultdict
from tqdm import tqdm

SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]
#we are implementing WordPiece tokenizer with delta-likelihood based merges using linked-list structure for efficient updates and lazy heap management and trie for fast tokenization.
class TrieNode:
    __slots__ = ("children","token")
    def __init__(self):
        self.children = {}
        self.token = None

class WordPieceDelta:
    def __init__(self, vocab_size):
        if vocab_size <= len(SPECIAL_TOKENS):
            raise ValueError("vocab_size must be > number of special tokens.")
        self.vocab_size = vocab_size

        # token string <-> id
        self.token2id = {}
        self.id2token = []

        # flat linked-list node arrays
        self.token_ids = []   # token id at node idx
        self.prev = []        # prev node idx or -1
        self.next = []        # next node idx or -1
        self.alive = []    
        self.weight = []      # integer weight (count of that node's word form)


        self.pair_occ = defaultdict(list)
        self.pair_freq = {}   # pair -> int

        self.token_counts = defaultdict(int)
        self.N = 0
        self.next_node_id = 0
        self.trie_root = None
        self.trained = False


    def _ensure_token(self, tok):
        tid = self.token2id.get(tok)
        if tid is None:
            tid = len(self.id2token)
            self.id2token.append(tok)
            self.token2id[tok] = tid
        return tid

    @staticmethod
    def _word_to_initial_tokens(word):
        if not word:
            return []
        toks = [word[0]]
        for ch in word[1:]:
            toks.append("##" + ch)
        return toks

    @staticmethod
    def read_word_counts(path):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        words = text.split()
        return Counter(words)

    def initialize(self, train_path):
        t0 = time.perf_counter()
        word_counts = self.read_word_counts(train_path)
        distinct = {}
        for w, c in word_counts.items():
            toks = tuple(self._word_to_initial_tokens(w))
            distinct[toks] = distinct.get(toks, 0) + c

        unique_chars = set()
        for toks in distinct.keys():
            for tk in toks:
                if tk.startswith("##"):
                    unique_chars.add(tk[2:])
                else:
                    unique_chars.add(tk)
        for s in SPECIAL_TOKENS:
            self._ensure_token(s)
        for ch in sorted(unique_chars):
            self._ensure_token(ch)
        for ch in sorted(unique_chars):
            self._ensure_token("##" + ch)

        # create linked-list nodes for each distinct tokenized word (weighted)
        for toks, cnt in distinct.items():
            if not toks:
                continue
            node_ids = []
            for tk in toks:
                tid = self._ensure_token(tk)
                nid = self.next_node_id
                self.next_node_id += 1
                self.token_ids.append(tid)
                self.prev.append(-1)
                self.next.append(-1)
                self.alive.append(True)
                self.weight.append(cnt)
                node_ids.append(nid)
                # update token_counts
                self.token_counts[tid] += cnt
                self.N += cnt
            # link nodes and record pair occurrences
            for i in range(len(node_ids) - 1):
                a = node_ids[i]
                b = node_ids[i+1]
                self.next[a] = b
                self.prev[b] = a
                pair = (self.token_ids[a], self.token_ids[b])
                self.pair_occ[pair].append(a)
                self.pair_freq[pair] = self.pair_freq.get(pair, 0) + cnt
        t1 = time.perf_counter()
        print(f"[init] nodes={self.next_node_id}, tokens={len(self.id2token)}, N={self.N}, time={t1-t0:.2f}s")


    @staticmethod
    def compute_likelihood(f_new, N):
        if f_new <= 0 or N <= 0:
            return -float("inf")
        return -f_new * math.log(f_new / N)

    def build_heap(self):
        self.heap = []
        self.heap_counter = 0
        for pair, freq in self.pair_freq.items():
            if freq > 0:
                score = self.compute_likelihood(freq, self.N)
                heapq.heappush(self.heap, (-score, self.heap_counter, pair))
                self.heap_counter += 1

    def _valid_occ(self, nid, pair):
        if nid < 0 or nid >= self.next_node_id:
            return False
        if not self.alive[nid]:
            return False
        nxt = self.next[nid]
        if nxt == -1 or not self.alive[nxt]:
            return False
        return (self.token_ids[nid], self.token_ids[nxt]) == pair


    def _merge_occurrence(self, nid, new_tid):
        """
        merge pair at nid (nid, nid.next) into single new node with token id new_tid.
        Returns merged node id and weight w (count of this occurrence).
        also updates pair_freq, token_counts, N accordingly.
        """
        if not self._valid_occ(nid, (self.token_ids[nid], self.token_ids[self.next[nid]])):
            return None, 0
        left = nid
        right = self.next[nid]
        w = self.weight[left]

        prev_n = self.prev[left]
        next_n = self.next[right]
        mid = self.next_node_id
        self.next_node_id += 1
        self.token_ids.append(new_tid)
        self.prev.append(prev_n if prev_n != -1 else -1)
        self.next.append(next_n if next_n != -1 else -1)
        self.alive.append(True)
        self.weight.append(w)

        # link neighbors to new node
        if prev_n != -1:
            self.next[prev_n] = mid
        if next_n != -1:
            self.prev[next_n] = mid

        # mark old nodes dead
        self.alive[left] = False
        self.alive[right] = False
        left_tid = self.token_ids[left]
        right_tid = self.token_ids[right]
        self.token_counts[left_tid] -= w
        if self.token_counts[left_tid] <= 0:
            self.token_counts.pop(left_tid, None)
        self.token_counts[right_tid] -= w
        if self.token_counts[right_tid] <= 0:
            self.token_counts.pop(right_tid, None)
        self.token_counts[new_tid] = self.token_counts.get(new_tid, 0) + w

        self.N -= w

        old_pair = (left_tid, right_tid)
        prev_val = self.pair_freq.get(old_pair, 0)
        if prev_val > 0:
            self.pair_freq[old_pair] = prev_val - w
            if self.pair_freq[old_pair] <= 0:
                self.pair_freq.pop(old_pair, None)
        # left-prev removed (prev,left) loses w
        if prev_n != -1:
            prev_pair = (self.token_ids[prev_n], left_tid)
            pv = self.pair_freq.get(prev_pair, 0)
            if pv > 0:
                self.pair_freq[prev_pair] = pv - w
                if self.pair_freq[prev_pair] <= 0:
                    self.pair_freq.pop(prev_pair, None)
        # right-next removed (right,next)
        if next_n != -1:
            rn_pair = (right_tid, self.token_ids[next_n])
            rv = self.pair_freq.get(rn_pair, 0)
            if rv > 0:
                self.pair_freq[rn_pair] = rv - w
                if self.pair_freq[rn_pair] <= 0:
                    self.pair_freq.pop(rn_pair, None)

        if prev_n != -1:
            new_left_pair = (self.token_ids[prev_n], new_tid)
            self.pair_freq[new_left_pair] = self.pair_freq.get(new_left_pair, 0) + w
            self.pair_occ[new_left_pair].append(prev_n)
        if next_n != -1:
            new_right_pair = (new_tid, self.token_ids[next_n])
            self.pair_freq[new_right_pair] = self.pair_freq.get(new_right_pair, 0) + w
            self.pair_occ[new_right_pair].append(mid)

        return mid, w


    def train(self, train_path):
        t_start = time.perf_counter()
        print("initializing from corpus :")
        self.initialize(train_path)

        print("Building initial heap (delta-likelihood scores):")
        self.build_heap()

        merges_needed = max(0, self.vocab_size - len(self.id2token))
        print(f"Planned merges: {merges_needed} (current vocab {len(self.id2token)})")

        performed = 0
        pbar = tqdm(total=merges_needed, desc="Merging (delta-L)")
        while performed < merges_needed and self.heap:
            negscore, _, pair = heapq.heappop(self.heap)
            popped_score = -negscore

            f_new = self.pair_freq.get(pair, 0)
            if f_new <= 0:
                continue
            current_score = self.compute_likelihood(f_new, self.N)

            if abs(current_score - popped_score) > 1e-8:
                heapq.heappush(self.heap, (-current_score, self.heap_counter, pair))
                self.heap_counter += 1
                continue
            tid1, tid2 = pair
            tok1 = self.id2token[tid1]
            tok2 = self.id2token[tid2]

            if tok1.startswith("##"):
                merged_tok = "##" + tok1[2:] + tok2.replace("##", "")
            else:
                merged_tok = tok1 + tok2.replace("##", "")

            merged_tid = self.token2id.get(merged_tok)
            if merged_tid is None:
                merged_tid = self._ensure_token(merged_tok)

            # f_new is sum of weights for pair - we need to iterate occurrences to merge them
            occ_list = list(self.pair_occ.get(pair, []))  # snapshot
            # iterate occurrences; validate each before merging
            for occ in occ_list:
                # stop if the pair no longer has frequency (all merged)
                if pair not in self.pair_freq:
                    break
                if not self._valid_occ(occ, pair):
                    continue
                # merge this occurrence
                mid, w = self._merge_occurrence(occ, merged_tid)
                if mid is None:
                    continue
                # after merging one occurrence, neighboring pairs got updated in _merge_occurrence
                # push their updated scores to heap (lazy)
                prev_n = self.prev[mid]
                next_n = self.next[mid]
                if prev_n != -1:
                    left_pair = (self.token_ids[prev_n], self.token_ids[mid])
                    if left_pair in self.pair_freq:
                        score = self.compute_likelihood(self.pair_freq[left_pair], self.N)
                        heapq.heappush(self.heap, (-score, self.heap_counter, left_pair))
                        self.heap_counter += 1
                if next_n != -1:
                    right_pair = (self.token_ids[mid], self.token_ids[next_n])
                    if right_pair in self.pair_freq:
                        score = self.compute_likelihood(self.pair_freq[right_pair], self.N)
                        heapq.heappush(self.heap, (-score, self.heap_counter, right_pair))
                        self.heap_counter += 1

            # cleanup: remove pair entries (they should be zeroed)
            self.pair_occ.pop(pair, None)
            self.pair_freq.pop(pair, None)

            performed += 1
            pbar.update(1)
        pbar.close()

        t_end = time.perf_counter()
        print(f"Training finished. merges performed {performed}. vocab size {len(self.id2token)}. time {t_end - t_start:.2f}s")

        self._build_trie()
        self.trained = True


    def _build_trie(self):
        root = TrieNode()
        for tok in self.id2token:
            node = root
            for ch in tok:
                if ch not in node.children:
                    node.children[ch] = TrieNode()
                node = node.children[ch]
            node.token = tok
        self.trie_root = root


    def tokenize(self, text):
        if not self.trained:
            raise RuntimeError("Tokenizer not trained. Call train() first.")
        out = []
        for w in text.strip().split():
            out.extend(self._tokenize_word(w))
        return out

    def _tokenize_word(self, word):
        if word in self.token2id:
            return [word]
        res = []
        start = 0
        L = len(word)
        while start < L:
            node = self.trie_root
            longest = None
            longest_len = 0
            i = start
            while i < L:
                ch = word[i]
                if ch not in node.children:
                    break
                node = node.children[ch]
                if node.token is not None:
                    cand = node.token
                    if (start == 0 and not cand.startswith("##")) or (start > 0 and cand.startswith("##")):
                        longest = cand
                        longest_len = len(cand.replace("##", ""))
                i += 1
            if longest is None:
                # fallback single char / ##char
                if start == 0:
                    sub = word[start]
                    res.append(sub if sub in self.token2id else "<unk>")
                else:
                    sub = "##" + word[start]
                    res.append(sub if sub in self.token2id else "<unk>")
                start += 1
            else:
                res.append(longest)
                start += longest_len
        final = []
        for i, t in enumerate(res):
            if i > 0 and not t.startswith("##") and t != "<unk>":
                final.append("##" + t)
            else:
                final.append(t)
        return final



    def detokenize(self, tokens):
        words = []
        cur = ""
        for t in tokens:
            if t in {"<pad>", "<s>", "</s>"}:
                continue
            if t == "<unk>":
                if cur:
                    words.append(cur); cur = ""
                words.append("<unk>")
                continue
            if t.startswith("##"):
                cur += t[2:]
            else:
                if cur:
                    words.append(cur)
                cur = t
        if cur:
            words.append(cur)
        return " ".join(words)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True, help="Path to training file")
    parser.add_argument("--input", type=str, required=True, help="Path to input file")
    parser.add_argument("--vocab_size", type=int, required=True, help="Target vocab size (incl. specials)")
    args = parser.parse_args()

    rollno = "220330"   # for uoutput file naming 

    tok = WordPieceDelta(args.vocab_size)
    t0 = time.perf_counter() # start time
    tok.train(args.train)
    t1 = time.perf_counter()# after training to calculate how much time ity took 

    # save vocab file with 220330 roll number 
    vocab_file = f"{rollno}_assignment2_wp_vocab_{args.vocab_size}.txt"
    with open(vocab_file, "w", encoding="utf-8") as f:
        for token in tok.id2token:
            f.write(token + "\n")
    print(f"Saved vocab to {vocab_file}")

    # tokenize input file
    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = tok.tokenize(text)
    tokens_file = f"{rollno}_assignment2_wp_tokens.txt"
    with open(tokens_file, "w", encoding="utf-8") as f:
        for t in tokens:
            f.write(t + "\n")
    print(f"Saved tokens to {tokens_file}")

    detok_text = tok.detokenize(tokens)
    detok_file = f"{rollno}_assignment2_wp_detokenized.txt"
    with open(detok_file, "w", encoding="utf-8") as f:
        f.write(detok_text)
    print(f"Saved detokenized to {detok_file}")

    t2 = time.perf_counter()
    print(f"Timings: train {t1-t0:.2f}s, tokenize+IO {t2-t1:.2f}s, total {t2-t0:.2f}s")

if __name__ == "__main__":
    main()
