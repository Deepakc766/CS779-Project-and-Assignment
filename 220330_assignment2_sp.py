
"""in this questuon,I have implemented sentence piece model using bpe algorithm with linked list approach for efficient merging of tokens.
I have re utilise the code of BPE that I have done in first part and rewritten whole code from scratch in this .py file 
Major changes taht I have done is:
1.in this code I have used linked list approach for merging of tokens which is more efficient than array based approach.
and its time complexity is O(n) where n is length of the text if we have used array based approach then its time complexity is O(n*m) where m is number of merges.
2.I have implemented Trie data structure for tokenization which is more efficient than simple dictionary based approach.
and its time complexity is O(m) where m is length of the longest token in the vocabulary.
3.I have added special tokens <pad>,<unk>,<s>,</s> in the vocabulary.
4.I have done normalisation of text using unicodedata library.

My approach is :
1. Read the training text and normalise it.
2. Initialize the vocabulary with special tokens and unique characters from the training text.
3. Build linked list nodes for each word in the training textto rduce the time complexity of merging tokens
4. Collect occurrences of adjacent token pairs using a dictionary of sets.
5. Use a max-heap to efficiently get the most frequent token pair instead of list and sortiq which is inefficient and takes O(n log n) time complexity each time we need to get the most frequent pair.
6. Merge the most frequent token pair and update the linked list nodes and occurrences dictionary.

"""

"""
Simple SentencePiece-style BPE using occurrence-linked nodes + Trie.
Whitespace marker: '_' (underscore)
Progress shown with tqdm.
"""
import argparse
import unicodedata
import heapq
import time
from collections import defaultdict
from tqdm import tqdm

Rollnumber = "220330"
SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]

class Node:
    __slots__ = ("tid", "prev", "next", "alive")
    def __init__(self, tid):
        self.tid = tid
        self.prev = None
        self.next = None
        self.alive = True

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.casefold()
    s = " ".join(s.split())
    return s.replace(" ", "_")

def initialize_vocab_and_ids(text_norm: str):
    id_to_token = {}
    tok2id = {}
    idx = 0
    for t in SPECIAL_TOKENS:
        id_to_token[idx] = t
        tok2id[t] = idx
        idx += 1
    for ch in sorted(set(text_norm)):
        if ch in tok2id:
            continue
        id_to_token[idx] = ch
        tok2id[ch] = idx
        idx += 1
    next_id = idx
    return id_to_token, tok2id, next_id

def build_corpus_nodes(text_norm: str, tok2id):
    parts = text_norm.split("_")
    heads = []
    for i, part in enumerate(parts):
        seg = part if i == 0 else "_" + part
        if not seg:
            continue
        head = None
        prev = None
        for ch in seg:
            tid = tok2id.get(ch, tok2id.get("<unk>"))
            node = Node(tid)
            if prev is None:
                head = node
            else:
                prev.next = node
                node.prev = prev
            prev = node
        if head:
            heads.append(head)
    return heads

def collect_pairs(heads):
    pair2occ = defaultdict(set)
    for head in heads:
        node = head
        while node and node.next:
            pair2occ[(node.tid, node.next.tid)].add(node)
            node = node.next
    return pair2occ

def push_pair_to_heap(heap, pair2occ, id_to_token, pair):
    occ = pair2occ.get(pair)
    if not occ or len(occ) <= 0:
        return
    ltid, rtid = pair
    heapq.heappush(heap, (-len(occ), id_to_token[ltid], id_to_token[rtid], pair))

def pop_best_pair(heap, pair2occ):
    while heap:
        negcnt, lstr, rstr, pair = heapq.heappop(heap)
        if len(pair2occ.get(pair, ())) == -negcnt and len(pair2occ[pair]) > 0:
            return pair, len(pair2occ[pair])
    return None, 0

def merge_tokens(pair, pair2occ, id_to_token, tok2id, next_id, heap):
    ltid, rtid = pair
    new_tok = id_to_token[ltid] + id_to_token[rtid]
    new_id = next_id
    id_to_token[new_id] = new_tok
    tok2id[new_tok] = new_id
    next_id += 1
    occ_nodes = list(pair2occ.get(pair, ()))
    pair2occ[pair].clear()
    for left_node in occ_nodes:
        if not left_node.alive:
            continue
        right_node = left_node.next
        if not right_node or not right_node.alive:
            continue
        if left_node.tid != ltid or right_node.tid != rtid:
            continue
        prev_node, next_node = left_node.prev, right_node.next
        if prev_node and prev_node.alive:
            pair2occ.get((prev_node.tid, left_node.tid), set()).discard(prev_node)
        if next_node and next_node.alive:
            pair2occ.get((right_node.tid, next_node.tid), set()).discard(right_node)
        left_node.tid = new_id
        right_node.alive = False
        left_node.next = next_node
        if next_node:
            next_node.prev = left_node
        if prev_node and prev_node.alive:
            pair2occ[(prev_node.tid, left_node.tid)].add(prev_node)
            push_pair_to_heap(heap, pair2occ, id_to_token, (prev_node.tid, left_node.tid))
        if next_node and next_node.alive:
            pair2occ[(left_node.tid, next_node.tid)].add(left_node)
            push_pair_to_heap(heap, pair2occ, id_to_token, (left_node.tid, next_node.tid))
    return new_tok, new_id, next_id

def train_sp(train_text_norm: str, vocab_size: int, time_debug=False):
    id_to_token, tok2id, next_id = initialize_vocab_and_ids(train_text_norm)
    heads = build_corpus_nodes(train_text_norm, tok2id)
    pair2occ = collect_pairs(heads)
    current_vocab_size = len(id_to_token)
    target_merges = vocab_size - current_vocab_size
    if target_merges <= 0:
        print(f"Current vocab size {current_vocab_size} >= target {vocab_size}, no merges needed.")
        return id_to_token, tok2id, [], heads
    print(f"Initial vocab size: {current_vocab_size}, target: {vocab_size}, merges to do: {target_merges}")
    heap = []
    for pair in pair2occ:
        push_pair_to_heap(heap, pair2occ, id_to_token, pair)
    merges = []
    pbar = tqdm(total=target_merges, desc="Merging pairs")
    start_time = time.time()
    while len(id_to_token) < vocab_size and heap:
        pair, count = pop_best_pair(heap, pair2occ)
        if not pair or count <= 1:
            break
        new_tok, new_id, next_id = merge_tokens(pair, pair2occ, id_to_token, tok2id, next_id, heap)
        merges.append((pair, new_tok, new_id))
        pbar.update(1)
        if time_debug and len(merges) % 500 == 0:
            print(f"[DEBUG] merges={len(merges)} vocab={len(id_to_token)} time={time.time()-start_time:.2f}s")
    pbar.close()
    return id_to_token, tok2id, merges, heads

class TrieNode:
    def __init__(self):
        self.children = {}
        self.tok_id = None
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    def insert(self, token_str, token_id):
        node = self.root
        for ch in token_str:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True
        node.tok_id = token_id
    def longest_match(self, s, start=0):
        node = self.root
        last_id = None
        last_pos = start
        pos = start
        while pos < len(s) and s[pos] in node.children:
            node = node.children[s[pos]]
            pos += 1
            if node.is_end:
                last_id = node.tok_id
                last_pos = pos
        return last_id, last_pos

def build_trie_from_id_to_token(id_to_token, vocab_size):
    trie = Trie()
    for i in range(min(vocab_size, max(id_to_token.keys()) + 1)):
        tok = id_to_token.get(i)
        if not tok or tok in SPECIAL_TOKENS:
            continue
        trie.insert(tok, i)
    return trie

def tokenise(norm_text: str, trie: Trie, tok2id: dict):
    token_ids = []
    i = 0
    while i < len(norm_text):
        tid, pos = trie.longest_match(norm_text, i)
        if tid is not None:
            token_ids.append(tid)
            i = pos
        else:
            ch = norm_text[i]
            tid_single = tok2id.get(ch, tok2id.get("<unk>", 1))
            token_ids.append(tid_single)
            i += 1
    return token_ids

def detokenise(token_ids, id_to_token):
    parts = []
    special_set = set(SPECIAL_TOKENS)
    for tid in token_ids:
        tok = id_to_token.get(tid)
        if not tok or tok in special_set:
            continue
        parts.append(tok)
    return "".join(parts).replace("_", " ")

def save_vocab_file(id_to_token, vocab_size, rollno):
    fname = f"{rollno}_assignment2_sp_vocab_{vocab_size}.txt"
    with open(fname, "w", encoding="utf-8") as fw:
        for i in range(vocab_size):
            tok = id_to_token.get(i, "<unk>")
            fw.write(tok + "\n")
    return fname

def save_tokens_file(token_ids, id_to_token, rollno):
    fname = f"{rollno}_assignment2_sp_tokens.txt"
    with open(fname, "w", encoding="utf-8") as fw:
        for tid in token_ids:
            fw.write(id_to_token.get(tid, "<unk>") + "\n")
    return fname

def save_detok_file(detok_text, rollno):
    fname = f"{rollno}_assignment2_sp_detokenized.txt"
    with open(fname, "w", encoding="utf-8") as fw:
        fw.write(detok_text)
    return fname

def main():
    parser = argparse.ArgumentParser(description="SentencePiece-style BPE tokenizers.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--rollno", type=str, default=Rollnumber)
    args = parser.parse_args()

    t0 = time.time()
    with open(args.train, "r", encoding="utf-8") as fr:
        train_raw = fr.read()
    train_norm = normalize_text(train_raw)

    id_to_token, tok2id, merges, heads = train_sp(train_norm, args.vocab_size)
    save_vocab_file(id_to_token, args.vocab_size, args.rollno)

    trie = build_trie_from_id_to_token(id_to_token, args.vocab_size)

    with open(args.input, "r", encoding="utf-8") as fr:
        input_raw = fr.read()
    input_norm = normalize_text(input_raw)

    token_ids = tokenise(input_norm, trie, tok2id)
    save_tokens_file(token_ids, id_to_token, args.rollno)

    detok_text = detokenise(token_ids, id_to_token)
    save_detok_file(detok_text, args.rollno)

    time_taken = time.time() - t0
    print(f"Done and  Time taken is : {time_taken:.2f}s")

if __name__ == "__main__":
    main()
