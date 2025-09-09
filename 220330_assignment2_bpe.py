
"""so for this assignment we have to implement byte pair encoding from scratch,This is my optimized approach which runns in 120 sectonds 
for vocab size of 5000 on a 26 mb txt file that I have extracted form the wikipedia using python script.
intially I had implement brute force approch which using using and rescaning the entire corpus for every merge operation which was taking lots of time  to complete for vocab size of 5000
so I decided to optimized by using doubly linked list to represent each word in the corpus and a hashmap to store the occurences of each pair in the corpus
and a max heap to get the most frequent pair in O(log n) time istead of O(n) time which was used in brute force approach using dictionary to store the pairs and their counts.
this we reducde time complexity of merge operation from o(n*m) to O(m log n) where n is the number of unique pairs and m is the number of merges.
so overall time complexity of the optimized approach is O(n + m log n) where n is the size of the corpus and m is the number of merges.
step wise my algiorithm is :
1.make a doubly linked list(so that traversal will be easy) to represent each word in the corpus.
Linked list will store the token ids of the characters in the word and pointers to the previous and next nodes and will also 
complexity of list for update  operations will be O(k) time where k is occurencs of taht pair.
in linkedlis apprach  I just cross out one word and fix its neighbors that reduce my extra working.
total complexity will be O(Nlong(M)*logM * merges) where N is the size of the corpus and M is the number of unique pairs and merges is the number of merges.
if we will compare with earlier brute force approach which was O(N*M*merges) where M is the number of unique pairs.
2.Initialize the vocabulary with reserved tokens (<pad>, <unk>, <s>, </s>) and  all unique characters (or UTF-8 bytes) from the training corpus.
3.build a hashmap to store the occurrences of each pair in the corpus.
4.Used a lazy max heap to efficiently get the most frequent pair instead od sorting list which is expensive steps time wise .
used max heap to store the pairs and their counts and used lazy deletion to remove the pairs which are not valid anymore.
5.Merge the most frequent pair and update the corpus, vocabulary, and pair occurrences.
6.Repeat steps 4-5 until the vocabulary size reaches the desired limit of 5000.


My sample output for my extracted tex file on 26 mb is 
Merge for step counts 100: ia</w> (count=30900)
Merge for step counts 200: ing (count=16411)
Merge for step counts 300: e.</w> (count=10337)
Merge for step counts 400: tive</w> (count=7336)
Merge for step counts 500: ourc (count=6169)
Merge for step counts 600: rap (count=5340)
Merge for step counts 700: ISBN</w> (count=4621)
Merge for step counts 800: lic</w> (count=4192)
Merge for step counts 900: more</w> (count=3672)
Merge for step counts 1000: ves</w> (count=3312)
Merge for step counts 1100: lation</w> (count=2998)
Merge for step counts 1200: ranc (count=2751)
Merge for step counts 1300: can</w> (count=2555)
Merge for step counts 1400: Mex (count=2396)
Merge for step counts 1500: es:</w> (count=2233)
Merge for step counts 1600: eat</w> (count=2088)
Merge for step counts 1700: sourcrce</w> (count=1945)
Merge for step counts 1800: secection</w> (count=1838)
Merge for step counts 1900: ies,</w> (count=1741)
Merge for step counts 2000: lations</w> (count=1652)
Merge for step counts 2100: alian</w> (count=1587)
Merge for step counts 2200: ivil</w> (count=1513)
Merge for step counts 2300: By</w> (count=1462)
Merge for step counts 2400: (199 (count=1404)
Merge for step counts 2500: thi (count=1339)
Merge for step counts 2600: ind</w> (count=1286)
Merge for step counts 2700: mpport (count=1241)
Training completed in 124.84 sec, 2713 merges.
Output files 220330.


"""


import argparse
import time
import heapq
from collections import defaultdict

#building a doubly linked list node
class Node:
    def __init__(self, token_id):
        self.token_id = token_id
        self.prev = None
        self.next = None

#intialising the vocab with reserved tokens
def initialize_vocab():
    reserved_tokens = ["<pad>", "<unk>", "<s>", "</s>"]
    vocab = {i: tok for i, tok in enumerate(reserved_tokens)}
    inverse_vocab = {tok: i for i, tok in vocab.items()}
    return vocab, inverse_vocab, len(vocab)


def build_corpus(text, vocab, inverse_vocab, next_token_id):
    words = [list(w) + ["</w>"] for w in text.strip().split()]
    corpus = []
    for w in words:
        head, prev = None, None
        for ch in w:
            if ch not in inverse_vocab:
                vocab[next_token_id] = ch
                inverse_vocab[ch] = next_token_id
                next_token_id += 1
            node = Node(inverse_vocab[ch])
            if prev:
                prev.next, node.prev = node, prev
            else:
                head = node
            prev = node
        corpus.append(head)
    return corpus, vocab, inverse_vocab, next_token_id


def count_pairs(corpus):
    pair_occurrences = defaultdict(set)
    for head in corpus:
        node = head
        while node and node.next:
            pair = (node.token_id, node.next.token_id)
            pair_occurrences[pair].add(node)
            node = node.next
    return pair_occurrences


def merge_pair(pair, new_token_id, vocab, inverse_vocab, pair_occurrences, heap):
    t1, t2 = pair
    new_token = vocab[t1] + vocab[t2]
    vocab[new_token_id] = new_token
    inverse_vocab[new_token] = new_token_id

    affected_nodes = list(pair_occurrences[pair])
    pair_occurrences[pair].clear()

    for node in affected_nodes:
        if not node.next or node.token_id != t1 or node.next.token_id != t2:
            continue

        # Merge into new token
        node.token_id = new_token_id
        removed = node.next
        node.next = removed.next
        if removed.next:
            removed.next.prev = node

        # Remove old pairs
        if node.prev:
            old = (node.prev.token_id, t1)
            pair_occurrences[old].discard(node.prev)
        if node.next:
            old = (t2, node.next.token_id)
            pair_occurrences[old].discard(node)

        # Add new neighbor pairs
        if node.prev:
            new = (node.prev.token_id, node.token_id)
            pair_occurrences[new].add(node.prev)
            heapq.heappush(heap, (-len(pair_occurrences[new]), new))
        if node.next:
            new = (node.token_id, node.next.token_id)
            pair_occurrences[new].add(node)
            heapq.heappush(heap, (-len(pair_occurrences[new]), new))

    return vocab, inverse_vocab

def train_bpe(text, vocab_size):
    #print("Trainsing started")
    vocab, inverse_vocab, next_token_id = initialize_vocab()
    corpus, vocab, inverse_vocab, next_token_id = build_corpus(text, vocab, inverse_vocab, next_token_id)

    pair_occurrences = count_pairs(corpus)
    heap = [(-len(nodes), pair) for pair, nodes in pair_occurrences.items()]
    heapq.heapify(heap)

    merges, steps = {}, 0
    start = time.time()

    while len(vocab) < vocab_size and heap:
        freq, pair = heapq.heappop(heap)
        freq = -freq
        if freq == 0 or len(pair_occurrences[pair]) != freq:
            continue

        merges[pair] = next_token_id
        vocab, inverse_vocab = merge_pair(pair, next_token_id, vocab, inverse_vocab, pair_occurrences, heap)
        next_token_id += 1
        steps += 1

        if steps % 100 == 0:
            print(f"Merge for step counts {steps}: {vocab[merges[pair]]} (count={freq})")
    #print("Training completed")
    print(f"Training completed in {time.time() - start:.2f} sec, {steps} merges.")
    return vocab, inverse_vocab, merges

def tokenize(text, inverse_vocab, merges):
    words = [list(w) + ["</w>"] for w in text.strip().split()]
    tokens = []

    for w in words:
        ids = [inverse_vocab[ch] for ch in w]
        merged = True
        while merged:
            merged, i = False, 0
            while i < len(ids) - 1:
                pair = (ids[i], ids[i+1])
                if pair in merges:
                    ids[i] = merges[pair]
                    ids.pop(i+1)
                    merged = True
                else:
                    i += 1
        tokens.extend(ids)
    return tokens


def detokenize(tokens, vocab):
    words, cur = [], []
    for tid in tokens:
        tok = vocab[tid]
        if tok.endswith("</w>"):
            cur.append(tok[:-4])
            words.append("".join(cur))
            cur = []
        else:
            cur.append(tok)
    if cur:
        words.append("".join(cur))
    return " ".join(words)


def save_outputs_to_files(vocab, tokens, detok, rollno, vocab_size):
    with open(f"{rollno}_assignment2_bpe_vocab_{vocab_size}.txt", "w", encoding="utf-8") as f:
        for tid, tok in vocab.items():
            f.write(f"{tid}\t{tok}\n")

    with open(f"{rollno}_assignment2_bpe_tokens.txt", "w", encoding="utf-8") as f:
        f.write(" ".join(map(str, tokens)))

    with open(f"{rollno}_assignment2_bpe_detokenized.txt", "w", encoding="utf-8") as f:
        f.write(detok)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    args = parser.parse_args()

    rollno = "220330"

    with open(args.train, "r", encoding="utf-8") as f:
        train_text = f.read()

    vocab, inverse_vocab, merges = train_bpe(train_text, args.vocab_size)

    with open(args.input, "r", encoding="utf-8") as f:
        input_text = f.read()

    tokens = tokenize(input_text, inverse_vocab, merges)
    detok = detokenize(tokens, vocab)

    save_outputs_to_files(vocab, tokens, detok, rollno, args.vocab_size)
    print(f"Output files {rollno}.")

if __name__ == "__main__":
    main()
