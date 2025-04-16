
############################################################
# CMPSC 442: Hidden Markov Models
############################################################

student_name = "Seokhyeon Bae"

############################################################
# Imports
from collections import defaultdict, Counter
import math
############################################################

# Include your imports here, if any are used.



############################################################
# Section 1: Hidden Markov Models
############################################################

def load_corpus(path):
    corpus = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            sentence = []
            for token_pos in line.split():
                if '=' not in token_pos:
                    continue  # skip malformed entries
                token, pos = token_pos.rsplit('=', 1)
                sentence.append((token, pos))
            corpus.append(sentence)
    return corpus


class Tagger(object):

    def __init__(self, sentences):
        self.initial_counts = Counter()
        self.transition_counts = defaultdict(Counter)
        self.emission_counts = defaultdict(Counter)
        self.tag_counts = Counter()
        self.vocab = set()
        self.tags = set()

        for sentence in sentences:
            if not sentence:
                continue
            first_token, first_tag = sentence[0]
            self.initial_counts[first_tag] += 1
            self.tag_counts[first_tag] += 1
            self.emission_counts[first_tag][first_token] += 1
            self.vocab.add(first_token)
            self.tags.add(first_tag)

            for i in range(1, len(sentence)):
                prev_token, prev_tag = sentence[i - 1]
                token, tag = sentence[i]
                self.transition_counts[prev_tag][tag] += 1
                self.tag_counts[tag] += 1
                self.emission_counts[tag][token] += 1
                self.vocab.add(token)
                self.tags.add(tag)

        self.total_sentences = len(sentences)
        self.tags = list(self.tags)
        self.vocab = list(self.vocab)
        self.num_tags = len(self.tags)
        self.num_tokens = len(self.vocab)

        # Laplace-smoothed probabilities
        self.initial_probs = {
            tag: (self.initial_counts[tag] + 1) / (self.total_sentences + self.num_tags)
            for tag in self.tags
        }

        self.transition_probs = {
            prev_tag: {
                tag: (self.transition_counts[prev_tag][tag] + 1) / (self.tag_counts[prev_tag] + self.num_tags)
                for tag in self.tags
            }
            for prev_tag in self.tags
        }

        self.emission_probs = {}
        for tag in self.tags:
            total = self.tag_counts[tag] + self.num_tokens
            self.emission_probs[tag] = {
                token: (self.emission_counts[tag][token] + 1) / total
                for token in self.vocab
            }
            if '<UNK>' not in self.emission_probs[tag]:
                self.emission_probs[tag]['<UNK>'] = 1 / total

    def most_probable_tags(self, tokens):
        result = []
        for token in tokens:
            best_tag = None
            best_prob = -1
            for tag in self.tags:
                # Use smoothed probability even for unknown tokens
                self.emission_probs[tag]['<UNK>'] = 1 / (self.tag_counts[tag] + self.num_tokens)
                prob = self.emission_probs[tag].get(token, 1 / (self.tag_counts[tag] + self.num_tokens))
                if prob > best_prob:
                    best_prob = prob
                    best_tag = tag
            result.append(best_tag)
        return result

    def viterbi_tags(self, tokens):
        n = len(tokens)
        V = [{} for _ in range(n)]
        backpointer = [{} for _ in range(n)]

        for tag in self.tags:
            emit_prob = self.emission_probs[tag].get(tokens[0], self.emission_probs[tag]['<UNK>'])
            V[0][tag] = math.log(self.initial_probs[tag]) + math.log(emit_prob)
            backpointer[0][tag] = None

        for t in range(1, n):
            for curr_tag in self.tags:
                max_prob = float('-inf')
                best_prev = None
                emit_prob = self.emission_probs[curr_tag].get(tokens[t], self.emission_probs[curr_tag]['<UNK>'])
                log_emit = math.log(emit_prob)
                for prev_tag in self.tags:
                    trans_prob = self.transition_probs[prev_tag][curr_tag]
                    prob = V[t-1][prev_tag] + math.log(trans_prob) + log_emit
                    if prob > max_prob:
                        max_prob = prob
                        best_prev = prev_tag
                V[t][curr_tag] = max_prob
                backpointer[t][curr_tag] = best_prev

        last_tag = max(V[n-1], key=V[n-1].get)
        path = [last_tag]
        for t in range(n-1, 0, -1):
            path.insert(0, backpointer[t][path[0]])
        return path
