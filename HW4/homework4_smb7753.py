############################################################
# CMPSC442: Classification
############################################################

student_name = "Seokhyeon Bae"

############################################################
# Imports
import email
from email.iterators import body_line_iterator
from email import policy
from email.parser import BytesParser   
import math 
from collections import Counter
import os

############################################################

# Include your imports here, if any are used.

############################################################
# Section 1: Spam Filter
############################################################


def load_tokens(email_path):
    content = []
    with open(email_path, 'r', encoding='utf-8', errors='ignore') as file:
        message = email.message_from_file(file, policy=policy.default)
        for line in body_line_iterator(message):
            line_tokens = line.split()
            content.extend(line_tokens)

    return content


def log_probs(email_paths, smoothing):
    word_counts = Counter()
    
    for path in email_paths:
        tokens = load_tokens(path)
        word_counts.update(tokens)

    vocab = set(word_counts.keys())
    total_count = sum(word_counts.values())
    vocab_size = len(vocab)
    denominator = total_count + smoothing * (vocab_size + 1)

    log_prob_dict = {}

    for word in vocab:
        count = word_counts[word]
        prob = (count + smoothing) / denominator
        log_prob_dict[word] = math.log(prob)

    unk_prob = smoothing / denominator
    log_prob_dict["<UNK>"] = math.log(unk_prob)

    return log_prob_dict

def logaddexp(a, b):
    if a > b:
        return a + math.log1p(math.exp(b - a))
    else:
        return b + math.log1p(math.exp(a - b))

class SpamFilter(object):

    def __init__(self, spam_dir, ham_dir, smoothing):
        self.smoothing = smoothing

        spam_paths = [os.path.join(spam_dir, fname) for fname in os.listdir(spam_dir)]
        ham_paths = [os.path.join(ham_dir, fname) for fname in os.listdir(ham_dir)]
        num_spam = len(spam_paths)
        num_ham = len(ham_paths)
        total = num_spam + num_ham

        self.log_prior_spam = math.log(num_spam / total)
        self.log_prior_ham = math.log(num_ham / total)

        vocab = set()
        for path in spam_paths + ham_paths:
            tokens = load_tokens(path)
            vocab.update(tokens)
        self.vocab = vocab

        self.spam_log_probs = log_probs(spam_paths, smoothing)
        self.ham_log_probs = log_probs(ham_paths, smoothing)
    
    def is_spam(self, email_path):
        tokens = load_tokens(email_path)

        word_counts = {}
        for token in tokens:
            token = token if token in self.vocab else "<UNK>"
            word_counts[token] = word_counts.get(token, 0) + 1

        log_prob_spam = self.log_prior_spam
        log_prob_ham = self.log_prior_ham

        for word, count in word_counts.items():
            log_prob_spam += count * self.spam_log_probs.get(word, self.spam_log_probs["<UNK>"])
            log_prob_ham += count * self.ham_log_probs.get(word, self.ham_log_probs["<UNK>"])

        return log_prob_spam > log_prob_ham

    # P(w|spam) and P(w|ham) = self.spam_log_probs[w] and self.sham_log_probs[w]
    def most_indicative_spam(self, n):
        indicative_scores = []

        for word in self.vocab:
            if word not in self.spam_log_probs or word not in self.ham_log_probs:
                continue

            pw_spam_log = self.spam_log_probs.get(word, self.spam_log_probs["<UNK>"])
            pw_ham_log = self.ham_log_probs.get(word, self.ham_log_probs["<UNK>"])

            log_pw = logaddexp(
                pw_spam_log + self.log_prior_spam,
                pw_ham_log + self.log_prior_ham
            )

            indicative_score = pw_spam_log - log_pw
            indicative_scores.append((indicative_score, word))

        indicative_scores.sort(reverse=True)
        return [word for _, word in indicative_scores[:n]]

    def most_indicative_ham(self, n):
        indicative_scores = []

        for word in self.vocab:
            if word not in self.spam_log_probs or word not in self.ham_log_probs:
                continue

            pw_spam_log = self.spam_log_probs.get(word, self.spam_log_probs["<UNK>"])
            pw_ham_log = self.ham_log_probs.get(word, self.ham_log_probs["<UNK>"])

            log_pw = logaddexp(
                pw_spam_log + self.log_prior_spam,
                pw_ham_log + self.log_prior_ham
            )

            indicative_score = pw_ham_log - log_pw
            indicative_scores.append((indicative_score, word))

        indicative_scores.sort(reverse=True)
        return [word for _, word in indicative_scores[:n]]


