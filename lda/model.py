from collections import Counter
import re
from typing import List, Tuple
import random
import numpy as np
from tqdm import trange


class DocumentPreprocessor:
    def __init__(self, nlp, min_token_occurances: int = 10):
        self.nlp = nlp
        self.min_token_occurances = min_token_occurances
        self.vocabulary = {}
        self.reverse_vocabulary = {}


    @staticmethod
    def _clean_text(text: str):
        """
        Cleans the text by making it lowercase and removing whitespaces

        Parameters:
        - text : The text to clean

        Returns:
        - str : Cleaned text
        """
        text = text.lower()
        text = text.strip()
        # remove whitespaces
        text = re.sub(r'(\s+)', ' ', text)

        return text

    def get_frequencies(self, cleaned_tokens: List[List]):
        """
        Returns the frequencies for each token

        Parameters:
        - cleaned_tokens (List[List]): A list of cleaned tokens.

        Returns:
        - Counter: Counter for tokens
        """
        frequencies = Counter()

        for current_tokens in cleaned_tokens:
            for token in current_tokens:
                frequencies[token] += 1
        return frequencies


    def remove_stopwords_and_lemmatize(self, corpus: List[List]):
        """
        clean the corpus text by lemmatizing and removing stopwords, punctuations and numbers and returns the cleaned corpus

        Parameters:
        - corpus (List[List]): list of all the documents

        Returns:
        - List[List] : List of cleaned document
        """

        cleaned_tokens = []

        for document in corpus:
            text = self._clean_text(document)
            doc = self.nlp(text)
            current_tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]

            if len(current_tokens) > 1:
                cleaned_tokens.append(current_tokens)


        return cleaned_tokens



    def build_vocab(self, frequencies: Counter, threshold: int = 10):
        """
        Builds the vocabulary based on the frequencies and if they are higher than a given threshold

        Parameters:
        - frequencies (Counter) : Counter of tokens
        - threshold (int) : The minimum number of token occurances
        """
        index = 0

        for token, count in frequencies.items():
            if count > threshold:
                self.vocabulary[token] = index
                self.reverse_vocabulary[index] = token
                index += 1


    def tokenize(self, corpus: List[List]) -> List[np.ndarray]:
        """
        Tokenizes the given corpus into integer tokens

        Parameters:
        - corpus (List[List]) : list of all the documents

        Returns:
        - List[np.ndarray]
        """

        tokenized_dataset = []
        num_tokens = 0

        for tokenized_document in self.remove_stopwords_and_lemmatize(corpus):
            current_document = []
            num_tokens += len(tokenized_document)
            for token in tokenized_document:
                if token in self.vocabulary:
                    current_document.append(self.vocabulary[token])
            tokenized_dataset.append(np.array(current_document))
        print("Number of total tokens: ", num_tokens)
        return tokenized_dataset

    def vocab_size(self) -> int:
        """
        Returns the size of the vocabulary

        Returns:
        - int: Vocabulary size
        """
        return len(self.vocabulary)

class LDA:
    def __init__(self, alpha, beta, n_topics, vocab_size):
        self.alpha = alpha
        self.beta = beta
        self.n_topics = n_topics
        self.vocab_size = vocab_size

    def initialize_topics(self, docs: List[List]):
        """
        Initializes a random topic for each word in each document

        Parameters:
        - docs (List[List]): A list of tokenized documents

        Returns:
        - List[List]: List of topics for each word in each document.
        """
        topics = [np.random.randint(low=0, high=self.n_topics, size=len(doc)) for doc in docs]
        return topics

    def initialize_ndk(self, docs: List[List], topics: List[List]) -> np.ndarray:
        """
        For each document d, initializes the number of words assigned to topic k

        Parameters:
        - docs (List[List]): A list of tokenized documents
        - topics (List[List]): List of topics for each word in each document

        Returns:
        - np.ndarray: 2D array representing the number of words assigned to each topic for each document.
        """
        num_docs = len(docs)
        ndk = np.array([[np.sum(topics[d] == k) for k in range(self.n_topics)] for d in range(num_docs)])
        return ndk

    def initialize_nkw(self, docs: List[List], topics: List[List]) -> np.ndarray:
        """
        Initializes the number of times word w is assigned to topic k

        Parameters:
        - docs (List[List]): A list of tokenized documents.
        - topics (List[List]): List of topics for each word in each document

        Returns:
        - np.ndarray: 2D array representing the number of times each word is assigned to each topic
        """
        nkw = np.zeros((self.n_topics, self.vocab_size))
        for doc_i, doc in enumerate(docs):
            for word_i, word in enumerate(doc):
                temp_topic = topics[doc_i][word_i]
                nkw[temp_topic, word] += 1
        return nkw

    def run(self, docs: List[List], num_iterations: int) -> Tuple[List[List], np.ndarray, np.ndarray, np.ndarray]:
        """
        Run collapsed Gibbs iterations for Latent Dirichlet Allocation (LDA) on tokenized documents

        Parameters:
        - docs (List[List]): A list of tokenized documents
        - num_iterations (int): Number of iterations for the Gibbs sampling algorithm

        Returns:
        - Tuple[List[List], np.ndarray, np.ndarray, np.ndarray]: Tuple containing topics, nkw, ndk, and nk
        """

        topics = self.initialize_topics(docs)
        num_docs = len(docs)
        ndk = self.initialize_ndk(docs, topics)
        nkw = self.initialize_nkw(docs, topics)

        #count the number of occurances of words for each topic, to be used later in the equation
        nk = np.sum(nkw, axis=1)
        topic_list = list(range(self.n_topics))

        for _ in trange(num_iterations):
            for doc_i, doc in enumerate(docs):
                for word_i, word in enumerate(doc):
                    #get the topic for the word
                    temp_topic = topics[doc_i][word_i]

                    #remove the topic assignments to this word since the equation is conditioned on all topic assignments except for this one
                    ndk[doc_i][temp_topic] -= 1
                    nkw[temp_topic][word] -= 1
                    nk[temp_topic] -= 1

                    # calculate the probabilities for the current word belonging to the K topics
                    p_z = ((ndk[doc_i, :] + self.alpha) * (nkw[:, word] + self.beta) / (nk[:] + self.beta * self.vocab_size))

                    # re sample the topics with probability distribution p_z
                    temp_topic = random.choices(topic_list, weights=p_z, k=1)[0]

                    topics[doc_i][word_i] = temp_topic
                    ndk[doc_i][temp_topic] += 1
                    nkw[temp_topic][word] += 1
                    nk[temp_topic] += 1

        return topics, nkw, ndk, nk


    def calculate_coherence_score(self, tokenized_dataset: List[List], phi: np.ndarray,  M: int = 20) -> np.ndarray:
        """
        Calculate coherence score given the tokenized dataset, phi and number of words to use

        Parameters:
        - tokenized_dataset (List[List]): A list of tokenized documents
        - phi (np.ndarray): The topic-document distribution matrix
        - M (int, optional): The number of words to consider for coherence calculation. Defaults to 20


        Returns:
        - np.ndarray: Array of coherence scores for each topic
        """


        document_frequency = np.zeros(self.vocab_size)
        co_document_frequency = np.zeros((self.vocab_size, self.vocab_size))

        for document in tokenized_dataset:
            for token in set(document):
                document_frequency[token] += 1


        for document in tokenized_dataset:

            # amount of same tokens does not matter, at least one is needed
            unique_tokens = list(set(document))
            for ind, token in enumerate(unique_tokens):
                for j in range(ind + 1, len(unique_tokens)):
                    other_token = unique_tokens[j]

                    # unique combination of tokens
                    co_document_frequency[token, other_token] += 1

        # V holds all the top words for each topic
        V = [np.argsort(phi[k])[::-1][:M] for k in range(self.n_topics)]
        coherent_scores = np.zeros(self.n_topics)

        # for each topic, calculate the coherent score
        for topic in range(self.n_topics):
            current_coherent_score = 0
            for m in range(2, M):
                for l in range(1, m - 1):
                    current_coherent_score += np.log((co_document_frequency[V[topic][m], V[topic][l]] + 1) / document_frequency[V[topic][l]])
            coherent_scores[topic] = current_coherent_score

        return coherent_scores


