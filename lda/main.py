from sklearn.datasets import fetch_20newsgroups
from model import DocumentPreprocessor, LDA
import spacy
import argparse

# Move constants to be default values
DEFAULT_NUM_DOCUMENTS = 3000
DEFAULT_NUM_TOPICS = 10
DEFAULT_ALPHA = 0.1
DEFAULT_BETA = 0.1
DEFAULT_NUM_ITERATIONS = 100

def parse_args():
    parser = argparse.ArgumentParser(description='Run LDA topic modeling')
    parser.add_argument('--num_documents', type=int, default=DEFAULT_NUM_DOCUMENTS,
                      help=f'Number of documents to process (default: {DEFAULT_NUM_DOCUMENTS})')
    parser.add_argument('--num_topics', type=int, default=DEFAULT_NUM_TOPICS,
                      help=f'Number of topics to extract (default: {DEFAULT_NUM_TOPICS})')
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA,
                      help=f'Alpha parameter for LDA (default: {DEFAULT_ALPHA})')
    parser.add_argument('--beta', type=float, default=DEFAULT_BETA,
                      help=f'Beta parameter for LDA (default: {DEFAULT_BETA})')
    parser.add_argument('--num_iterations', type=int, default=DEFAULT_NUM_ITERATIONS,
                      help=f'Number of iterations for LDA (default: {DEFAULT_NUM_ITERATIONS})')
    return parser.parse_args()

def main():
    args = parse_args()
    
    dataset = fetch_20newsgroups(subset='all', shuffle=True, remove=('headers', 'footers', 'quotes'))

    nlp = spacy.load("en_core_web_sm")
    preprocessor = DocumentPreprocessor(nlp=nlp, min_token_occurances=2)
    cleaned = preprocessor.remove_stopwords_and_lemmatize(dataset.data[:args.num_documents])
    freqs = preprocessor.get_frequencies(cleaned)
    preprocessor.build_vocab(freqs)
    vocab_size = preprocessor.vocab_size()
    print("vocab_size", vocab_size)
    tokenized_dataset = preprocessor.tokenize(dataset.data[:args.num_documents])
    
    lda = LDA(alpha=args.alpha, beta=args.beta, n_topics=args.num_topics, vocab_size=vocab_size)
    topics, nkw, ndk, nk = lda.run(docs=tokenized_dataset, num_iterations=args.num_iterations)

if __name__ == "__main__":
    main()
