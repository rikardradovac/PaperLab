# Latent Dirichlet Allocation (LDA) Implementation

This is a Python implementation of Latent Dirichlet Allocation (LDA) using collapsed Gibbs sampling, based on the seminal paper:

> Blei, David M., Andrew Y. Ng, and Michael I. Jordan. "Latent Dirichlet Allocation." Journal of Machine Learning Research 3 (2003): 993-1022. [[PDF]](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)

## Features

- Document preprocessing using spaCy
- Collapsed Gibbs sampling implementation
- Topic coherence scoring
- Configurable hyperparameters (α, β)
- Command-line interface for easy experimentation

## Usage

1. Install the required dependencies:

````

pip install numpy spacy tqdm scikit-learn
python -m spacy download en_core_web_sm

````
2. Run the LDA model:

````
python main.py --num_documents 3000 --num_topics 10 --alpha 0.1 --beta 0.1 --num_iterations 100
````


## Parameters

- `num_documents`: Number of documents to process (default: 3000)
- `num_topics`: Number of topics to extract (default: 10)
- `alpha`: Dirichlet prior on document-topic distribution (default: 0.1)
- `beta`: Dirichlet prior on topic-word distribution (default: 0.1)
- `num_iterations`: Number of Gibbs sampling iterations (default: 100)

## Implementation Details

The implementation includes:
- Document preprocessing with stopword removal and lemmatization
- Vocabulary building with frequency thresholding
- Collapsed Gibbs sampling for LDA inference
- Topic coherence calculation

## License

MIT