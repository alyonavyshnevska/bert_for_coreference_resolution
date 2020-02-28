## Bert for Coreference Resolution
Joshi et al 2019


## What do we learn from context? Probing for sentence structure in contextualized word represenattions 

Tenney et al 2019


## The Illustrated Transformer (blog post)

Jay Allamar: [Blogpost](http://jalammar.github.io/illustrated-transformer/)

1. Stack of encoders, same size stack of decoders. In Bert base: 12 encoder layers, in bert large: 24.

2. Simplified verison with 2:

![](graphics/transformer.png)

3. Multi-headed self-attention: 
    - useful if we’re translating a sentence like “The animal didn’t cross the street because it was too tired”. Which word “it” refers to.


![](graphics/transformer_multi-headed_self-attention-recap.png)

4. Each layer applies self-attention, and passes its results through a feed-forward network, and then hands it off to the next encoder.





## The Illustrated BERT
Jay Allamar: [Blogpost](http://jalammar.github.io/illustrated-bert/)


1. W0rd Embeddings (word2vec) VS contextualised word-embeddings (ELMo, BERT)
Word embeddings use fixed embedding for each word. 
ELMo looks at the entire sentence before assigning each word in it an embedding. So embeddings are not fixed. 

2. ELMo/BERT

![](graphics/bert_elmo.png)

- ELMo uses bi-directional LSTMs. 
- The OpenAI Transformer (GPT) is made up of the decoder stack from the Transformer. No encoder-decoder attention, only self-attention. ELMo’s language model was bi-directional, but the openAI transformer only trains a forward language model. 
- BERT user bi-directional transformer (also referred to as "Transformer encoder"). Its language model looks both forward and backwards (“is conditioned on both left and right context”).   
- Bert is pre-trained on 2 tasks:  
    - 1. language modeling task masks 15% of words in the input and asks the model to predict the missing word.   
    - 2. Bert's 2. task is pre-trained on is a two-sentence classification task: can sentence A follow sent B?  
- BERT doesn’t look at words as tokens. Rather, it looks at WordPieces.

Transformers deal with long-term dependancies better than LSTMs.

- btw, Bidirectional Encoder Representations  from Transformers

