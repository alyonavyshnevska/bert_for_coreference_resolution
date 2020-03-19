## Experimental Design
Group: Olena Vyshnevska, Patrick Kahardipraja

##### Span Representation and Long-Range Coreference
We attempt to investigate to what extent the span representations proposed using BERT embeddings in [1] can encode coreference information, and whether it is able to encode non-local coreference phenomena or is it just simply modeling local coreference?

In order to analyse this, we consider 2 kinds of span representations:
1. BERT-based span representations finetuned on OntoNotes in [1] with first and last word-pieces (concatenated with the attention version of all word pieces in the span).
2. pre-trained BERT embeddings (not finetuned on OntoNotes) for all tokens within the mention span, which is then passed through a convolutional layer (with kernel width of 3 and 5) to incorporate the local context and followed by self-attention pooling operator to produce a fixed-length span representations. This is to model head words, inspired by approach from [2].

Both span representations will be then used as inputs for coreference arc prediction task [3], where a probing model (in this case a simple FFNN) is used to predict coreference relations. The probing model is designed with limited capacity to focus on what information that can be extracted from the span representations. The probing model itself has a sigmoid output layer, which is trained to minimize binary cross entropy. Each negative samples (*w_entity*, *wb*) will be generated for every positive samples (*wa*,*wb*) where *wb* occurs after *wa* and *w_entity* is a token that occurs before *wb* and belong to a difference coreference cluster, to ensure a balanced data. By comparing the performance of the probing model using these two span representations, we can hypothesize to what extent that the proposed span representation in [1] can capture coreference information. We will also experiment with mention span separation distance to see how the probing model performs and whether if there is a degradation of accuracy and F1 score of the probing model with distant spans.


###### References
1. [BERT for Coreference Resolution: Baselines and Analysis][1]
2. [What do you learn from context? Probing for sentence structure in contextualized word representations][2]
3. [Linguistic Knowledge and Transferability of Contextual Representations][3]


[1]:https://arxiv.org/pdf/1908.09091.pdf
[2]:https://arxiv.org/pdf/1905.06316.pdf
[3]:https://www.aclweb.org/anthology/N19-1112.pdf






