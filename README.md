# LT2316 H20 Assignment A1

Name: Lawin Khalid

## Notes on Part 1.

I first tried to install and use flair library for both tokenization and features. However, after installing flair and allennlp (to use ELMo vectors), pushing the data to the GPU resulted in `AssertionError: The NVIDIA driver on your system is too old`, most likey due to the requirements upgrade of torch from 1.2 to 1.6. Finally I had to uninstall flair to be able to make use of the GPU.

Instead, I used scispacy for tokenization and features, installed as below:

`pip3 install --user scispacy==0.2.5`  
`pip3 install --user https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_core_sci_md-0.2.5.tar.gz`

Reasons for choosing scispacy were (a) it has shown to work well for biomedical texts, (b) we can easily extract character offsets of every token from the scispacy output, (c) it offers POS-tags and FastText vectors.

Since it's slow to iterate through and process every token in the dataset (at least with my code), I opted to extract POS-tags in Part 1 to avoid having to re-parse the sentences with scispacy again in Part 2. Optimally, also the vectors would be extracted in the same run, and the only reason I left it to Part 2 was the design of the assignment. Thus, in the parsing loop, the dataframes are filled, as well as max_sample_length and POS-tags.


## Notes on Part 2.

Scispacy was also used for FastText vectors. As far as my understanding, these vectors are trained on scientific texts, which I argue would be beneficial for our dataset. I used the medium sized scispacy model, since the small model did not include vectors. The large sized model would likely capture more words which would improve performance of the model. However, for the sake of this assignment there was no need to download excess data. Another reason for choosing scispacy over e.g. torchtext vectors, was that scispacy could handle capitalized words, where torchtext would treat them as unknown words. I opted to not lowercase the tokens, because entities often are capitalized (although far from always true for our dataset). Moreover, I find FastText superior to word2vec for this task, since it better handles OOV words, and understands affixes, due to its design of character n-grams. The latter is especially important for this task where the suffixes of many entities are somewhat similar.

Arguments added to the function `extract_features` are: (a) the POS-tags extracted in Part 1, (b) the vocab, needed to get the vectors, since the mapping of index to vectors in the scispacy model does not correspond to our mapping of index to words, (c) device for pushing the data to the GPU.

In this part the POS-tags get converted to indices, and the vectors are extracted from the scispacy model. Furthermore, I included the token_id of the word, as well as the previous and following word, as features. The nature of the corpus, where drug-drug-interaction is discussed, provides a relatively monotone language, where entities are often (but far from always) preceded/succeeded by specific tokens, such as *and*, or parentheses. Hence, I reason that the immediate context would be valuable as features.

In total, we have 5 features which were all concatenated. The FastText vectors have 200 dimensions, resulting in a feature length of 204 for each token.



## Notes on Part Bonus.

The bonus parts were straight forward using matplotlib, except the last one which I skipped. There wasn't many choices to be made and documented. As for `plot_split_ner_distribution` (not bonus part), I excluded the much larger negative class to avoid tiny bars for the positive classes, which would hurt the visual overview of the distribution. In hindsight, I could've used log scales for all histograms.
