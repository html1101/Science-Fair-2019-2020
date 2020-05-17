# Predicting and Understanding Genomic Sequences Using Neural Networks

By Sarah Cross

# **Table of Contents**

[**Table of Contents**](#_5xkfttsmvtnp) **1**

**[Background(Shorter Version)](#_g6tcxx9r3z2e) 2**

**[Background(Long Version)](#_18opa8ilz7ht) 10**

**[Rationale](#_glr8c9k8nyqk) 23**

**[Introduction](#_vuphz5xh1mhu) 25**

[**Purpose**](#_mlwc67nayefc) **27**

[**Hypothesis**](#_g9rhfhmt81vr) **28**

**[The Code](#_q0wwycs6h2po) 37**

**[Procedure](#_3jvkw7xashb3) 45**

**[Materials](#_9a9dnc4izwji) 47**

**[Conclusion](#_31mumto2p5xw) 48**

[Problems Encountered](#_91rbj9mbplno) 49

[Future Expansions](#_thxgy0kqfcx8) 51

[Practical Applications](#_fo869sc5waep) 52

[Bibliography](#_jjpc3bimz6fa) 53

# Background(Shorter Version)

The virion, or the complete form of a virus outside of the host cell, is enveloped, spherical, and contains one copy of the positive-sense RNA genome. A virus spreads by injecting its genome into the host cell, and this genome is then translated into another viral protein, RNA, or ribosomal nucleic acid. This then forms a new virus, which leaves the cell to go and infect other cells. Viruses depend on the host ribosome to make its protein, and the host ribosome only read mRNA, so viruses are forced to translate their genome into mRNA. From there the ribosome then creates the viral proteins. Positive RNA is when the genome is translated directly into mRNA, meaning that the ribosome can directly read the genome without the virus having to translate the viral protein or DNA into mRNA. Viruses must learn to adapt ecological niches or they become extinct.

Potential cancer treatments have been suggested by modulating transcription factors and &quot;emphasiz[ing] agents with established clinical efficacy or with promising effects in preclinical models.&quot;

Deep Neural Networks have proven to be incredibly accurate at obtaining sites of transcription factor(TF) binding(TFBS). The Deep Motif Dashboard, for example, provides a suite of visualization strategies to extract motifs or sequence patterns from deep neural networks for TFBS classification without the use of external sources such as protein maps.

Transcription factors are regulatory proteins that bind to DNA(turn genes &quot;on&quot; and &quot;off&quot;). Given an input DNA sequence, DeMo classifies whether or not there&#39;s a binding site for a TF. Given a TF of interest and a dataset made of samples of positive and negative TFBS sequences, the model tests three DL architectures to classify the sequences: CNN, RNN, and CNN-RNN structures. Secondly, DeMo attempts to understand why they perform the way they do by measuring nucleotide importance with saliency maps, measuring critical sequences positions for the classifier using temporal output scores, and generating class-specific motif patterns with class optimization.

Chromatin immunoprecipitation(ChIP-seq) technologies, or the precipitation of a protein antigen used to isolate and concentrate a protein from a sample containing thousands of proteins, makes finding site locations available for hundreds of different TFs. Unfortunately, ChIP-seq experiments are slow, expensive, and they can&#39;t find patterns common across binding sites(although they can find the binding sites).

Using a deep neural network, three different architectures were tested: CNN, RNN, and a combination of the two, CNN-RNN. The raw nucleotide base characters are used as input to output an output vector of a fixed dimension, which is linearly fed to a softmax function. The final output returns a binary classification task of size 1x1 returning whether the input is a positive or negative binding site. The models use the stochastic gradient algorithm Adam with a mini-batch size of 256 sequences and a dropout regularization method. DeMo attempts to understand which parts of the DNA sequence are the most influential for classification by using saliency maps. Given a sequence of length and class , a DNN model provides a score function of . Because it&#39;s difficult to directly see the influence of each nucleotide on the complex, highly non-linear score function, given , can be approximated by a linear function by computing the first-order Taylor expansion(a representation of a function as an infinite sum of terms that are calculated from the values of the function&#39;s derivatives at a single point).

With bacterial genomes, codons are very easy to identify and find. Unfortunately there&#39;s more space in between real genes in eukaryotic genes(62% of human genome is intergenic), there are introns which interrupt the sequencing of DNA, codon bias(not all codons are used equally frequently), exon-intron boundaries, and many other problems.

Biologists in recent years have also been able to identify transcription factors(regulatory proteins that bind to a particular sequence), DNase I hypersensitive sites(sites sensitive to cleavage by the DNase I enzyme), and histone marks(chemical modifications to histone proteins). Given no additional outside information, the RNN attempts to predict whether a given feature will be present given only the sequence at the nucleotide-level.

Other models have been published in the past. DeepSEA, for example, is a deep neural network consisting of a mixture of convolutional neural networks and pooling layers which worked on a fixed length sequence, along with a sigmoid output layer to compute the probability of seeing a particular genome feature. DeepBind used a deep CNN with 4 stages of convolution, rectification, pooling, and nonlinear activation functions.

Basset, published a few months later, used a deep CNN architecture to learn the functional activity of genomics sequences.

DanQ, published in December 2015, incorporated a convolutional layer and a bidirectional long short term memory(LSTM) recurrent neural network(RNN) on top of a max pooling layer.

All of the following neural networks utilized one-hot encoding of the input sequence into the convolution layer. One-hot encoding expands integer inputs into arrays of a fixed length; for example, given that the nucleotide bases A, T, C, and G are given integer values, each letter in the sequence of DNA of length _L_ would be transformed into a 4 x _L_ matrix. Each tool used a convolutional layer first, which requires fixed sizes. By introducing several sets of convolutional layers, more parameters are introduced, thus a first approach should avoid using convolutional layers.

The following loss function was used by the researchers:

For which _n_ is the number of training samples(about the length of the genome), |V| is the size of the prediction set(4 for the 4 nucleotide bases), is the predicted probability of the predicted word at time _t_ being word _i_, and is a one-hot vector of the real word at time _t_.

For character-level processing, a bidirectional GRU is used because a genetic sequence can be regulated by both the beginning and end sequences. The outputs then go through a softmax layer; for each K task, a prediction of 0 or 1 is made which indicates whether or not a particular feature should be observed for the input sequence. The overall loss is the geometric average of the perplexity across all K tasks. For the multitask prediction problem, |V| equals 2, the number of classes for each class.

The researchers used a random subset of the dataset used in the DanQ and DeepSEA papers. The dataset was collected from experiments in the ENCODE and Roadmap Epigenomics data releases. They randomly chose 80,000 sequence-label pairs for the training set, 8000 for the testing set, and 2000 for the validation set. Each sequence has a length of 1000.

At the time, training an RNN on all 1000 characters in each example proved to be particularly difficult due to computational limitations. As a result, the RNNs below were trained on the middle 100 characters of each sequence. The baseline model was trained on the same set of truncated sequences.

The first two datasets are used as testing in order to filter out any model not capable of performing well on these genomes.

For the _E. coli_ and _P. falciparum_ genomes, the researchers used a dropout keep rate of 1, a batch size of 50, and 1 epoch. They tested: &quot;GRU, LSTM, and simple RNN architectures, learning rates of 0.0001 and 0.0008, sequence lengths of 50, 100, 500, and 1000, and 2 and 3 layers.&quot; All combinations of the architectures returned perplexities of 3.679 for _E. coli_ and 2.938 for _P. falciparum_. Although the perplexity was low, training for more than 1 epoch showed a reduction in the perplexity; however, in this experiment the number of epochs was reduced in order to test the hyperparameters and their influence on the perplexity.

Using a learning rate of 0.0008 was much faster than using a learning rate of 0.0001, although this is not surprising considering that smaller perplexities are possible and thus increasing the learning rate in this instance would not lead to diverging from the local minima.

Using 2 or 3 layers did not significantly influence the model&#39;s ability to return smaller perplexities, and, although genomes often involve long-range interactions, sequences of 50 or 100-length sequences performed better than 500 or 1000-length sequences. Overall, character-level genome prediction results proved that a 2-layer network with GRUs would be most appropriate for modeling a genomic sequence and that longer input sequences do not necessarily mean lower perplexities.

Logistic regression was used with default parameters excluding the regularization constant of the regularization term set to , and, instead of using raw error rate, the researchers looked at the F1 score defined as:

For which precision is the proportion of correct positive predictions and recall is the proportion of positive examples that received a positive prediction. A perfect score would be 1(meaning the model has reached perfect precision and recall) and its worst at 0.

Ultimately, the final model used a learning rate of 0.001, 2 hidden layers, an embedding size of 2(because there were only 4 unique characters), a dropout keep probability of 0.95, hidden layer sizes of 128, a bidirectional RNN GRU with tanh activation functions, an input sequence of length 100, an Adam optimizer, and hidden layer with a size of 128.

The model, after training for 61 epochs on 80,000 training examples for 3 days on the NVIDIA GRID K520 GPU, plateaued with an average validation loss trapped at a perplexity of 1.087.

In comparison, the RNN beat the logistic regression model in 94.6% of the tasks, giving an average F1 score of 0.135 in comparison to the logistic regression model&#39;s average score of 0.044.

In conclusion, this project constructed deep bidirectional RNNs capable of predicting and capturing complex patterns in 100-character sequences of the human genome. The researchers concluded that more improvements could be made to their model:

1. Creating more data, for example, extending the 100-character sequences to 200-character sequences.
2. Modifying the hyperparameters further.
3. Designing an RNN cell suited for the long-range information associated with genomic sequences such as the clockwork RNN architecture.
4. Finding a distributed representation of genomic &quot;words&quot; and initializing a word-based RNN model.

Sequence prediction and classification are problems Recurrent Neural Networks(RNNs) are both designed and used for. In theory RNNs have the ability to cope with temporal dependencies; however, when long-term memory is required, they are difficult to train correctly.

#

# Background(Long Version)

The virion, or the complete form of a virus outside of the host cell, is enveloped, spherical, and contains one copy of the positive-sense RNA genome. A virus spreads by injecting its genome into the host cell, and this genome is then translated into another viral protein, RNA, or ribosomal nucleic acid. This then forms a new virus, which leaves the cell to go and infect other cells. Viruses depend on the host ribosome to make its protein, and the host ribosome only read mRNA, so viruses are forced to translate their genome into mRNA. From there the ribosome then creates the viral proteins. Positive RNA is when the genome is translated directly into mRNA, meaning that the ribosome can directly read the genome without the virus having to translate the viral protein or DNA into mRNA. Viruses must learn to adapt ecological niches or they become extinct. Scientists have found that several bird species are capable of generating extremely high viremias, which, as a result, causes mosquitoes to become infected as well. Viremia is a medical condition in which viruses enter the bloodstream, giving them access to the entire body. Because mosquitoes bite other animals in order to feed off of their blood, the virus then spreads to them. Viremia in Crows can spread to more than mL of blood, and the mortality is almost uniform. There are many different lineages of WNV, but only the first two lineages are universally accepted.

Potential cancer treatments have been suggested by modulating transcription factors and &quot;emphasiz[ing] agents with established clinical efficacy or with promising effects in preclinical models.&quot; Anand S. Bhagwat and Christopher R. Vakoc, from Cold Spring Harbor Laboratory, have been working on pharmacologically elevating the function of specific TFs related to tumor suppression pathways.

Deep Neural Networks have proven to be incredibly accurate at obtaining sites of transcription factor(TF) binding(TFBS).

The Deep Motif Dashboard provides a suite of visualization strategies to extract motifs or sequence patterns from deep neural networks for TFBS classification without the use of external sources such as protein maps.

Finding a test sequence saliency map using first-order derivatives to find the importance of each nucleotide in TFBS classification.

Understanding genomic sequences is important because of its high correlation of genes with diseases and drugs. By understanding genomic sequences, we can better understand how mutations such as single nucleotide polymorphisms or methylation can change phenotypes and allow scientists to design more effective treatments for genetic illnesses.

Transcription factors are regulatory proteins that bind to DNA(turn genes &quot;on&quot; and &quot;off&quot;). Given an input DNA sequence, DeMo classifies whether or not there&#39;s a binding site for a TF. Given a TF of interest and a dataset made of samples of positive and negative TFBS sequences, the model tests three DL architectures to classify the sequences:

- CNN
- RNN
- CNN-RNN

Secondly, DeMo attempts to understand why they perform the way they do by measuring nucleotide importance with saliency maps, measuring critical sequences positions for the classifier using temporal output scores, and generating class-specific motif patterns with class optimization.

Chromatin immunoprecipitation(ChIP-seq) technologies, or the precipitation of a protein antigen used to isolate and concentrate a protein from a sample containing thousands of proteins, makes finding site locations available for hundreds of different TFs. Unfortunately, ChIP-seq experiments are slow, expensive, and they can&#39;t find patterns common across binding sites(although they can find the binding sites). Thus we cannot find why TFs bind to certain locations, and there is a need for computational methods capable of making accurate binding site classifications that can identify and understand patterns that influence the binding site locations.

Using a deep neural network, three different architectures were tested: CNN, RNN, and a combination of the two, CNN-RNN. The raw nucleotide base characters are used as input to output an output vector of a fixed dimension, which is linearly fed to a softmax function. The final output returns a binary classification task of size 1x1 returning whether the input is a positive or negative binding site. The models use the stochastic gradient algorithm Adam with a mini-batch size of 256 sequences and a dropout regularization method.

However, deep neural networks are often criticized for their &quot;black box&quot; structure in that we cannot completely understand how they make predictions. DeMo attempts to understand which parts of the DNA sequence are the most influential for classification by using saliency maps. Given a sequence of length and class , a DNN model provides a score function of . Because it&#39;s difficult to directly see the influence of each nucleotide on the complex, highly non-linear score function, given , can be approximated by a linear function by computing the first-order Taylor expansion(a representation of a function as an infinite sum of terms that are calculated from the values of the function&#39;s derivatives at a single point):

For which _w_ is the derivative of with respect to the sequence _X_ at the point :

saliency map

The saliency map is simply a weighted sum of the input nucleotides where each weight indicates the influence of that nucleotide position on the output score.

We can find genes by looking for the start codon(ATG) and any of the three end codons. Problem is that there are 6 reading frames. Three in one direction, three in the reverse direction.

GC content is the percentage of nucleotides in a genome that are G or C. For example, a GC content of 50% would mean there is a termination codon every 64 bp. Random DNA won&#39;t show many open reading frames longer than 50 codons in length.

With bacterial genomes, codons are very easy to identify and find. Unfortunately there&#39;s more space in between real genes in eukaryotic genes(62% of human genome is intergenic), there are introns which interrupt the sequencing of DNA, codon bias(not all codons are used equally frequently), exon-intron boundaries, and many other problems. The number or frequency of codons do not necessarily dictate the complexity of an organism. _Escherichia coli_ have 317 codons, humans have 450 codons, and _Saccharomyces cerevisiae_ have 483 codons in their genome.

Researchers at Stanford University investigated how RNN architecture can be used to learn sequential patterns in genomic sequences, giving promising results for epigenetics, or the study of how the genome is regulated by external factors.

Biologists in recent years have also been able to identify transcription factors(regulatory proteins that bind to a particular sequence), DNase I hypersensitive sites(sites sensitive to cleavage by the DNase I enzyme), and histone marks(chemical modifications to histone proteins). Given no additional outside information, the RNN attempts to predict whether a given feature will be present given only the sequence at the nucleotide-level.

Other models have been published in the past. DeepSEA, for example, is a deep neural network consisting of a mixture of convolutional neural networks and pooling layers which worked on a fixed length sequence, along with a sigmoid output layer to compute the probability of seeing a particular genome feature. DeepBind used a deep CNN with 4 stages of convolution, rectification, pooling, and nonlinear activation functions. Although DeepBind incorporated varying-length input, it used additional information pertaining to the features of an input sequence.

Basset, published a few months later, used a deep CNN architecture to learn the functional activity of genomics sequences.

DanQ, published in December 2015, incorporated a convolutional layer and a bidirectional long short term memory(LSTM) recurrent neural network(RNN) on top of a max pooling layer.

All of the following neural networks utilized one-hot encoding of the input sequence into the convolution layer. One-hot encoding expands integer inputs into arrays of a fixed length; for example, given that the nucleotide bases A, T, C, and G are given integer values, each letter in the sequence of DNA of length _L_ would be transformed into a 4 x _L_ matrix. Each tool used a convolutional layer first, which requires fixed sizes. When the length of a particular sequence cannot be discerned or is subject to change, this method becomes invalid. By introducing several sets of convolutional layers, more parameters are introduced, thus a first approach should avoid using convolutional layers.

The following loss function was used by the researchers:

For which _n_ is the number of training samples(about the length of the genome), |V| is the size of the prediction set(4 for the 4 nucleotide bases), is the predicted probability of the predicted word at time _t_ being word _i_, and is a one-hot vector of the real word at time _t_.

For character-level processing, a bidirectional GRU is used because a genetic sequence can be regulated by both the beginning and end sequences. The outputs then go through a softmax layer; for each K task, a prediction of 0 or 1 is made which indicates whether or not a particular feature should be observed for the input sequence. The overall loss is the geometric average of the perplexity across all K tasks. For the multitask prediction problem, |V| equals 2, the number of classes for each class.

For the character-level prediction genome task, the following datasets were prepared in order to test the ability of the neural network to detect sequences:

- A genome with a length of 30,000 where the repeated unit is AGCTTGAGGC
- A genome with a length of 30,000 random genome
- A genome with a length of 4,639,675 for _E. coli_
- A genome with a length of 23,264,338 for _P. palsifarm_(the most common parasite used to spread malaria)

The researchers used a random subset of the dataset used in the DanQ and DeepSEA papers. The dataset was collected from experiments in the ENCODE and Roadmap Epigenomics data releases. They randomly chose 80,000 sequence-label pairs for the training set, 8000 for the testing set, and 2000 for the validation set. Each sequence has a length of 1000.

At the time, training an RNN on all 1000 characters in each example proved to be particularly difficult due to computational limitations. As a result, the RNNs below were trained on the middle 100 characters of each sequence. The baseline model was trained on the same set of truncated sequences.

The first two datasets are used as testing in order to filter out any model not capable of performing well on these genomes.

The first model utilized an LSTM structure, a dropout keep rate of 1, a batch size of 50, a sequence length of 50, a learning rate of 0.002, and 10 epochs. The perplexity of the random genome returned as 4.003, meaning that the model returns the right character 1000 times out of 4003, or about 24.9% of the time. Considering that each letter(&quot;A,&quot; &quot;T,&quot; &quot;C,&quot; and &quot;G&quot;) has an equal probability of being selected in the random genome, this is a good score. The perplexity of the repeated genome was 1.002, or about 99.8% of the time the model predicted the right score. Since the sequence repeats without noise, the model should be able to put all the probability mass on a single character given the previous few characters.

For the _E. coli_ and _P. falciparum_ genomes, the researchers used a dropout keep rate of 1, a batch size of 50, and 1 epoch. They tested: &quot;GRU, LSTM, and simple RNN architectures, learning rates of 0.0001 and 0.0008, sequence lengths of 50, 100, 500, and 1000, and 2 and 3 layers&quot;(4). All combinations of the architectures returned perplexities of 3.679 for _E. coli_ and 2.938 for _P. falciparum_. Although the perplexity was low, training for more than 1 epoch showed a reduction in the perplexity; however, in this experiment the number of epochs was reduced in order to test the hyperparameters and their influence on the perplexity.

Using a learning rate of 0.0008 was much faster than using a learning rate of 0.0001, although this is not surprising considering that smaller perplexities are possible and thus increasing the learning rate in this instance would not lead to diverging from the local minima.

Using 2 or 3 layers did not significantly influence the model&#39;s ability to return smaller perplexities, and, although genomes often involve long-range interactions, sequences of 50 or 100-length sequences performed better than 500 or 1000-length sequences. RNNs and GRUs surprisingly outperformed LSTMs.

On their own, the above results were not useful because genomes have certain repeated structures in certain regions, and an unequal distribution of the nucleotide bases sets a naturally low perplexity. Overall, character-level genome prediction results proved that a 2-layer network with GRUs would be most appropriate for modeling a genomic sequence and that longer input sequences do not necessarily mean lower perplexities.

The researchers also experimented with different levels of feature mapping. Mapping each of the 4 nucleotide bases to integers proved to perform quite poorly because the mapping did not encapsulate how the characters interacted with one another. Then they attempted a different type of feature mapping in which a k-mer bag of words was used, in which each version of an input sequence was counted. For _k_ = 1, 2, 3, 4, 5, there were 1364 features. For example, in the sequence &quot;ACTGG,&quot; feature mapping produced a length-1364 vector in which entries corresponding to the k-mers A, C, T, AC, CT, TG, GG, ACT, CTG, TGG, ACTG, CTGG, and ACTGG would equal 1; entries corresponding to G would equal two. Any other entries would equal 0.

Logistic regression was used with default parameters excluding the regularization constant of the regularization term set to , and, instead of using raw error rate, the researchers looked at the F1 score defined as:

For which precision is the proportion of correct positive predictions and recall is the proportion of positive examples that received a positive prediction. A perfect score would be 1(meaning the model has reached perfect precision and recall) and its worst at 0.

In order to account for the fact that &quot;less than 1% of the training examples were labeled 1&quot;(5), the baseline and RNN models only were compared when at least 1% of the training examples were labeled 1.

Ultimately, the final model used a learning rate of 0.001, 2 hidden layers, an embedding size of 2(because there were only 4 unique characters), a dropout keep probability of 0.95, hidden layer sizes of 128, a bidirectional RNN GRU with tanh activation functions, an input sequence of length 100, an Adam optimizer, and hidden layer with a size of 128.

The model, after training for 61 epochs on 80,000 training examples for 3 days on the NVIDIA GRID K520 GPU, plateaued with an average validation loss trapped at a perplexity of 1.087.

In comparison, the RNN beat the logistic regression model in 94.6% of the tasks, giving an average F1 score of 0.135 in comparison to the logistic regression model&#39;s average score of 0.044.

In conclusion, this project constructed deep bidirectional RNNs capable of predicting and capturing complex patterns in 100-character sequences of the human genome. The researchers concluded that more improvements could be made to their model:

1. Creating more data, for example, extending the 100-character sequences to 200-character sequences.
2. Modifying the hyperparameters further.
3. Designing an RNN cell suited for the long-range information associated with genomic sequences such as the clockwork RNN architecture.
4. Finding a distributed representation of genomic &quot;words&quot; and initializing a word-based RNN model.

Sequence prediction and classification are problems Recurrent Neural Networks(RNNs) are both designed and used for. In theory RNNs have the ability to cope with temporal dependencies; however, when long-term memory is required, they are difficult to train correctly. Using the Clockwork RNN introduces a new model in which &quot;the hidden layer is partitions into separate modules, each processing inputs at its own temporal granularity, making computations only at its prescribed clock rate.&quot; It reduces the number of RNN parameters by half, making it perfect for longer sequences or massive datasets and speeds up network evaluation significantly.

The researchers tested CW-RNNS on two supervised learning tasks: sequence generation where a target auto signal must be outputted without any kind of input and spoken word classification using the TIMIT dataset. Their modifications consisted of adding forward connections and partitioning neurons to the original simple RNN(SRN): &quot;There are forward connections from the input to hidden layer, and from the hidden to output layer . . . the neurons in the hidden layer are partitioned into _g_ modules of size _k._&quot; Each module is assigned a clock period in which every module is interconnected but &quot;the recurrent connections from module _j_ to module _i_ exists only if the period is smaller than .&quot;

The main difference between a CW-RNN and an SRN is that at each time step _t_, only the output of modules _i_ that satisfy are executed. At every forward pass step, only the block-rows of the hidden weight and input matrices are used for evaluation in the modulus equation, and the corresponding parts of the output vector are calculated. The low-clock-rate modules process, retain, and output the long-term information, while higher speed modules focus on high-frequency, local information.

The backward pass is the same as the SRN; however, the error propagates only from modules executed at time step _t_. The error of non-propaged modules is then passed on and copied into the back-propagated error.

# Rationale

Predicting the DNA sequence is an important step in understanding many biological factors. Being able to understand the pattern in deoxyribonucleic acid allows for further technologies such as predicting and understanding transcription factor binding sites. Using architectures such as convolutional layers in neural networks, computers can accurately compile the important features of a model and, using recurrent layers such as gated recurrent units(GRUs) or long short term memory units(LSTMs), networks have the ability to retain information. Although previous sources have attempted to accurately predict the DNA sequence, restrictions on time and computational efficiency have significantly diminished research in this area. By understanding and finding patterns to genomic sequences, we can better understand how mutations such as single nucleotide polymorphisms or how DNA methylation can change phenotypes and allow scientists to design more effective treatments for genetic illnesses. Artificial RNA sequences such as shRNA can silence target gene expression; by identifying locations where certain biomarkers are such as transcription factors, histone markers, or DNase sites, illnesses with a basis in the genome have potential treatments. In addition, treatments have the ability to be specialized for different people and can reduce the risk of silencing important features. For example, in illnesses such as Crohn&#39;s disease, a genetic basis can influence the disease and its severity; however, there are many different factors, making the disease different for different people and making blindly silencing features potentially deadly.

Although CRISPR is a growing technology with the ability to impact millions, it is necessary to pinpoint the specific genes causing illness, and, in order to do so, an understanding of the letters being changed through CRISPR is instrumental from both an ethical and genetic standpoint.

# Introduction

AI, or Artificial Intelligence, is a field of study for many. Some, such as Bill Gates and Elon Musk, believe that AI will destroy humans; however, being able to work together would be the best possible solution, as humans can interact with the physical world easily, while robots find the virtual world has favorable conditions for learning. Machine learning is a type of artificial intelligence which is capable of following instructions not explicitly given to it. Neural networks are a kind of machine learning which enable computers to predict the values of a certain situation, given inputs and their expected outputs. Their roots are derived from our own brains and how billions of neurons communicate with one another. Neural networks(NNs) are a relatively recent addition to the world of artificial intelligence which has taken the world by storm. In biology, NNs have proven incredibly useful as they are capable of tasks such as identifying skin lesions or classifying genomic sequences. Their ability to make complex networks out of a simple perceptron replicates the processes occurring in our brain as axioms interact with one another and allow us to make conclusions and decipher problems. This project uses neural networks in order to identify 919 chromatin features within a genomic sequence: 125 DNase features, 690 transcription factor binding sites, and 104 histone predictions. There are more than 10,000 people who suffer from monogenic disorders, or disorders involving a single error in the genetic code. Viruses kill. Diseases are unstoppable in their relentless pursuit to evolve and destroy our immune system over and over again. Yet there is hope. Using machine learning, we can double our efforts in finding the cure to illnesses while neural networks can aid in the tedious task of finding and identifying chromatin features in the genome.

# Purpose

Is it possible to predict chromatin features in a genomic sequence, and, if so, how can this technology aid in diagnosing illnesses?

# Hypothesis

The researcher hypothesizes that, by extracting features such as transcription factors in a genomic sequence in coordination with deep neural network methods including gated recurrent units and clockwork RNNs, genomic sequence prediction and classification is possible and understandable using features described in the Deep Motif Dashboard(DeMo), created by researchers at the University of Virginia. Using the first-order Taylor expansion, a representation of the model as an infinite sum of terms calculated from the values of the model&#39;s derivatives at a single point, a saliency map can be derived in order to determine the influence of a nucleotide position on the output score. Combining two works focusing separately on optimizing different elements of genomic sequence prediction and comprehension should aid in creating a complex model capable of extracting features accurately.

![](RackMultipart20200517-4-1alg1fx_html_6e43d211ccdaf211.gif)

![](RackMultipart20200517-4-1alg1fx_html_92e2a893a5c44546.png)

Sample inputs and outputs for the NN - the right is a feature map of the first 200 outputs. Each black dot corresponds to a chromatin feature present. To the left is a feature map of the first 200 inputs for which each letter corresponds to a different nucleotide base - yellow representing A, green representing T, blue representing C, and purple representing G.

![](RackMultipart20200517-4-1alg1fx_html_41028c1cb69f1cfc.png)

Source: https://science.sciencemag.org/content/sci/356/6337/489/F1.large.jpg

Batch 1, Epoch 1:

![](RackMultipart20200517-4-1alg1fx_html_578f3bac1a98182e.png)

Batch 1, Epoch 10:

![](RackMultipart20200517-4-1alg1fx_html_b280de08e8b23a5a.png)

A feature map of the differences between the output and the predicted output of the neural network.

![](RackMultipart20200517-4-1alg1fx_html_f43b448b89140c37.png)

This feature map compares the differences between the validation output and the training output.

![](RackMultipart20200517-4-1alg1fx_html_b25d53e9d5087571.gif)

![](RackMultipart20200517-4-1alg1fx_html_49b5d6b18e7c0cfa.png)

The &quot;blue screen of death&quot; encountered in response to a memory overload.

# The Code

This code works given train.mat, a file taken from DeepSea&#39;s database. The Python libraries TensorFlow, h5py, os, NumPy, and Matplotlib must be installed beforehand.

**transcriptionTesting.py**

# Imports

import tensorflow as tf

print(tf.\_\_version\_\_)

import h5py, os

import numpy as np

import matplotlib.pyplot as plt

fileI =&quot;train.mat&quot;# In some cases(eg, VSC running .ipynb file) this path may need to be defined statically.

fileVal =&quot;valid.mat&quot;

print(&quot;Everything is more or less working.&quot;)

# Loading data

# This cell uses a LOT of RAM and battery power.

# train.mat contains 3.4GB of data, but, in some cases,

# RAM usage is much higher. This could be caused by:

&quot;&quot;&quot;

- Other processes

- The source running the .ipynb file(eg VSC has many extra features == extra RAM usage)

- Multiple variables storing the file; shallow copying may account for this, but it still remains a problem.

&quot;&quot;&quot;

print(os.path.exists(fileI))

f = h5py.File(fileI, &quot;r+&quot;)

print(f.keys())

inputs = f.get(&quot;trainxdata&quot;)[()]

print(inputs.shape)

print(inputs[0])

outputs = f.get(&quot;traindata&quot;)[()]

print(&quot;Loaded train data.&quot;)

# Outputs are 919 chromatin features.

# 125 DNase features, 690 TF features, and 104 histone predictions.

# Our inputs, on the other hand,

# Are of length (1000, 4)

# Deleting the initial training array(f) aids in decreasing memory usage a little.

del f

print(inputs[0])

threshold =int(len(inputs) \*3/4)

inputVal = np.transpose(inputs, (2, 0, 1))[threshold:]

outputVal = outputs[:,threshold:].T

print(inputVal.shape, outputVal.shape)

inputs = np.transpose(inputs, (2, 0, 1))[:threshold]

outputs = outputs[:,:threshold].T

print(inputs.shape, outputs.shape)

# Here is some data to show

# This is the input; we can only show a certain

# Amount of letters before it doesn&#39;t look nice anymore,

# So here is a barcode-like representation:

# plt.imshow(inputs[8][:100])

# plt.show()

# Here is the output(validation set), displayed as a bar code.

defargmaxFunc(a):

return np.array([np.argmax(i) for i in a])

# argmaxFunc = np.vectorize(myFunc)

defdraw(inputs, outputs, max\_len=200):

subplots = plt.subplots(200, 2)

for i inrange(len(subplots[1])):

subplots[1][i][0].set\_axis\_off()

subplots[1][i][0].imshow(outputs[i].reshape((1, -1)), aspect=&quot;auto&quot;, cmap=&quot;binary&quot;, interpolation=None)

subplots[1][i][1].set\_axis\_off()

subplots[1][i][1].imshow(argmaxFunc(inputs[i]).reshape((1, -1)), aspect=&quot;auto&quot;, interpolation=None)

return plt

draw(inputVal, outputVal).show()

# Looking at the data

infoFile =&quot;journal.pcbi.1007616.s007.xlsx&quot;

import xlrd

fullCellData = []

workbook = xlrd.open\_workbook(infoFile)

worksheet = workbook.sheet\_by\_index(0)

for row inrange(1, worksheet.nrows):

fullCellData.append({

&quot;cell&quot;: worksheet.cell\_value(row,0),

&quot;regulatory element&quot;: worksheet.cell\_value(row,1),

&quot;treatment&quot;: worksheet.cell\_value(row, 2)

})

# Alright, we&#39;re going to

# - Create a function which takes in the output array,

# - Puts it in alignment with all the other info we have,

# - And graphs the normalized probability of that occurring.

# fullCellData = np.array(fullCellData)

defvisualizeOutput(output):

# Taken in output, let&#39;s try a histogram.

val =0# Where the data will appear

# We can plot the data as a simple scatter plot, as shown here.

plt.plot(output, color=&quot;green&quot;, marker=&quot;o&quot;, markersize=2)

plt.show()

# We can transform it into a pie chart, which will take a little more work....

chart = fullCellData

# Now we can show a chart

for i inrange(len(chart)):

chart[i][&quot;output&quot;] = output[i]

chart.sort(key=lambdaval: val[&quot;output&quot;], reverse=True)

# print(chart[:10])

outputPush = []

for i inrange(len(chart)):

if i %100==0:

print(&quot;%d percent through&quot;%int(i /len(chart)\*100))

outputPush.append(list(chart[i].values()))

plt.table(cellText=outputPush,

loc=&quot;center&quot;,

colLabels=[&quot;Cell Lines&quot;, &quot;Regulatory Element&quot;, &quot;Treatment&quot;, &quot;Probability&quot;])

plt.savefig(&quot;PDF.pdf&quot;)

plt.show()

# np.savetxt(&quot;info.csv&quot;, outputVal[0], delimiter=&quot;,&quot;)

visualizeOutput(outputVal[0])

# # Now there is some more file assorting.

# # We have a file with the names of all the chromatin features we&#39;re looking for; problem is, they are in files.txt.

# fileO = []

# timesBefore = 1

# for i in open(&quot;files.txt&quot;, &quot;r+&quot;).read().split(&quot;\n&quot;):

# if len(i.split(&quot;; &quot;)) \&gt;= 6:

# if len(fileO) \&gt; 0:

# cellName = i.split(&quot;; &quot;)[6][5:]

# if cellName != fileO[-1]:

# fileO.append(cellName)

# else:

# print(&quot;Skipped &quot; + str(timesBefore))

# timesBefore += 1

# else:

# fileO.append(i.split(&quot;; &quot;)[6][5:])

# else:

# print(&quot;SKIPPED&quot;)

# print(len(fileO), fileO)

# Testing whether we have a GPU or not

# As of TF 2, GPU support is used by default, so this only applies

# If we have TF version \&lt; 2.

ifint(tf.\_\_version\_\_[0]) \&lt;2:

if tf.test.is\_gpu\_available():

rnn = tf.keras.layers.CuDNNGRU # This checks if it can use CuDNNGRU.

print(&quot;GPU support enabled.&quot;)

else:

import functools

rnn = functools.partial(

tf.keras.layers.GRU, recurrent\_activation=&#39;tanh&#39;)

print(&quot;GPU not found, defaulting to CPU.&quot;)

else:

rnn = tf.keras.layers.GRU

if tf.test.is\_gpu\_available():

print(&quot;GPU support enabled.&quot;)

else:

print(&quot;GPU will NOT be used. Make sure Cuda is in your PATH.&quot;)

# F1 metric

deff1\_metric(y\_true, y\_pred):

true\_positives = K.sum(K.round(K.clip(y\_true \* y\_pred, 0, 1)))

possible\_positives = K.sum(K.round(K.clip(y\_true, 0, 1)))

predicted\_positives = K.sum(K.round(K.clip(y\_pred, 0, 1)))

precision = true\_positives / (predicted\_positives + K.epsilon())

recall = true\_positives / (possible\_positives + K.epsilon())

f1\_val =2\*(precision\*recall)/(precision+recall+K.epsilon())

return f1\_val

# Now we&#39;re going to work on the actual model.

# --------TEST MODEL----------

defModel(input\_shape, output\_shape, unit1=128, unit2=128):

inputs = tf.keras.Input(input\_shape[1:])

# Simple RNN

recurrent = rnn(unit1, return\_sequences=True)(inputs)

recurrent1 = rnn(unit2, return\_sequences=False)(recurrent)

dense1 = tf.keras.layers.Dense(output\_shape[1], activation=&quot;sigmoid&quot;)(recurrent1) # return\_state

# Dense layer

# dense = tf.keras.layers.Dense(919)(recurrentLayer) #919

print(dense1.shape, inputs.shape)

model = tf.keras.Model(inputs=inputs, outputs=dense1)

return model

# This very simple model is merely a proof of purpose.

# Just to see whether the shapes work, whether the GRU

# Performs more or less correctly, etc.

# Actual training

# Creating the model

print(tf.\_\_version\_\_)

import tensorflow.keras.backend as K

print(inputs.shape, outputs.shape)

print(len(inputs[0][0]))

model = Model(inputs.shape, outputs.shape)

# We will need to reverse the shape in order for this to work

# Work properly, but first let&#39;s check whether this works.

accReadings = []

f1Readings = []

perplexity = []

classMyCallback(tf.keras.callbacks.Callback):

defon\_train\_batch\_end(self, batch, logs=None):

# print(logs[&quot;loss&quot;], tf.math.exp(logs[&quot;loss&quot;]))

perplexity.append(np.exp(logs[&quot;loss&quot;]))

accReadings.append(logs[&quot;loss&quot;])

print(np.exp(logs[&quot;loss&quot;]))

print(&quot;Batch ended with a perplexity of %2d&quot;% np.exp(logs[&quot;loss&quot;]))

defon\_epoch\_end(self, batch, logs=None):

draw(inputVal, model.predict(inputVal)).show()

model.compile(

loss=tf.keras.losses.BinaryCrossentropy(),

optimizer=tf.keras.optimizers.Adam(),

metrics=[&quot;accuracy&quot;, f1\_metric])

model.summary()

#

# Now that the model is created, we finally

# Can try out the training.

# Proper model

# Taken from DeepSea model

defModel(input\_shape, output\_shape):

# Input

model = tf.keras.Sequential()

model.add(tf.keras.Input(input\_shape[1:]))

&quot;&quot;&quot;

&quot;The basic layer types in out model are convolution layer, pooling layer and fully connected layer. A convolution layer computes output by one-dimensional convolution operation with a specified number of kernels . . . . In the first convolution layer, each kernel can be considered as a position weight matrix(PWM) and the convolution operation is equivilent to computing the PWM scores with a moving window with step size one on the sequence.&quot;

&quot;&quot;&quot;

&quot;&quot;&quot;Here is the code in Lua:

model:add(nn.SpatialConvolutionMM(nfeats, nkernels[1], 1, 8, 1, 1, 0):cuda())

model:add(nn.Threshold(0, 1e-6):cuda())

model:add(nn.SpatialMaxPooling(1,4,1,4):cuda())

model:add(nn.Dropout(0.2):cuda())

model:add(nn.SpatialConvolutionMM(nkernels[1], nkernels[2], 1, 8, 1, 1, 0):cuda())

model:add(nn.Threshold(0, 1e-6):cuda())

model:add(nn.SpatialMaxPooling(1,4,1,4):cuda())

model:add(nn.Dropout(0.2):cuda())

model:add(nn.SpatialConvolutionMM(nkernels[2], nkernels[3], 1, 8, 1, 1, 0):cuda())

model:add(nn.Threshold(0, 1e-6):cuda())

model:add(nn.Dropout(0.5):cuda())

nchannel = math.floor((math.floor((width-7)/4.0)-7)/4.0)-7

model:add(nn.Reshape(nkernels[3]\*nchannel))

model:add(nn.Linear(nkernels[3]\*nchannel, noutputs))

model:add(nn.Threshold(0, 1e-6):cuda())

model:add(nn.Linear(noutputs , noutputs):cuda())

model:add(nn.Sigmoid():cuda())

&quot;&quot;&quot;

nkernels = [4, 320,480,960]

dropout = [0.2, 0.2, 0.5]

# We have 3 rounds of convolution. Each one contains a convolution layer with output of kernel size nkernels[i], a threshold(which we can implement later...), and a dropout.

model.add(tf.keras.layers.Conv1D(4, 2, 1, &quot;valid&quot;))

model.add(tf.keras.layers.MaxPooling1D(2))

model.add(tf.keras.layers.Dropout(dropout[i]))

model.summary()

return model

print(inputs.shape, outputs.shape)

model = Model(inputs.shape, outputs.shape)

# Procedure

1) Obtain and download the data from:

- The Deep Motif Dashboard(https://media.githubusercontent.com/media/QData/DeepMotif/master/data/deepbind.tar.gz)

- The ENCODE database(http://deepsea.princeton.edu/media/code/deepsea\_train\_bundle.v0.9.tar.gz)

- The _E. coli_ genome(ftp://ftp.ensemblgenomes.org/pub/bacteria/release-46/fasta/bacteria\_0\_collection/escherichia\_coli\_str\_k\_12\_substr\_mg1655/dna/)

2) Download Pip, Python 3.7.x, and the following Python libraries:

- timeit

- TensorFlow

- OS

- h5py

- NumPy

- Keras

- Matplotlib

3) Begin programming, using Python as a mainframe.

a) Work on a character-level prediction neural network. This neural network should take in N number of input letters and output N output letters, using the k-mer bag of words method.

b) Work on a binary classification prediction neural network, taking in N number of input letters and outputting whether there are clear transcription factors in the DNA.

c) Work on modification of the neural network by implementing graphs using the Matplotlib library.

d) Work on efficiency of the neural network by comparing the speed of the GPU and CPU, running a simple test(by timing how long it takes each processing unit to run a convolutional 32x7x7x3 filter over random 100x100x100x3 vectors) and utilizing whichever works fastest. Although typically the GPU works significantly faster, in some cases and in past bugs TensorFlow has been revealed to have efficiency problems with GPUs not configured correctly or certain older GPUs.

# Materials

- 1 computer(the researcher is using an HP Envy 13 with an NVIDIA GeForce MX250 GPU)

# Conclusion

Ultimately this project resulted in creating a neural network capable of understanding genomic sequences and predicting 919 chromatin features: 125 DNase features, 670 transcription factor features, and 104 histone features. Using data scraped from the ENCODE and Roadmap epigenomics databases, the final product depended heavily on DeepSea by building on their dataset and working on incorporating different techniques used in other studies such as recurrent neural networks and gated recurrent units(GRUs). This project also focused particularly on comprehension and analysis of the data by using the &quot;Matplotlib&quot; library in order to render the inputs and outputs as one dimensional feature maps and aligning the different chromatin features with their corresponding cell types, regulatory elements, and treatments.

## Problems Encountered

The researcher encountered a multitude of problems while working on this project. DeepSea utilized a desktop PC with massive computing power while the researcher used a smaller PC with a relatively high-end GPU not only because of the expense but also as a proof of purpose; neural networks can be written on low-end PCs and perform similarly because ultimately, even with a lower-end processor, NNs can still grasp the concepts and prove that patterns do exist in data. Previously the researcher used .py files; however, they were found to be time-expensive due to the massive size of the training file and the time it took to load which significantly slowed advancements in writing the code. The researcher ultimately used .ipynb files, a type of Python file which allows for running pieces at one time. Different Python environments contributed to errors as different Tensorflow environments allowed for different capabilities and restrictions. The newest version of Tensorflow, 2.1, allows for Tensorflow add ons such as a built-in F1 metric; however, tf.contrib, referenced multiple times in the original code, was removed from that version of Tensorflow, meaning that either the researcher could enable the F1 metric and scrap the tf.contrib components or disable the F1 metric altogether and keep the original code. Ultimately the researcher incorporated the F1 metric by hand and downgraded Tensorflow to a more usable version less focused on detail.

The researcher also found that past research in this area has proven to be very general and not explained in sufficient detail to be potentially replicated by other researchers. Dr. Jessez Zhang, for example, provided an accurate description of his experiments with enough parameters defined to be replicated. DeepSea, on the other hand, provided specific code; however, the code was most likely not the final version used because they designed a convolutional neural network with invalid parameters. Not only were the filters set to obscene values, the CNN was set to take in a three dimensional input when the research paper explicitly stated that the CNN would take in a two dimensional input and the reshaping of the input in the code corresponded to reshaping the data to a two dimensional input.

##


## Future Expansions

Potentially this project, given more time and more processing power, could attempt to use more parameters, or perceptrons; because the memory limit was already very close to capacity, the level of complexity the model could have was very low. Potentially this project could also aid in designing antibodies to combat viruses such as the coronavirus.

## Practical Applications

This project could be applied in modern medicine in order to use technologies such as CRISPR or shRNA to silence features such as transcription factors. For example, the treatment for the coronavirus lies in targeting biomarkers and silencing the part that allows them to attack our immune systems and render us useless. This also could provide the cure to rare illnesses in which very little time has been spent determining the factors which influence these diseases such as Crohn&#39;s disease.

## Bibliography

Brown, Terence A. &quot;Understanding a Genome Sequence.&quot; Genomes. 2nd Edition., U.S. National Library of Medicine, 1 Jan. 1970, www.ncbi.nlm.nih.gov/books/NBK21136/.

Culex Pipiens(Diptera: Culicidae) to Transmit West Nile Virus.&quot; Journal of Medical Entomology, vol. 39, no. 1, Jan. 2002, pp. 221–225., doi:10.1603/0022-2585-39.1.221.

&quot;Escherichia Coli Str. K-12 Substr. MG1655.&quot; Escherichia Coli Str. K-12 Substr. MG1655 - Ensembl Genomes 46, bacteria.ensembl.org/Escherichia\_coli\_str\_k\_12\_substr\_mg1655/Info/Index.

Koutník, Jan. &quot;A Clockwork RNN.&quot; Cornell University, 14 Feb. 2014, arxiv.org/abs/1402.3511.

Lanchantin, Jack, et al. &quot;Deep Motif Dashboard: Visualizing and Understanding Genomic Sequences Using Deep Neural Networks.&quot; Pacific Symposium on Biocomputing. Pacific Symposium on Biocomputing, U.S. National Library of Medicine, 2017, www.ncbi.nlm.nih.gov/pmc/articles/PMC5787355/.

Redell, Michele S, and David J Tweardy. &quot;Targeting Transcription Factors in Cancer: Challenges and Evolving Strategies.&quot; ScienceDirect, Elsevier, 31 Oct. 2006, www.sciencedirect.com/science/article/abs/pii/S1740674906000588.

Zhang, Jesse M, and Govinda M Kamath. &quot;Learning the Language of the Genome Using RNNs.&quot; Deep Learning for Natural Language Processing, cs224d.stanford.edu/reports/jessesz.pdf.

Zhou, Jian, and Olga G Troyanskaya. &quot;Predicting Effects of Noncoding Variants with Deep Learning-Based Sequence Model.&quot; _Nature Methods_, U.S. National Library of Medicine, Oct. 2015, www.ncbi.nlm.nih.gov/pmc/articles/PMC4768299/.