

# In this tutorial, we will show how to use DeepArchitect to implement
# search spaces from the literature.
# We will take the textual description of a paper in the architecture search
# literature and we will show how can we write it in DeepArchitect.
# This exercise serves to show to the reader the process by which one would go
# about writing existing search spaces from the literature in DeepArchitect.

# NOTE: that the search space simply encodes the set of architecture that can
# be considered. it says nothing about how are these architectures going to
# be evaluated.
# for most cases that we have seen in the literature, this can also be done
# under our framework, but we leave it for another time.

# ### Zoph and Le 2017

# We will follow the description of the search space as it is done in the paper.
# We make no effort to implement idiosyncracies in the actual implementation.
# Let us take one of the first architecture search papers: Zoph and Le, 2017.
# In section 4.1, they describe their search space:


# In our framework, if one layer has many input layers then all input layers are
# concatenated in the depth dimension. Skip connections can cause “compilation
# failures” where one layer is not compatible with another layer, or one layer
# may not have any input or output. To circumvent these issues, we employ three
# simple techniques. First, if a layer is not connected to any input layer then
# the image is used as the input layer. Second, at the final layer we take all
# layer outputs that have not been connected and concatenate them before sending
# this final hiddenstate to the classifier. Lastly, if input layers to be
# concatenated have different sizes, we pad the small layers with zeros so that the
# concatenated layers have the same sizes.


# 4.1 LEARNING CONVOLUTIONAL ARCHITECTURES FOR CIFAR-10

# Search space: Our search space consists of convolutional architectures, with
# rectified linear units as non-linearities (Nair & Hinton, 2010),
# batch normalization (Ioffe & Szegedy, 2015) and skip connections between layers
# (Section 3.3). For every convolutional layer, the controller RNN has to select
# a filter height in [1, 3, 5, 7], a filter width in [1, 3, 5, 7], and a number of
# filters in [24, 36, 48,64]. For strides, we perform two sets of experiments,
# one where we fix the strides to be 1, and one where we allow the controller to
# predict the strides in [1, 2, 3].

# NOTE: this is an important part of teh model. I think that it is most important
# to consider it.

# Dataset: In these experiments we use the CIFAR-10 dataset with data preprocessing and augmentation
# procedures that are in line with other previous results. We first preprocess the data by
# whitening all the images. Additionally, we upsample each image then choose a random 32x32 crop
# of this upsampled image. Finally, we use random horizontal flips on this 32x32 cropped image.



### Negrinho and Gordon, 2017

# TODO: short description of what DeepArchitect is about.
# This was the original paper for DeepArchitect having

#### TODO: do the example for DeepArchitect




# Let us take one of the examples from original DeepArchitect paper
# https://arxiv.org/abs/1704.08792.
# The main ideas from writing a DSL to express search spaces came from this paper.
# The ideas were considerably extended for this current implementation
# (e.g., multi-input multi-output modules, hyperparameter sharing, general
# framework support, ...).
# The original DeepArchitect paper presented the following search space

# def Module_fn(filter_ns, filter_ls, keep_ps, repeat_ns):
#     b = RepeatTied(
#     Concat([
#         Conv2D(filter_ns, filter_ls, [1], ["SAME"]),
#         MaybeSwap_fn( ReLU(), BatchNormalization() ),
#         Optional_fn( Dropout(keep_ps) )
#     ]), repeat_ns)
#     return b

# filter_nums = range(48, 129, 16)
# repeat_nums = [2 ** i for i in xrange(6)]
# mult_fn = lambda ls, alpha: list(alpha * np.array(ls))
# M = Concat([MH,
#         Conv2D(filter_nums, [3, 5, 7], [2], ["SAME"]),
#         Module_fn(filter_nums, [3, 5], [0.5, 0.9], repeat_nums),
#         Conv2D(filter_nums, [3, 5, 7], [2], ["SAME"]),
#         Module_fn(mult_fn(filter_nums, 2), [3, 5], [0.5, 0.9], repeat_nums),
#         Affine([num_classes], aff_initers) ])


# ### NOTE: I could develop this one.


# ### NOTE: now talk about a different search space.

# Neural Architecture Search with Reinforcement Learning (Zoph and Le. 2016)
# https://arxiv.org/abs/1611.01578


####

The description of the search space employed is spread out across the paper.
The main idea of the paper is to write architectures as hierarchical compositions
of motifs.

Architectures in this case are just motifs at a higher level of composition.
In the description of the paper, a motif is generated by connecting lower level
using an acyclical directed graph.
All motifs have a single input and a single output.
If multiple edges come into a node, the inputs are merged into a single
input and then passed as input to the motif.
Each of the lower motifs is placed in one of the nodes of the graph and the

Given some number of higher level motifs, these can be constructed

There are the high-level ideas for the paper.


NOTE: the motif is generated by putting the elements in topological order and
picking which ones should be there.



To paraphrase and summarize what they describe in the paper, they start with a
set of six operations (which they call primitives).


This is used to search for a cell that is then used in an architecture for
evaluation. Be


# We consider the following six primitives at the bottom level of the hierarchy (` = 1, M` = 6):
# • 1 × 1 convolution of C channels
# • 3 × 3 depthwise convolution
# • 3 × 3 separable convolution of C channels
# • 3 × 3 max-pooling
# • 3 × 3 average-pooling
# • identity

### TODO: what is going to be the plan for the other parts of the model.
# I think that this is going to be useful.

# NOTE: multiplying the initial number of filters.
# how to extract a search space from it. that seems kind of tricky?


For the hierarchical representation, we use three levels (L = 3),
with M1 = 6, M2 = 6, M3 = 1. Each of the level-2 motifs is a graph with
# |G(2)| = 4 nodes, and the level-3 motif is a graph with |G(3)| = 5 nodes.
# Each level-2 motif is followed by a 1×1 convolution
# with the same number of channels as on the motif input to reduce the number
# of parameters. For the flat representation, we used a graph with 11 nodes to
# achieve a comparable number of edges.


# NOTE: the sampling of the motifs is going to be important.
# I think that this is important.


The two main ideas are the

# Number of lower level motifs. (number of acyclical graphs between these models)


# L = 3;



# This is the hierarchical composition.


It is arguably more complicated to understand the textual description of the
search space in the paper than to express it in DeepArchitect.
This is a strong indicator of value of DeepArchitect for greatly improving the
reproducibility and reusability of architecture search research.

# NOTE: it would

# NOTE: most of these models where used.

# TODO: show some information in the model.


# TODO: show three architectures sampled from the search space.
# I think that this is going to be interesting.


#### Genetic CNN,

# TODO: distinguish place where we are talking and places where someone else is
# talking.

# TODO: another aspect that I want to show off is how to take the existing
# parts of the model, and put them together.
# I think that this is the right way of go




# and many more. If you would like me to write a search space that is not
# covered in this model, just let me know and I will consider adding it.


#

# In this tutorial we showed that it is possible to use the language constructs
# that we defined in DeepArchitect to implement the search spaces over architectures.
# In many cases, the textual description of the search space in the paper is
# more complex than the description in DeepArchitect, show casing the capabilities
# of our framework.


# The papers were chosen part based on chronology and part based on the number
# of citations that they


# TODO: searching for activation functions.

# We believe that DeepArchitect will make research in architecture search
# dramatically easier, allowing researchers to more easily build on previous
# work.

# We are confident that the primitives that were introduced are sufficient to
# represent a very large number of search spaces.

# TODO: check with max what does he have for the concatenation of search
# spaces.

# what do they talk about depth wise and separable convolution.

# in merging stuff, I think that it is a good idea to have them run.
# TODO: what is a depth

# they end up having three levels.
# NOTE: it should be easier to maintain if they are in multiple files.
# it should generate a high level tutorial file, but I don't know exactly
# what should be the structure for it.


# let us get started from scratch.
# we will use

# NOTE: this is just the search space for the cell, if we wish
# to write something