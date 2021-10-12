# Input representation

## The Data

The data used for the model is comprised of over 150,000 human games, each represented in a series of turns also known as phases.

*Present data information structure explanation*

## Input Data

There's probably no surprise that the data needs to be transformed or represented in a way that is more understandable and could be used as an input to the model.

In the original paper, the authors used two things as inputs: **current board state** and **previous phase orders**.

Both the board state and orders are represented as one-hot-encoded feature vectors for each province.

**The board state** is encoded according to the province type, whether there is a unit on that province, the power owning that unit, ect.

**The previous phase orders** are encoded in such a way to should help infer the alignment among the powers, thus the encoded information includes: unit type, the owning power, order type, ect.

A full and concise representation shceme of the board state and previous phase orders is presented below:

```{image} ./images/representation_scheme.png
:alt: representation_scheme
:align: center
```






