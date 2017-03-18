## Data Processing

### Overview of initial pre-processing
Our initial method of processing the raw reddit data uses the pandas(pd) library. 
We start by reading in our different files into a DataFrame(df) and remove extraneous data.
After this we are ready to start cleaning up our data. 
#### Initial Clean
The initial clean consists of
1. Remove commments that are deleted. 
2. Document whether a comment is a root comment. 
3. Replace digits, contractions, links. 
4. Remove large comments. 

After the initial clean, we generate a word frequency dictionary by tokenizing our comments
and tracking the word usage. This word frequency is used to provide a 'validity' score for 
our comments. 

#### Validity score

The validity score relies on the word frequncy dictionary and the pyEnchant library. At this
point the validity is a function of indpendent word usage only. We keep a penalty score
which increases for words that are not in the pyEnchant english dictionary. The penalty
is the inverse of the number of occurences of the specific word in all of the processed comments.

### Drawbacks of method

This method is not very efficient. Specifically the computation of sentence scores is time consuming.
The order of operations is being explored and the ability to save progress needs to be added. 

### Further Analysis
A natural question that arises is how much data should be used to generate the word dictionaries. 
Due to the size of the files, we will probably need to modularize this as much as possible.


## Using sqlite 
