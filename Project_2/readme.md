PROJECT 2 README:
=================

Part 1:
----------

This function was programmed in Python 2.7. There are some functions that will not work in Python 3.

To run the code, pull up a command prompt and run:

```
python gibbsSampler.python
```
Then, you will be prompted to give an input. This is in the form:

```
targetNode evidence1=state1 evidence2=state2 -u nRun -d nDrop
```
Note: Both the order and the fact that there are no spaces in `evidence1=state1` is important.

Once this input is given, the function will print the probability of each possible value that targetNode can take. There may be intermediate text in between.