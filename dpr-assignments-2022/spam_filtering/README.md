# Spam Filtering using Naive Bayes
In this assignment we will implement a spam filter using Naive Bayes.

If you're new to Python you might want to do a Python tutorial first. For NumPy and scikit-learn, have a look at https://numpy.org/doc/stable/user/index.html and https://scikit-learn.org/stable/user_guide.html, respectively.

# Results
1. For the general case, `|X|^n*|C|` parameters are required to estimate the full joint probability distribution `P(x_1,...,x_n | c)`, where `n` is the message length, `X` the set of possible tokens, and `C` the set of classes. For the naive assumption, we only have to estimate `|X|*|C|` parameters.
2. In practice we don't have infinite amount of data and therefore it's generally impossible to get a good estimate of our likelihood parameters. The Naive Bayes assumption resolves the curse of dimensionality by assuming  that the words in a message are conditionally independent. This assumption, of course, is not true in reality:  words often appear together and can have different meanings depending on the context.  But it’s a useful simplification that leads to practical results.
3. Smoothing ensures that if a token `t` does not occur in class `S`/`H`, the associated likelihood values `P(t | H)` / `P(t | S)` do not equal to zero. Otherwise, if the training data does not contain a certain word `t` for the spam class `S`, any message containing `t` would never be classified as spam.
4. You should achieve a train accuracy of about 99% and a test accuracy of about 98%.
