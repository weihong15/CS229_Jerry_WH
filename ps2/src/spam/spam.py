import collections

import numpy as np

import util
import svm

def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    return message.lower().split(' ')
    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use 
    to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    store = {}
    for i, message in enumerate(messages):
        for word in get_words(message):
            try:
                store[word].add(i)
            except KeyError:
                store[word] = {i}
    # remove list of words that appear in fewer than 5 messages
    s = {k for k, v in store.items() if len(v) >= 5}
    #return {k : v for v, k in enumerate(s)}
    # a sorted order of dict for debugging
    return {k : v for v, k in enumerate(sorted(list(s)))}
    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message.
    Each row in the resulting array should correspond to each message
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***
    # Create an array[i,j] like this where M_i is the i-th message:
    #        word-0  word-1  word-2
    #  M_0      1      2       0
    #  M_1      3      0       1
    #  M_2      4      1       3
    z = np.zeros((len(messages), len(word_dictionary)))
    for i, message in enumerate(messages):
        for word in get_words(message):
            if word in word_dictionary: 
                z[i][word_dictionary[word]] += 1
    return z
    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """
    # *** START CODE HERE ***
    # cast to boolean for proper invert
    labels_ = np.array(labels, dtype=np.bool_)
    M, V = matrix.shape
    # phi_{k|y=0}, phi_{k|y=1}, each length-V array
    phi_k = np.zeros((2, V))
    # numerator with Laplace smoothing
    phi_k[0] = np.sum((matrix.T * ~labels_).T, axis = 0) + 1
    phi_k[1] = np.sum((matrix.T * labels_).T, axis = 0) + 1
    # denominator with Laplace smoothing
    phi_k[0] /= (np.sum((matrix.T * ~labels_).T) + V)
    phi_k[1] /= (np.sum((matrix.T * labels_).T) + V)
    phi_y = labels_.sum() / M
    return phi_k, phi_y
    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    # *** START CODE HERE ***
    # P(y=1|x) = P(x|y=1) * P(y=1) / P(x); ignore the denomiator
    #          => Π(phi_k[1]^x_i) * phi_y ;
    # P(y=0|x) = P(x|y=0) * P(y=0) / P(x);
    #          => Π(phi_k[0]^x_i) * (1-phi_y)
    # log(P(y=1|x)) = sigma(log(phi_k[1]) * x_i) + log(phi_y)
    # log(P(y=0|x)) = sigma(log(phi_k[0]) * x_i) + log(1-phi_y)
    phi_k, phi_y = model
    M, _ = matrix.shape
    pred = np.zeros(M)
    for i in range(M):
        # Use dot product to calculate sigma(log(phi_k[1]) * x_i); add log(1 - phi_y) and log(phi_y)
        prob_zero = np.dot(np.log(phi_k[0]), matrix[i, :]) + np.log(1 - phi_y)
        prob_one = np.dot(np.log(phi_k[1]), matrix[i, :]) + np.log(phi_y)
        if prob_one > prob_zero:
            pred[i] = 1
    return pred

def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    phi_k, _ = model
    # claculate the top 5 tokens ranked by log-indicative
    top_tokens = np.argsort(np.log(phi_k[1]) - np.log(phi_k[0]))[-5:][::-1]
    idx_to_word = {v: k for k, v in dictionary.items()}
    return [idx_to_word[idx] for idx in top_tokens]
    # *** END CODE HERE ***


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider

    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***
    prediction_batch = np.array([svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, r) for r in radius_to_consider])
    correctness = [val_labels == prediction for prediction in prediction_batch]
    correct_predict = [sum(val_labels == prediction) for prediction in prediction_batch]
    best_r = np.argmax(correct_predict)
    return radius_to_consider[best_r]
    # *** END CODE HERE ***


def main():
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')

    dictionary = create_dictionary(train_messages)

    print('Size of dictionary: ', len(dictionary))

    util.write_json('spam_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('spam_sample_train_matrix', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('spam_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('spam_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('spam_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))


if __name__ == "__main__":
    main()