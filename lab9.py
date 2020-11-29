# MIT 6.034 Lab 9: Boosting (Adaboost)
# Written by 6.034 staff

from math import log as ln
from utils import *


#### Part 1: Helper functions ##################################################

def initialize_weights(training_points):
    """Assigns every training point a weight equal to 1/N, where N is the number
    of training points.  Returns a dictionary mapping points to weights."""
    N = len(training_points)
    return {point: make_fraction(1,N) for point in training_points}

def calculate_error_rates(point_to_weight, classifier_to_misclassified):
    """Given a dictionary mapping training points to their weights, and another
    dictionary mapping classifiers to the training points they misclassify,
    returns a dictionary mapping classifiers to their error rates."""
    classifiers = classifier_to_misclassified.keys()
    return {classifier: sum([point_to_weight[point] for point in classifier_to_misclassified[classifier]]) for classifier in classifiers}

def pick_best_classifier(classifier_to_error_rate, use_smallest_error=True):
    """Given a dictionary mapping classifiers to their error rates, returns the
    best* classifier, or raises NoGoodClassifiersError if best* classifier has
    error rate 1/2.  best* means 'smallest error rate' if use_smallest_error
    is True, otherwise 'error rate furthest from 1/2'."""
    
    if use_smallest_error:
        best_classifier = min(classifier_to_error_rate.keys(), key=lambda x: classifier_to_error_rate[x])
    else:
        best_classifier = max(classifier_to_error_rate.keys(), key=lambda x: abs(make_fraction(0.5) - classifier_to_error_rate[x]))

    if classifier_to_error_rate[best_classifier] != make_fraction(0.5):
        return best_classifier

    raise NoGoodClassifiersError

def calculate_voting_power(error_rate):
    """Given a classifier's error rate (a number), returns the voting power
    (aka alpha, or coefficient) for that classifier."""
    if error_rate == 0:
        return INF
    if error_rate == 1:
        return -1 * INF

    return 0.5 * ln( ( (1 - error_rate) / error_rate ) )

def get_overall_misclassifications(H, training_points, classifier_to_misclassified):
    """Given an overall classifier H, a list of all training points, and a
    dictionary mapping classifiers to the training points they misclassify,
    returns a set containing the training points that H misclassifies.
    H is represented as a list of (classifier, voting_power) tuples."""
    
    '''
     Basically, you are looking at the sum of voting powers. 
     You should iterate through the points and for each point, 
     you have to see if a classifier misclassifies a point, 
     and you subtract the voting power and if it doesn't, 
     you add the voting power. If voting powers are less than 0, 
     then that point is misclassfied.
    '''

    misclassified = set()
    for point in training_points:
        vote = 0
        for (classifier, voting_power) in H:
            if point in classifier_to_misclassified[classifier]:
                vote = vote - voting_power
            else:
                vote = vote + voting_power

        if vote <= 0:
            misclassified.add(point)

    return misclassified

def is_good_enough(H, training_points, classifier_to_misclassified, mistake_tolerance=0):
    """Given an overall classifier H, a list of all training points, a
    dictionary mapping classifiers to the training points they misclassify, and
    a mistake tolerance (the maximum number of allowed misclassifications),
    returns False if H misclassifies more points than the tolerance allows,
    otherwise True.  H is represented as a list of (classifier, voting_power)
    tuples."""
    mistakes = len(get_overall_misclassifications(H, training_points, classifier_to_misclassified))
    return mistakes <= mistake_tolerance

def update_weights(point_to_weight, misclassified_points, error_rate):
    """Given a dictionary mapping training points to their old weights, a list
    of training points misclassified by the current weak classifier, and the
    error rate of the current weak classifier, returns a dictionary mapping
    training points to their new weights.  This function is allowed (but not
    required) to modify the input dictionary point_to_weight."""
    for point in point_to_weight:
        if point in misclassified_points:
            point_to_weight[point] = make_fraction(0.5) * make_fraction(1, error_rate) * point_to_weight[point]
        else:
            point_to_weight[point] = make_fraction(0.5) * make_fraction(1, 1 - error_rate) * point_to_weight[point]
    return point_to_weight



#### Part 2: Adaboost ##########################################################

def adaboost(training_points, classifier_to_misclassified,
             use_smallest_error=True, mistake_tolerance=0, max_rounds=INF):
    """Performs the Adaboost algorithm for up to max_rounds rounds.
    Returns the resulting overall classifier H, represented as a list of
    (classifier, voting_power) tuples."""
    
    """
    1. Initialize all training points' weights.
    2. Compute the error rate of each weak classifier.
    3. Pick the "best" weak classifier h, by some definition of "best."
    4. Use the error rate of h to compute the voting power for h.
    5. Append h, along with its voting power, to the ensemble classifier H.
    6. Update weights in preparation for the next round.
    7. Repeat steps 2-7 until no good classifier remains, we have reached some max number of iterations, or H is "good enough."
    """

    # step 1 
    point_to_weight = initialize_weights(training_points)
    iterations = 0
    H = []
    while iterations < max_rounds:
        # step 2
        classifier_to_error_rate = calculate_error_rates(point_to_weight, classifier_to_misclassified)
        #step 3
        try:
            best_classifier = pick_best_classifier(classifier_to_error_rate, use_smallest_error)
            error_rate = classifier_to_error_rate[best_classifier]
        except:
            return H
        # step 4-5
        H.append((best_classifier, calculate_voting_power(error_rate)))
        # step 6
        if is_good_enough(H, training_points, classifier_to_misclassified, mistake_tolerance):
            return H
        misclassified_points = classifier_to_misclassified[best_classifier]
        point_to_weight = update_weights(point_to_weight, misclassified_points, error_rate)
        iterations += 1

    return H

#### SURVEY ####################################################################

NAME = "Nabib Ahmed"
COLLABORATORS = "None"
HOW_MANY_HOURS_THIS_LAB_TOOK = 3
WHAT_I_FOUND_INTERESTING = "The modular structure"
WHAT_I_FOUND_BORING = "Nothing"
SUGGESTIONS = "None"
