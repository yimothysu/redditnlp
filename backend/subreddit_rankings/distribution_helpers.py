import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore

def get_percentile_of_val_in_distribution(values, target_value):
    percentile = percentileofscore(values, target_value, kind='rank')
    return round(percentile, 4)

def get_toxicity_grade(toxicity):
    # A+ means extremely civil and non-toxic subreddit. very respectful
    # F- means extremely toxic subreddit 

    # toxicity is from [0, 1]
    # I want to convert it to a [0, 100] scale
    toxicity *= 100
    if (toxicity <= 0.05): return "A+" # 0.05 = 1 / 2000
    elif (toxicity <= 0.1): return "A" # 0.1 = 1 / 1000
    elif(toxicity <= 0.5): return "B+" # 0.5 = 1 / 200
    elif (toxicity <= 1): return "B" 
    elif (toxicity <= 5): return "B-"
    elif (toxicity <= 10): return "C+"
    elif (toxicity <= 15): return "C"
    elif (toxicity <= 20): return "C-"
    elif (toxicity <= 30): return "D+"
    elif (toxicity <= 40): return "D"
    elif (toxicity <= 50): return "D-"
    else: return "F"

def get_positive_content_grade(positive_content_score):
    # A+ means very positive content subreddit
    # F means very negative content subreddit 

    # positive_content is from [0, 1]
    # I want to convert it to a [0, 100] scale
    positive_content_score *= 100
    if (positive_content_score >= 92): return "A+" 
    elif (positive_content_score >= 84): return "A" 
    elif(positive_content_score >= 75): return "B+" 
    elif (positive_content_score >= 68): return "B" 
    elif (positive_content_score >= 60): return "B-"
    elif (positive_content_score >= 52): return "C+"
    elif (positive_content_score >= 43): return "C"
    elif (positive_content_score >= 35): return "C-"
    elif (positive_content_score >= 26): return "D+"
    elif (positive_content_score >= 17): return "D"
    elif (positive_content_score >= 9): return "D-"
    else: return "F"


def get_distribution_statistics(values):
    mean = np.mean(values)
    median = np.median(values)
    lower_quartile = np.percentile(values, 25)  
    upper_quartile = np.percentile(values, 75)  
    std_dev = np.std(values, ddof=0) 
    min_value = np.min(values)
    max_value = np.max(values)

    print(f"Mean: {mean:.4f}")
    print(f"Median: {median:.4f}")
    print(f"Lower Quartile (Q1): {lower_quartile:.4f}")
    print(f"Upper Quartile (Q3): {upper_quartile:.4f}")
    print(f"Standard Deviation: {std_dev:.4f}")
    print(f"Min: {min_value:.4f}")
    print(f"Max: {max_value:.4f}")

def plot_distribution(values, xlabel, ylabel, title, range_min, range_max, num_bins=50):
    plt.hist(values, bins=num_bins, range=(range_min, range_max), edgecolor='black', alpha=0.75)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

