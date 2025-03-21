import distribution_helpers as helpers
import matplotlib.pyplot as plt

positive_content_dict = {}

with open("top_subreddits_positive_content_score.txt", "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split(" ") 
        if len(parts) == 2: 
            subreddit_name, score = parts
            positive_content_dict[subreddit_name] = round(float(score), 6)

print('len(positive_content_dict): ', len(positive_content_dict))

# in ascending order
sorted_positive_content_dict = dict(sorted(positive_content_dict.items(), key=lambda item: item[1], reverse=True))
#print(sorted_positive_content_dict) 

# helpers.get_distribution_statistics(list(sorted_positive_content_dict.values()))
# helpers.plot_distribution(sorted_positive_content_dict.values(), 
#                           xlabel='Positive Content Score Range (-1 to 1)',
#                           ylabel='Frequency',
#                           title='Distribution of Subreddits\' Positive Content Scores',
#                           range_min=-1,
#                           range_max=1)

normalized_values = [(x + 1) / 2 for x in list(sorted_positive_content_dict.values())]
helpers.get_distribution_statistics(normalized_values)
helpers.plot_distribution(normalized_values, 
                          xlabel='Positive Content Score Range (0 to 1)',
                          ylabel='Frequency',
                          title='Distribution of Subreddits\' Positive Content Scores',
                          range_min=0,
                          range_max=1)

grade_distribution = {}
for x in normalized_values:
    grade = helpers.get_positive_content_grade(x)
    if (grade not in grade_distribution): grade_distribution[grade] = 0
    grade_distribution[grade] += 1

total_count = len(normalized_values)
for grade, count in grade_distribution.items():
    percent = round((count / total_count) * 100, 1)
    print(grade, ': ', count, ' ---> ', percent, '%')

grades = list(grade_distribution.keys())
counts = list(grade_distribution.values())

plt.figure(figsize=(8, 5))
plt.bar(grades, counts, color='skyblue', edgecolor='black')
plt.xlabel('Grade')
plt.ylabel('Count')
plt.title('Positive Content Grade Distribution')
plt.xticks(rotation=45)
plt.show()
