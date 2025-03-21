import distribution_helpers as helpers 
import matplotlib.pyplot as plt

toxicity_dict = {}

with open("top_subreddits_toxicity_score.txt", "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split(" ") 
        if len(parts) == 2: 
            subreddit_name, score = parts
            toxicity_dict[subreddit_name] = round(float(score), 4)

print('len(toxic_dict): ', len(toxicity_dict))

sorted_toxicity_dict = dict(sorted(toxicity_dict.items(), key=lambda item: item[1], reverse=True))
#print(sorted_toxic_dict) 

helpers.get_distribution_statistics(list(sorted_toxicity_dict.values()))
helpers.plot_distribution(sorted_toxicity_dict.values(), 
                          xlabel='Toxicity Score Range (0 to 1)',
                          ylabel='Frequency',
                          title='Distribution of Subreddits\' Toxicity Scores',
                          range_min=0,
                          range_max=1)

grade_distribution = {}
for subreddit, toxicity_score in sorted_toxicity_dict.items():
    grade = helpers.get_toxicity_grade(toxicity_score)
    if (grade not in grade_distribution): grade_distribution[grade] = 0
    grade_distribution[grade] += 1
    print('r/', subreddit, ': ', grade)

total_count = len(toxicity_dict)
for grade, count in grade_distribution.items():
    percent = round((count / total_count) * 100, 1)
    print(grade, ': ', count, ' ---> ', percent, '%')

grades = list(grade_distribution.keys())
counts = list(grade_distribution.values())

plt.figure(figsize=(8, 5))
plt.bar(grades, counts, color='skyblue', edgecolor='black')
plt.xlabel('Grade')
plt.ylabel('Count')
plt.title('Toxicity Grade Distribution')
plt.xticks(rotation=45)
plt.show()