# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # TDT4305 Project 1 - RDD Tasks

# %%
from datetime import datetime

# %% [markdown]
# ## Loading data files

# %%
reviews = sc.textFile('../data/yelp_top_reviewers_with_reviews.csv')     .zipWithIndex()     .filter(lambda x: x[1] > 0)     .map(lambda x: x[0].replace('"', '').split('\t'))
# "review_id","user_id","business_id","review_text","review_date"


# %%
businesses = sc.textFile('../data/yelp_businesses.csv')     .zipWithIndex()     .filter(lambda x: x[1] > 0)     .map(lambda x: x[0].replace('"', '').split('\t'))
# "business_id","name","address","city","state","postal_code",
# "latitude","longitude","stars","review_count","categories"


# %%
friendships = sc.textFile('../data/yelp_top_users_friendship_graph.csv')     .zipWithIndex()     .filter(lambda x: x[1] > 0)     .map(lambda x: x[0].replace('"', '').split(','))
# "src_user_id","dst_user_id"

# %% [markdown]
# ## Task 1
# %% [markdown]
# ### Counting number of rows

# %%
review_count = reviews.count()
business_count = businesses.count()
friendship_count = friendships.count()


# %%
review_count


# %%
business_count


# %%
friendship_count


# %%
with open('solutions/task1.csv', 'w') as f:
    f.write(f'review_count, business_count, friendship_count\n{review_count}, {business_count}, {friendship_count}')

# %% [markdown]
# ## Task 2
# %% [markdown]
# ### a) Finding number of distinct users

# %%
f_users = friendships.map(lambda row: row[0])
r_users = reviews.map(lambda row: row[1])

all_users = sc.union([f_users, r_users])
distinct_users = all_users.distinct().count()


# %%
with open('solutions/task2a.csv', 'w') as f:
    f.write(str(distinct_users))

# %% [markdown]
# ### b) Average numbers of characters in a review

# %%
avg_characters_in_review = reviews.map(lambda row: len(row[3])).mean()


# %%
with open('solutions/task2b.csv', 'w') as f:
    f.write(str(avg_characters_in_review))

# %% [markdown]
# ### c) Top 10 businesses by amount of reviews

# %%
top_businesses_by_review = reviews.map(lambda row: (row[2], 1)).reduceByKey(lambda x, y: x + y)     .sortBy(lambda row: row[1], ascending=False)     .map(lambda row: row[0])


# %%
top_businesses_by_review.take(10)


# %%
top_businesses_by_review.zipWithIndex().filter(lambda row: row[1] < 10).map(lambda row: row[0])     .coalesce(1).saveAsTextFile('solutions/raw/task2c')

# %% [markdown]
# ### d) Reviews per year

# %%
reviews_per_year = reviews.map(lambda row: (datetime.fromtimestamp(float(row[4])).year, 1))     .reduceByKey(lambda x, y: x + y)     .sortBy(lambda x: x[0])


# %%
reviews_per_year.collect()


# %%
reviews_per_year.map(lambda row: ','.join(str(d) for d in row)).coalesce(1).saveAsTextFile('solutions/raw/task2d')

# %% [markdown]
# ### e) First and last review

# %%
dates = reviews.map(lambda row: float(row[4]))

first_review = datetime.fromtimestamp(dates.sortBy(lambda x: x).first())
last_review = datetime.fromtimestamp(dates.sortBy(lambda x: -x).first())


# %%
first_review.strftime("%d.%m.%Y, %H:%M:%S")


# %%
last_review.strftime("%d.%m.%Y, %H:%M:%S")


# %%
with open('solutions/task2e.csv', 'w') as f:
    f.write(f'first_review, last_review\n{first_review}, {last_review}')

# %% [markdown]
# ### f) Pearson Correlation Coefficient

# %%
count_and_length = reviews.map(lambda row: (row[1], len(row[3])))     .aggregateByKey((0, 0), lambda x, y: (x[0] + 1, x[1] + y), lambda x, y: (x[0] + y[0], x[1] + y[1]))     .map(lambda row: (row[0], row[1][0], row[1][1] / row[1][0]))


# %%
count_and_length.first()


# %%
# (count, review_count_sum, review_length_sum)
agg = count_and_length.map(lambda row: (row[1], row[2]))     .aggregate((0, 0, 0),
                lambda x, y: (x[0] + 1, x[1] + y[0], x[2] + y[1]),
                lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2])
              )
x_avg = agg[1] / agg[0]
y_avg = agg[2] / agg[0]


# %%
pcc_agg = count_and_length.map(lambda row: (row[1], row[2]))     .map(lambda row: (
                         (row[0] - x_avg) * (row[1] - y_avg),
                         (row[0] - x_avg) ** 2,
                         (row[1] - y_avg) ** 2
                     )) \
    .reduce(lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2]))

pcc = pcc_agg[0] / ((pcc_agg[1] ** 0.5) * (pcc_agg[2] ** 0.5))


# %%
pcc


# %%
with open('solutions/task2f.csv', 'w') as f:
    f.write(str(pcc))

# %% [markdown]
# ## Task 3
# %% [markdown]
# ### a) Average rating by city

# %%
avg_rating_by_city = businesses.map(lambda row: (row[3], (float(row[8]), 1)))     .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))     .map(lambda row: (row[0], row[1][0] / row[1][1]))     .sortBy(lambda x: x[0])


# %%
avg_rating_by_city.collect()


# %%
avg_rating_by_city.map(lambda row: ','.join(str(r) for r in row)).coalesce(1)     .saveAsTextFile('solutions/raw/task3a')

# %% [markdown]
# ### b) Top 10 most frequent categories

# %%
top_categories = businesses.flatMap(lambda row: row[10].split(','))     .map(lambda row: (row.strip(), 1))     .reduceByKey(lambda x, y: x + y)     .sortBy(lambda row: row[1], ascending=False)


# %%
top_categories.take(10)


# %%
top_categories.zipWithIndex().filter(lambda row: row[1] < 10)     .map(lambda row: row[0][0]).coalesce(1).saveAsTextFile('solutions/raw/task3b')

# %% [markdown]
# ### c) Geographical centroid

# %%
pc_centroids = businesses.map(lambda row: (row[5], (1, float(row[6]), float(row[7]))))     .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2]))     .map(lambda row: (row[0], (row[1][1] / row[1][0], row[1][2] / row[1][0])))


# %%
pc_centroids.take(5)


# %%
pc_centroids.map(lambda row: f'{row[0]}, {row[1][0]}, {row[1][1]}').coalesce(1)     .saveAsTextFile('solutions/raw/task3c')

# %% [markdown]
# ## Task 4
# %% [markdown]
# ### a) Top in and out degrees

# %%
in_out_degrees = friendships.flatMap(lambda row: [(row[0], (0, 1)), (row[1], (1, 0))])         .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))


# %%
top_in = in_out_degrees.map(lambda row: (row[0], row[1][0]))     .sortBy(lambda row: row[1], ascending=False)


# %%
top_in.take(10)


# %%
top_out = in_out_degrees.map(lambda row: (row[0], row[1][1]))     .sortBy(lambda row: row[1], ascending=False)


# %%
top_out.take(10)


# %%
top_in.zipWithIndex().filter(lambda row: row[1] < 10).map(lambda row: row[0][0])     .coalesce(1).saveAsTextFile('solutions/raw/task4a-in')

top_out.zipWithIndex().filter(lambda row: row[1] < 10).map(lambda row: row[0][0])     .coalesce(1).saveAsTextFile('solutions/raw/task4a-out')

# %% [markdown]
# ### b) Mean and median in and out degrees

# %%
mean_in = in_out_degrees.map(lambda row: row[1][0]).mean()
mean_out = in_out_degrees.map(lambda row: row[1][1]).mean()


# %%
mean_in


# %%
mean_out


# %%
count = in_out_degrees.count()

top_in = in_out_degrees.map(lambda row: row[1][0])     .sortBy(lambda row: row, ascending=False)     .zipWithIndex()     .map(lambda row: (row[1], row[0]))

top_out = in_out_degrees.map(lambda row: row[1][1])     .sortBy(lambda row: row, ascending=False)     .zipWithIndex()     .map(lambda row: (row[1], row[0]))

if count % 2 == 0:
    l = count // 2
    r = l + 1
    median_in = (top_in.lookup(l)[0] + top_in.lookup(r)[0]) / 2
    median_out = (top_out.lookup(l)[0] + top_out.lookup(r)[0]) / 2
else:
    mid = count // 2
    median_in = top_in.lookup(mid)[0]
    median_out = top_out.lookup(mid)[0]


# %%
median_in


# %%
median_out


# %%
with open('solutions/task4b-in.csv', 'w') as f:
    f.write(f'mean, median\n{mean_in},{median_in}')
    
with open('solutions/task4b-out.csv', 'w') as f:
    f.write(f'mean, median\n{mean_out},{median_out}')

