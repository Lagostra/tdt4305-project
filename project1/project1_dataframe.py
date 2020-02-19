# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # TDT4305 Project 1 - DataFrame Tasks
# %% [markdown]
# ## Task 5

# %%
reviews = spark.read.format('csv').option('delimiter', '\t').option('header', 'true')    .load('../data/yelp_top_reviewers_with_reviews.csv')
businesses = spark.read.format('csv').option('delimiter', '\t').option('header', 'true')    .load('../data/yelp_businesses.csv')
friendships = spark.read.csv('../data/yelp_top_users_friendship_graph.csv', header='true')


# %%
reviews.createTempView('reviews')
businesses.createTempView('businesses')
friendships.createTempView('friendships')


# %%
reviews.printSchema()


# %%
businesses.printSchema()


# %%
friendships.printSchema()

# %% [markdown]
# ## Task 6

# %%
spark.sql('SELECT * FROM reviews INNER JOIN businesses ON reviews.business_id = businesses.business_id')     .createTempView('business_review')


# %%
spark.sql('SELECT * FROM business_review LIMIT 5').show()


# %%
user_no_reviews = spark.sql('SELECT user_id, count(user_id) AS no_reviews ' +
                            'FROM reviews GROUP BY user_id ORDER BY no_reviews DESC ' + 
                            'LIMIT 20')
user_no_reviews.show()


# %%
user_no_reviews.write.format('csv').save('solutions/raw/task6c')

