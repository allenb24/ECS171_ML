# ECS171_ML
## Contributors
Allen Benjamin, Adib Guedoir, Aldo Sandoval

## Relevant links
Jupyter Notebook: 

Dataset: https://data.world/nolanoreilly495/nba-data-with-salaries-1996-2017


# Introduction
The National Basketball Association (NBA) is one of the largest sports organizations in the world and is worth over 90 billion dollars. Every season teams are set a league-wide team salary cap, currently set to 137 million dollars to build a championship-caliber team. Although the objective of the game has remained the same, the style of play seems to change constantly, emphasizing different attributes.

With limited cap space, teams have to decide how to distribute their funds appropriately. Signing a star player to the team greatly impacts the available funding, whereas signing a role player allows for more financial flexibility. 
We found this topic interesting because we are basketball fans and understand that the financial aspect of the NBA is one of the main reasons that players end up playing for a chosen team. Obviously, we want the team that we support to do the best.

Our project will make projections on what a worthy salary is for an NBA player based on their performance and position relative to the current state of the NBA. This will allow general managers to utilize machine learning to assist their financial decisions by appropriately allocating money for players who fit the team’s desires without overspending.



# Methods

## Data Exploration
In our data exploration, we first got a general overview of the correlation between each attribute amongst each other by dropping our ‘Players’ column and then visualizing the data with a correlation matrix, which is represented by a heatmap. 
Moreover, we plotted another heatmap to display the correlation between salary and the different aspects of the game, such as offense, defense, playmaking, and availability.


We set the data we'd like to analyze by first defining our dependent and independent variables. Our dependent variable 'y' is the annual salary that each player receives. Our independent variable 'X' holds the values that represent important statistics in a basketball game, including position, year, age, games played, games started, player efficiency rating, points, assists, offensive/defensive/total rebounds, offensive and defensive efficiency, steals, blocks, turnovers, shooting percentage, and minutes played.

## Preprocessing
For accurate results, we first needed to clean our data. To analyze salaries for each position, we assigned the individual positions and merged the ones that represented multi-position players (e.g. a player with the position ‘SF-C’ would be converted to ‘SF’). Next, we defined the attributes that we wanted to analyze by selecting the ones we felt impacted a player’s value the most. 

```
stats = ['Player','Salary','Pos', 'Year', 'Age', 'G', 'GS', 'PER', 'PTS/G', 'AST/G', 'ORB/G', 'DRB/G', 'OBPM', 'DBPM', 'TRB/G', 'STL/G', 'BLK/G', 'TOV/G', '2P%', '3P%', 'eFG%', 'FT%', 'MP/G'] 
```

We then took the instances that held ‘N/A’ values within the attributes and filled them in with the average value for that column.

Of the nearly 13,000 rows of data, approximately 2,000 of them contained instances where the earned salary for a player was 0. For the purpose of this project, we only want to analyze the players who have an active salary, so we removed those rows from our data.

Next, we broke down the attributes into more specific aspects of basketball such as offense, defense, playmaking, and availability. 

Furthermore, we clustered specific ranges of years that had the smallest difference in salary cap in the NBA. These three groups include the years 1996-2000, 2001-2007, and 2008-2017. To further classify our data, we broke it down by each position ('PG', 'SG', 'SF', 'PF', and 'C').

```
score_eff = ['Salary','PTS/G','2P%', '3P%', 'eFG%', 'FT%', 'PER', 'OBPM'] # Scoring and Efficiency
defense = ['Salary','ORB/G', 'DRB/G','TRB/G', 'STL/G', 'BLK/G', 'DBPM'] # Defensive
play_making = ['Salary','AST/G','OBPM','TOV/G'] # Playmaking
availability = ['Salary', 'G', 'GS', 'MP/G'] # Avaliablity
```

## Model
Since our objective was to find which in-game statistics were the best indicators for the total salary a player earned for the year we decided to do a linear regression model. Our hope was that the model would be able to identify the most important feature that correlated to the target. For our features matrix ‘X’, we dropped the columns that were strings and did not make sense to include in the data which were ‘Player’, ‘Position’, and ‘Salary’. The ‘Salary’ column we set up as our target vector ‘y’. We then split our data using the ‘train_test_split()’ function into 80% for training and 20% for testing. We then standardized our data using the ‘MinMaxScaler()’ for both our training and testing data. However, for our training data, we both transformed and fit the data while only transforming the testing data. Using the linear regression model from sklearn, ‘LinearRegression()’ we create a model labeled ‘model1’. We then trained the model with the standardized data and the ‘y_training’ data, using the function model1.fit(X_train_standardized, y_train). After training the model we got our prediction ‘yhat_train’ for the training data and ‘yhat_test’ for our testing data. To see how well our model did we used the Mean Squared Error (MSE) metric for both data sets. However, we had extremely large numbers for both sets, our MSE for the training data was 8677232253245.942 and our MSE for the testing data was 9004849118587.037. 

This model was not good. For our next model, we decided to use a Polynomial Regression model of degree 2. Similarly to the linear regression model we removed the unnecessary columns, standardized the data sets then applied the Polynomial model using ‘PolynomialFeatures()’. After we transformed and fitted out training data and only transformed out test data. We also used MSE as our measuring metric and got an MSE of 6515867731483.319 for our training data and an MSE of 6985161144957.786 for our testing data. Again this is not good. 

We realized that a possible reason our model is just not good is because our target ‘Salary’ has varied over time due to inflation and increased Salary Cap. After getting the average salary for every year we determined 3 groups for which we would train and test individually due to their similar salary average. The first group was from the years 1996-2000 with an average salary of $2 million, the second group was from 2001-2007 with an average salary of $3 million, and the third group was from 2008-2017 with an average salary of $4 million dollars. We then used the same linear regression model as above with a MinMax scaler.  As a result, we got 2 MSE for each of the years. Although we saw a decrease in 2 out of the 3 groups our MSEs were still very large at a scale of 10^13.

Believing there was still too much discrepancy in our data we further divided our data. For this model, we still used linear regression on each of the three groups. However, we divided the groups by position meaning in total we had 15 different groups which would lead to 15 different models. We hope that the model’s MSE will decrease. After following the same steps as our previous linear regression model. Our result MSE was still extremely large.

In our final model, we decided to do feature selection on our data and divide the in-game statistics we mentioned into 4 categories that describe the aspects of the game. The aspects were ‘Offense’, ‘Defense’, ‘Play Making’, and ‘Availability’.  While still maintaining the 3 groups divided by their average salary we ran a linear regression model with the data frame modified to only feature the attributes from the specific aspects. Once again we found our MSEs to be very large and saw no significant improvement. In the same data frames, we ran a model of polynomial regression to degree 2 and saw worse performance than the linear equivalent.


# Results

## Data Exploration

## Preprocessing

## Model


# Discussion

# Conclusion
Overall, we found this to be a very fun project, as the topic is related to something our team really enjoys. We found reasonable results when constructing our correlation matrices, as they seemed to accurately represent the correlation between certain statistics. For example, a player with a high 3-point shooting percentage was projected to also have a higher free throw percentage. 

However, we did have some shortcomings. Our MSE was very large and we believed that it was due to the large number that is the salary of the player, which is typically in millions.

## Collaboration Statement
Allen Benjamin, Collaborator

Adib Guedoir, Collaborator

Aldo Sandoval, Collaborator


We all worked together as a team and consistently met through Discord to work on our project. We communicated with each other throughout and each contributed to the milestones. Everyone was on the same page about what we wanted to accomplish, and we worked as a team by contributing code and making suggestions on how to approach each section, from preprocessing to our final model. There was no lack of effort by anyone throughout the project and we had great team chemistry.
