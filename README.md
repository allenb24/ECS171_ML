# ECS171_ML
## Contributors
Allen Benjamin, Adib Guedoir, Aldo Sandoval

## Relevant links
Jupyter Notebook: 

Dataset: https://data.world/nolanoreilly495/nba-data-with-salaries-1996-2017

# Abstract
The National Basketball Association (NBA) is one of the largest sports organizations in the world and is worth over 90 billion dollars. Every season teams are set a league-wide team salary cap, currently set to 137 million dollars to build a championship-caliber team. Although the objective of the game has remained the same, the style of play seems to change constantly, emphasizing different attributes. With limited cap space, teams have to decide how to distribute their funds appropriately. Signing a star player to the team greatly impacts the available funding, whereas signing a role player allows for more financial flexibility. Our project will make projections using Regression Models a branch of Supervised Machine Learning on what a worthy salary is for an NBA player based on their performance and position relative to the current state of the NBA. We will use an NBA data set which contains 12,378 rows and 51 columns of information about each team’s roster from the years 1996-2017. Each player that was listed on that roster that season is also included, along with information about them such as, their position, age, and in-game statistics. These in-game metrics are used to analytically describe the player’s performance.  


# Introduction
We believed this would be a fun and intresting project beacuse it combines two of our favorite things sports and computers. While this group does not share any favorite teams in fact our favorite teams are rivals we hoped that applying Machine learning to the data set filled with in NBA players' in game stats would indicate how the game has evolved over time and which apsects are more valuable in present day. It was our desire that after building an accurate model to predict the salary, we would be able to input new players along with their stats and have the model determine their monetary value. Then I would be able to brag how our rival team overpaid for a player. We also hoped the this project would be a great way to apply things we learned in class to the real world.


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



# Results

## Data Exploration
This is the heatmap of the correlation between all aspects of our data:
![Alt text](download.png)

These are the heatmaps for the specific aspects of the game (offense, defense, playmaking, availability)
![Alt text](download-1.png)

![Alt text](download-2.png)

![Alt text](download-3.png)

![Alt text](download-4.png)


## Preprocessing
After cleaning our data and filtering the positions, features, period, and N/A values, we have this:
```
NBA DATA SET
Positions:  ['C' 'PF' 'SF' 'PG' 'SG']
Features:  ['Player', 'Salary', 'Pos', 'Year', 'Age', 'G', 'GS', 'PER', 'PTS/G', 'AST/G', 'ORB/G', 'DRB/G', 'OBPM', 'DBPM', 'TRB/G', 'STL/G', 'BLK/G', 'TOV/G', '2P%', '3P%', 'eFG%', 'FT%', 'MP/G']
Data Period:  [2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007, 2006, 2005, 2004, 2003, 2002, 2001, 2000, 1999, 1998, 1997, 1996]
Columns w/ NA:  ['PER', '2P%', '3P%', 'eFG%', 'FT%']
Columns w/ NA after cleaning:  []
Data Rows:  9409
Data Columns:  23
```

We then obtained the average salary for each player for each year and then grouped them into different ranges:
```
Average Salaries:  [1710539.0, 1980512.0, 2289899.0, 2511917.0, 2823992.0, 3358028.0, 3372248.0, 3577774.0, 3601343.0, 3675149.0, 3784542.0, 3902860.0, 4225603.0, 4581743.0, 4458596.0, 4336169.0, 4169421.0, 4189944.0, 5042071.0, 4166249.0, 4508850.0, 5657306.0]
Group 1:  [2000 1999 1998 1997 1996]
Group 2:  [2007 2006 2005 2004 2003 2002 2001]
Group 3:  [2017 2016 2015 2014 2013 2012 2011 2010 2009 2008]
```

Next, we broke them down by each position and then categorized them into different aspects of basketball:
```
Years 1996-2000

Aspect OFFENSE
Stats:  ['3P%', 'OBPM', 'FT%', 'PER', 'eFG%', '2P%', 'PTS/G']

Aspect DEFENSE
Stats:  ['DRB/G', 'ORB/G', 'DBPM', 'STL/G', 'BLK/G', 'TRB/G']

Aspect PLAY MAKING
Stats:  ['AST/G', 'TOV/G', 'OBPM']

Aspect AVAILIBILTY
Stats:  ['G', 'GS', 'MP/G']
```

## Model
These are our outputs for our first linear regression model:
```Linear Regression Model
Data Features:  ['Year', 'Age', 'G', 'GS', 'PER', 'PTS/G', 'AST/G', 'ORB/G', 'DRB/G', 'OBPM', 'DBPM', 'TRB/G', 'STL/G', 'BLK/G', 'TOV/G', '2P%', '3P%', 'eFG%', 'FT%', 'MP/G']
Target: Salary
TRAINING Mean Squared Error is:  8677232253245.942
TESTING  Mean Squared Error is:  9004849118587.037
```

Outputs for the polynomial model (2nd degree):
```Polynomial Regression Model Degree 2
Data Features:  ['Year', 'Age', 'G', 'GS', 'PER', 'PTS/G', 'AST/G', 'ORB/G', 'DRB/G', 'OBPM', 'DBPM', 'TRB/G', 'STL/G', 'BLK/G', 'TOV/G', '2P%', '3P%', 'eFG%', 'FT%', 'MP/G']
Target: Salary
TRAINING Mean Squared Error:  6515867731483.319
TESTING  Mean Squared Error:  6985161144957.786
```

Outputs for grouped years with the average salary:
```Linear Model for Salary Groups

Years 1996-2000 
TRAINING Mean Squared Error:  3807582698253.4463
TESTING  Mean Squared Error:  2964517928759.518

Years 2001-2007 
TRAINING Mean Squared Error:  6518703415016.366
TESTING  Mean Squared Error:  7732462331460.306

Years 2008-2017 
TRAINING Mean Squared Error:  11420931577355.373
TESTING  Mean Squared Error:  9397297468441.938
```

Outputs for linear model with grouped years for each position:
```Linear Model for Salary Groups and Positions



Years 1996-2000

Position C
TRAINING Mean Squared Error:  3898677781471.917
TESTING  Mean Squared Error:  5263299564826.124

Position PF
TRAINING Mean Squared Error:  3276620310097.7544
TESTING  Mean Squared Error:  4420139042493.459

Position SF
TRAINING Mean Squared Error:  2318524051960.8296
TESTING  Mean Squared Error:  1652905118328.126

Position PG
TRAINING Mean Squared Error:  1748011107848.1685
TESTING  Mean Squared Error:  3101092759412.6753

Position SG
TRAINING Mean Squared Error:  4740619379431.312
TESTING  Mean Squared Error:  2410032063627.0234



Years 2001-2007

Position C
TRAINING Mean Squared Error:  6282403862322.076
TESTING  Mean Squared Error:  6880601471262.061

Position PF
TRAINING Mean Squared Error:  7598832705954.246
TESTING  Mean Squared Error:  8091724994512.151

Position SF
TRAINING Mean Squared Error:  6446549877582.223
TESTING  Mean Squared Error:  8058025187882.841

Position PG
TRAINING Mean Squared Error:  4707963663312.306
TESTING  Mean Squared Error:  5569293667382.371

Position SG
TRAINING Mean Squared Error:  6353148627482.31
TESTING  Mean Squared Error:  4422556054135.364



Years 2008-2017

Position C
TRAINING Mean Squared Error:  13235380171699.508
TESTING  Mean Squared Error:  11370869813766.316

Position PF
TRAINING Mean Squared Error:  9040324927062.46
TESTING  Mean Squared Error:  12765402908786.254

Position SF
TRAINING Mean Squared Error:  10367026948065.81
TESTING  Mean Squared Error:  13107560881954.268

Position PG
TRAINING Mean Squared Error:  8548812549261.228
TESTING  Mean Squared Error:  10527745559148.787

Position SG
TRAINING Mean Squared Error:  10297387641426.854
TESTING  Mean Squared Error:  10849179089457.8
```

Outputs of linear regression for grouped years with different aspects of the game:
```Linear Model for Salary Groups and Specific Statistics



Years 1996-2000

Aspect OFFENSE
Stats:  ['3P%', 'OBPM', 'FT%', 'PER', 'eFG%', '2P%', 'PTS/G']
TRAINING Mean Squared Error:  4862163147797.9
TESTING  Mean Squared Error:  3866172462549.382

Aspect DEFENSE
Stats:  ['DRB/G', 'ORB/G', 'DBPM', 'STL/G', 'BLK/G', 'TRB/G']
TRAINING Mean Squared Error:  4878064325894.496
TESTING  Mean Squared Error:  3397144784734.2983

Aspect PLAY MAKING
Stats:  ['AST/G', 'TOV/G', 'OBPM']
TRAINING Mean Squared Error:  5769655248139.349
TESTING  Mean Squared Error:  4063849604211.781

Aspect AVAILIBILTY
Stats:  ['G', 'GS', 'MP/G']
TRAINING Mean Squared Error:  5518561691544.87
TESTING  Mean Squared Error:  3623977016904.963



Years 2001-2007

Aspect OFFENSE
Stats:  ['3P%', 'OBPM', 'FT%', 'PER', 'eFG%', '2P%', 'PTS/G']
TRAINING Mean Squared Error:  8886551663738.182
TESTING  Mean Squared Error:  10452430302321.023

Aspect DEFENSE
Stats:  ['DRB/G', 'ORB/G', 'DBPM', 'STL/G', 'BLK/G', 'TRB/G']
TRAINING Mean Squared Error:  9322991725995.299
TESTING  Mean Squared Error:  10897387903693.875

Aspect PLAY MAKING
Stats:  ['AST/G', 'TOV/G', 'OBPM']
TRAINING Mean Squared Error:  9902700620412.766
TESTING  Mean Squared Error:  11872235022760.121

Aspect AVAILIBILTY
Stats:  ['G', 'GS', 'MP/G']
TRAINING Mean Squared Error:  9409583138135.666
TESTING  Mean Squared Error:  11206553119904.953



Years 2008-2017

Aspect OFFENSE
Stats:  ['3P%', 'OBPM', 'FT%', 'PER', 'eFG%', '2P%', 'PTS/G']
TRAINING Mean Squared Error:  14946094476376.242
TESTING  Mean Squared Error:  11864016001379.627

Aspect DEFENSE
Stats:  ['DRB/G', 'ORB/G', 'DBPM', 'STL/G', 'BLK/G', 'TRB/G']
TRAINING Mean Squared Error:  15721439432896.457
TESTING  Mean Squared Error:  12850302318646.842

Aspect PLAY MAKING
Stats:  ['AST/G', 'TOV/G', 'OBPM']
TRAINING Mean Squared Error:  17115088757551.688
TESTING  Mean Squared Error:  13948398386212.086

Aspect AVAILIBILTY
Stats:  ['G', 'GS', 'MP/G']
TRAINING Mean Squared Error:  16232899420678.766
TESTING  Mean Squared Error:  12802396591989.824
```

Outputs for polynomial model with groups for specific aspects of the game:
```
Polynimal Model for Groups and Specific Stats

Years 1996-2000

Aspect OFFENSE
Stats:  ['3P%', 'OBPM', 'FT%', 'PER', 'eFG%', '2P%', 'PTS/G']
TRAINING Mean Squared Error:  4359740108621.801
TESTING  Mean Squared Error:  3816321990336.7847

Aspect DEFENSE
Stats:  ['DRB/G', 'ORB/G', 'DBPM', 'STL/G', 'BLK/G', 'TRB/G']
TRAINING Mean Squared Error:  4685853779179.045
TESTING  Mean Squared Error:  3299870372242.6377

Aspect PLAY MAKING
Stats:  ['AST/G', 'TOV/G', 'OBPM']
TRAINING Mean Squared Error:  5425837030134.952
TESTING  Mean Squared Error:  3925876920447.0835

Aspect AVAILIBILTY
Stats:  ['G', 'GS', 'MP/G']
TRAINING Mean Squared Error:  5331127643652.879
TESTING  Mean Squared Error:  3496787120178.607

Years 2001-2007

Aspect OFFENSE
Stats:  ['3P%', 'OBPM', 'FT%', 'PER', 'eFG%', '2P%', 'PTS/G']
TRAINING Mean Squared Error:  8586837558025.07
TESTING  Mean Squared Error:  10171812826191.656

Aspect DEFENSE
Stats:  ['DRB/G', 'ORB/G', 'DBPM', 'STL/G', 'BLK/G', 'TRB/G']
TRAINING Mean Squared Error:  8862867137258.416
TESTING  Mean Squared Error:  10834465573243.734

Aspect PLAY MAKING
Stats:  ['AST/G', 'TOV/G', 'OBPM']
TRAINING Mean Squared Error:  9644655566904.281
TESTING  Mean Squared Error:  11221487660998.725

Aspect AVAILIBILTY
Stats:  ['G', 'GS', 'MP/G']
TRAINING Mean Squared Error:  8996739982794.117
TESTING  Mean Squared Error:  10816680827810.77

Years 2008-2017

Aspect OFFENSE
Stats:  ['3P%', 'OBPM', 'FT%', 'PER', 'eFG%', '2P%', 'PTS/G']
TRAINING Mean Squared Error:  14570758118426.64
TESTING  Mean Squared Error:  11664740527330.775

Aspect DEFENSE
Stats:  ['DRB/G', 'ORB/G', 'DBPM', 'STL/G', 'BLK/G', 'TRB/G']
TRAINING Mean Squared Error:  15014155081565.258
TESTING  Mean Squared Error:  12368165777048.021

Aspect PLAY MAKING
Stats:  ['AST/G', 'TOV/G', 'OBPM']
TRAINING Mean Squared Error:  16476106375221.81
TESTING  Mean Squared Error:  13114356767145.271

Aspect AVAILIBILTY
Stats:  ['G', 'GS', 'MP/G']
TRAINING Mean Squared Error:  15136902088488.074
TESTING  Mean Squared Error:  11922041299156.035
```


# Discussion
Since our objective was to find which in-game statistics were the best indicators for the total salary a player earned for the year we decided to do a linear regression model. Our hope was that the model would be able to identify the most important feature that correlated to the target. For our features matrix ‘X’, we dropped the columns that were strings and did not make sense to include in the data which were ‘Player’, ‘Position’, and ‘Salary’. The ‘Salary’ column we set up as our target vector ‘y’. We then split our data using the ‘train_test_split()’ function into 80% for training and 20% for testing. We then standardized our data using the ‘MinMaxScaler()’ for both our training and testing data. However, for our training data, we both transformed and fit the data while only transforming the testing data. Using the linear regression model from sklearn, ‘LinearRegression()’ we create a model labeled ‘model1’. We then trained the model with the standardized data and the ‘y_training’ data, using the function model1.fit(X_train_standardized, y_train). After training the model we got our prediction ‘yhat_train’ for the training data and ‘yhat_test’ for our testing data. To see how well our model did we used the Mean Squared Error (MSE) metric for both data sets. However, we had extremely large numbers for both sets, our MSE for the training data was 8677232253245.942 and our MSE for the testing data was 9004849118587.037. 

This model was not good. For our next model, we decided to use a Polynomial Regression model of degree 2. Similarly to the linear regression model we removed the unnecessary columns, standardized the data sets then applied the Polynomial model using ‘PolynomialFeatures()’. After we transformed and fitted out training data and only transformed out test data. We also used MSE as our measuring metric and got an MSE of 6515867731483.319 for our training data and an MSE of 6985161144957.786 for our testing data. Again this is not good. 

We realized that a possible reason our model is just not good is because our target ‘Salary’ has varied over time due to inflation and increased Salary Cap. After getting the average salary for every year we determined 3 groups for which we would train and test individually due to their similar salary average. The first group was from the years 1996-2000 with an average salary of $2 million, the second group was from 2001-2007 with an average salary of $3 million, and the third group was from 2008-2017 with an average salary of $4 million dollars. We then used the same linear regression model as above with a MinMax scaler.  As a result, we got 2 MSE for each of the years. Although we saw a decrease in 2 out of the 3 groups our MSEs were still very large at a scale of 10^13.

Believing there was still too much discrepancy in our data we further divided our data. For this model, we still used linear regression on each of the three groups. However, we divided the groups by position meaning in total we had 15 different groups which would lead to 15 different models. We hope that the model’s MSE will decrease. After following the same steps as our previous linear regression model. Our result MSE was still extremely large.

In our final model, we decided to do feature selection on our data and divide the in-game statistics we mentioned into 4 categories that describe the aspects of the game. The aspects were ‘Offense’, ‘Defense’, ‘Play Making’, and ‘Availability’.  While still maintaining the 3 groups divided by their average salary we ran a linear regression model with the data frame modified to only feature the attributes from the specific aspects. Once again we found our MSEs to be very large and saw no significant improvement. In the same data frames, we ran a model of polynomial regression to degree 2 and saw worse performance than the linear equivalent.

# Conclusion
Overall, we found this to be a very fun project, as the topic is related to something our team really enjoys. We found reasonable results when constructing our correlation matrices, as they seemed to accurately represent the correlation between certain statistics. For example, a player with a high 3-point shooting percentage was projected to also have a higher free throw percentage. 

However, we did have some shortcomings. Our MSE was very large and we believed that it was due to the large number that is the salary of the player, which is typically in millions.

## Collaboration Statement
Allen Benjamin, Collaborator

Adib Guedoir, Collaborator

Aldo Sandoval, Collaborator


We all worked together as a team and consistently met through Discord to work on our project. We communicated with each other throughout and each contributed to the milestones. Everyone was on the same page about what we wanted to accomplish, and we worked as a team by contributing code and making suggestions on how to approach each section, from preprocessing to our final model. There was no lack of effort by anyone throughout the project and we had great team chemistry.
