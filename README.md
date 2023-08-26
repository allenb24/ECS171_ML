## ECS171_ML
Allen Benjamin, Adib Guedoir, Aldo Sandoval

Jupyter Notebook: https://colab.research.google.com/drive/1ovXUoYBr06Ob2OakHg9UBX_TWNf-wrYY?usp=sharing
Data: https://data.world/nolanoreilly495/nba-data-with-salaries-1996-2017

Team project using machine learning to analyze various information about NBA players to predict a worthy salary.

## Data Exploration
We set the data we'd like to analyze by first defining our dependent and independent variables. Our dependent variable 'y' is the annual salary that each player receives. Our independent variable 'X' holds the values that represent important statistics in a basketball game, including position, year, age, games played, games started, player efficiency rating, points, assists, offensive/defensive/total rebounds, offensive and defensive efficiency, steals, blocks, turnovers, shooting percentage, and minutes played.

## Preprocessing
To preprocess our data, we must consider multiple aspects with the first being scaling. We did some preprocessing, as we did label encoding for the different positions in basketball and did feature selection by choosing the categories that we deemed important. For label encoding, we set each position as a number (which is reflected in the NBA, where they call the PG position the 'one' and the C position the 'five'). These positions are represented as follows: 'PG': 1, 'SG': 2, 'SF' : 3, 'PF': 4, 'C': 5. 

The numbers that we got from the in-game statistics are very spread out, as the number of points or minutes played are on a much higher scale than steals or blocks. For this reason, we will scale the data appropriately.

We then split the data into training and testing sets.

Next, to train it, we will determine which model is appropriate to train it. We will likely begin with linear regression and then determine how to proceed from there.
