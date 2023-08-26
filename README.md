# ECS171_ML
Team project using machine learning to analyze various information about NBA players to predict a worthy salary.

We preprocessed the data by first setting our dependent and independent variables. Our dependent variable 'y' is the annual salary that each player receives. Our independent variable 'X' holds the values that represent important statistics in a basketball game, including position, age, games played, games started, player efficiency rating, points, assists, offensive/defensive/total rebounds, offensive and defensive efficiency, steals, blocks, turnovers, shooting percentage, and minutes played.

Next, we split the positions column in a form of label encoding, where we set each position a number (which is reflective in the NBA, where they call the PG position the 'one' and the C position the 'five'). These positions are represented as follows: 'PG': 1, 'SG': 2, 'SF' : 3, 'PF': 4, 'C': 5.

We then split the data into training and testing sets.
