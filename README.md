# Stockplots

## About:
**Stockplots** is a web-app where users can search for a stock and observe its price history. Users can also enter a future date to see the projected price of a stock, calculated using a linear regression model. The data is obtained from [Kaggle](https://www.kaggle.com/datasets/dgawlik/nyse), and it spans from 2010 to 2016.

## Getting Started:
1. Clone this repository:  
`git clone git@github.com:ishuagrawal/Stockplots.git`
2. Change directory to the repo: `cd Stockplots`
3. Install the following packages to your Python3 interpreter:
    * matplotlib
    * Flask
    * scikit-learn
    * pandas
4. Run `app_starter.py`

## Next Steps:
I plan to improve this webapp by the following ways: 
* Gather stock data using the Yahoo Finance API instead of the dataset to retrieve more recent data, including more information about each stock
* Extend the functionality of my app by allowing users to save a stock to their watchlist
* Improve the linear regression model such that it uses more features to better predict the stock's future price

## Screenshots:

![Login/Register Page](readme-images/login_register.png "Login/Register")
![Home Page](readme-images/Home.png "Home Page")
![Search Page](readme-images/Search.png "Search Page")
![Speculate Page](readme-images/Speculate.png "Speculate Page")
