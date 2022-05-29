from io import BytesIO

import numpy as np
from flask import Flask, redirect, render_template, request, session, url_for, send_file
import os
import sqlite3 as sl
import pandas as pd
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
db = "mystocks.db"
prices = pd.read_csv("data/prices-split-adjusted.csv")  # stores the price history
companies = pd.read_csv("data/securities.csv")  # stores info about each company


# Run when program starts
@app.before_first_request
def run():
    global prices

    # Create tables
    conn = sl.connect(db)
    curs = conn.cursor()
    stmt1 = "CREATE TABLE IF NOT EXISTS users('email' PRIMARY KEY, 'username', 'password')"  # Create user table
    curs.execute(stmt1)
    conn.commit()
    conn.close()

    prices['date_conv'] = pd.to_datetime(prices['date'])  # Create new column with date as datetime object


@app.route("/")
def home():
    # if user is logged in, then display their food
    if not session.get("logged_in"):
        return render_template("auth.html", message=None)
    else:
        return render_template("home.html", message=None)


# Logs in a user
@app.route("/login", methods=["POST", "GET"])
def login():
    email = request.form["email-login"]
    password = request.form["pwd-login"]
    error = check_login(email, password)    # check if email and password match
    if error:
        return render_template("auth.html", message=error)
    else:
        return redirect(url_for("home"))


# Registers a new user
@app.route("/register", methods=["POST", "GET"])
def register():
    email = request.form["email-register"]
    username = request.form["name-register"]
    password = request.form["pwd-register"]
    password_ = request.form["pwd-confirm"]
    error = check_registration(email, username, password, password_)    # check for input form errors
    if error:
        return render_template("auth.html", message=error)
    else:
        create_user(email, username, password)
        return redirect(url_for("home"))


# Logs out the user
@app.route("/logout", methods=["POST", "GET"])
def logout():
    # Logged out -> clear session
    if request.method == "POST":
        session["logged_in"] = False
        session.pop("email", None)
        session.pop("username", None)

    return redirect(url_for("home"))


# User can search for a stock on a date and retrieve info, including its closing price
@app.route("/search", methods=["POST", "GET"])
def search():
    global prices
    global companies

    ticker = request.form["ticker"]
    date = request.form["date"]
    dates = prices["date"].unique()

    error = isValidTicker(ticker)   # check if ticker is valid
    if error is not None:
        return render_template("home.html", message=error)

    # if trading day is closed, go back to most recent trading day
    days = 0
    error = None
    while not(date in dates):
        date = datetime.strptime(date, "%Y-%m-%d")
        date = date - timedelta(days=1)
        date = datetime.strftime(date, "%Y-%m-%d")
        days += 1
        if days == 5:   # if more than 5 days back, date is invalid
            error = f"Error: {ticker} was not traded on {date}. Please choose a different day."
            return render_template("home.html", message=error)

    # find the closing price and volume, given the ticker symbol and date
    data = prices[(prices["date"] == date) & (prices["symbol"] == ticker)]
    data2 = companies[companies["Ticker symbol"] == ticker]
    price = data["close"].to_string(index=False)
    volume = data["volume"].to_string(index=False)
    company = data2["Security"].to_string(index=False)
    sector = data2["GICS Sector"].to_string(index=False)

    return render_template("home.html", ticker=ticker, date=date, price=price, company=company, sector=sector, volume=volume, mode="search")


# checks to see if ticker is valid: if invalid, then returns error
def isValidTicker(ticker):
    global prices

    tickers = prices["symbol"].unique()
    error = None

    if ticker not in tickers:
        error = f"Error: {ticker} is not a valid ticker."
    return error


# User can search for a stock and see what its projected price will be for a certain day
@app.route("/speculate", methods=["POST", "GET"])
def speculate():
    global prices
    global companies

    ticker = request.form["ticker"]
    date = request.form["date"]

    error = isValidTicker(ticker)   # check if ticker is valid
    if error is not None:
        return render_template("home.html", message=error)

    # convert date input to number of days
    first_date = datetime(2010, 1, 4)
    delta = datetime.strptime(date, '%Y-%m-%d') - first_date
    pred_x = np.array(delta.days)
    pred_x = pred_x.reshape(-1, 1)
    prices['date_days'] = prices.apply(lambda row: getNumDays(row), axis=1)     # Create new column with number of days from 1/4/2010

    # Training linear regression model to predict price
    data = prices[prices["symbol"] == ticker]
    x = data["date_days"].values.reshape(-1, 1) # input: num of days since 2010/1/4
    y = data["close"].values
    y = y.astype(np.float)  # output: float

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    model = LinearRegression().fit(x, y)
    prediction = model.predict(pred_x)
    price = prediction[0]   # stores predicted price for the user's inputted date

    data = companies[companies["Ticker symbol"] == ticker]
    company = data["Security"].to_string(index=False)
    sector = data["GICS Sector"].to_string(index=False)

    return render_template("home.html", ticker=ticker, date=date, price=price, company=company, sector=sector, mode="speculate")


# Send search graph to flask
@app.route("/fig_search/<symbol>/<date>/<price>")
def fig_search(symbol, date, price):
    fig = graph_search(symbol, date, price)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')


# Send specualte graph to flask
@app.route("/fig_speculate/<symbol>/<date>/<price>")
def fig_speculate(symbol, date, price):
    fig = graph_speculate(symbol, date, price)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')


# creates a plot of the ticker's closing price over time, along with the price at user's inputted date
def graph_search(ticker, date, price):
    global prices
    data = prices[prices["symbol"] == ticker]

    date = datetime.strptime(date, '%Y-%m-%d')
    date = date.date()
    fig, ax = plt.subplots(1, 1)

    ax.plot(data["date_conv"], data["close"], label="All dates")   # plots stock price over time
    ax.plot([date], [float(price)], marker="*", markersize=10, mfc="red", label=date)   # marks price at the user's inputted date

    ax.set(title=ticker, xlabel="date", ylabel="closing price ($)")
    ax.legend()
    fig.tight_layout()
    return fig


# creates a plot of the ticker's closing price over time, along with the predicted price at user's inputted date
def graph_speculate(ticker, date, price):
    global prices
    data = prices[prices["symbol"] == ticker]

    date = datetime.strptime(date, '%Y-%m-%d')
    fig, ax = plt.subplots(1, 1)

    ax.plot(data["date_conv"], data["close"], label="All dates")   # plots stock price over time
    ax.plot([date], [float(price)], marker="*", markersize=10, mfc="red", label="Prediction")   # marks predicted price at the user's inputted date

    ax.set(title=ticker, xlabel="date", ylabel="closing price ($)")
    ax.legend()
    fig.tight_layout()
    return fig


# validates input fields from login form and performs login for user
def check_login(email, password):
    conn = sl.connect(db)
    curs = conn.cursor()

    # check if user exists that matches email and password (input fields)
    v = (email, password)
    stmt1 = "SELECT * FROM users WHERE email = ? AND password = ?"
    curs.execute(stmt1, v)
    data = curs.fetchone()

    conn.commit()
    conn.close()

    # no matches -> user does not exist
    if not data:
        return "Error: email or password is incorrect!"
    else:
        # logs in user by storing email and username
        session["logged_in"] = True
        session["email"] = data[0]
        session["username"] = data[1]
        return None


# validates input fields from register form
def check_registration(email, username, pwd, confirm_pwd):
    # passwords must match
    if pwd != confirm_pwd:
        return "Error: passwords do not match!"

    conn = sl.connect(db)
    curs = conn.cursor()

    # check if user associated with input email exists
    v = (email,)
    stmt1 = "SELECT * FROM users WHERE email = ?"
    curs.execute(stmt1, v)
    data = curs.fetchone()

    conn.commit()
    conn.close()

    # matches -> user already exists with that email
    if data:
        return "Error: email has already been registered. Please login!"
    else:
        # logs in user by storing email and username
        session["logged_in"] = True
        session["email"] = email
        session["username"] = username
        return None


# Creates a new user in the users table
def create_user(email, username, password):
    conn = sl.connect(db)
    curs = conn.cursor()

    v = (email, username, password)
    stmt = "INSERT INTO users (email, username, password) VALUES(?, ?, ?)"  # Insert user
    curs.execute(stmt, v)

    conn.commit()
    conn.close()


# Returns number of days from 2010/1/4
def getNumDays(row):
    first_date = datetime(2010, 1, 4)
    delta = row["date_conv"] - first_date
    return delta.days


if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True)
