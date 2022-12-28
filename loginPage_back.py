from flask import Flask, render_template, redirect, url_for, request, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
app = Flask(__name__, template_folder='template')

#
# app.config['MYSQL_HOST'] = 'localhost'
# app.config['MYSQL_USER'] = 'root'
# # DATABASE PASSWORD
# app.config['MYSQL_PASSWORD'] = ''
# # DATABASE NAME
# app.config['MYSQL_DB'] = 'aditisDb'

# mysql = MySQL(app)


@app.route('/')
def home():
    return render_template('Login.html')

# REDIRECT WHEN SUCCESS
@app.route('/success')
def success():
    return render_template('success.html')

# REDIRECT WHEN FAIL
@app.route('/fail')
def fail():
    return render_template('fail.html')

# TO OPEN WHEN CLICKED ON LOGIN
@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        if(username=="anup" and password=="hackathon"):
            return redirect(url_for('/success'))
        else:
            return '<script> alert("incorrect Credentials") </script>'
    else:
        return '<script> alert("Something Went wrong") </script>'


if __name__ == 'main':
    app.run(debug=True)