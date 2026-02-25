import os
import io
import base64
from datetime import datetime
from flask import Flask, render_template, request, redirect, session, flash, url_for,jsonify
import mysql.connector
import pandas as pd
import plotly.express as px
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import OpenAI 
import json
import google.generativeai as genai
import time



from utils.nlp_utils import extract_entities, get_keywords
from config import DB_CONFIG, SECRET_KEY

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key =SECRET_KEY

load_dotenv()
print("loaded:",os.getenv("GEMINI_API_KEY"))

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model=genai.GenerativeModel("gemini-2.5-flash")
response=model.generate_content("Hello! this is test.")
print(response.text)
# ---------- DB helper ----------
def get_db_connection():
    return mysql.connector.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        database=DB_CONFIG["database"],
        autocommit=False
    )

# ---------- Routes ----------
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        db = get_db_connection()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
        user = cursor.fetchone()
        cursor.close()
        db.close()
        if user:
            session["user"] = user["username"]
            return redirect(url_for("dashboard"))
        flash("Invalid credentials", "danger")
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        db = get_db_connection()
        cursor = db.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
            db.commit()
            flash("Account created. Please log in.", "success")
            return redirect(url_for("login"))
        except mysql.connector.Error as e:
            db.rollback()
            flash("Username already exists or database error.", "danger")
        finally:
            cursor.close()
            db.close()
    return render_template("signup.html")

@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        username = request.form.get("username")
        new_password = request.form.get("new_password")
        db = get_db_connection()
        cursor = db.cursor()
        cursor.execute("SELECT id FROM users WHERE username=%s", (username,))
        user = cursor.fetchone()
        if user:
            cursor.execute("UPDATE users SET password=%s WHERE username=%s", (new_password, username))
            db.commit()
            flash("Password updated. Please login.", "success")
            cursor.close()
            db.close()
            return redirect(url_for("login"))
        else:
            flash("Username not found.", "danger")
        cursor.close()
        db.close()
    return render_template("forgot_password.html")

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", username=session["user"])

# SEARCH: 8 fields, 4 in first row, 4 in second; show results only after clicking search
@app.route("/search", methods=["GET", "POST"])
def search():
    if "user" not in session:
        return redirect(url_for("login"))
    results = []
    if request.method == "POST":
        # allowed filter fields mapping to DB column names used in HTML form
        filters = {}
        keys = ["victim_name","victim_age","victim_gender","weapon_used",
                "crime_domain","police_deployed","city","crime_description","date_of_occurence"]
        for k in keys:
            v = request.form.get(k)
            if v:
                filters[k] = v
        db = get_db_connection()
        cursor = db.cursor(dictionary=True)
        query = "SELECT * FROM cases WHERE 1=1"
        params = []
        for col, val in filters.items():
            if col == "date_of_occurence":
                query += " AND date_of_occurence = %s"
                params.append(val)
            else:
                query += f" AND {col} = %s"
                params.append(val)
        cursor.execute(query, tuple(params))
        results = cursor.fetchall()
        cursor.close()
        db.close()
    return render_template("search.html", results=results)

# CASES: list and add new case
@app.route("/cases", methods=["GET", "POST"])
def cases():
    if "user" not in session:
        return redirect(url_for("login"))
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    if request.method == "POST":
        # collect all fields sent from add-case form
        columns = ["date_reported","date_of_occurence","city","crime_code",
                   "crime_description","victim_name","victim_age","victim_gender","weapon_used","crime_domain",
                   "police_deployed","case_closed_date","case_closed"]
        data = []
        for c in columns:
            val = request.form.get(c)
            # convert empty strings to None
            if c==("date_of_occurence" or "date_reported") and val is not None and val!="":
                 val=val.replace("T","")+":00"
            data.append(val if val != "" else None)
        placeholders = ", ".join(["%s"] * len(columns))
        col_names = ", ".join(columns)
      
        cursor.execute(f"INSERT INTO cases ({col_names}) VALUES ({placeholders})", tuple(data))
        db.commit()
          

    cursor.execute("SELECT * FROM cases ORDER BY report_number DESC LIMIT 5000")
    all_cases = cursor.fetchall()
    cursor.close()
    db.close()
    return render_template("cases.html", cases=all_cases)



@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    if "user" not in session:
        return redirect(url_for("login"))

    suspect = None
    network_img_base64 = None

    if request.method == "POST":
        description = request.form.get("crime_description")
        keywords = get_keywords(description)

        db = get_db_connection()
        cursor = db.cursor(dictionary=True)

        # ---------- Predict suspect ----------
        try:
            cursor.execute("""
                SELECT victim_name, COUNT(*) AS freq
                FROM cases
                WHERE MATCH(crime_description) AGAINST(%s IN NATURAL LANGUAGE MODE)
                GROUP BY victim_name
                ORDER BY freq DESC LIMIT 1
            """, (keywords,))
            row = cursor.fetchone()
            suspect = row["victim_name"] if row else "Unknown"
        except mysql.connector.Error:
            cursor.execute("""
                SELECT victim_name, COUNT(*) AS freq
                FROM cases
                WHERE crime_description LIKE %s
                GROUP BY victim_name
                ORDER BY freq DESC LIMIT 1
            """, (f"%{keywords}%",))
            row = cursor.fetchone()
            suspect = row["victim_name"] if row else "Unknown"

        # ---------- Find related cases ----------
        # Get crime domain of most relevant case
        cursor.execute("""
            SELECT crime_domain 
            FROM cases 
            WHERE crime_description LIKE %s 
            LIMIT 1
        """, (f"%{keywords}%",))
        domain_row = cursor.fetchone()
        related_domain = domain_row["crime_domain"] if domain_row else None

        if related_domain:
            # Fetch only similar-domain cases (limit to 50 for clarity)
            df = pd.read_sql("""
                SELECT report_number, crime_domain, victim_name, crime_description
                FROM cases
                WHERE crime_domain = %s
                ORDER BY report_number DESC
                LIMIT 50
            """, con=db, params=[related_domain])
        else:
            df = pd.DataFrame()
        import re
        import nltk
        nltk.download('stopwords',quiet=True)
        from nltk.corpus import stopwords
        stop_words=set(stopwords.words("english"))
        
        # ---------- Build smaller, meaningful graph ----------
        

        if not df.empty:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            des=df["crime_description"].fillna("").astype(str).tolist()
            vectorizer=TfidfVectorizer(stop_words="english")
            tfidf_matrix=vectorizer.fit_transform(des)
            similarity=cosine_similarity(tfidf_matrix)
            G=nx.Graph()
            for idx,row in df.iterrows():
                G.add_node(int(row["report_number"]),label=row["victim_name"],domain=row["crime_domain"])
            for i in range(len(df)):
                for j in range(i+1,len(df)):
                    if similarity[i,j]>0.15:
                        G.add_edge(int(df.loc[i,"report_number"]),int(df.loc[j,"report_number"]),weight=similarity[i,j])
            print("nodes:",len(G.nodes()))
            print("edges:",len(G.edges()))


            plt.figure(figsize=(10,8))
            pos=nx.spring_layout(G,k=0.4,iterations=50)
          
            nx.draw_networkx_nodes(G,pos,node_size=500,node_color="#66ccff",edgecolors="black",linewidths=0.5,alpha=0.9)
           
            nx.draw_networkx_edges(G,pos,width=[d["weight"]*6 for(_,_,d)in G.edges(data=True)],alpha=0.5,edge_color="gray")
            
            nx.draw_networkx_labels(G,pos,font_size=7,font_color="black")
            plt.title("Crime Realtionship graph",fontsize=12,weight="bold")
            plt.axis("off")
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            plt.close()
            buf.seek(0)
            network_img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            cursor.close()
            db.close()

    return render_template("analyze.html", suspect=suspect, network_graph_html=network_img_base64)


@app.route("/chatbot", methods=["POST"])
def chatbot():
    user_message = request.json.get("message", "")
    if not user_message:
        return jsonify({"response": "Please enter a valid question."})

    # Fetch context from database
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT victim_name, crime_domain, city, crime_description ,report_number,date_of_occurence FROM cases")
    cases = cursor.fetchall()
    cursor.close()
    db.close()

    # Build text context for Gemini
    context = "\n".join([
        f"Victim: {c['victim_name']},Report_Number:{c['report_number']},Date of Occurence:{c['date_of_occurence']}, Domain: {c['crime_domain']}, City: {c['city']}, Description: {c['crime_description']}"
        for c in cases if c.get('crime_description')
    ])

    prompt = f"""
    You are a crime analysis assistant. Answer the user's question using only this data:
    {context}

    Question: {user_message}
    """

    try:
        start_time=time.time()
        model= genai.GenerativeModel("gemini-2.5-flash")
        response=model.generate_content(prompt)
        reply = response.text.strip()
        end_time=time.time()
        response_time=end_time-start_time
        
        answer_weight=len(reply.split())if reply else 0
        
    except Exception as e:
        reply = f"⚠️ Error: {str(e)}"
        response_time=None
        answer_weight=None

    return jsonify({"response": reply,"response_time_seconds":round(response_time,3)if response_time else None,"answer_weight":answer_weight})
    
# VISUALIZATION: bar and line graph of offense counts and trends
@app.route("/visualization")
def visualization():
    if "user" not in session:
        return redirect(url_for("login"))
    db = get_db_connection()
    df = pd.read_sql("SELECT crime_domain, COUNT(*) as count FROM cases GROUP BY crime_domain", con=db)
    db.close()
    df=df.sort_values(by="count",ascending=False)
    fig = px.bar(df, x="crime_domain", y="count", title="Cases by Crime Domain",color="crime_domain",text="count")
    fig.update_layout(xaxis_title="crime Type",yaxis_title="Number of cases", showlegend=False)
    fig.update_traces(textposition="outside")
    graph_html = fig.to_html(full_html=False)
    return render_template("visualization.html", graph_html=graph_html)

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

# Run locally
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
