▶️ How to Run the Project (Local Setup)

1️⃣ Clone the repository
git clone https://github.com/yourusername/AI-Clue-Organizer.git

 2️⃣ Navigate into the project folder
cd AI-Clue-Organizer

 3️⃣ (Optional) Create virtual environment
python -m venv venv
venv\Scripts\activate   # On Windows

 4️⃣ Install required dependencies
pip install flask
pip install mysql-connector-python

 5️⃣ Configure database
 Update your MySQL username, password, and database name in config.py
 Make sure MySQL server is running

 6️⃣ Run the Flask application
python app.py

 7️⃣ Open in browser
http://127.0.0.1:5000

 Access cases page
http://127.0.0.1:5000/cases
