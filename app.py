from flask import Flask, render_template, request, jsonify
from chatbot import ChatBot

app = Flask(__name__)

# Initialize chatbot
print("Loading model...")
chatbot = ChatBot()
print("Ready!")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["GET", "POST"])
def get_bot_response():
    user_text = request.args.get('msg') if request.method == "GET" else request.form['msg']
    response, tag = chatbot.chat(user_text)
    return jsonify({
        "response": response,
        "tag": tag
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)