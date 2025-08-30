from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
from psycopg2.extras import RealDictCursor
import re
import os
from dotenv import load_dotenv
from openai import OpenAI
import math

# ✅ Load .env
load_dotenv()

# ✅ Flask app
app = Flask(__name__)
CORS(app)

# 🔐 OpenAI Config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
MODEL_NAME = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"

# 🛢️ PostgreSQL Config
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": int(os.getenv("DB_PORT"))
}

# 📞 Contact Info
CONTACTS = {
    "founder": {"name": "Divya Khandal", "email": "divz333@gmail.com", "phone": "9166167005", "role": "Founder"},
    "gm": {"name": "Mr. Maan Singh", "email": "mansinghr4@gmail.com", "phone": "9829854896", "role": "General Manager"}
}

# -------------------- Helper Functions --------------------

def is_hindi(text):
    return re.search('[\u0900-\u097F]', text) is not None

def smart_filter(content, query, max_sentences=3):
    sentences = re.split(r'(?<=[.?!])\s+', content.strip())
    query_words = query.lower().split()
    scored = [(sum(1 for w in query_words if w in s.lower()), s) 
              for s in sentences if any(w in s.lower() for w in query_words)]
    scored.sort(reverse=True)
    filtered = [s for _, s in scored]
    return " ".join(filtered[:max_sentences]) if filtered else " ".join(sentences[:max_sentences])

def contact_response(msg):
    msg = msg.lower()
    if "founder" in msg or "divya" in msg:
        return f"👩‍💼 Founder: {CONTACTS['founder']['name']}\n📧 {CONTACTS['founder']['email']}\n📞 {CONTACTS['founder']['phone']}"
    elif "general manager" in msg or "maan singh" in msg or "gm" in msg:
        return f"👨‍💼 GM: {CONTACTS['gm']['name']}\n📧 {CONTACTS['gm']['email']}\n📞 {CONTACTS['gm']['phone']}"
    elif "contact" in msg:
        return f"📞 Founder: {CONTACTS['founder']['phone']} | GM: {CONTACTS['gm']['phone']}\n📧 Emails: {CONTACTS['founder']['email']}, {CONTACTS['gm']['email']}"
    return None

def cosine_similarity(vec1, vec2):
    dot = sum(a*b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a*a for a in vec1))
    norm2 = math.sqrt(sum(b*b for b in vec2))
    return dot / (norm1 * norm2)

# -------------------- Database + Semantic Search --------------------

def search_database(query):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT title, url, content FROM dhonk_pages LIMIT 20")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        if not rows:
            return None

        # ✅ Generate embedding for query
        query_embedding = client.embeddings.create(model=EMBED_MODEL, input=query).data[0].embedding

        # ✅ Compare with DB rows
        best_match, best_score = None, -1
        for row in rows:
            text = row['content'] or ""
            if not text.strip():
                continue
            doc_embedding = client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding
            score = cosine_similarity(query_embedding, doc_embedding)
            if score > best_score:
                best_score, best_match = score, row

        if best_score > 0.75:  # threshold
            return best_match
        return None

    except Exception as e:
        print("DB Error:", e)
        return None

# -------------------- System Prompts --------------------

system_prompt_en = "You are ONLY an AI assistant for Dhonk Craft. Be concise and helpful."
system_prompt_hi = "आप Dhonk Craft के लिए एक सहायक बॉट हैं। संक्षिप्त और मददगार जवाब दें।"

# -------------------- Routes --------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "✅ Dhonk Craft Backend with OpenAI is running!"})

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message", "").strip()
    if not user_msg:
        return jsonify({"answer": "❌ Please type something."}), 400

    # 1. Intent Handling
    from intent_handler import detect_intent, get_intent_response
    intent = detect_intent(user_msg)
    intent_response = get_intent_response(intent)
    if intent_response:
        return jsonify({"answer": intent_response})

    # 2. Contact Info
    contact_reply = contact_response(user_msg)
    if contact_reply:
        return jsonify({"answer": contact_reply})

    # 3. DB + Semantic Search
    db_result = search_database(user_msg)
    if db_result:
        short_answer = smart_filter(db_result['content'], user_msg)
        if db_result['url']:
            short_answer += f"\n\n🔗 [More Info]({db_result['url']})"
        return jsonify({"answer": short_answer})

    # 4. Fallback → LLM
    try:
        system_prompt = system_prompt_hi if is_hindi(user_msg) else system_prompt_en
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_msg}],
            temperature=0.6
        )
        reply = response.choices[0].message.content
        return jsonify({"answer": reply})

    except Exception as e:
        return jsonify({"answer": f"❌ OpenAI Error: {str(e)}"}), 500

# -------------------- Run --------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
