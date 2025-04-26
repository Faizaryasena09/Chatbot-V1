import os
import json
import torch
import ast
import re
from transformers import pipeline
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
import tensorflow as tf

emotion_detector = pipeline("sentiment-analysis")
qa_pipeline = pipeline("question-answering")
math_model = pipeline("text2text-generation", model="t5-small")

MODEL_PATH = "./indo-gpt2"
BRAIN_FILE = "brain.json"
TRAIN_FILE = "train_data.txt"
CHAT_LOG_FILE = "chat_history.txt"

factory = StopWordRemoverFactory()
stop_words_indonesian = factory.get_stop_words()

vectorizer = TfidfVectorizer(stop_words=stop_words_indonesian)
nmf = NMF(n_components=3, random_state=42)

tokenizer = AutoTokenizer.from_pretrained("cahya/gpt2-small-indonesian-522M")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

def split_to_topics(text):
    pattern = r"(?:^|\n)([A-Z][^\n:]{2,50}):\s*"
    matches = list(re.finditer(pattern, text))

    result = []

    if matches and matches[0].start() > 0:
        first_part = text[:matches[0].start()].strip()
        if first_part:
            result.append((
                "Apa yang dimaksud dengan topik utama ini?",
                first_part
            ))

    for i in range(len(matches)):
        title = matches[i].group(1).strip()
        start = matches[i].end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        pertanyaan = generate_question_from_heading(title)
        result.append((pertanyaan, content))

    return result

def save_chat_history(prompt, reply):
    with open(CHAT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"User: {prompt}\nBot: {reply}\n\n")

def generate_question_from_heading(title):
    title_lower = title.lower()
    
    if title_lower.startswith("apa itu"):
        return title + "?"
    
    elif title_lower.startswith("fungsi"):
        return f"Apa fungsi dari {title_lower.replace('fungsi', '').strip()}?"
    
    elif title_lower.startswith("tips") or title_lower.startswith("cara"):
        return f"Apa saja tips untuk {title_lower}? Apa cara terbaik untuk {title_lower}?"
    
    elif title_lower.startswith("komponen"):
        return f"Apa saja komponen yang terdapat pada {title_lower.replace('komponen', '').strip()}?"
    
    elif title_lower.startswith("jenis"):
        return f"Apa saja jenis dari {title.lower()}? Bagaimana cara membedakan jenis-jenis {title_lower.replace('jenis', '').strip()}?"
    
    elif title_lower.startswith("manfaat"):
        return f"Apa manfaat dari {title.lower()}? Bagaimana {title.lower()} bisa memberikan manfaat dalam kehidupan sehari-hari?"
    
    elif title_lower.startswith("sejarah"):
        return f"Bagaimana sejarah dari {title.lower()}? Apa yang terjadi pada {title.lower()} di masa lalu?"
    
    elif title_lower.startswith("proses"):
        return f"Bagaimana proses {title.lower()} dilakukan? Apa langkah-langkah dalam {title.lower()}?"
    
    elif title_lower.startswith("perbedaan"):
        return f"Apa perbedaan antara {title_lower.replace('perbedaan', '').strip()} dan {title_lower.replace('perbedaan', '').strip()}?"
    
    elif title_lower.startswith("keuntungan"):
        return f"Apa keuntungan dari {title.lower()}? Mengapa {title.lower()} lebih menguntungkan dibandingkan yang lain?"
    
    else:
        return f"Apa yang kamu ketahui tentang {title}?"

def detect_topic_keywords(texts, num_words=3):
    factory = StopWordRemoverFactory()
    stop_words = factory.get_stop_words()

    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf = vectorizer.fit_transform(texts)
    nmf = NMF(n_components=len(texts), random_state=42)
    W = nmf.fit_transform(tfidf)
    H = nmf.components_

    words = vectorizer.get_feature_names_out()
    topic_keywords = []
    for topic in H:
        top_words = [words[i] for i in topic.argsort()[-num_words:]]
        topic_keywords.append(top_words)
    return topic_keywords

def auto_generate_knowledge(text):
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
    if not paragraphs:
        return []

    topic_keywords = detect_topic_keywords(paragraphs)

    knowledge = []
    for i, para in enumerate(paragraphs):
        keywords = topic_keywords[i] if i < len(topic_keywords) else ["topik"]
        topik_str = " ".join(keywords)
        pertanyaan = f"Apa yang kamu ketahui tentang {topik_str}?"
        knowledge.append((pertanyaan, para))

    return knowledge

def detect_topics(texts, num_topics=3, num_words=3):
    factory = StopWordRemoverFactory()
    stop_words = factory.get_stop_words()

    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf = vectorizer.fit_transform(texts)
    nmf = NMF(n_components=min(num_topics, len(texts)), random_state=42)
    W = nmf.fit_transform(tfidf)
    H = nmf.components_

    words = vectorizer.get_feature_names_out()
    topics = []
    for topic in H:
        top_words = [words[i] for i in topic.argsort()[-num_words:]]
        topics.append(" ".join(top_words))
    return topics

def extract_topics(texts, num_topics=3, num_words=3):
    tfidf = vectorizer.fit_transform(texts)
    W = nmf.fit_transform(tfidf)
    H = nmf.components_
    
    words = vectorizer.get_feature_names_out()
    topics = []
    for topic in H:
        top_words = [words[i] for i in topic.argsort()[-num_words:]]
        topics.append(" ".join(top_words))
    return topics

def extract_keywords(text, top_n=3):
    vectorizer = TfidfVectorizer(stop_words='indonesian', max_features=50)
    X = vectorizer.fit_transform([text])
    keywords = vectorizer.get_feature_names_out()
    scores = X.toarray()[0]
    keyword_score_pairs = list(zip(keywords, scores))
    keyword_score_pairs.sort(key=lambda x: x[1], reverse=True)
    return [kw for kw, score in keyword_score_pairs[:top_n]]

def load_model():
    if os.path.exists(os.path.join(MODEL_PATH, "config.json")):
        print("✅ Model diload dari lokal.")
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    else:
        print("📥 Model IndoGPT2 diunduh dari HuggingFace.")
        model = AutoModelForCausalLM.from_pretrained("cahya/gpt2-small-indonesian-522M")
    model.resize_token_embeddings(len(tokenizer))
    return model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

model = load_model()

def load_brain():
    if os.path.exists(BRAIN_FILE):
        with open(BRAIN_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_brain(brain):
    with open(BRAIN_FILE, "w", encoding="utf-8") as f:
        json.dump(brain, f, ensure_ascii=False, indent=2)

def add_knowledge(question, answer):
    brain = load_brain()
    question = question.strip().lower()
    answer = answer.strip()

    if not question or not answer:
        print("❌ Pertanyaan atau jawaban tidak valid.")
        return

    brain[question] = answer
    save_brain(brain)

    with open(TRAIN_FILE, "a", encoding="utf-8") as f:
        f.write(f"<s> {question} {answer}</s>\n")

def add_knowledge_from_text_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

            knowledge_pairs = split_to_topics(content)

            if not knowledge_pairs:
                print("❌ Tidak ada pengetahuan yang valid ditemukan.")
                return

            for pertanyaan, jawaban in knowledge_pairs:
                add_knowledge(pertanyaan, jawaban)
                print(f"✅ Disimpan: {pertanyaan}")

    except Exception as e:
        print(f"❌ Terjadi kesalahan saat membaca file: {e}")

def train_model():
    global model
    if not os.path.exists(TRAIN_FILE) or os.stat(TRAIN_FILE).st_size == 0:
        print("❌ Tidak ada data untuk training.")
        return

    print("🔁 Melatih ulang model...")

    datasets = [TRAIN_FILE, CHAT_LOG_FILE]

    dataset = load_dataset("text", data_files={ "train": datasets })

    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

    tokenized = dataset["train"].map(tokenize_function, batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=MODEL_PATH,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=1,
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    print("✅ Training selesai dan model disimpan.")
    
    model = load_model()
    print("🔄 Model telah dimuat ulang.")

chat_history = []

def detect_emotion(text):
    emotion_keywords = {
        "marah": ["marah", "kesal", "geram", "frustrasi", "jengkel"],
        "senang": ["senang", "bahagia", "gembira", "senyum", "ceria"],
        "sedih": ["sedih", "murung", "kecewa", "menangis", "putus asa"],
        "kecewa": ["kecewa", "frustrasi", "terluka", "patah hati"],
        "benci": ["benci", "dendam", "jengkel", "kesal"],
        "takut": ["takut", "cemas", "khawatir", "gelisah"]
    }
    
    for emotion, keywords in emotion_keywords.items():
        if any(keyword in text.lower() for keyword in keywords):
            return emotion
    
    result = emotion_detector(text)
    emotion_label = result[0]['label'].lower()
    
    if 'anger' in emotion_label:
        return "marah"
    elif 'joy' in emotion_label:
        return "senang"
    elif 'sadness' in emotion_label:
        return "sedih"
    elif 'fear' in emotion_label:
        return "takut"
    elif 'disgust' in emotion_label:
        return "benci"
    elif 'surprise' in emotion_label:
        return "kecewa"
    else:
        return "netral"

def detect_greeting(text):
    greetings = [
        "halo", "hai", "assalamualaikum", "hi", "selamat pagi", "selamat malam", 
        "selamat sore", "selamat datang", "hai semuanya", "apa kabar", "apa khabar", 
        "good morning", "good evening", "good night", "hey", "yo", "greetings", 
        "halo semua", "selamat siang", "hi semua", "pagi", "malam"
    ]
    
    text = text.lower()
    
    for greet in greetings:
        if re.search(r'\b' + re.escape(greet) + r'\b', text):
            return True
    return False

def try_math_eval(prompt):
    try:
        expr = re.sub(r"[^0-9\\+\\-\\*/\\.\\(\\)]", "", prompt)
        if expr:
            result = eval(expr)
            return f"Hasilnya adalah {result}"
    except Exception as e:
        pass
    return None

def is_math_expression(expr):
    try:
        node = ast.parse(expr, mode='eval')
        for n in ast.walk(node):
            if not isinstance(n, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Load)):
                return False
        return True
    except:
        return False

def clean_math_prompt(prompt):
    keywords = [
        "berapa", "hitung", "hasil dari", "berapa hasil", "berapa ya", "berapakah", 
        "tambah", "kurang", "kali", "bagi", "nilai", "berapa nilai", "berapa banyak", 
        "berapa total", "jumlahkan", "solusi dari", "cari", "dapatkan", "hitungkan", 
        "bagaimana cara", "apa hasil", "apa itu", "berapa hasil"
    ]
    
    prompt = prompt.lower()
    
    for key in keywords:
        prompt = prompt.replace(key, "")
    
    return prompt.strip().replace("=", "").replace("?", "").replace("?", "").replace(" ", "")

def generate_reply(prompt):
    brain = load_brain()
    key = prompt.strip().lower()

    if key in brain:
        return random.choice(brain[key]) if isinstance(brain[key], list) else brain[key]

    if detect_greeting(prompt):
        if detect_greeting(prompt):
         return random.choice([
        "👋 Hai juga! Ada yang bisa aku bantu?",
        "Halo! Senang ngobrol sama kamu 😊",
        "Hai! Gimana kabarnya?",
        "Hey! Apa yang bisa aku bantu hari ini? 👋",
        "Salam! Ada yang bisa aku bantu? 😄",
        "Halo! Aku senang bisa ngobrol dengan kamu!",
        "Hi! Ada yang ingin kamu bicarakan? 😊",
        "Hai! Semoga harimu menyenankan! 😎",
        "Selamat datang! Ada yang bisa aku bantu? 👋",
        "Hey, apa kabar? Semoga hari ini menyenankan! 🌞",
        "Halo! Senang sekali bisa ngobrol denganmu! 😁",
        "Halo! Semangat ya hari ini! Apa yang bisa aku bantu? 💪",
        "Hai! Apa yang bisa kita diskusikan hari ini? 🤔"
    ])

    math_result = try_math_eval(prompt)
    if math_result:
        return math_result

    emotion = detect_emotion(prompt)
    preface = {
    "marah": random.choice([
        "😠 Wah, kamu keliatan marah. Ceritain aja ya.",
        "😥 Tenang, aku dengerin kok.",
        "😡 Aku bisa ngerasain kemarahanmu. Ayo cerita.",
        "😤 Gak apa-apa, kadang emosi itu perlu dikeluarin.",
        "🤬 Marah banget ya? Ceritain, biar lega!"
    ]),
    "sedih": random.choice([
        "😢 Aku turut sedih. Mau cerita?",
        "💙 Semoga kamu lekas membaik.",
        "😔 Aku tahu ini sulit, apa yang bisa aku bantu?",
        "😓 Sedih banget ya. Aku siap dengerin.",
        "😞 Jangan khawatir, kamu nggak sendiri. Mau cerita?"
    ]),
    "senang": random.choice([
        "😊 Wah senangnya!",
        "😄 Seneng denger kabar baik!",
        "😁 Senang banget denger kabarmu!",
        "🎉 Yeay, kabar baik banget! Senang banget!",
        "🤩 Wah, keren banget! Kamu pasti merasa bahagia."
    ]),
    "netral": "" 
}.get(emotion, "")

    context = "\n".join(chat_history[-6:] + [f"User: {prompt}", "Bot:"])

    inputs = tokenizer(context, return_tensors="pt", truncation=True, padding=True, max_length=256)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=80,
        do_sample=True,
        top_k=40,
        top_p=0.9,
        temperature=0.8,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    reply = decoded.split("Bot:")[-1].strip()
    if not reply or len(reply.split()) < 2:
        reply = random.choice([
            "Hmmm... bisa dijelasin lagi?",
            "Aku kurang nangkep. Maksudmu gimana?",
            "Coba diulang ya, biar aku bisa bantu lebih baik."
        ])

    if '?' in prompt:
        answer = qa_pipeline(question=prompt, context=context)
        if answer['score'] > 0.5:
            reply = answer['answer']

    return preface + " " + reply.strip()

print("🤖 Selamat datang di AI Chatbot Bahasa Indonesia (Generatif + Belajar Mandiri)")
print("Perintah:\n  /add → tambah data\n  /train → latih ulang AI\n  /addtext → tambah dari file\n  /save → simpan chat\n  /clear → reset memory\n  /exit → keluar\n")

while True:
    try:
        user_input = input("Kamu: ").strip()
        if user_input == "/exit":
            print("👋 Sampai jumpa!")
            break
        elif user_input == "/add":
            q = input("Pertanyaan: ")
            a = input("Jawaban: ")
            add_knowledge(q, a)
            print("✅ Pengetahuan baru disimpan.")
        elif user_input == "/train":
            train_model()
        elif user_input == "/addtext":
            file_path = input("Masukkan path file teks yang ingin digunakan: ")
            if os.path.exists(file_path):
                add_knowledge_from_text_file(file_path)
                train_model()
            else:
                print(f"❌ File tidak ditemukan: {file_path}")
        elif user_input == "/save":
            with open(CHAT_LOG_FILE, "w", encoding="utf-8") as f:
                f.write("\n".join(chat_history))
            print("💾 Riwayat chat disimpan.")
        elif user_input == "/clear":
            open(BRAIN_FILE, "w").write("{}")
            open(TRAIN_FILE, "w").close()
            print("🧠 Memori dan data pelatihan direset.")
        elif user_input == "/reset":
            import shutil
            if os.path.exists(MODEL_PATH):
                shutil.rmtree(MODEL_PATH)
            open(BRAIN_FILE, "w").write("{}")
            open(TRAIN_FILE, "w").close()
            model = load_model()
            print("🔄 Model direset ke versi awal. Pengetahuan dan data training dihapus.")
        else:
            chat_history.append(f"User: {user_input}")
            response = generate_reply(user_input)
            chat_history.append(f"Bot: {response}")
            print("🤖", response)
    except KeyboardInterrupt:
        print("\n👋 Sampai jumpa!")
        break
    except Exception as e:
        print(f"❌ Terjadi kesalahan: {e}")
