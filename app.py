import streamlit as st
import pandas as pd
from pathlib import Path

# ===== CONFIG =====
st.set_page_config(
    page_title="Perpustakaan Novel",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== LOAD CSS =====
with open(Path("assets/style.css")) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ===== LOAD DATA =====
buku = pd.read_csv("buku.csv")
rating = pd.read_csv("rating.csv")

# Gabungkan rating ke buku
buku = buku.merge(
    rating.groupby("id_buku")["rating"].mean().reset_index(),
    left_on="id",
    right_on="id_buku",
    how="left"
)
buku["rating"] = buku["rating"].fillna(0)

# ===== HELPER CARD =====
def render_card(row):
    stars = "★" * round(row["rating"]) + "☆" * (5 - round(row["rating"]))
    return f"""
    <div class="card">
        <div>
            <h4>{row['judul']}</h4>
            <div class="author">{row['pengarang']}</div>
        </div>
        <span class="genre">{row['kategori']}</span>
        <div class="rating">{stars}</div>
    </div>
    """

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown("<div class='sidebar-logo'>📘 Perpustakaan</div>", unsafe_allow_html=True)
    st.markdown("""
        <a href='/?page=home' class='sidebar-link'>🏠 Home</a>
        <a href='/?page=katalog' class='sidebar-link'>📚 Katalog</a>
        <a href='/?page=tentang' class='sidebar-link'>ℹ️ Tentang</a>
    """, unsafe_allow_html=True)

page = st.query_params.get("page", "home")
col_main, col_chat = st.columns([3.5, 1.5], gap="large")

# ===== HOME =====
if page == "home":
    with col_main:
        st.markdown("<div class='main-title'>📚 New Releases</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub-title'>Novel terbaru & rekomendasi populer</div>", unsafe_allow_html=True)

        home_data = buku.sort_values("rating", ascending=False).head(9)
        cols = st.columns([1,1,1], gap="large")
        for i, row in home_data.iterrows():
            with cols[i % 3]:
                st.markdown(render_card(row), unsafe_allow_html=True)

# ===== KATALOG =====
elif page == "katalog":
    with col_main:
        st.markdown("<div class='main-title'>📖 Katalog Novel</div>", unsafe_allow_html=True)
        q = st.text_input("🔍 Cari judul atau pengarang")

        data = buku.sort_values("judul")
        if q:
            data = data[
                data["judul"].str.contains(q, case=False) |
                data["pengarang"].str.contains(q, case=False)
            ]

        cols = st.columns([1,1,1], gap="large")
        for i, row in data.iterrows():
            with cols[i % 3]:
                st.markdown(render_card(row), unsafe_allow_html=True)

# ===== TENTANG =====
else:
    with col_main:
        st.markdown("<div class='main-title'>ℹ️ Tentang</div>", unsafe_allow_html=True)
        st.write("""
        **Perpustakaan Novel Digital**
        - Rekomendasi berbasis rating & genre  
        - Chatbot TF-IDF (offline, tanpa API mahal)  
        - UI modern berbasis Streamlit  
        """)

# =================================================================
# ====================== CHATBOT TF-IDF ===========================
# =================================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===== SESSION STATE =====
if "chat" not in st.session_state:
    st.session_state.chat = []

if "last_recommendation" not in st.session_state:
    st.session_state.last_recommendation = None

# ===== TF-IDF DATASET =====
corpus = []
for _, row in buku.iterrows():
    corpus.append(
        f"rekomendasi novel {row['kategori']} berjudul {row['judul']} karya {row['pengarang']}"
    )

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(corpus)

def chatbot_tfidf(user_text):
    query_vec = vectorizer.transform([user_text])
    similarity = cosine_similarity(query_vec, tfidf_matrix)
    idx = similarity.argmax()
    score = similarity[0][idx]

    if score < 0.15:
        return None

    row = buku.iloc[idx]
    return (
        f"📚 Aku rekomendasikan novel ini:<br>"
        f"<b>{row['judul']}</b><br>"
        f"Genre: {row['kategori']}<br>"
        f"Penulis: {row['pengarang']}"
    )

# ===== CHAT UI =====
with col_chat:
    st.markdown("## 💬 Chat Assistant")

    # INIT STATE
    if "chat" not in st.session_state:
        st.session_state.chat = []

    if "last_recommendation" not in st.session_state:
        st.session_state.last_recommendation = None

    # TAMPILKAN CHAT
    for role, msg in st.session_state.chat:
        cls = "chat-user" if role == "user" else "chat-bot"
        st.markdown(
            f"<div class='{cls}'>{msg}</div>",
            unsafe_allow_html=True
        )

    # ===== FORM CHAT =====
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Tulis pesan...",
            placeholder="contoh: rekomendasi novel bagus dong"
        )
        submit = st.form_submit_button("Kirim")

    if submit and user_input:
        text = user_input.lower()
        st.session_state.chat.append(("user", user_input))

        # ===== LOGIKA CHATBOT =====
        if any(k in text for k in ["rating", "tertinggi", "terbaik", "bagus"]):
            top = buku.sort_values("rating", ascending=False).head(5)
            st.session_state.last_recommendation = top
            reply = (
                "⭐ <b>Novel rating tertinggi:</b><br>" +
                "<br>".join(f"- {x}" for x in top["judul"])
            )

        elif any(g.lower() in text for g in buku["kategori"].unique()):
            genre = next(
                g for g in buku["kategori"].unique()
                if g.lower() in text
            )
            hasil = buku[buku["kategori"].str.lower() == genre.lower()].head(5)
            st.session_state.last_recommendation = hasil
            reply = (
                f"📚 <b>Rekomendasi genre {genre}:</b><br>" +
                "<br>".join(f"- {x}" for x in hasil["judul"])
            )

        elif any(k in text for k in ["penulis", "pengarang", "siapa"]):
            if st.session_state.last_recommendation is not None:
                rows = st.session_state.last_recommendation
                reply = (
                    "✍️ <b>Penulis dari rekomendasi tadi:</b><br>" +
                    "<br>".join(
                        f"- {j} — {p}"
                        for j, p in zip(rows["judul"], rows["pengarang"])
                    )
                )
            else:
                reply = "🙂 Coba minta rekomendasi novelnya dulu ya."

        else:
            tfidf_reply = chatbot_tfidf(text)
            reply = tfidf_reply if tfidf_reply else (
                "🤔 Aku bisa bantu dengan:<br>"
                "• rekomendasi novel romance<br>"
                "• novel rating tertinggi<br>"
                "• novel bagus untuk dibaca"
            )

        st.session_state.chat.append(("bot", reply))
        st.rerun()


