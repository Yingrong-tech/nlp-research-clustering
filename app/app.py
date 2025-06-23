import streamlit as st
import pandas as pd
import torch
import textwrap
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
import faiss

CLUSTERS = {
    "— Select a topic —": None,
    "Pose Estimation": 15,
    "Reinforcement Learning & Control": 7,
    "Facial Analysis & Biometrics": 1,
    "Multilingual NLP & LLMs": 15,
}

device = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def load_models():
    # login(token="XXXX")

    retriever = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2b-it",
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    return retriever, tokenizer, llm


@st.cache_resource
def load_index(cluster_id):
    # Load CSV and filter
    df = pd.read_csv("../output/cluster_sample_10000_v2.csv")
    subdf = df[df.cluster_id == cluster_id].reset_index(drop=True)

    # Embed & index
    retriever, _, _ = load_models()
    summaries = subdf.summary.tolist()
    embeddings = retriever.encode(summaries, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return subdf, index


# Grab your heavy objects once
retriever_model, tokenizer, llm_model = load_models()

# --- UI ---
# st.title("💬 RAG Chatbot with Fast Cluster Switching")
st.title('💬 ScrapeChat')
st.write("Select a cluster, then ask questions. Gemma-7b-it will respond using the selected cluster's context.")

cluster_name = st.selectbox("Choose a knowledge cluster", list(CLUSTERS.keys()))
cluster_id = CLUSTERS[cluster_name]

if cluster_id is None:
    st.info("Please select a cluster above.")
    st.stop()

# Build / reuse only the FAISS index for this cluster
subdf, index = load_index(cluster_id)


def fetch_context(prompt, k=20):
    k = min(k, len(subdf))
    q_emb = retriever_model.encode([prompt], normalize_embeddings=True)
    _, I = index.search(q_emb, k)
    return I[0].tolist()


def build_prompt(indices, query):
    snippets = []
    for i in indices:
        row = subdf.iloc[i]
        title = row.title
        truncated = textwrap.shorten(row.summary, width=1200, placeholder="…")
        snippets.append(f"{title}: {truncated}")
    ctx = "\n\n".join(snippets)
    return (
        "Use the context below to answer the question.\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )


def format_references(indices):
    lines = []
    for i in indices:
        row = subdf.iloc[i]
        lines.append(f"📄 '{row.title}', {row.published_date}")
    return "\n\n".join(lines)


def generate_answer(prompt, **gen_kwargs):
    idxs = fetch_context(prompt)
    full_prompt = build_prompt(idxs, prompt)

    tokenized = tokenizer(
        full_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=8192,
    )
    inputs = {k: v.to(device) for k, v in tokenized.items()}

    out = llm_model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        **gen_kwargs
    )
    resp = tokenizer.decode(out[0], skip_special_tokens=True)
    return resp.split("Answer:")[-1].strip() if "Answer:" in resp else resp.strip()


# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_q := st.chat_input("Ask me anything…"):
    # show user message
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    # references
    idxs = fetch_context(user_q)
    refs = format_references(idxs)
    st.session_state.messages.append({"role": "assistant", "content": f"**References:**\n\n{refs}"})
    with st.chat_message("assistant"):
        st.markdown(f"**References:**\n\n{refs}")

    # answer
    ans = generate_answer(user_q)
    st.session_state.messages.append({"role": "assistant", "content": ans})
    with st.chat_message("assistant"):
        st.markdown(ans)
