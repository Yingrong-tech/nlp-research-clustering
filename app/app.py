import streamlit as st
import pandas as pd
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain import HuggingFacePipeline
from transformers import pipeline
import plotly.express as px
import plotly.graph_objects as go
import torch
import textwrap
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
import faiss

if torch.cuda.is_available():
    DEVICE = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


class Config:
    DEVICE: str = DEVICE
    HF_TOKEN: str = "XXXXX"
    # PRETRAIN_MODEL: str = # "google/gemma-7b-it" This is quite big
    PRETRAIN_MODEL: str = "distilgpt2"  # Good for testing
    BASE_OUTPUT_PATH: str = '/Users/waritboonmasiri/PycharmProjects/nlp-research-clustering/output/'
    INDEX_PATH: str = "/Users/waritboonmasiri/PycharmProjects/nlp-research-clustering/output/faiss2/faiss_index"
    CLUSTER_PATH: str = "/Users/waritboonmasiri/PycharmProjects/nlp-research-clustering/output/df_research_clustered_10000_xy.csv"


# Cluster name extract from keywords, see eda.py
cluster_name_map = {
    0: "nmt, machine, based",
    1: "information, based, time",
    2: "agent, tasks, policy",
    3: "LLM related topic: tasks, reasoning, knowledge",
    4: "time, methods, detection",
    5: "Hardware related topic about training: pruning, quantization, hardware",
    6: "causal, matrix, based",
    7: "relations, student, relation",
    8: "target, source, adaptation",
    9: "optimization, search, regret",
    10: "cluster, time, algorithm",
    11: "Graph based machine learning: node, hyperbolic, gnns",
    12: "network, images, semantic",
    13: "language, word, text",
    14: "images, texture, person",
    15: "Pose Estimation: pose, detection, object",
    16: "networks, spike, temporal",
    17: "neural, networks, network",
    18: "dialogue, semantics, based",
    19: "tasks, question, meta",
}

CLUSTERS = {
    "— Select a topic —": None,
    **{name: cid for cid, name in cluster_name_map.items()}
}


@st.cache_resource
def load_selected_clustered_and_index(cluster_id):
    df = pd.read_csv(Config.CLUSTER_PATH)
    subdf = df[df.cluster_id == cluster_id].reset_index(drop=True)

    retriever, _, _ = load_chat_models()
    summaries = subdf.summary.tolist()
    embeddings = retriever.encode(summaries, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return subdf, index


@st.cache_resource
def load_chat_models():
    login(token=Config.HF_TOKEN)

    retriever = SentenceTransformer("all-MiniLM-L6-v2", device=Config.DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(Config.PRETRAIN_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        Config.PRETRAIN_MODEL,
        device_map={"": Config.DEVICE},
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    return retriever, tokenizer, llm


def fetch_context(base_df, retriever_model, index, prompt, k=20):
    k = min(k, len(base_df))
    q_emb = retriever_model.encode([prompt], normalize_embeddings=True)
    _, I = index.search(q_emb, k)
    return I[0].tolist()


def build_prompt(subdf, indices, query):
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


def format_references(subdf, indices):
    lines = []
    for i in indices:
        row = subdf.iloc[i]
        lines.append(f"📄 '{row.title}', {row.published_date}")
    return "\n".join(lines)


def generate_answer(base_df, retriever_model, index, tokenizer, llm_model, prompt, **gen_kwargs):
    idxs = fetch_context(base_df, retriever_model, index, prompt)
    full_prompt = build_prompt(base_df, idxs, prompt)

    tokenized = tokenizer(
        full_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=8192,
    )
    inputs = {k: v.to(Config.DEVICE) for k, v in tokenized.items()}

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


@st.cache_resource
def load_vectorstore() -> FAISS:
    embed = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vs = FAISS.load_local(Config.INDEX_PATH, embed, allow_dangerous_deserialization=True)
    return vs


@st.cache_resource
def load_qa(_vs: FAISS) -> RetrievalQA:
    # load LLM pipeline
    hf_pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        device_map=Config.DEVICE
    )
    llm = HuggingFacePipeline(pipeline=hf_pipe)
    # build QA
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=_vs.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        verbose=False,
    )
    return qa


@st.cache_resource
def load_cluster() -> pd.DataFrame:
    with open(Config.CLUSTER_PATH, "rb") as f:
        clustered_data = pd.read_csv(f)
    return clustered_data


def qa_tab(vs, qa):
    st.header("📚 Q&A over Cluster Research Summaries")
    query = st.text_input("Your question", "")
    if st.button("Ask") and query:
        with st.spinner("🔍 Retrieving answer…"):
            res = qa({"query": query})
        st.subheader("🏷️ Answer")
        st.write(res["result"])

        retrieved_ids = []
        for doc in res["source_documents"]:
            first_line = doc.page_content.split("\n", 1)[0]
            if first_line.lower().startswith("id:"):
                _, raw_id = first_line.split(":", 1)
                retrieved_ids.append(raw_id.strip())
            else:
                retrieved_ids.append(None)

        st.session_state["retrieved_ids"] = retrieved_ids

        st.subheader("📄 Source Chunks")
        for i, doc in enumerate(res["source_documents"], start=1):
            first_line = doc.page_content.split("\n", 1)[0]
            display_id = first_line.split(":", 1)[1].strip() if ":" in first_line else "?"
            with st.expander(f"Chunk #{i} — {display_id}"):
                st.text(doc.page_content)


def cluster_tab(df_cluster: pd.DataFrame):
    st.header("🔎 Cluster TSNE Scatter Plot")
    st.write("Faded points are NOT in the last answer.  Bright points are retrieved.")

    retrieved = set(st.session_state.get("retrieved_ids", []))
    df_cluster["highlight"] = df_cluster["id"].isin(retrieved)
    df_cluster["cluster_str"] = df_cluster["cluster_id"].map(cluster_name_map)

    fig = px.scatter(
        df_cluster,
        x="x",
        y="y",
        color="cluster_str",
        hover_name="title",
        hover_data={
            "id": True,
            "authors": True,
            "published_date": True,
            "cluster_str": False,
            "x": False,
            "y": False,
        },
        title="TSNE Projection of All Documents",
        labels={"cluster_str": "cluster_id"},
        width=1400,
        height=800,
    )
    fig.update_traces(marker=dict(size=6, opacity=0.2))

    df_h = df_cluster[df_cluster["highlight"]]
    if not df_h.empty:
        fig.add_trace(
            go.Scatter(
                x=df_h["x"],
                y=df_h["y"],
                mode="markers",
                marker=dict(
                    size=16,
                    opacity=1.0,
                    line=dict(width=2, color="black"),
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
    st.plotly_chart(fig, use_container_width=False)


def main():
    st.set_page_config(
        page_title="🧠 Research Explorer",
        layout="wide",
    )
    st.title("🧠 Research Explorer")
    vs = load_vectorstore()
    qa = load_qa(vs)
    df_cluster = load_cluster()

    tab1, tab2, tab3 = st.tabs(["Chat", "Q&A", "Cluster Visualization"])
    with tab1:
        retriever_model, tokenizer, llm_model = load_chat_models()
        st.title("💬 RAG Chatbot with Fast Cluster Switching")
        st.write("Select a cluster, then ask questions. Gemma-7b-it will respond using the selected cluster's context.")

        cluster_name = st.selectbox("Choose a knowledge cluster", list(CLUSTERS.keys()))
        cluster_id = CLUSTERS[cluster_name]

        if cluster_id is None:
            st.info("Please select a cluster above.")
            st.stop()

        subdf, index = load_selected_clustered_and_index(cluster_id)

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

            idxs = fetch_context(subdf, retriever_model, index, user_q)
            refs = format_references(subdf, idxs)
            st.session_state.messages.append({"role": "assistant", "content": f"**References:**\n{refs}"})
            with st.chat_message("assistant"):
                st.markdown(f"**References:**\n{refs}")

            ans = generate_answer(subdf, retriever_model, index, tokenizer, llm_model, user_q)
            st.session_state.messages.append({"role": "assistant", "content": ans})
            with st.chat_message("assistant"):
                st.markdown(ans)
    with tab2:
        qa_tab(vs, qa)
    with tab3:
        cluster_tab(df_cluster)


if __name__ == "__main__":
    main()
