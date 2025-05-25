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

# def build_prompt(subdf, indices, query):
#     snippets = []
#     for i in indices:
#         row = subdf.iloc[i]
#         title = row.title
#         truncated = textwrap.shorten(row.summary, width=1200, placeholder="…")
#         snippets.append(f"{title}: {truncated}")
#     ctx = "\n\n".join(snippets)
#     return (
#         "Use the context below to answer the question.\n\n"
#         f"Context:\n{ctx}\n\n"
#         f"Question: {query}\n"
#         "Answer:"
#     )


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

    tab1, tab2 = st.tabs(["Research Chat", "Cluster Visualization"])
    with tab1:
        if 'chat_answers_history' not in st.session_state:
            st.session_state['chat_answers_history'] = []
        if 'user_prompt_history' not in st.session_state:
            st.session_state['user_prompt_history'] = []
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []
        if 'select_summary' not in st.session_state:
            st.session_state['select_summary'] = []
        if "source_documents" not in st.session_state:
            st.session_state['source_documents'] = []
        if "retrieved_ids" not in st.session_state:
            st.session_state['retrieved_ids'] = []
        col1, col2 = st.columns([3, 2])
        col1.subheader("QA ChatBot")
        col2.subheader("📄Research Source")

        if st.session_state["chat_answers_history"]:
            for i, j in zip(st.session_state["chat_answers_history"], st.session_state["user_prompt_history"]):
                message1 = col1.chat_message("user")
                message1.write(j)
                message2 = col1.chat_message("assistant")
                message2.write(i)

        with col1:
            prompt = col1.chat_input("Enter your questions here")
            if prompt:
                with st.spinner("Thinking......"):
                    res = qa({"query": f"User ask the following {prompt}"})
                    retrieved_ids = []
                    for doc in res["source_documents"]:
                        first_line = doc.page_content.split("\n", 1)[0]
                        if first_line.lower().startswith("id:"):
                            _, raw_id = first_line.split(":", 1)
                            retrieved_ids.append(raw_id.strip())
                        else:
                            retrieved_ids.append(None)
                    st.session_state["retrieved_ids"] = retrieved_ids
                    st.session_state["source_documents"] = res["source_documents"]
                    output = {
                        'answer': res["result"]
                    }
                    st.session_state["chat_answers_history"].append(output['answer'])
                    st.session_state["user_prompt_history"].append(prompt)
                    st.session_state["chat_history"].append((prompt, output['answer']))

        with col2:
            source_docs = st.session_state["source_documents"]
            for i, doc in enumerate(source_docs, start=1):
                first_line = doc.page_content.split("\n", 1)[0]
                display_id = first_line.split(":", 1)[1].strip() if ":" in first_line else "?"
                with st.expander(f"Chunk #{i} — {display_id}"):
                    st.text(doc.page_content)

    with tab2:
        cluster_tab(df_cluster)


if __name__ == "__main__":
    main()
