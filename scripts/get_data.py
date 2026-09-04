import json
import time

import hdbscan
import numpy as np
import spacy
import umap
from datasets import load_dataset
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

TARGET_CHUNKS = 100000
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
UMAP_DIMENSIONS = 10
HDBSCAN_MIN_CLUSTER_SIZE = 30


def extract_chunks():
    print("[1] Đang tải mô hình spaCy...")
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    print("[*] Đang kết nối kho dữ liệu (CNN News + Daily Dialog)...")
    dataset_cnn = load_dataset(
        "abisee/cnn_dailymail",
        "3.0.0",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    dataset_dialog = load_dataset(
        "roskoN/dailydialog", split="train", streaming=True, trust_remote_code=True
    )

    unique_chunks = {}

    print(f"[*] Bắt đầu trích xuất... Mục tiêu: {TARGET_CHUNKS} cụm/câu.")
    start_time = time.time()
    pbar = tqdm(total=TARGET_CHUNKS, desc="Trích xuất cụm từ", unit="chunk")

    def text_generator():
        iter_cnn = iter(dataset_cnn)
        iter_dialog = iter(dataset_dialog)

        while True:
            try:
                yield next(iter_cnn)["article"]
                dialog_lines = next(iter_dialog)["utterances"]
                yield " ".join(dialog_lines)
            except StopIteration:
                break

    def add_chunk(chunk_text, chunk_type):
        if (
            chunk_text
            and chunk_text not in unique_chunks
            and any(c.isalpha() for c in chunk_text)
        ):
            unique_chunks[chunk_text] = {
                "type": chunk_type,
            }
            pbar.update(1)
            return len(unique_chunks) >= TARGET_CHUNKS
        return False

    for doc in nlp.pipe(text_generator(), batch_size=20):
        is_full = False

        for sent in doc.sents:
            words = [
                token.text.lower()
                for token in sent
                if not token.is_punct and not token.is_space
            ]

            if 2 <= len(words) <= 4:
                raw_sentence = " ".join(words).replace(" '", "'").replace(" n't", "n't")

                is_full = add_chunk(raw_sentence, "conversational_routine")
                if is_full:
                    break

                continue

            for token in sent:
                chunk_text = None

                if token.pos_ == "VERB":
                    prt = [c for c in token.children if c.dep_ == "prt"]
                    dobj = [c for c in token.children if c.dep_ == "dobj"]
                    prep = [c for c in token.children if c.dep_ == "prep"]

                    if prt and dobj:  # Verb + Particle + Dobj (VD: turn it off)
                        chunk_text = f"{token.text.lower()} {prt[0].text.lower()} {dobj[0].text.lower()}"
                    elif prt:  # Phrasal verb: Verb + Particle (VD: give up)
                        chunk_text = f"{token.text.lower()} {prt[0].text.lower()}"
                    elif prep:
                        pobj = [c for c in prep[0].children if c.dep_ == "pobj"]
                        if pobj:  # Verb + Prep + Object (VD: depend on it)
                            chunk_text = f"{token.text.lower()} {prep[0].text.lower()} {pobj[0].text.lower()}"
                    elif dobj:  # Fallback: Verb + Dobj (VD: play soccer)
                        chunk_text = f"{token.text.lower()} {dobj[0].text.lower()}"

                elif token.pos_ == "NOUN":
                    # Lấy tính từ (amod) hoặc danh từ ghép (compound) đi kèm
                    modifiers = [
                        c.text.lower()
                        for c in token.children
                        if c.dep_ in ("amod", "compound")
                    ]
                    if modifiers:
                        chunk_text = f"{' '.join(modifiers)} {token.text.lower()}"

                if chunk_text:
                    is_full = add_chunk(chunk_text, "grammatical_collocation")
                    if is_full:
                        break

            if is_full:
                break
        if is_full:
            break

    pbar.close()
    print(
        f"\n[+] Hoàn thành trích xuất {TARGET_CHUNKS} cụm/câu sau {time.time() - start_time:.2f} giây."
    )
    return list(unique_chunks.keys()), list(unique_chunks.values())


def build_galaxy():
    chunks, metadata = extract_chunks()
    print(f"\n[2] Tải mô hình AI Local: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"[*] Đang Vector hóa (Embedding) {TARGET_CHUNKS} cụm từ...")
    start_time = time.time()
    embeddings = model.encode(chunks, batch_size=512, show_progress_bar=True)
    print(f"[+] Nhúng xong sau {time.time() - start_time:.2f} giây.")

    print(f"\n[3] Khởi động UMAP ép về {UMAP_DIMENSIONS} chiều (Metric: Cosine)...")
    start_time = time.time()
    reducer = umap.UMAP(n_neighbors=30, n_components=UMAP_DIMENSIONS, metric="cosine")
    embeddings_reduced = reducer.fit_transform(embeddings)
    print(f"[+] Ép chiều xong sau {time.time() - start_time:.2f} giây.")

    print("\n[4] Khởi động HDBSCAN quét mật độ xây cây phân cấp...")
    start_time = time.time()
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, min_samples=5, metric="euclidean"
    )
    clusterer.fit(embeddings_reduced)
    print(f"[+] Xây cây phân cấp xong sau {time.time() - start_time:.2f} giây.")

    tree_df = clusterer.condensed_tree_.to_pandas()

    tree_df.to_csv("galaxy_tree.csv", index=False)
    type_map = {"conversational_routine": 0, "grammatical_collocation": 1}
    with open("galaxy_nodes.json", "w", encoding="utf-8") as f:
        json.dump(
            [
                {"chunk": c, "type": type_map.get(m["type"], 0)}
                for c, m in zip(chunks, metadata)
            ],
            f,
            ensure_ascii=False,
        )

    analyze_tree(tree_df, clusterer.labels_, TARGET_CHUNKS)


def analyze_tree(tree_df, labels, total_chunks):
    print("\n==========================================")
    print(" BÁO CÁO CẤU TRÚC NGÂN HÀ (TREE STATS)")
    print("==========================================")

    unique_clusters = set(labels)
    num_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
    noise_count = np.sum(labels == -1)

    print(f"- Tổng số Chòm sao (Clusters) phân rã: {num_clusters}")
    print(
        f"- Rác vũ trụ (Noise/Outliers bị loại): {noise_count} ({noise_count / total_chunks * 100:.1f}%)"
    )

    branching = tree_df.groupby("parent").size()

    print(f"- Branching Factor Trung bình (Số con/cha): {branching.mean():.2f}")
    print(f"- Branching Factor Nhỏ nhất: {branching.min()}")
    print(f"- Branching Factor Lớn nhất (Siêu Tinh Vực nổ): {branching.max()}")

    depth_levels = len(tree_df["lambda_val"].unique())
    print(f"- Độ sâu (LOD Levels / Lambda cuts): {depth_levels} tầng chi tiết.")
    print("==========================================")
    print(
        "[V] Hoàn tất! File cây cấu trúc lưu tại 'galaxy_tree.csv' và 'galaxy_nodes.json'."
    )


if __name__ == "__main__":
    build_galaxy()
