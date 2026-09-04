import json
import time

import hdbscan
import numpy as np
import spacy
import umap
from datasets import load_dataset
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
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
                chunk_tokens = []

                if token.pos_ == "VERB":
                    chunk_tokens.append(token)

                    # 1. Tìm tân ngữ trực tiếp (VD: grant PERMISSION)
                    dobj = [c for c in token.children if c.dep_ == "dobj"]
                    if dobj:
                        obj = dobj[0]
                        chunk_tokens.append(obj)
                        # Bắt thêm mạo từ (a, the), đại từ sở hữu (my, their) và tính từ của tân ngữ
                        modifiers = [
                            c
                            for c in obj.children
                            if c.dep_ in ("det", "poss", "amod", "compound")
                        ]
                        chunk_tokens.extend(modifiers)

                        # Bắt giới từ đi theo sau tân ngữ (VD: permission TO)
                        obj_prep = [c for c in obj.children if c.dep_ == "prep"]
                        if obj_prep:
                            chunk_tokens.append(obj_prep[0])

                    # 2. Tìm giới từ bám trực tiếp vào động từ (VD: grant access FROM)
                    v_prep = [c for c in token.children if c.dep_ == "prep"]
                    if v_prep:
                        chunk_tokens.append(v_prep[0])
                        # Nếu động từ KHÔNG CÓ tân ngữ, ta lấy luôn vế sau giới từ (VD: depend on THE SYSTEM)
                        if not dobj:
                            pobj = [c for c in v_prep[0].children if c.dep_ == "pobj"]
                            if pobj:
                                chunk_tokens.append(pobj[0])
                                modifiers = [
                                    c
                                    for c in pobj[0].children
                                    if c.dep_ in ("det", "poss", "amod", "compound")
                                ]
                                chunk_tokens.extend(modifiers)

                    # 3. Gom luôn Phrasal verbs (VD: turn OFF)
                    prt = [c for c in token.children if c.dep_ == "prt"]
                    if prt:
                        chunk_tokens.extend(prt)

                elif token.pos_ == "NOUN":
                    # Xử lý Cụm Danh Từ dài: Tính từ + Danh từ + Giới từ + Tân ngữ (VD: a strong impact ON society)
                    modifiers = [
                        c for c in token.children if c.dep_ in ("amod", "compound")
                    ]
                    prep = [c for c in token.children if c.dep_ == "prep"]

                    if modifiers and prep:
                        chunk_tokens.append(token)
                        chunk_tokens.extend(modifiers)
                        chunk_tokens.append(prep[0])
                        pobj = [c for c in prep[0].children if c.dep_ == "pobj"]
                        if pobj:
                            chunk_tokens.append(pobj[0])

                # LỌC: Bắt buộc độ dài từ 3 đến 6 từ để tạo thành Collocation hoàn chỉnh
                if chunk_tokens and 3 <= len(chunk_tokens) <= 6:
                    # Sắp xếp các từ lại theo đúng thứ tự xuất hiện gốc trong câu (.i)
                    chunk_tokens = sorted(set(chunk_tokens), key=lambda x: x.i)
                    chunk_text = " ".join([t.text.lower() for t in chunk_tokens])

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

    print("\n[5] Xây dựng Cấu trúc Ngân hà 3 Tầng (Bottom-Up)...")
    labels = clusterer.labels_
    valid_labels = sorted(set(labels) - {-1})
    num_leaf_clusters = len(valid_labels)

    # Tầng 3 (Chòm sao Lá): Lấy tâm (centroid) của các cụm HDBSCAN
    leaf_centroids = []
    for label in valid_labels:
        # Dùng embeddings gốc (không gian AI đa chiều) để đảm bảo ngữ nghĩa chuẩn nhất
        cluster_points = embeddings[labels == label]
        leaf_centroids.append(cluster_points.mean(axis=0))
    leaf_centroids = np.array(leaf_centroids)

    # Tầng 2 (Tiểu tinh): Gom lên thành 100 cụm
    n_sub = min(100, num_leaf_clusters)
    print(
        f"[*] Đang gom {num_leaf_clusters} Chòm sao thành {n_sub} Tiểu tinh (Layer 2)..."
    )
    kmeans_l2 = KMeans(n_clusters=n_sub, random_state=42, n_init="auto")
    l2_mapping = kmeans_l2.fit_predict(leaf_centroids)

    # Lấy tâm của Tầng 2 để gom tiếp lên Tầng 1
    l2_centroids = []
    for i in range(n_sub):
        points = leaf_centroids[l2_mapping == i]
        l2_centroids.append(points.mean(axis=0))
    l2_centroids = np.array(l2_centroids)

    # Tầng 1 (Đại tinh): Gom lên thành 10 cụm
    n_super = min(10, n_sub)
    print(f"[*] Đang gom {n_sub} Tiểu tinh thành {n_super} Đại tinh (Layer 1)...")
    kmeans_l1 = KMeans(n_clusters=n_super, random_state=42, n_init="auto")
    l1_mapping = kmeans_l1.fit_predict(l2_centroids)

    # Tạo từ điển map ID nhanh (Layer 3 -> Layer 2 -> Layer 1)
    label_to_l2 = {valid_labels[i]: l2_mapping[i] for i in range(num_leaf_clusters)}
    l2_to_l1 = {i: l1_mapping[i] for i in range(n_sub)}

    type_map = {"conversational_routine": 0, "grammatical_collocation": 1}
    galaxy_nodes = []
    for c, m, label in zip(chunks, metadata, labels):
        is_noise = label == -1
        node = {
            "chunk": c,
            "type": type_map.get(m["type"], 0),
            "l3_cluster": int(label),
            "l2_cluster": int(label_to_l2[label]) if not is_noise else -1,
            "l1_cluster": int(l2_to_l1[label_to_l2[label]]) if not is_noise else -1,
        }
        galaxy_nodes.append(node)

    with open("galaxy_nodes.json", "w", encoding="utf-8") as f:
        json.dump(galaxy_nodes, f, ensure_ascii=False)

    print(
        "\n[V] Hoàn tất! File cây cấu trúc lưu tại 'galaxy_tree.csv' và 'galaxy_nodes.json'."
    )
    print(
        "=> BẢN ĐỒ NGÂN HÀ ĐÃ CHUẨN: 10 Đại tinh -> 100 Tiểu tinh -> Các Chòm sao HDBSCAN."
    )


if __name__ == "__main__":
    build_galaxy()
