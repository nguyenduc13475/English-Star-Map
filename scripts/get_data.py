import json
import time

import hdbscan
import numpy as np
import spacy
import umap
from datasets import load_dataset
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

TARGET_CHUNKS = 100000
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
UMAP_DIMENSIONS = 5
HDBSCAN_MIN_CLUSTER_SIZE = 15


def extract_chunks():
    print("[1] Đang tải mô hình spaCy...")
    # Tắt NER để tăng tốc, bắt buộc GIỮ LẠI 'parser' và 'senter' để tách câu (sents)
    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    print("[*] Đang kết nối kho dữ liệu (CNN News + Daily Dialog)...")
    # Nguồn 1: Academic & News (Nghị luận, công việc, tin tức)
    dataset_cnn = load_dataset(
        "abisee/cnn_dailymail", "3.0.0", split="train", streaming=True
    )
    # Nguồn 2: Daily Dialog (Giao tiếp đời thường, phản xạ ngắn)
    dataset_dialog = load_dataset("roskoN/dailydialog", split="train", streaming=True)

    unique_chunks = {}

    print(f"[*] Bắt đầu trích xuất... Mục tiêu: {TARGET_CHUNKS} cụm/câu.")
    start_time = time.time()

    # Hàm trộn data: 1 bài CNN xen kẽ 1 đoạn hội thoại
    def text_generator():
        iter_cnn = iter(dataset_cnn)
        iter_dialog = iter(dataset_dialog)

        while True:
            try:
                # Trả về 1 bài báo (Text dài)
                yield next(iter_cnn)["article"]
                # Trả về 1 đoạn hội thoại (Nối các câu chat lại với nhau)
                dialog_lines = next(iter_dialog)["dialog"]
                yield " ".join(dialog_lines)
            except StopIteration:
                break

    # Hàm phụ trợ để thêm chunk và check limit cho code gọn
    def add_chunk(chunk_text, context, chunk_type):
        # Lọc rác: Bỏ dấu cách, dấu nháy đơn, kiểm tra xem còn lại toàn chữ cái không
        if chunk_text and chunk_text not in unique_chunks:
            check_str = chunk_text.replace(" ", "").replace("'", "")
            if check_str.isalpha():
                unique_chunks[chunk_text] = {
                    "context": context.replace("\n", " ").strip() + "...",
                    "type": chunk_type,  # Gắn tag để biết là câu ngắn hay cụm ngữ pháp
                }
                if len(unique_chunks) % 10000 == 0:
                    print(
                        f"    -> Đã gom được {len(unique_chunks)} / {TARGET_CHUNKS} dữ liệu..."
                    )
                return len(unique_chunks) >= TARGET_CHUNKS
        return False

    # Chạy Pipeline
    for doc in nlp.pipe(text_generator(), batch_size=1000):
        is_full = False

        # [THAY ĐỔI LỚN]: Duyệt qua TỪNG CÂU (Sentence) thay vì từng chữ
        for sent in doc.sents:
            # Bóc các từ trong câu (bỏ dấu câu và khoảng trắng)
            words = [
                token.text.lower()
                for token in sent
                if not token.is_punct and not token.is_space
            ]

            # ---------------------------------------------------------
            # LUẬT 1: LƯỚI BẮT CÂU NGẮN (2 đến 4 chữ) - Bắt trọn ổ phản xạ
            # ---------------------------------------------------------
            if 2 <= len(words) <= 4:
                # Nối lại thành câu. Sửa lỗi dính chữ (vd: "i 'm" -> "i'm", "do n't" -> "don't")
                raw_sentence = " ".join(words).replace(" '", "'").replace(" n't", "n't")

                # Lưu nguyên cả câu này làm 1 chunk
                is_full = add_chunk(
                    raw_sentence, sent.text[:150], "conversational_routine"
                )
                if is_full:
                    break

                # Đã bắt trọn ổ câu ngắn thì bỏ qua không cần mổ xẻ ngữ pháp nữa
                continue

            # ---------------------------------------------------------
            # LUẬT 2: MỔ XẺ NGỮ PHÁP (Dành cho các câu dài, cấu trúc phức tạp)
            # ---------------------------------------------------------
            for token in sent:
                chunk_text = None

                # Pattern: VERB + dobj
                if token.pos_ == "VERB":
                    for child in token.children:
                        if child.dep_ == "dobj":
                            chunk_text = f"{token.text.lower()} {child.text.lower()}"
                            break

                # Pattern: ADJ + NOUN
                elif token.pos_ == "NOUN":
                    for child in token.children:
                        if child.dep_ == "amod":
                            chunk_text = f"{child.text.lower()} {token.text.lower()}"
                            break

                if chunk_text:
                    is_full = add_chunk(
                        chunk_text, sent.text[:150], "grammatical_collocation"
                    )
                    if is_full:
                        break

            if is_full:
                break
        if is_full:
            break

    print(
        f"[+] Hoàn thành trích xuất {TARGET_CHUNKS} cụm/câu sau {time.time() - start_time:.2f} giây."
    )
    return list(unique_chunks.keys()), list(unique_chunks.values())


def build_galaxy():
    # 1. Trích xuất dữ liệu
    chunks, metadata = extract_chunks()

    # 2. Embedding
    print(f"\n[2] Tải mô hình AI Local: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("[*] Đang Vector hóa (Embedding) 100.000 cụm từ...")
    start_time = time.time()
    embeddings = model.encode(chunks, batch_size=512, show_progress_bar=True)
    print(f"[+] Nhúng xong sau {time.time() - start_time:.2f} giây.")

    # 3. UMAP Giảm chiều
    print(f"\n[3] Khởi động UMAP ép về {UMAP_DIMENSIONS} chiều (Metric: Cosine)...")
    start_time = time.time()
    reducer = umap.UMAP(
        n_neighbors=30, n_components=UMAP_DIMENSIONS, metric="cosine", random_state=42
    )
    embeddings_reduced = reducer.fit_transform(embeddings)
    print(f"[+] Ép chiều xong sau {time.time() - start_time:.2f} giây.")

    # 4. HDBSCAN Phân cụm
    print("\n[4] Khởi động HDBSCAN quét mật độ xây cây phân cấp...")
    start_time = time.time()
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, min_samples=5, metric="euclidean"
    )
    clusterer.fit(embeddings_reduced)
    print(f"[+] Xây cây phân cấp xong sau {time.time() - start_time:.2f} giây.")

    # 5. Phân tích Cây & Lưu File
    tree_df = clusterer.condensed_tree_.to_pandas()

    # Xuất ra JSON/CSV
    tree_df.to_csv("galaxy_tree.csv", index=False)
    with open("galaxy_nodes.json", "w", encoding="utf-8") as f:
        json.dump(
            [{"chunk": c, "meta": m} for c, m in zip(chunks, metadata)],
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 6. Thống kê
    analyze_tree(tree_df, clusterer.labels_, TARGET_CHUNKS)


def analyze_tree(tree_df, labels, total_chunks):
    print("\n==========================================")
    print(" BÁO CÁO CẤU TRÚC NGÂN HÀ (TREE STATS)")
    print("==========================================")

    # Số chòm sao thực tế
    unique_clusters = set(labels)
    num_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
    noise_count = np.sum(labels == -1)

    print(f"- Tổng số Chòm sao (Clusters) phân rã: {num_clusters}")
    print(
        f"- Rác vũ trụ (Noise/Outliers bị loại): {noise_count} ({noise_count / total_chunks * 100:.1f}%)"
    )

    # Phân tích Branching Factor (Độ phân nhánh)
    branching = tree_df.groupby("parent").size()

    print(f"- Branching Factor Trung bình (Số con/cha): {branching.mean():.2f}")
    print(f"- Branching Factor Nhỏ nhất: {branching.min()}")
    print(f"- Branching Factor Lớn nhất (Siêu Tinh Vực nổ): {branching.max()}")

    # Chiều sâu của cây
    depth_levels = len(tree_df["lambda_val"].unique())
    print(f"- Độ sâu (LOD Levels / Lambda cuts): {depth_levels} tầng chi tiết.")
    print("==========================================")
    print(
        "[V] Hoàn tất! File cây cấu trúc lưu tại 'galaxy_tree.csv' và 'galaxy_nodes.json'."
    )


if __name__ == "__main__":
    build_galaxy()
