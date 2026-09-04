import json
from collections import defaultdict

import numpy as np
import pandas as pd


def get_actual_tree_depth(tree_df):
    cluster_links = tree_df[tree_df["child_size"] > 1]
    if cluster_links.empty:
        return 1

    children_map = {}
    for _, row in cluster_links.iterrows():
        p = int(row["parent"])
        c = int(row["child"])
        if p not in children_map:
            children_map[p] = []
        children_map[p].append(c)

    parents = set(cluster_links["parent"])
    children = set(cluster_links["child"])
    roots = parents - children

    if not roots:
        return 1

    max_d = 0
    stack = [(r, 1) for r in roots]
    while stack:
        curr, d = stack.pop()
        max_d = max(max_d, d)
        if curr in children_map:
            for child in children_map[curr]:
                stack.append((child, d + 1))
    return max_d


def analyze_existing_galaxy():
    try:
        tree_df = pd.read_csv("galaxy_tree.csv")
        with open("galaxy_nodes.json", "r", encoding="utf-8") as f:
            nodes = json.load(f)
    except FileNotFoundError:
        print(
            "Lỗi: Không tìm thấy 'galaxy_tree.csv' hoặc 'galaxy_nodes.json'. Hãy chạy scripts/get_data.py trước."
        )
        return

    total_chunks = len(nodes)
    if total_chunks == 0 or "l3_cluster" not in nodes[0]:
        print(
            "Lỗi: Dữ liệu chưa có định dạng Tầng (l1, l2, l3). Hãy chạy lại file get_data.py mới nhất."
        )
        return

    print("\n==========================================")
    print(" TỔNG QUAN BẢN ĐỒ NGÂN HÀ (STAR MAP STATS)")
    print("==========================================")

    # Đọc nhãn các tầng
    l1_labels = np.array([n.get("l1_cluster", -1) for n in nodes])
    l2_labels = np.array([n.get("l2_cluster", -1) for n in nodes])
    l3_labels = np.array([n.get("l3_cluster", -1) for n in nodes])

    noise_count = np.sum(l3_labels == -1)

    unique_l1 = len(set(l1_labels) - {-1})
    unique_l2 = len(set(l2_labels) - {-1})
    unique_l3 = len(set(l3_labels) - {-1})

    print(f"- Tổng số Từ vựng/Cụm từ: {total_chunks}")
    print(
        f"- Rác vũ trụ (Noise): {noise_count} cụm ({noise_count / total_chunks * 100:.1f}%)"
    )
    print("- Các từ vựng hợp lệ được chia vào:")
    print(f"   + {unique_l1} Đại tinh (Tầng 1)")
    print(f"   + {unique_l2} Tiểu tinh (Tầng 2)")
    print(f"   + {unique_l3} Chòm sao  (Tầng 3)")

    print("\n[PHÂN BỐ CẤU TRÚC CHIA ĐỂ TRỊ (BFS)]")
    # Tính toán phân bố
    l1_to_l2 = defaultdict(set)
    l2_to_l3 = defaultdict(set)
    l3_to_words = defaultdict(int)

    for n in nodes:
        l1, l2, l3 = (
            n.get("l1_cluster", -1),
            n.get("l2_cluster", -1),
            n.get("l3_cluster", -1),
        )
        if l1 != -1 and l2 != -1 and l3 != -1:
            l1_to_l2[l1].add(l2)
            l2_to_l3[l2].add(l3)
            l3_to_words[l3] += 1

    l2_per_l1 = [len(v) for v in l1_to_l2.values()] if l1_to_l2 else [0]
    l3_per_l2 = [len(v) for v in l2_to_l3.values()] if l2_to_l3 else [0]
    words_per_l3 = list(l3_to_words.values()) if l3_to_words else [0]

    print(
        f"- 1 Đại tinh chứa trung bình: {np.mean(l2_per_l1):.1f} Tiểu tinh (Min: {np.min(l2_per_l1)}, Max: {np.max(l2_per_l1)})"
    )
    print(
        f"- 1 Tiểu tinh chứa trung bình: {np.mean(l3_per_l2):.1f} Chòm sao (Min: {np.min(l3_per_l2)}, Max: {np.max(l3_per_l2)})"
    )
    print(
        f"- 1 Chòm sao chứa trung bình: {np.mean(words_per_l3):.1f} Cụm từ (Min: {np.min(words_per_l3)}, Max: {np.max(words_per_l3)})"
    )

    # Đếm phân loại cụm từ
    type_0_count = sum(1 for n in nodes if n.get("type") == 0)
    type_1_count = sum(1 for n in nodes if n.get("type") == 1)

    type_0_noise = sum(
        1 for n in nodes if n.get("type") == 0 and n.get("l3_cluster") == -1
    )
    type_1_noise = sum(
        1 for n in nodes if n.get("type") == 1 and n.get("l3_cluster") == -1
    )

    print("\n[PHÂN LOẠI CỤM TỪ]")
    print(
        f"- Conversational Routine (Giao tiếp mảng): {type_0_count} cụm ({type_0_count / total_chunks * 100:.1f}% tổng)"
    )
    if type_0_count > 0:
        print(
            f"   => Trong đó là rác (Noise): {type_0_noise} cụm ({type_0_noise / type_0_count * 100:.1f}%)"
        )

    print(
        f"- Grammatical Collocation (Cấu trúc ngữ pháp): {type_1_count} cụm ({type_1_count / total_chunks * 100:.1f}% tổng)"
    )
    if type_1_count > 0:
        print(
            f"   => Trong đó là rác (Noise): {type_1_noise} cụm ({type_1_noise / type_1_count * 100:.1f}%)"
        )
    print("------------------------------------------")

    # Thống kê nhánh cây cũ của HDBSCAN
    cluster_links_only = tree_df[tree_df["child_size"] > 1]
    print("\n[THỐNG KÊ QUÁ TRÌNH PHÂN RÃ CỦA HDBSCAN (Tầng Đáy)]")
    if not cluster_links_only.empty:
        branching = tree_df.groupby("parent").size()
        print(f"- Branching Factor Trung bình (Số con/cha): {branching.mean():.2f}")
        print(f"- Branching Factor Nhỏ nhất: {branching.min()}")
        print(f"- Branching Factor Lớn nhất (Siêu Tinh Vực nổ): {branching.max()}")
    else:
        print("- Cây không có nhánh (phẳng hoàn toàn).")

    actual_depth = get_actual_tree_depth(tree_df)
    print(f"- Độ sâu gia phả cây (Topological Depth): {actual_depth} tầng")
    print("==========================================")


if __name__ == "__main__":
    analyze_existing_galaxy()
