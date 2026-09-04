import json

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

    print("\n==========================================")
    print(" BÁO CÁO CẤU TRÚC NGÂN HÀ (TREE STATS)")
    print("==========================================")

    if total_chunks > 0 and "cluster" in nodes[0]:
        labels = np.array([n["cluster"] for n in nodes])
        unique_clusters = set(labels)
        num_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
        noise_count = np.sum(labels == -1)

        print(f"- Tổng số Chòm sao (Leaf Clusters): {num_clusters}")
        print(
            f"- Rác vũ trụ (Noise/Outliers bị loại): {noise_count} ({noise_count / total_chunks * 100:.1f}%)"
        )
    else:
        print("- Tổng số Chòm sao: (Chưa có data nhãn)")
        print("- Rác vũ trụ: (Chưa có data nhãn)")

    type_0_count = sum(1 for n in nodes if n.get("type") == 0)
    type_1_count = sum(1 for n in nodes if n.get("type") == 1)

    print("\n[PHÂN LOẠI CỤM TỪ]")
    print(
        f"- Conversational Routine (Giao tiếp mảng): {type_0_count} cụm ({type_0_count / total_chunks * 100:.1f}%)"
    )
    print(
        f"- Grammatical Collocation (Cấu trúc ngữ pháp): {type_1_count} cụm ({type_1_count / total_chunks * 100:.1f}%)"
    )
    print("------------------------------------------")

    cluster_links_only = tree_df[tree_df["child_size"] > 1]
    if not cluster_links_only.empty:
        branching = cluster_links_only.groupby("parent").size()
        print(f"- Branching Factor Trung bình (Số cụm con/cha): {branching.mean():.2f}")
        print(f"- Branching Factor Nhỏ nhất: {branching.min()}")
        print(f"- Branching Factor Lớn nhất: {branching.max()}")
    else:
        print("- Cây không có nhánh (phẳng hoàn toàn).")

    actual_depth = get_actual_tree_depth(tree_df)
    print(f"- Độ sâu gia phả cây (Topological Depth): {actual_depth} tầng")
    print("==========================================")


if __name__ == "__main__":
    analyze_existing_galaxy()
