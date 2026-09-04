import json

import numpy as np
import pandas as pd


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

        print(f"- Tổng số Chòm sao (Clusters) phân rã: {num_clusters}")
        print(
            f"- Rác vũ trụ (Noise/Outliers bị loại): {noise_count} ({noise_count / total_chunks * 100:.1f}%)"
        )
    else:
        print(
            "- Tổng số Chòm sao: (Không hiển thị được do file JSON cũ chưa lưu nhãn cluster)"
        )
        print(
            "- Rác vũ trụ: (Hãy chạy lại get_data.py bản mới nếu cần xem thông số này)"
        )

    branching = tree_df.groupby("parent").size()

    print(f"- Branching Factor Trung bình (Số con/cha): {branching.mean():.2f}")
    print(f"- Branching Factor Nhỏ nhất: {branching.min()}")
    print(f"- Branching Factor Lớn nhất (Siêu Tinh Vực nổ): {branching.max()}")

    depth_levels = len(tree_df["lambda_val"].unique())
    print(f"- Độ sâu (LOD Levels / Lambda cuts): {depth_levels} tầng chi tiết.")
    print("==========================================")


if __name__ == "__main__":
    analyze_existing_galaxy()
