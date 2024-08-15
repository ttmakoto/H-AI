import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# 車線の幅 (m)
lane_width = 6.0
# 同じ車線内での点の幅 (m)
same_lane_width = 2.0
# 障害物の半径 (m)
obstacle_radius = 1.5

# 左レーンと右レーンの2Dポイントを生成
left_lane_start = np.array([0, 0])
right_lane_start = np.array([0, lane_width])

num_points = 10
left_points = np.array([left_lane_start + np.array([i * same_lane_width, np.random.uniform(-0.5, 0.5)]) for i in range(num_points)])
right_points = np.array([right_lane_start + np.array([i * same_lane_width, np.random.uniform(-0.5, 0.5)]) for i in range(num_points)])

# 障害物の追加 (中心位置)
obstacles = np.array([
    [3, 1.5],  # 障害物1
    [13, 4.5]   # 障害物2
])

# 全てのポイントを結合
points = np.vstack((left_points, right_points, obstacles))

# ドロネー三角形分割を実行
tri = Delaunay(points)

# 経路の中点を保存するリスト
path = []

# 各三角形の辺の中点を計算
for simplex in tri.simplices:
    # 三角形の頂点
    vertices = points[simplex]
    
    # 三角形の各辺の中点を計算
    for i in range(3):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % 3]
        mid_point = (p1 + p2) / 2

        # 点が同じ車線のものか確認
        in_left = np.any(np.all(left_points == p1, axis=1)) and np.any(np.all(left_points == p2, axis=1))
        in_right = np.any(np.all(right_points == p1, axis=1)) and np.any(np.all(right_points == p2, axis=1))

        # 左同士、右同士の中点をスキップ
        if in_left or in_right:
            continue

        # 障害物からの距離を計算
        distances = np.linalg.norm(obstacles - mid_point, axis=1)

        # 中点が障害物の円形に含まれていないか確認
        if np.min(distances) > obstacle_radius * 1.2:
            path.append(mid_point)

# 経路の中点をnumpy配列に変換
path = np.array(path)

# 中点同士の距離を計算
def calculate_distances(points):
    num_points = len(points)
    distances = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(i + 1, num_points):
            distances[i, j] = np.linalg.norm(points[i] - points[j])
            distances[j, i] = distances[i, j]
    return distances

# 線分と円の交差判定
def does_line_intersect_circle(p1, p2, circle_center, radius):
    # 線分のベクトル
    line_vec = p2 - p1
    # 線分の長さの2乗
    line_len_sq = np.dot(line_vec, line_vec)
    # 点p1から円の中心までのベクトル
    p1_to_circle = circle_center - p1
    # 線分の直線上の点に投影
    projection = np.dot(p1_to_circle, line_vec) / line_len_sq
    # 投影点が線分上にあるか確認
    projection = np.clip(projection, 0, 1)
    nearest_point = p1 + projection * line_vec
    # 最近点から円の中心までの距離
    distance_to_circle = np.linalg.norm(nearest_point - circle_center)
    return distance_to_circle <= radius

# 最も近い中点同士を見つける
def find_nearest_neighbors(points):
    distances = calculate_distances(points)
    nearest_neighbors = []
    for i in range(len(points)):
        # 自分を除外して最小距離のインデックスを見つける
        nearest_index = np.argmin(np.delete(distances[i], i))
        nearest_neighbors.append((i, nearest_index))
    return nearest_neighbors

# 中点同士の最も近い組み合わせを見つける
nearest_neighbors = find_nearest_neighbors(path)

# 交差しない中点同士の線分のみを選択
valid_lines = []
used_pairs = set()  # すでに使用したペアを保存
for p1_idx, p2_idx in nearest_neighbors:
    p1 = path[p1_idx]
    p2 = path[p2_idx]
    if (p1_idx, p2_idx) in used_pairs or (p2_idx, p1_idx) in used_pairs:
        continue
    intersects = any(does_line_intersect_circle(p1, p2, obs, obstacle_radius) for obs in obstacles)
    if not intersects:
        valid_lines.append((p1, p2))
        used_pairs.add((p1_idx, p2_idx))  # ペアを記録

# プロット
plt.figure()

# 経路の中点をプロット (点のみ表示)
if path.size > 0:  # path が空でないことを確認
    plt.plot(path[:, 0], path[:, 1], 'ro', label="Midpoints")

# ドロネー三角形分割のプロット
plt.triplot(points[:, 0], points[:, 1], tri.simplices, color='blue')

# 左レーンと右レーンのポイントをプロット
plt.plot(left_points[:, 0], left_points[:, 1], 'o', color='green', label="Left Lane Points")
plt.plot(right_points[:, 0], right_points[:, 1], 'o', color='black', label="Right Lane Points")

# 障害物を円形としてプロット
for obs in obstacles:
    circle = plt.Circle(obs, obstacle_radius, color='red', fill=True, alpha=0.5, label='Obstacle')
    plt.gca().add_artist(circle)

# 交差しない中点同士を線で結ぶ
for p1, p2 in valid_lines:
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b--')

# プロットの装飾
plt.gca().set_aspect('equal')
plt.title('Delaunay Triangulation with Midpoints and Obstacles')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # 凡例を図の外に配置
plt.show()