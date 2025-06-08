import random
import math

# 計算兩點之間距離
def distance(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

# 計算整條路徑長度
def total_distance(tour, cities):
    dist = 0
    for i in range(len(tour)):
        dist += distance(cities[tour[i - 1]], cities[tour[i]])
    return dist

# 建立一個鄰居解（交換兩個城市）
def get_neighbor(tour):
    new_tour = tour[:]
    i, j = random.sample(range(len(tour)), 2)
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour

# 主程式
def hill_climb(cities, max_iterations=10000):
    n = len(cities)
    tour = list(range(n))
    random.shuffle(tour)
    
    current_distance = total_distance(tour, cities)
    
    for iteration in range(max_iterations):
        neighbor = get_neighbor(tour)
        neighbor_distance = total_distance(neighbor, cities)
        
        # 如果鄰居更好，接受
        if neighbor_distance < current_distance:
            tour = neighbor
            current_distance = neighbor_distance
            
    return tour, current_distance

# 測試範例
cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(10)]

best_tour, best_distance = hill_climb(cities)

print("最佳路徑:", best_tour)
print("最短距離:", best_distance)
