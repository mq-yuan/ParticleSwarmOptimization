import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict
import time
from dataclasses import dataclass

# ==================== Problem Definition Classes ====================

@dataclass
class TSPResult:
    """结果数据类"""
    best_distance: float
    best_path: List[int]
    convergence_history: List[float]
    computation_time: float
    iterations: int

class BaseTSP(ABC):
    """TSP问题基类"""
    
    def __init__(self, distance_matrix: np.ndarray):
        """
        初始化TSP问题
        
        Args:
            distance_matrix: 距离矩阵，shape为(n_cities, n_cities)
        """
        self.distance_matrix = distance_matrix
        self.n_cities = len(distance_matrix)
        
    @abstractmethod
    def calculate_distance(self, path: List[int]) -> float:
        """计算路径距离"""
        pass
    
    def is_valid_path(self, path: List[int]) -> bool:
        """检查路径是否有效"""
        return (len(path) == self.n_cities and 
                set(path) == set(range(self.n_cities)))
    
    @classmethod
    def create_random_instance(cls, n_cities: int, seed: Optional[int] = None):
        """创建随机TSP实例"""
        if seed is not None:
            np.random.seed(seed)
        
        # 生成随机城市坐标
        coords = np.random.rand(n_cities, 2) * 100
        
        # 计算欧几里得距离矩阵
        distance_matrix = np.zeros((n_cities, n_cities))
        for i in range(n_cities):
            for j in range(n_cities):
                if i != j:
                    distance_matrix[i][j] = np.sqrt(
                        (coords[i][0] - coords[j][0])**2 + 
                        (coords[i][1] - coords[j][1])**2
                    )
        
        return cls(distance_matrix)

class TSP(BaseTSP):
    """对称TSP问题"""
    
    def calculate_distance(self, path: List[int]) -> float:
        """计算对称TSP路径距离"""
        if not self.is_valid_path(path):
            return float('inf')
        
        total_distance = 0.0
        for i in range(len(path)):
            from_city = path[i]
            to_city = path[(i + 1) % len(path)]
            total_distance += self.distance_matrix[from_city][to_city]
        
        return total_distance

class ATSP(BaseTSP):
    """非对称TSP问题"""
    
    def __init__(self, distance_matrix: np.ndarray):
        super().__init__(distance_matrix)
        # 验证矩阵是否为非对称
        if np.allclose(distance_matrix, distance_matrix.T):
            print("Warning: 距离矩阵是对称的，但被视为ATSP")
    
    def calculate_distance(self, path: List[int]) -> float:
        """计算非对称TSP路径距离"""
        return TSP.calculate_distance(self, path)  # 计算逻辑相同
    
    @classmethod
    def create_random_instance(cls, n_cities: int, seed: Optional[int] = None):
        """创建随机ATSP实例"""
        if seed is not None:
            np.random.seed(seed)
        
        # 生成非对称距离矩阵
        distance_matrix = np.random.rand(n_cities, n_cities) * 100
        np.fill_diagonal(distance_matrix, 0)  # 对角线设为0
        
        return cls(distance_matrix)

# ==================== PSO Algorithm Classes ====================

class Particle:
    """粒子类"""
    
    def __init__(self, n_cities: int):
        self.n_cities = n_cities
        self.position = list(range(n_cities))  # 当前位置（路径）
        self.velocity = []  # 速度（交换操作序列）
        self.best_position = self.position.copy()  # 个体最优位置
        self.best_fitness = float('inf')  # 个体最优适应度
        self.fitness = float('inf')  # 当前适应度
        
        # 随机初始化位置
        np.random.shuffle(self.position)
        self.best_position = self.position.copy()
    
    def update_best(self):
        """更新个体最优"""
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position.copy()

class BasePSO(ABC):
    """PSO算法基类"""
    
    def __init__(self, 
                 tsp_problem: BaseTSP,
                 n_particles: int = 30,
                 max_iterations: int = 1000,
                 w: float = 0.5,
                 c1: float = 1.5,
                 c2: float = 1.5,
                 seed: Optional[int] = None):
        
        self.tsp_problem = tsp_problem
        self.n_particles = n_particles
        self.max_iterations = max_iterations
        self.w = w  # 惯性权重
        self.c1 = c1  # 认知系数
        self.c2 = c2  # 社会系数
        
        if seed is not None:
            np.random.seed(seed)
        
        # 初始化粒子群
        self.particles = [Particle(tsp_problem.n_cities) for _ in range(n_particles)]
        
        # 全局最优
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        
        # 收敛历史
        self.convergence_history = []
        
    def evaluate_particle(self, particle: Particle):
        """评估粒子适应度"""
        particle.fitness = self.tsp_problem.calculate_distance(particle.position)
        particle.update_best()
        
        # 更新全局最优
        if particle.fitness < self.global_best_fitness:
            self.global_best_fitness = particle.fitness
            self.global_best_position = particle.position.copy()
    
    def generate_swap_sequence(self, current_path: List[int], target_path: List[int]) -> List[Tuple[int, int]]:
        """生成从当前路径到目标路径的交换序列"""
        current = current_path.copy()
        swaps = []
        
        for i in range(len(target_path)):
            if current[i] != target_path[i]:
                # 找到目标城市在当前位置
                target_city = target_path[i]
                target_index = current.index(target_city)
                
                # 交换
                if target_index != i:
                    swaps.append((i, target_index))
                    current[i], current[target_index] = current[target_index], current[i]
        
        return swaps
    
    def apply_swap_sequence(self, path: List[int], swaps: List[Tuple[int, int]], factor: float = 1.0) -> List[int]:
        """应用交换序列到路径"""
        result = path.copy()
        n_swaps = max(1, int(len(swaps) * factor))
        
        for i in range(min(n_swaps, len(swaps))):
            if np.random.random() < factor:
                idx1, idx2 = swaps[i]
                if 0 <= idx1 < len(result) and 0 <= idx2 < len(result):
                    result[idx1], result[idx2] = result[idx2], result[idx1]
        
        return result
    
    def update_velocity_and_position(self, particle: Particle):
        """更新粒子速度和位置"""
        # 生成朝向个体最优的交换序列
        cognitive_swaps = self.generate_swap_sequence(particle.position, particle.best_position)
        
        # 生成朝向全局最优的交换序列
        social_swaps = self.generate_swap_sequence(particle.position, self.global_best_position)
        
        # 更新位置
        new_position = particle.position.copy()
        
        # 应用惯性（保持部分当前位置）
        # 应用认知部分
        if cognitive_swaps and np.random.random() < self.c1 / (self.c1 + self.c2):
            new_position = self.apply_swap_sequence(new_position, cognitive_swaps, self.c1)
        
        # 应用社会部分
        if social_swaps and np.random.random() < self.c2 / (self.c1 + self.c2):
            new_position = self.apply_swap_sequence(new_position, social_swaps, self.c2)
        
        # 随机扰动（惯性的体现）
        if np.random.random() < self.w:
            i, j = np.random.choice(len(new_position), 2, replace=False)
            new_position[i], new_position[j] = new_position[j], new_position[i]
        
        particle.position = new_position
    
    def solve(self) -> TSPResult:
        """求解TSP问题"""
        start_time = time.time()
        
        # 初始化评估
        for particle in self.particles:
            self.evaluate_particle(particle)
        
        # 迭代优化
        for iteration in range(self.max_iterations):
            for particle in self.particles:
                self.update_velocity_and_position(particle)
                self.evaluate_particle(particle)
            
            # 记录收敛历史
            self.convergence_history.append(self.global_best_fitness)
            
            # 可选：早停策略
            if iteration > 50 and len(set(self.convergence_history[-20:])) == 1:
                break
        
        computation_time = time.time() - start_time
        
        return TSPResult(
            best_distance=self.global_best_fitness,
            best_path=self.global_best_position,
            convergence_history=self.convergence_history,
            computation_time=computation_time,
            iterations=len(self.convergence_history)
        )

# ==================== Enhanced PSO Classes ====================

class EnhancedPSO(BasePSO):
    """增强PSO算法"""
    
    def __init__(self, 
                 tsp_problem: BaseTSP,
                 n_particles: int = 30,
                 max_iterations: int = 1000,
                 w: float = 0.5,
                 c1: float = 1.5,
                 c2: float = 1.5,
                 enable_local_search: bool = True,
                 enable_adaptive_params: bool = True,
                 enable_chaos: bool = True,
                 seed: Optional[int] = None):
        
        super().__init__(tsp_problem, n_particles, max_iterations, w, c1, c2, seed)
        
        self.enable_local_search = enable_local_search
        self.enable_adaptive_params = enable_adaptive_params
        self.enable_chaos = enable_chaos
        
        # 自适应参数
        self.initial_w = w
        self.initial_c1 = c1
        self.initial_c2 = c2
        
        # 混沌参数
        self.chaos_value = 0.5
    
    def local_search_2opt(self, path: List[int]) -> List[int]:
        """2-opt局部搜索"""
        best_path = path.copy()
        best_distance = self.tsp_problem.calculate_distance(best_path)
        
        improved = True
        while improved:
            improved = False
            for i in range(len(path) - 1):
                for j in range(i + 2, len(path)):
                    # 创建新路径
                    new_path = path.copy()
                    new_path[i:j] = reversed(new_path[i:j])
                    
                    new_distance = self.tsp_problem.calculate_distance(new_path)
                    if new_distance < best_distance:
                        best_path = new_path
                        best_distance = new_distance
                        improved = True
        
        return best_path
    
    def update_adaptive_parameters(self, iteration: int):
        """自适应参数更新"""
        if not self.enable_adaptive_params:
            return
        
        # 线性递减惯性权重
        self.w = self.initial_w - (self.initial_w - 0.1) * iteration / self.max_iterations
        
        # 动态调整加速系数
        self.c1 = self.initial_c1 - (self.initial_c1 - 0.5) * iteration / self.max_iterations
        self.c2 = self.initial_c2 + (2.5 - self.initial_c2) * iteration / self.max_iterations
    
    def chaos_operation(self, particle: Particle):
        """混沌操作"""
        if not self.enable_chaos:
            return
        
        # 更新混沌值
        self.chaos_value = 4 * self.chaos_value * (1 - self.chaos_value)
        
        # 以较小概率应用混沌扰动
        if np.random.random() < 0.1 * self.chaos_value:
            n_swaps = max(1, int(len(particle.position) * 0.1))
            for _ in range(n_swaps):
                i, j = np.random.choice(len(particle.position), 2, replace=False)
                particle.position[i], particle.position[j] = particle.position[j], particle.position[i]
    
    def solve(self) -> TSPResult:
        """增强求解方法"""
        start_time = time.time()
        
        # 初始化评估
        for particle in self.particles:
            self.evaluate_particle(particle)
        
        # 迭代优化
        for iteration in range(self.max_iterations):
            # 自适应参数更新
            self.update_adaptive_parameters(iteration)
            
            for particle in self.particles:
                # 标准PSO更新
                self.update_velocity_and_position(particle)
                
                # 局部搜索增强
                if self.enable_local_search and np.random.random() < 0.3:
                    particle.position = self.local_search_2opt(particle.position)
                
                # 混沌操作
                self.chaos_operation(particle)
                
                # 评估
                self.evaluate_particle(particle)
            
            # 记录收敛历史
            self.convergence_history.append(self.global_best_fitness)
            
            # 早停策略
            if iteration > 100 and len(set(self.convergence_history[-30:])) == 1:
                break
        
        computation_time = time.time() - start_time
        
        return TSPResult(
            best_distance=self.global_best_fitness,
            best_path=self.global_best_position,
            convergence_history=self.convergence_history,
            computation_time=computation_time,
            iterations=len(self.convergence_history)
        )

# ==================== Visualization Module ====================

class PSOVisualizer:
    """PSO可视化类"""
    
    @staticmethod
    def plot_convergence(results: Dict[str, TSPResult], title: str = "PSO Convergence Comparison"):
        """绘制收敛曲线"""
        plt.figure(figsize=(12, 8))
        
        for name, result in results.items():
            plt.plot(result.convergence_history, label=f"{name} (Final: {result.best_distance:.2f})")
        
        plt.xlabel('Iteration')
        plt.ylabel('Best Distance')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    @staticmethod
    def plot_performance_comparison(results: Dict[str, TSPResult]):
        """绘制性能对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        names = list(results.keys())
        
        # 最优距离对比
        distances = [results[name].best_distance for name in names]
        axes[0, 0].bar(names, distances)
        axes[0, 0].set_title('Best Distance Comparison')
        axes[0, 0].set_ylabel('Distance')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 计算时间对比
        times = [results[name].computation_time for name in names]
        axes[0, 1].bar(names, times)
        axes[0, 1].set_title('Computation Time Comparison')
        axes[0, 1].set_ylabel('Time (seconds)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 迭代次数对比
        iterations = [results[name].iterations for name in names]
        axes[1, 0].bar(names, iterations)
        axes[1, 0].set_title('Iterations Comparison')
        axes[1, 0].set_ylabel('Iterations')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 收敛速度对比（前50%迭代的改进）
        improvements = []
        for name in names:
            history = results[name].convergence_history
            if len(history) > 10:
                initial = history[0]
                mid_point = history[len(history)//2]
                improvement = (initial - mid_point) / initial * 100
                improvements.append(max(0, improvement))
            else:
                improvements.append(0)
        
        axes[1, 1].bar(names, improvements)
        axes[1, 1].set_title('Convergence Speed (% improvement in first 50% iterations)')
        axes[1, 1].set_ylabel('Improvement %')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

# ==================== Testing and Demo ====================

def run_comprehensive_test():
    """运行综合测试"""
    print("=== PSO-TSP 综合测试 ===\n")
    
    # 创建测试问题
    print("创建测试问题...")
    n_cities = 20
    tsp = TSP.create_random_instance(n_cities, seed=42)
    atsp = ATSP.create_random_instance(n_cities, seed=42)
    
    # 测试参数
    test_params = {
        'n_particles': 30,
        'max_iterations': 200,
        'seed': 42
    }
    
    results = {}
    
    # 标准PSO求解TSP
    print("运行标准PSO求解TSP...")
    pso_tsp = BasePSO(tsp, **test_params)
    results['Standard PSO (TSP)'] = pso_tsp.solve()
    
    # 标准PSO求解ATSP  
    print("运行标准PSO求解ATSP...")
    pso_atsp = BasePSO(atsp, **test_params)
    results['Standard PSO (ATSP)'] = pso_atsp.solve()
    
    # 增强PSO求解TSP（全部增强）
    print("运行增强PSO求解TSP（全部增强）...")
    enhanced_pso_all = EnhancedPSO(tsp, **test_params)
    results['Enhanced PSO (All features)'] = enhanced_pso_all.solve()
    
    # 仅局部搜索增强
    print("运行增强PSO（仅局部搜索）...")
    enhanced_pso_ls = EnhancedPSO(tsp, enable_local_search=True, 
                                  enable_adaptive_params=False, 
                                  enable_chaos=False, **test_params)
    results['Enhanced PSO (Local Search only)'] = enhanced_pso_ls.solve()
    
    # 仅自适应参数增强
    print("运行增强PSO（仅自适应参数）...")
    enhanced_pso_adaptive = EnhancedPSO(tsp, enable_local_search=False, 
                                       enable_adaptive_params=True, 
                                       enable_chaos=False, **test_params)
    results['Enhanced PSO (Adaptive only)'] = enhanced_pso_adaptive.solve()
    
    # 仅混沌操作增强
    print("运行增强PSO（仅混沌操作）...")
    enhanced_pso_chaos = EnhancedPSO(tsp, enable_local_search=False, 
                                    enable_adaptive_params=False, 
                                    enable_chaos=True, **test_params)
    results['Enhanced PSO (Chaos only)'] = enhanced_pso_chaos.solve()
    
    # 打印结果
    print("\n=== 测试结果 ===")
    for name, result in results.items():
        print(f"{name}:")
        print(f"  最优距离: {result.best_distance:.2f}")
        print(f"  计算时间: {result.computation_time:.2f}s")
        print(f"  迭代次数: {result.iterations}")
        print(f"  最优路径: {result.best_path[:10]}..." if len(result.best_path) > 10 else f"  最优路径: {result.best_path}")
        print()
    
    # 可视化
    print("生成可视化图表...")
    visualizer = PSOVisualizer()
    visualizer.plot_convergence(results, "PSO算法收敛性对比")
    visualizer.plot_performance_comparison(results)
    
    return results

if __name__ == "__main__":
    # 运行测试
    results = run_comprehensive_test()
    
    print("测试完成！")
    
    # 额外的小规模快速测试
    print("\n=== 快速验证测试 ===")
    small_tsp = TSP.create_random_instance(10, seed=123)
    quick_pso = EnhancedPSO(small_tsp, n_particles=20, max_iterations=50, seed=123)
    quick_result = quick_pso.solve()
    
    print("10城市TSP测试:")
    print(f"最优距离: {quick_result.best_distance:.2f}")
    print(f"最优路径: {quick_result.best_path}")
    print(f"计算时间: {quick_result.computation_time:.2f}s")
