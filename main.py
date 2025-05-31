import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
import time
from dataclasses import dataclass
import argparse # 新增
import os       # 新增

# ==================== Problem Definition Classes ====================

@dataclass
class TSPResult:
    """Data class for results"""
    best_distance: float
    best_path: List[int]
    convergence_history: List[float]
    computation_time: float
    iterations: int
    problem_coords: Optional[np.ndarray] = None # 新增: 用于绘图

class BaseTSP(ABC):
    """Base class for TSP problems"""
    
    def __init__(self, distance_matrix: np.ndarray, coords: Optional[np.ndarray] = None):
        """
        Initialize TSP problem
        
        Args:
            distance_matrix: Distance matrix, shape (n_cities, n_cities)
            coords: Optional city coordinates, shape (n_cities, 2)
        """
        self.distance_matrix = distance_matrix
        self.n_cities = len(distance_matrix)
        self.coords = coords # 新增
        
    @abstractmethod
    def calculate_distance(self, path: List[int]) -> float:
        """Calculate path distance"""
        pass
    
    def is_valid_path(self, path: List[int]) -> bool:
        """Check if path is valid"""
        return (len(path) == self.n_cities and 
                set(path) == set(range(self.n_cities)))
    
    @classmethod
    def create_random_instance(cls, n_cities: int, seed: Optional[int] = None):
        """Create random TSP instance"""
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random city coordinates
        coords = np.random.rand(n_cities, 2) * 100
        
        # Calculate Euclidean distance matrix
        distance_matrix = np.zeros((n_cities, n_cities))
        for i in range(n_cities):
            for j in range(n_cities):
                if i != j:
                    distance_matrix[i][j] = np.sqrt(
                        (coords[i][0] - coords[j][0])**2 + 
                        (coords[i][1] - coords[j][1])**2
                    )
        
        return cls(distance_matrix, coords) # 修改: 传递 coords

class TSP(BaseTSP):
    """Symmetric TSP problem"""
    
    def calculate_distance(self, path: List[int]) -> float:
        """Calculate symmetric TSP path distance"""
        if not self.is_valid_path(path):
            return float('inf')
        
        total_distance = 0.0
        for i in range(len(path)):
            from_city = path[i]
            to_city = path[(i + 1) % len(path)]
            total_distance += self.distance_matrix[from_city][to_city]
        
        return total_distance

class ATSP(BaseTSP):
    """Asymmetric TSP problem"""
    
    def __init__(self, distance_matrix: np.ndarray, coords: Optional[np.ndarray] = None): # 修改: 增加 coords
        super().__init__(distance_matrix, coords) # 修改: 传递 coords
        # Validate if matrix is asymmetric
        if np.allclose(distance_matrix, distance_matrix.T):
            print("Warning: Distance matrix is symmetric but treated as ATSP.")
    
    def calculate_distance(self, path: List[int]) -> float:
        """Calculate asymmetric TSP path distance"""
        return TSP.calculate_distance(self, path)  # Calculation logic is the same
    
    @classmethod
    def create_random_instance(cls, n_cities: int, seed: Optional[int] = None):
        """Create random ATSP instance"""
        if seed is not None:
            np.random.seed(seed)
        
        # Generate asymmetric distance matrix
        distance_matrix = np.random.rand(n_cities, n_cities) * 100
        np.fill_diagonal(distance_matrix, 0)  # Set diagonal to 0
        
        # For ATSP, coordinates might not be directly used for distance,
        # but can be generated for visualization consistency if needed.
        # Here, we don't generate specific coords for ATSP's non-Euclidean nature
        # unless a specific visualization need arises that maps it to 2D.
        # For now, coords will be None for ATSP unless explicitly provided.
        coords = np.random.rand(n_cities, 2) * 100 # 可选：为ATSP也生成坐标用于潜在的可视化
        
        return cls(distance_matrix, coords) # 修改: 传递 coords

# ==================== PSO Algorithm Classes ====================

class Particle:
    """Particle class"""
    
    def __init__(self, n_cities: int):
        self.n_cities = n_cities
        self.position = list(range(n_cities))  # Current position (path)
        self.velocity = []  # Velocity (swap operation sequence)
        self.best_position = self.position.copy()  # Personal best position
        self.best_fitness = float('inf')  # Personal best fitness
        self.fitness = float('inf')  # Current fitness
        
        # Randomly initialize position
        np.random.shuffle(self.position)
        self.best_position = self.position.copy()
    
    def update_best(self):
        """Update personal best"""
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position.copy()

class BasePSO(ABC):
    """Base class for PSO algorithm"""
    
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
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive coefficient
        self.c2 = c2  # Social coefficient
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize particle swarm
        self.particles = [Particle(tsp_problem.n_cities) for _ in range(n_particles)]
        
        # Global best
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        
        # Convergence history
        self.convergence_history = []
        
    def evaluate_particle(self, particle: Particle):
        """Evaluate particle fitness"""
        particle.fitness = self.tsp_problem.calculate_distance(particle.position)
        particle.update_best()
        
        # Update global best
        if particle.fitness < self.global_best_fitness:
            self.global_best_fitness = particle.fitness
            self.global_best_position = particle.position.copy()
    
    def generate_swap_sequence(self, current_path: List[int], target_path: List[int]) -> List[Tuple[int, int]]:
        """Generate swap sequence from current path to target path"""
        current = current_path.copy()
        swaps = []
        
        for i in range(len(target_path)):
            if current[i] != target_path[i]:
                target_city = target_path[i]
                try:
                    target_index = current.index(target_city)
                except ValueError: # Should not happen with valid paths
                    continue 
                
                if target_index != i:
                    swaps.append((i, target_index))
                    current[i], current[target_index] = current[target_index], current[i]
        
        return swaps
    
    def apply_swap_sequence(self, path: List[int], swaps: List[Tuple[int, int]], factor: float = 1.0) -> List[int]:
        """Apply swap sequence to path"""
        result = path.copy()
        # Apply a portion of swaps based on the factor, can be stochastic
        for swap_idx in range(len(swaps)):
            if np.random.random() < factor: # Apply this swap with probability 'factor'
                idx1, idx2 = swaps[swap_idx]
                if 0 <= idx1 < len(result) and 0 <= idx2 < len(result):
                    result[idx1], result[idx2] = result[idx2], result[idx1]
        return result
    
    def update_velocity_and_position(self, particle: Particle):
        """Update particle velocity and position"""
        new_position = particle.position.copy()

        # Inertia component (simplified: a chance to apply random swaps or keep parts)
        if np.random.random() < self.w:
            # Apply a few random swaps to represent inertia/exploration
            num_random_swaps = np.random.randint(1, max(2, self.tsp_problem.n_cities // 10))
            for _ in range(num_random_swaps):
                if len(new_position) >= 2:
                    i, j = np.random.choice(len(new_position), 2, replace=False)
                    new_position[i], new_position[j] = new_position[j], new_position[i]
        
        # Cognitive component (move towards personal best)
        cognitive_swaps = self.generate_swap_sequence(new_position, particle.best_position)
        if cognitive_swaps and np.random.random() < self.c1: # Simplified probability
             new_position = self.apply_swap_sequence(new_position, cognitive_swaps, 0.5 + np.random.random()*0.5) # Apply a random portion

        # Social component (move towards global best)
        social_swaps = self.generate_swap_sequence(new_position, self.global_best_position)
        if social_swaps and np.random.random() < self.c2: # Simplified probability
            new_position = self.apply_swap_sequence(new_position, social_swaps, 0.5 + np.random.random()*0.5) # Apply a random portion

        particle.position = new_position
        # Ensure path validity after updates (all cities visited once)
        if not self.tsp_problem.is_valid_path(particle.position):
            # If path becomes invalid, try to repair or revert (simplistic repair: shuffle)
            # A more robust repair might be needed for complex scenarios
            np.random.shuffle(particle.position)


    def solve(self) -> TSPResult:
        """Solve TSP problem"""
        start_time = time.time()
        
        # Initial evaluation
        for particle in self.particles:
            self.evaluate_particle(particle)
        
        if self.global_best_position is None and self.particles: # Initialize global best if not set
            self.global_best_position = self.particles[0].best_position
            self.global_best_fitness = self.particles[0].best_fitness
            for particle in self.particles: # Re-check after first eval
                 if particle.best_fitness < self.global_best_fitness:
                    self.global_best_fitness = particle.best_fitness
                    self.global_best_position = particle.best_position.copy()

        # Iterative optimization
        actual_iterations = 0
        for iteration in range(self.max_iterations):
            actual_iterations += 1
            for particle in self.particles:
                if self.global_best_position is None: # Safety check
                    print("Error: Global best position is None during iteration.")
                    # Attempt to re-initialize or handle error
                    if particle.best_position:
                         self.global_best_position = particle.best_position.copy()
                         self.global_best_fitness = particle.best_fitness
                    else: # Skip update if no valid global best
                         continue
                self.update_velocity_and_position(particle)
                self.evaluate_particle(particle)
            
            self.convergence_history.append(self.global_best_fitness)
            
            if iteration > 50 and len(set(self.convergence_history[-20:])) == 1:
                # print(f"Early stopping at iteration {iteration+1} due to convergence.")
                break
        
        computation_time = time.time() - start_time
        
        return TSPResult(
            best_distance=self.global_best_fitness,
            best_path=self.global_best_position if self.global_best_position else [],
            convergence_history=self.convergence_history,
            computation_time=computation_time,
            iterations=actual_iterations,
            problem_coords=self.tsp_problem.coords # 新增
        )

# ==================== Enhanced PSO Classes ====================

class EnhancedPSO(BasePSO):
    """Enhanced PSO algorithm"""
    
    def __init__(self, 
                 tsp_problem: BaseTSP,
                 n_particles: int = 30,
                 max_iterations: int = 1000,
                 w: float = 0.7, # Adjusted default
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
        
        self.initial_w = w
        self.min_w = 0.2 # Min inertia weight
        self.initial_c1 = c1
        self.final_c1 = 0.5 
        self.initial_c2 = c2
        self.final_c2 = 2.5
        
        self.chaos_value = np.random.random() # Initialize chaos value randomly
    
    def local_search_2opt(self, path: List[int]) -> List[int]:
        """2-opt local search"""
        if not path: return []
        best_path = path.copy()
        best_distance = self.tsp_problem.calculate_distance(best_path)
        
        improved = True
        while improved:
            improved = False
            for i in range(len(best_path) - 1): # Use best_path for iteration
                for j in range(i + 2, len(best_path)):
                    if j >= len(best_path): continue

                    new_path = best_path.copy()
                    segment_to_reverse = best_path[i+1:j+1] # Corrected slicing for 2-opt
                    segment_to_reverse.reverse()
                    new_path = best_path[:i+1] + segment_to_reverse + best_path[j+1:]

                    if not self.tsp_problem.is_valid_path(new_path): continue # Skip invalid paths

                    new_distance = self.tsp_problem.calculate_distance(new_path)
                    if new_distance < best_distance:
                        best_path = new_path
                        best_distance = new_distance
                        improved = True
                        break # Exit inner loop and restart search from the new best_path
                if improved:
                    break # Exit outer loop and restart search
        return best_path
    
    def update_adaptive_parameters(self, iteration: int):
        """Adaptive parameter update"""
        if not self.enable_adaptive_params:
            return
        
        progress = iteration / self.max_iterations
        # Linearly decreasing inertia weight
        self.w = self.initial_w - (self.initial_w - self.min_w) * progress
        
        # Dynamically adjust acceleration coefficients
        self.c1 = self.initial_c1 - (self.initial_c1 - self.final_c1) * progress
        self.c2 = self.initial_c2 + (self.final_c2 - self.initial_c2) * progress
    
    def chaos_operation(self, particle: Particle):
        """Chaos operation"""
        if not self.enable_chaos or not particle.position:
            return
        
        # Logistic map for chaos
        self.chaos_value = 4 * self.chaos_value * (1 - self.chaos_value)
        
        # Apply chaos perturbation with a small probability based on chaos value
        if np.random.random() < 0.1 * (0.5 + self.chaos_value * 0.5): # Scaled probability
            n_swaps = max(1, int(len(particle.position) * 0.05 * self.chaos_value)) # Fewer swaps
            for _ in range(n_swaps):
                if len(particle.position) >=2:
                    i, j = np.random.choice(len(particle.position), 2, replace=False)
                    particle.position[i], particle.position[j] = particle.position[j], particle.position[i]
    
    def solve(self) -> TSPResult:
        """Enhanced solving method"""
        start_time = time.time()
        
        for particle in self.particles:
            self.evaluate_particle(particle)

        if self.global_best_position is None and self.particles: # Initialize global best if not set
            self.global_best_position = self.particles[0].best_position
            self.global_best_fitness = self.particles[0].best_fitness
            for particle in self.particles: # Re-check after first eval
                 if particle.best_fitness < self.global_best_fitness:
                    self.global_best_fitness = particle.best_fitness
                    self.global_best_position = particle.best_position.copy()
        
        actual_iterations = 0
        for iteration in range(self.max_iterations):
            actual_iterations +=1
            self.update_adaptive_parameters(iteration)
            
            for particle in self.particles:
                if self.global_best_position is None: continue # Skip if no global best

                self.update_velocity_and_position(particle) # Standard PSO update
                
                if self.enable_local_search and np.random.random() < 0.3: # Apply 2-opt with some probability
                    particle.position = self.local_search_2opt(particle.position)
                
                self.chaos_operation(particle) # Apply chaos
                
                self.evaluate_particle(particle) # Evaluate after all modifications
            
            self.convergence_history.append(self.global_best_fitness)
            
            if iteration > 100 and len(set(self.convergence_history[-30:])) == 1:
                # print(f"Enhanced PSO: Early stopping at iteration {iteration+1} due to convergence.")
                break
        
        computation_time = time.time() - start_time
        
        return TSPResult(
            best_distance=self.global_best_fitness,
            best_path=self.global_best_position if self.global_best_position else [],
            convergence_history=self.convergence_history,
            computation_time=computation_time,
            iterations=actual_iterations,
            problem_coords=self.tsp_problem.coords # 新增
        )

# ==================== Visualization Module ====================

class PSOVisualizer:
    """PSO visualization class"""
    
    @staticmethod
    def plot_convergence(results: Dict[str, TSPResult], title: str = "PSO Convergence Comparison"):
        """Plot convergence curves"""
        fig = plt.figure(figsize=(12, 8))
        
        for name, result in results.items():
            plt.plot(result.convergence_history, label=f"{name} (Final: {result.best_distance:.2f})")
        
        plt.xlabel('Iteration')
        plt.ylabel('Best Distance')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        return fig

    @staticmethod
    def plot_performance_comparison(results: Dict[str, TSPResult]):
        """Plot performance comparison charts"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('PSO Performance Comparison', fontsize=16)
        
        names = list(results.keys())
        bar_colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum', 'lightsalmon'] # 定义颜色列表以便复用

        # Best Distance Comparison
        distances = [results[name].best_distance for name in names]
        axes[0, 0].bar(names, distances, color=bar_colors[:len(names)])
        axes[0, 0].set_title('Best Distance Comparison')
        axes[0, 0].set_ylabel('Distance')
        # 修改X轴刻度标签的旋转和对齐方式
        plt.setp(axes[0, 0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Computation Time Comparison
        times = [results[name].computation_time for name in names]
        axes[0, 1].bar(names, times, color=bar_colors[:len(names)])
        axes[0, 1].set_title('Computation Time Comparison')
        axes[0, 1].set_ylabel('Time (seconds)')
        # 修改X轴刻度标签的旋转和对齐方式
        plt.setp(axes[0, 1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Iterations Comparison
        iterations = [results[name].iterations for name in names]
        axes[1, 0].bar(names, iterations, color=bar_colors[:len(names)])
        axes[1, 0].set_title('Iterations Comparison')
        axes[1, 0].set_ylabel('Iterations')
        # 修改X轴刻度标签的旋转和对齐方式
        plt.setp(axes[1, 0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Convergence Speed Comparison
        improvements = []
        for name in names:
            history = results[name].convergence_history
            if len(history) > 10: 
                initial = history[0]
                mid_point_idx = min(len(history)-1, len(history)//2) 
                mid_point = history[mid_point_idx]
                improvement = (initial - mid_point) / initial * 100 if initial > 0 else 0
                improvements.append(max(0, improvement))
            else:
                improvements.append(0)
        
        axes[1, 1].bar(names, improvements, color=bar_colors[:len(names)])
        axes[1, 1].set_title('Convergence Speed (% impr. in first 50% iter.)')
        axes[1, 1].set_ylabel('Improvement (%)')
        # 修改X轴刻度标签的旋转和对齐方式
        plt.setp(axes[1, 1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        plt.tight_layout(rect=[0, 0, 1, 0.96]) 
        return fig

    @staticmethod
    def plot_best_path(result: TSPResult, title_prefix: str = "Best Path"):
        """Plot the best path found for a TSP instance"""
        if result.best_path is None or not result.best_path or result.problem_coords is None:
            print(f"Cannot plot best path for {title_prefix}: Missing path or coordinates.")
            return None

        fig = plt.figure(figsize=(8, 8))
        coords = result.problem_coords
        path = result.best_path

        plt.plot(coords[:, 0], coords[:, 1], 'o', markersize=5, label='Cities')

        for i in range(len(path)):
            start_node = path[i]
            end_node = path[(i + 1) % len(path)]
            plt.plot([coords[start_node, 0], coords[end_node, 0]],
                     [coords[start_node, 1], coords[end_node, 1]],
                     'b-', alpha=0.7) 

        if path:
            plt.plot(coords[path[0], 0], coords[path[0], 1], 'go', markersize=10, label='Start City') 

        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(f'{title_prefix} (Distance: {result.best_distance:.2f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal') 
        return fig

# ==================== Main Execution and CLI ====================
def run_specific_pso(pso_class: type, problem: BaseTSP, params: Dict[str, Any], enhancements: Optional[Dict[str, bool]] = None) -> TSPResult:
    """Helper to run a specific PSO variant"""
    if enhancements is None:
        solver = pso_class(problem, **params)
    else:
        solver = pso_class(problem, **params, **enhancements)
    return solver.solve()

def main():
    parser = argparse.ArgumentParser(description="Run PSO algorithms for TSP.")
    parser.add_argument('--n_cities', type=int, default=20, help="Number of cities for the TSP problem.")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument('--pso_type', type=str, default='enhanced_all_tsp', # 默认运行一个TSP变体以便快速测试
                        choices=['base_tsp', 'base_atsp', 'enhanced_all_tsp',
                                 'enhanced_ls_tsp', 'enhanced_adaptive_tsp', 'enhanced_chaos_tsp',
                                 'enhanced_all_atsp', 'enhanced_ls_atsp', 'enhanced_adaptive_atsp',
                                 'enhanced_chaos_atsp', 'all_tsp', 'all_atsp', 'all'], # 新增all_tsp和all_atsp选项
                        help="Type of PSO algorithm to run.")
    parser.add_argument('--max_iterations', type=int, default=200, help="Maximum number of iterations.")
    parser.add_argument('--n_particles', type=int, default=30, help="Number of particles.")

    parser.add_argument('--show_plots', action=argparse.BooleanOptionalAction, default=True, help="Show visualization plots.")
    parser.add_argument('--save_plots', action=argparse.BooleanOptionalAction, default=False, help="Save visualization plots to files.")
    parser.add_argument('--output_dir', type=str, default='pso_plots', help="Directory to save plots.")

    args = parser.parse_args()

    if args.save_plots and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    print("=== PSO-TSP Execution with CLI Parameters ===")
    print(f"Number of cities: {args.n_cities}")
    print(f"Seed: {args.seed if args.seed is not None else 'Not set (random)'}")
    print(f"PSO Type: {args.pso_type}")
    print(f"Max Iterations: {args.max_iterations}")
    print(f"Number of Particles: {args.n_particles}")
    print(f"Show Plots: {args.show_plots}")
    print(f"Save Plots: {args.save_plots}")
    if args.save_plots:
        print(f"Output Directory: {args.output_dir}")
    print("-" * 30)

    tsp_problem = TSP.create_random_instance(args.n_cities, seed=args.seed)
    atsp_problem = ATSP.create_random_instance(args.n_cities, seed=args.seed)

    pso_params = {
        'n_particles': args.n_particles,
        'max_iterations': args.max_iterations,
        'seed': args.seed
    }

    results: Dict[str, TSPResult] = {}
    plot_figs: Dict[str, plt.Figure] = {}

    pso_configs = {
        'base_tsp': {'class': BasePSO, 'problem': tsp_problem, 'enhancements': {}, 'name': 'Standard PSO (TSP)'},
        'enhanced_all_tsp': {'class': EnhancedPSO, 'problem': tsp_problem,
                             'enhancements': {'enable_local_search': True, 'enable_adaptive_params': True, 'enable_chaos': True},
                             'name': 'Enhanced PSO (All Features, TSP)'},
        'enhanced_ls_tsp': {'class': EnhancedPSO, 'problem': tsp_problem,
                            'enhancements': {'enable_local_search': True, 'enable_adaptive_params': False, 'enable_chaos': False},
                            'name': 'Enhanced PSO (Local Search, TSP)'},
        'enhanced_adaptive_tsp': {'class': EnhancedPSO, 'problem': tsp_problem,
                                  'enhancements': {'enable_local_search': False, 'enable_adaptive_params': True, 'enable_chaos': False},
                                  'name': 'Enhanced PSO (Adaptive Params, TSP)'},
        'enhanced_chaos_tsp': {'class': EnhancedPSO, 'problem': tsp_problem,
                               'enhancements': {'enable_local_search': False, 'enable_adaptive_params': False, 'enable_chaos': True},
                               'name': 'Enhanced PSO (Chaos, TSP)'},

        'base_atsp': {'class': BasePSO, 'problem': atsp_problem, 'enhancements': {}, 'name': 'Standard PSO (ATSP)'},
        'enhanced_all_atsp': {'class': EnhancedPSO, 'problem': atsp_problem,
                             'enhancements': {'enable_local_search': True, 'enable_adaptive_params': True, 'enable_chaos': True},
                             'name': 'Enhanced PSO (All Features, ATSP)'},
        'enhanced_ls_atsp': {'class': EnhancedPSO, 'problem': atsp_problem,
                            'enhancements': {'enable_local_search': True, 'enable_adaptive_params': False, 'enable_chaos': False},
                            'name': 'Enhanced PSO (Local Search, ATSP)'},
        'enhanced_adaptive_atsp': {'class': EnhancedPSO, 'problem': atsp_problem,
                                  'enhancements': {'enable_local_search': False, 'enable_adaptive_params': True, 'enable_chaos': False},
                                  'name': 'Enhanced PSO (Adaptive Params, ATSP)'},
        'enhanced_chaos_atsp': {'class': EnhancedPSO, 'problem': atsp_problem,
                               'enhancements': {'enable_local_search': False, 'enable_adaptive_params': False, 'enable_chaos': True},
                               'name': 'Enhanced PSO (Chaos, ATSP)'},
    }

    types_to_run = []
    if args.pso_type == 'all':
        types_to_run = list(pso_configs.keys())
    elif args.pso_type == 'all_tsp':
        types_to_run = [k for k in pso_configs if "TSP" in pso_configs[k]['name']]
    elif args.pso_type == 'all_atsp':
        types_to_run = [k for k in pso_configs if "ATSP" in pso_configs[k]['name']]
    elif args.pso_type in pso_configs:
        types_to_run = [args.pso_type]
    else:
        print(f"Error: Unknown PSO type '{args.pso_type}'. Exiting.")
        return

    for pso_key in types_to_run:
        config = pso_configs[pso_key]
        print(f"\nRunning {config['name']}...")
        result = run_specific_pso(config['class'], config['problem'], pso_params, config.get('enhancements', {}))
        results[config['name']] = result
        print(f"  Best Distance: {result.best_distance:.2f}")
        print(f"  Computation Time: {result.computation_time:.2f}s")
        print(f"  Iterations: {result.iterations}")
        path_preview = result.best_path[:10] if result.best_path and len(result.best_path) > 10 else result.best_path
        print(f"  Best Path (preview): {path_preview}{'...' if path_preview and len(result.best_path) > 10 else ''}")

        if result.problem_coords is not None:
            path_fig_title = f"Best Path for {config['name']}"
            path_fig = PSOVisualizer.plot_best_path(result, title_prefix=path_fig_title)
            if path_fig:
                 plot_figs[f"best_path_{pso_key.replace(' ', '_')}"] = path_fig # 使用更安全的文件名

    # --- 修改后的可视化生成部分 ---
    if not results:
        print("No results to visualize.")
    else:
        print("\nGenerating comparison visualizations...")

        # 筛选TSP和ATSP的结果
        results_tsp_only = {name: res for name, res in results.items() if "TSP)" in name}
        results_atsp_only = {name: res for name, res in results.items() if "ATSP)" in name}

        # 1. 仅 TSP 结果的比较图
        if results_tsp_only:
            print("\nGenerating comparison visualizations for TSP results only...")
            if len(results_tsp_only) > 0:
                conv_fig_tsp = PSOVisualizer.plot_convergence(results_tsp_only, title="PSO Convergence Comparison (TSP only)")
                plot_figs["convergence_comparison_tsp_only"] = conv_fig_tsp
            if len(results_tsp_only) > 1:
                perf_fig_tsp = PSOVisualizer.plot_performance_comparison(results_tsp_only)
                # 更新性能比较图的标题
                perf_fig_tsp.suptitle('PSO Performance Comparison (TSP only)', fontsize=16)
                plot_figs["performance_comparison_tsp_only"] = perf_fig_tsp
        else:
            print("No TSP results to generate TSP-only comparison plots.")

        # 2. 仅 ATSP 结果的比较图
        if results_atsp_only:
            print("\nGenerating comparison visualizations for ATSP results only...")
            if len(results_atsp_only) > 0:
                conv_fig_atsp = PSOVisualizer.plot_convergence(results_atsp_only, title="PSO Convergence Comparison (ATSP only)")
                plot_figs["convergence_comparison_atsp_only"] = conv_fig_atsp
            if len(results_atsp_only) > 1:
                perf_fig_atsp = PSOVisualizer.plot_performance_comparison(results_atsp_only)
                # 更新性能比较图的标题
                perf_fig_atsp.suptitle('PSO Performance Comparison (ATSP only)', fontsize=16)
                plot_figs["performance_comparison_atsp_only"] = perf_fig_atsp
        else:
            print("No ATSP results to generate ATSP-only comparison plots.")

        # 3. 所有结果的比较图 (如果同时有TSP和ATSP结果，或者用户选择了运行多种类型的算法)
        # 仅当结果集中同时包含TSP和ATSP，或者运行了多种基础/增强算法时，这个“全部”才有意义
        # 或者，如果用户明确运行了 'all' 类型的实验，也应显示
        should_plot_all_comparison = len(results) > 1 and (len(results_tsp_only) > 0 and len(results_atsp_only) > 0 or len(results) > len(results_tsp_only) or len(results) > len(results_atsp_only))
        if args.pso_type == 'all': # 如果用户指定了 'all'，则总是绘制总的比较图
            should_plot_all_comparison = True

        if should_plot_all_comparison and len(results) > 0 : # 确保至少有一个结果
             print("\nGenerating comparison visualizations for ALL combined results...")
             conv_fig_all = PSOVisualizer.plot_convergence(results, title="PSO Algorithm Convergence Comparison (All Results)")
             plot_figs["convergence_comparison_all_results"] = conv_fig_all
             if len(results) > 1:
                perf_fig_all = PSOVisualizer.plot_performance_comparison(results)
                # 更新性能比较图的标题
                perf_fig_all.suptitle('PSO Performance Comparison (All Results)', fontsize=16)
                plot_figs["performance_comparison_all_results"] = perf_fig_all
        elif len(results_tsp_only) <=1 and len(results_atsp_only) <=1 and len(results) > 0 : # 如果只有单个TSP或ATSP实验，则总图无意义
            pass #单个实验的结果已经在TSP-only或ATSP-only中绘制（如果适用）
        elif len(results) > 0 and not results_tsp_only and not results_atsp_only:
            # 处理一些边缘情况，比如运行的算法名称不包含(TSP)或(ATSP)
             print("\nGenerating comparison visualizations for ALL collected results (generic)...")
             conv_fig_all = PSOVisualizer.plot_convergence(results, title="PSO Algorithm Convergence Comparison (Collected)")
             plot_figs["convergence_comparison_collected"] = conv_fig_all
             if len(results) > 1:
                perf_fig_all = PSOVisualizer.plot_performance_comparison(results)
                perf_fig_all.suptitle('PSO Performance Comparison (Collected)', fontsize=16)
                plot_figs["performance_comparison_collected"] = perf_fig_all


    if plot_figs:
        for fig_name, fig_object in plot_figs.items():
            if args.save_plots:
                save_path = os.path.join(args.output_dir, f"{fig_name.replace(' ', '_').replace('(', '').replace(')', '')}.png") # 文件名净化
                try:
                    fig_object.savefig(save_path)
                    print(f"Saved plot: {save_path}")
                except Exception as e:
                    print(f"Error saving plot {save_path}: {e}")

            if args.show_plots:
                plt.figure(fig_object.number)
                plt.show()

            plt.close(fig_object)

    print("\nExecution finished!")


if __name__ == "__main__":
    main()

