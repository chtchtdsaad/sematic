import numpy as np
import torch
import matplotlib.pyplot as plt
from core import RSEEnv, D3QNAgent, CONFIG
from attack import OptimalAdversary


class D3QNTrainer:
    """
    D3QN训练器工具类
    封装训练过程，提供训练、测试、可视化等功能
    """
    
    def __init__(self, config=None):
        """
        初始化训练器
        Args:
            config: 配置字典，如果为None则使用默认CONFIG
        """
        self.config = config if config is not None else CONFIG
        self.env = RSEEnv()

        self.agent = D3QNAgent()
        self.attacker = OptimalAdversary(agent=self.agent, 
                                            epsilon=self.config['attack_PGD_epsilon'], 
                                            steps=self.config['attack_PGD_steps'], 
                                            alpha=self.config['attack_PGD_alpha'])
        self.training_history = {
            'avg_costs': [],
            'episode_rewards': [],
            'epsilon_history': []
        }
        
    def train(self, num_episodes=None, steps_per_episode=None, verbose=True):
        """
        执行训练过程
        Args:
            num_episodes: 训练轮数，默认使用配置中的值
            steps_per_episode: 每轮步数，默认使用配置中的值
            verbose: 是否打印训练信息
        Returns:
            training_history: 包含训练过程数据的字典
        """
        num_episodes = num_episodes or self.config['num_episodes']
        steps_per_episode = steps_per_episode or self.config['steps_per_episode']
        
        if verbose:
            print("=" * 50)
            print("开始训练 D3QN...")
            print(f"训练轮数: {num_episodes}")
            print(f"每轮步数: {steps_per_episode}")
            print("=" * 50)
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_cost = 0
            episode_reward = 0
            
            for t in range(steps_per_episode):
                # 选择动作
                action = self.agent.select_action(state)
                
                # 执行动作
                next_state, reward, terminated, truncated, info = self.env.step(action)
                
                # 存储经验
                self.agent.buffer.push(state, action, reward, next_state, False)
                
                # 更新网络
                self.agent.update()
                
                # 更新状态
                state = next_state
                episode_cost += info['cost']
                episode_reward += reward
                
                # 定期更新 Target Net
                if t % self.config['target_update_freq'] == 0:
                    self.agent.update_target_net()
            
            # 记录本轮数据
            avg_cost = episode_cost / steps_per_episode
            self.training_history['avg_costs'].append(avg_cost)
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['epsilon_history'].append(self.agent.epsilon)
            
            # 衰减 Epsilon
            self.agent.epsilon = max(
                self.config['epsilon_end'], 
                self.agent.epsilon * self.config['epsilon_decay']
            )
            
            # 打印进度
            if verbose and (episode + 1) % 30 == 0:
                print(f"Episode {episode+1}/{num_episodes} | "
                      f"Avg Cost: {avg_cost:.2f} | "
                      f"Total Reward: {episode_reward:.2f} | "
                      f"Epsilon: {self.agent.epsilon:.3f}")
        
        if verbose:
            print("=" * 50)
            print("训练结束！")
            print("=" * 50)
        
        return self.training_history
    
    def test(self, num_episodes=10, render=False, verbose=True):
        """
        测试训练好的智能体
        Args:
            num_episodes: 测试轮数
            render: 是否渲染（暂不支持）
            verbose: 是否打印详细信息
        Returns:
            test_results: 包含测试结果的字典
        """
        if verbose:
            print("\n" + "=" * 50)
            print("开始测试训练好的智能体...")
            print("=" * 50)
        
        test_results = {
            'episode_costs': [],
            'episode_rewards': [],
            'action_distribution': [0, 0, 0, 0]  # 4个动作的使用次数
        }
        
        # 保存原始epsilon，测试时使用贪婪策略
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_cost = 0
            episode_reward = 0
            episode_actions = []
            
            for t in range(self.config['steps_per_episode']):
                # 使用贪婪策略选择动作
                action = self.agent.select_action(state)
                episode_actions.append(action)
                test_results['action_distribution'][action] += 1
                
                # 执行动作
                next_state, reward, terminated, truncated, info = self.env.step(action)
                
                state = next_state
                episode_cost += info['cost']
                episode_reward += reward
            
            avg_cost = episode_cost / self.config['steps_per_episode']
            test_results['episode_costs'].append(avg_cost)
            test_results['episode_rewards'].append(episode_reward)
            
            if verbose and (episode+1)%20 == 0 :
                print(f"Test Episode {episode+1}/{num_episodes} | "
                      f"Avg Cost: {avg_cost:.2f} | "
                      f"Total Reward: {episode_reward:.2f}")
        
        # 恢复原始epsilon
        self.agent.epsilon = original_epsilon
        
        # 计算统计信息
        test_results['mean_cost'] = np.mean(test_results['episode_costs'])
        test_results['std_cost'] = np.std(test_results['episode_costs'])
        test_results['mean_reward'] = np.mean(test_results['episode_rewards'])
        test_results['std_reward'] = np.std(test_results['episode_rewards'])
        
        if verbose:
            
            print("=" * 50)
            print("测试结果统计:")
            print(f"平均Cost: {test_results['mean_cost']:.2f} ± {test_results['std_cost']:.2f}")
            print(f"平均Reward: {test_results['mean_reward']:.2f} ± {test_results['std_reward']:.2f}")
            print(f"动作分布: {test_results['action_distribution']}")
            total_actions = sum(test_results['action_distribution'])
            action_probs = [count/total_actions for count in test_results['action_distribution']]
            print(f"动作概率: [Action 0: {action_probs[0]:.2%}, "
                  f"Action 1: {action_probs[1]:.2%}, "
                  f"Action 2: {action_probs[2]:.2%}, "
                  f"Action 3: {action_probs[3]:.2%}]")
            print("=" * 50)
        
        return test_results


    
    def test_attack(self, num_episodes=100, render=False, verbose=True, method='PGD'):
        """
        测试训练好的智能体
        Args:
            num_episodes: 测试轮数
            render: 是否渲染（暂不支持）
            verbose: 是否打印详细信息
        Returns:
            test_results: 包含测试结果的字典
        """
        if verbose:
            print("\n" + "=" * 50)
            print("开始测试攻击过的智能体...")
            print("=" * 50)
        
        test_results = {
            'episode_costs': [],
            'episode_rewards': [],
            'action_distribution': [0, 0, 0, 0]  # 4个动作的使用次数
        }
        
        # 保存原始epsilon，测试时使用贪婪策略
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()  #state 为np.array
            episode_cost = 0
            episode_reward = 0
            episode_actions = []
            
            for t in range(self.config['steps_per_episode']):
                # 使用贪婪策略选择动作
                if method == 'random':
                    random_attack = np.random.randint(-self.config['attack_PGD_epsilon'], self.config['attack_PGD_epsilon'], size=state.shape)
                    random_attack_state = np.clip(state + random_attack, 0, 20) - state
                    action = self.agent.select_action(state + random_attack_state)
                else:
                    perturbed_state = self.attacker.generate_attack(state)
                    action = self.agent.select_action(perturbed_state)
                episode_actions.append(action)
                test_results['action_distribution'][action] += 1
                
                # 执行动作
                next_state, reward, terminated, truncated, info = self.env.step(action)
                
                state = next_state


                episode_cost += info['cost']
                episode_reward += reward
            
            avg_cost = episode_cost / self.config['steps_per_episode']
            test_results['episode_costs'].append(avg_cost)
            test_results['episode_rewards'].append(episode_reward)
            
            if verbose and (episode+1)%20 == 0 :
                print(f"Test Episode {episode+1}/{num_episodes} | "
                      f"Avg Cost: {avg_cost:.2f} | "
                      f"Total Reward: {episode_reward:.2f}")
        
        # 恢复原始epsilon
        self.agent.epsilon = original_epsilon
        
        # 计算统计信息
        test_results['mean_cost'] = np.mean(test_results['episode_costs'])
        test_results['std_cost'] = np.std(test_results['episode_costs'])
        test_results['mean_reward'] = np.mean(test_results['episode_rewards'])
        test_results['std_reward'] = np.std(test_results['episode_rewards'])
        
        if verbose:
            
            print("=" * 50)
            print("测试结果统计:")
            print(f"平均Cost: {test_results['mean_cost']:.2f} ± {test_results['std_cost']:.2f}")
            print(f"平均Reward: {test_results['mean_reward']:.2f} ± {test_results['std_reward']:.2f}")
            print(f"动作分布: {test_results['action_distribution']}")
            total_actions = sum(test_results['action_distribution'])
            action_probs = [count/total_actions for count in test_results['action_distribution']]
            print(f"动作概率: [Action 0: {action_probs[0]:.2%}, "
                  f"Action 1: {action_probs[1]:.2%}, "
                  f"Action 2: {action_probs[2]:.2%}, "
                  f"Action 3: {action_probs[3]:.2%}]")
            print("=" * 50)
        
        return test_results

    def test_random_strategy(self, num_episodes=100, verbose=True):
        """
        Random Strategy 测试: 每次以相等概率选择四个动作之一
        Args:
            num_episodes: 测试轮数
            verbose: 是否打印详细信息
        Returns:
            test_results: 包含测试结果的字典（与test()格式一致）
        """
        import random
        
        if verbose:
            print("\n" + "=" * 50)
            print("开始测试 Random Strategy...")
            print("=" * 50)
        
        test_results = {
            'episode_costs': [],
            'episode_rewards': [],
            'action_distribution': [0, 0, 0, 0]
        }
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_cost = 0
            episode_reward = 0
            
            for t in range(self.config['steps_per_episode']):
                # Random Strategy: 以相等概率随机选择动作
                action = random.randint(0, 3)
                test_results['action_distribution'][action] += 1
                
                # 执行动作
                next_state, reward, terminated, truncated, info = self.env.step(action)
                
                state = next_state
                episode_cost += info['cost']
                episode_reward += reward
            
            avg_cost = episode_cost / self.config['steps_per_episode']
            test_results['episode_costs'].append(avg_cost)
            test_results['episode_rewards'].append(episode_reward)
            
            if verbose and (episode + 1) % 20 == 0:
                print(f"Test Episode {episode+1}/{num_episodes} | "
                      f"Avg Cost: {avg_cost:.2f} | "
                      f"Total Reward: {episode_reward:.2f}")
        
        # 计算统计信息
        test_results['mean_cost'] = np.mean(test_results['episode_costs'])
        test_results['std_cost'] = np.std(test_results['episode_costs'])
        test_results['mean_reward'] = np.mean(test_results['episode_rewards'])
        test_results['std_reward'] = np.std(test_results['episode_rewards'])
        
        if verbose:
            print("=" * 50)
            print("Random Strategy 测试结果统计:")
            print(f"平均Cost: {test_results['mean_cost']:.2f} ± {test_results['std_cost']:.2f}")
            print(f"平均Reward: {test_results['mean_reward']:.2f} ± {test_results['std_reward']:.2f}")
            print(f"动作分布: {test_results['action_distribution']}")
            total_actions = sum(test_results['action_distribution'])
            action_probs = [count/total_actions for count in test_results['action_distribution']]
            print(f"动作概率: [Action 0: {action_probs[0]:.2%}, "
                  f"Action 1: {action_probs[1]:.2%}, "
                  f"Action 2: {action_probs[2]:.2%}, "
                  f"Action 3: {action_probs[3]:.2%}]")
            print("=" * 50)
        
        return test_results

    def test_round_robin_strategy(self, num_episodes=100, verbose=True):
        """
        Round-Robin Strategy 测试: 周期性地采用动作 a0, a1, a2, a3
        Args:
            num_episodes: 测试轮数
            verbose: 是否打印详细信息
        Returns:
            test_results: 包含测试结果的字典（与test()格式一致）
        """
        if verbose:
            print("\n" + "=" * 50)
            print("开始测试 Round-Robin Strategy...")
            print("=" * 50)
        
        test_results = {
            'episode_costs': [],
            'episode_rewards': [],
            'action_distribution': [0, 0, 0, 0]
        }
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_cost = 0
            episode_reward = 0
            
            for t in range(self.config['steps_per_episode']):
                # Round-Robin Strategy: 周期性选择动作 0, 1, 2, 3, 0, 1, 2, 3, ...
                action = t % 4
                test_results['action_distribution'][action] += 1
                
                # 执行动作
                next_state, reward, terminated, truncated, info = self.env.step(action)
                
                state = next_state
                episode_cost += info['cost']
                episode_reward += reward
            
            avg_cost = episode_cost / self.config['steps_per_episode']
            test_results['episode_costs'].append(avg_cost)
            test_results['episode_rewards'].append(episode_reward)
            
            if verbose and (episode + 1) % 20 == 0:
                print(f"Test Episode {episode+1}/{num_episodes} | "
                      f"Avg Cost: {avg_cost:.2f} | "
                      f"Total Reward: {episode_reward:.2f}")
        
        # 计算统计信息
        test_results['mean_cost'] = np.mean(test_results['episode_costs'])
        test_results['std_cost'] = np.std(test_results['episode_costs'])
        test_results['mean_reward'] = np.mean(test_results['episode_rewards'])
        test_results['std_reward'] = np.std(test_results['episode_rewards'])
        
        if verbose:
            print("=" * 50)
            print("Round-Robin Strategy 测试结果统计:")
            print(f"平均Cost: {test_results['mean_cost']:.2f} ± {test_results['std_cost']:.2f}")
            print(f"平均Reward: {test_results['mean_reward']:.2f} ± {test_results['std_reward']:.2f}")
            print(f"动作分布: {test_results['action_distribution']}")
            total_actions = sum(test_results['action_distribution'])
            action_probs = [count/total_actions for count in test_results['action_distribution']]
            print(f"动作概率: [Action 0: {action_probs[0]:.2%}, "
                  f"Action 1: {action_probs[1]:.2%}, "
                  f"Action 2: {action_probs[2]:.2%}, "
                  f"Action 3: {action_probs[3]:.2%}]")
            print("=" * 50)
        
        return test_results

    def test_min_energy_strategy(self, num_episodes=100, verbose=True):
        """
        Minimum Energy Strategy 测试: 始终采用动作 a2 或 a3（LF模式，最小化能量消耗）
        根据论文设定，a2和a3是LF模式（energy_lf=2.0），而a0和a1是HF模式（energy_hf=7.0）
        这里采用 a2 作为最小能量策略（S1->Ch1）
        Args:
            num_episodes: 测试轮数
            verbose: 是否打印详细信息
        Returns:
            test_results: 包含测试结果的字典（与test()格式一致）
        """
        if verbose:
            print("\n" + "=" * 50)
            print("开始测试 Minimum Energy Strategy...")
            print("=" * 50)
        
        test_results = {
            'episode_costs': [],
            'episode_rewards': [],
            'action_distribution': [0, 0, 0, 0]
        }
        

        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_cost = 0
            episode_reward = 0
            
            for t in range(self.config['steps_per_episode']):

            # 最小能量动作: a2,a3 (LF: S1->Ch1, energy=2.0)
                min_energy_action = 2 if t % 2 == 0 else 3
                action = min_energy_action
                test_results['action_distribution'][action] += 1
                
                # 执行动作
                next_state, reward, terminated, truncated, info = self.env.step(action)
                
                state = next_state

                
                episode_cost += info['cost']
                episode_reward += reward
                if (t+1) % 20 == 0:
                    print(f"cost: {info['cost']}")
            
            avg_cost = episode_cost / self.config['steps_per_episode']
            test_results['episode_costs'].append(avg_cost)
            test_results['episode_rewards'].append(episode_reward)
            
            if verbose and (episode + 1) % 20 == 0:
                print(f"Test Episode {episode+1}/{num_episodes} | "
                      f"Avg Cost: {avg_cost:.2f} | "
                      f"Total Reward: {episode_reward:.2f}")
        
        # 计算统计信息
        test_results['mean_cost'] = np.mean(test_results['episode_costs'])
        test_results['std_cost'] = np.std(test_results['episode_costs'])
        test_results['mean_reward'] = np.mean(test_results['episode_rewards'])
        test_results['std_reward'] = np.std(test_results['episode_rewards'])
        
        if verbose:
            print("=" * 50)
            print("Minimum Energy Strategy 测试结果统计:")
            print(f"平均Cost: {test_results['mean_cost']:.2f} ± {test_results['std_cost']:.2f}")
            print(f"平均Reward: {test_results['mean_reward']:.2f} ± {test_results['std_reward']:.2f}")
            print(f"动作分布: {test_results['action_distribution']}")
            total_actions = sum(test_results['action_distribution'])
            action_probs = [count/total_actions for count in test_results['action_distribution']]
            print(f"动作概率: [Action 0: {action_probs[0]:.2%}, "
                  f"Action 1: {action_probs[1]:.2%}, "
                  f"Action 2: {action_probs[2]:.2%}, "
                  f"Action 3: {action_probs[3]:.2%}]")
            print("=" * 50)
        
        return test_results
    
    def plot_training_curves(self, save_path=None):
        """
        绘制训练曲线
        Args:
            save_path: 保存路径，如果为None则直接显示
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 平均Cost曲线
        axes[0, 0].plot(self.training_history['avg_costs'], label='Average Cost', color='blue')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Average Cost')
        axes[0, 0].set_title('Average Cost over Episodes')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # 2. Episode总Reward曲线
        axes[0, 1].plot(self.training_history['episode_rewards'], label='Episode Reward', color='green')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Total Reward')
        axes[0, 1].set_title('Episode Reward over Episodes')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # 3. Epsilon衰减曲线
        axes[1, 0].plot(self.training_history['epsilon_history'], label='Epsilon', color='red')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Epsilon')
        axes[1, 0].set_title('Epsilon Decay over Episodes')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # 4. Cost滑动平均 (平滑曲线)
        if len(self.training_history['avg_costs']) > 10:
            window_size = min(50, len(self.training_history['avg_costs']) // 10)
            smoothed_costs = np.convolve(
                self.training_history['avg_costs'], 
                np.ones(window_size)/window_size, 
                mode='valid'
            )
            axes[1, 1].plot(smoothed_costs, label=f'Smoothed Cost (window={window_size})', color='purple')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Smoothed Average Cost')
            axes[1, 1].set_title('Smoothed Cost Trend')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练曲线已保存到: {save_path}")
        else:
            plt.show()
    
    def save_model(self, model_path='d3qn_model.pth'):
        """
        保存模型
        Args:
            model_path: 模型保存路径
        """
        torch.save(self.agent.eval_net.state_dict(), model_path)
        print(f"模型已保存为: {model_path}")
    
    def load_model(self, model_path='d3qn_model.pth'):
        """
        加载模型
        Args:
            model_path: 模型路径
        """
        self.agent.eval_net.load_state_dict(torch.load(model_path))
        self.agent.target_net.load_state_dict(self.agent.eval_net.state_dict())
        print(f"模型已从 {model_path} 加载")
    
    def evaluate_policy(self, num_steps=1000, verbose=True):
        """
        评估当前策略的性能
        Args:
            num_steps: 评估步数
            verbose: 是否打印信息
        Returns:
            evaluation_metrics: 评估指标字典
        """
        if verbose:
            print("\n" + "=" * 50)
            print("开始策略评估...")
            print("=" * 50)
        
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0
        
        state, _ = self.env.reset()
        total_cost = 0
        total_reward = 0
        action_counts = [0, 0, 0, 0]
        
        for step in range(num_steps):
            action = self.agent.select_action(state)
            action_counts[action] += 1
            
            next_state, reward, terminated, truncated, info = self.env.step(action)
            
            total_cost += info['cost']
            total_reward += reward
            state = next_state
        
        self.agent.epsilon = original_epsilon
        
        evaluation_metrics = {
            'avg_cost_per_step': total_cost / num_steps,
            'avg_reward_per_step': total_reward / num_steps,
            'action_distribution': action_counts,
            'action_probabilities': [count/num_steps for count in action_counts]
        }
        
        if verbose:
            print(f"评估步数: {num_steps}")
            print(f"平均Step Cost: {evaluation_metrics['avg_cost_per_step']:.4f}")
            print(f"平均Step Reward: {evaluation_metrics['avg_reward_per_step']:.4f}")
            print(f"动作分布: {evaluation_metrics['action_distribution']}")
            print(f"动作概率: {[f'{p:.2%}' for p in evaluation_metrics['action_probabilities']]}")
            print("=" * 50)
        
        return evaluation_metrics


# ==========================================
# 测试函数示例
# ==========================================
def test_trainer_basic():
    """基础训练测试"""
    print("\n【测试1: 基础训练流程】")
    trainer = D3QNTrainer()
    
    # 训练较少的轮数用于快速测试
    trainer.train(num_episodes=50, verbose=True)
    
    # 测试
    test_results = trainer.test(num_episodes=5, verbose=True)
    
    # 保存模型
    trainer.save_model('test_model.pth')
    
    return trainer


def test_trainer_with_custom_config():
    """自定义配置测试"""
    print("\n【测试2: 自定义配置训练】")
    
    # 修改配置
    custom_config = CONFIG.copy()
    custom_config['num_episodes'] = 100
    custom_config['lr'] = 5e-4
    
    trainer = D3QNTrainer(config=custom_config)
    trainer.train(num_episodes=30, verbose=True)
    
    return trainer


def test_model_save_load():
    """模型保存和加载测试"""
    print("\n【测试3: 模型保存与加载】")
    
    # 训练并保存
    trainer1 = D3QNTrainer()
    trainer1.train(num_episodes=20, verbose=False)
    trainer1.save_model('temp_model.pth')
    results_before = trainer1.evaluate_policy(num_steps=100, verbose=False)
    
    # 加载模型
    trainer2 = D3QNTrainer()
    trainer2.load_model('temp_model.pth')
    results_after = trainer2.evaluate_policy(num_steps=100, verbose=False)
    
    print("保存前评估结果:", results_before['avg_cost_per_step'])
    print("加载后评估结果:", results_after['avg_cost_per_step'])
    print("差异:", abs(results_before['avg_cost_per_step'] - results_after['avg_cost_per_step']))
    
    return trainer2


def test_visualization():
    """可视化测试"""
    print("\n【测试4: 训练曲线可视化】")
    
    trainer = D3QNTrainer()
    trainer.train(num_episodes=100, verbose=False)
    
    # 绘制并保存训练曲线
    trainer.plot_training_curves(save_path='training_curves.png')
    
    return trainer


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始运行D3QNTrainer工具类测试套件")
    print("=" * 60)
    
    try:
        trainer1 = test_trainer_basic()
        print("✓ 测试1通过")
    except Exception as e:
        print(f"✗ 测试1失败: {e}")
    
    try:
        trainer2 = test_trainer_with_custom_config()
        print("✓ 测试2通过")
    except Exception as e:
        print(f"✗ 测试2失败: {e}")
    
    try:
        trainer3 = test_model_save_load()
        print("✓ 测试3通过")
    except Exception as e:
        print(f"✗ 测试3失败: {e}")
    
    try:
        trainer4 = test_visualization()
        print("✓ 测试4通过")
    except Exception as e:
        print(f"✗ 测试4失败: {e}")
    
    print("\n" + "=" * 60)
    print("测试套件运行完成")
    print("=" * 60)


# ==========================================
# 主函数：简单使用示例
# ==========================================
if __name__ == "__main__":
    # 方式1: 简单使用
    # print("【方式1: 简单使用】")
    # trainer = D3QNTrainer()
    # trainer.train(num_episodes=100)
    # trainer.test(num_episodes=10)
    # trainer.plot_training_curves()
    # trainer.save_model('d3qn_trained_model.pth')
    
    # 方式2: 运行完整测试套件
    run_all_tests()
