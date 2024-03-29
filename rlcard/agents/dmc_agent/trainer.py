import os
import pprint
import threading
import time
import timeit
from collections import deque

import torch
from torch import multiprocessing as mp
from torch import nn

from .file_writer import FileWriter
from .model import DMCModel, RuleModel, MultiModel
from .utils import act, create_buffers, create_optimizers, get_batch, log


def compute_loss(logits, targets):
    '''MSE compute loss'''
    loss = ((logits - targets)**2).mean()
    return loss

def learn(position,
          actor_models,
          agent,
          batch,
          optimizer,
          training_device,
          max_grad_norm,
          mean_episode_return_buf,
          lock):
    """Performs a learning (optimization) step."""
    device = torch.device('cuda:'+str(training_device))
    # 将 batch_size * unroll_length 个数据组装好
    obs_x = torch.flatten(batch['obs_x'].to(device), 0, 1).float()
    obs_z = torch.flatten(batch['obs_z'].to(device), 0, 1).float()
    obs_action = torch.flatten(batch['obs_action'].to(device), 0, 1).float()
    target = torch.flatten(batch['target'].to(device), 0, 1)
    # 计算本 batch 中 payoffs 的平均值
    episode_returns = batch['episode_return'][batch['done']]
    mean_episode_return_buf[position].append(torch.mean(episode_returns).to(device))

    with lock:
        values = agent.forward(obs_x, obs_z, obs_action)
        loss = compute_loss(values, target)
        stats = {
            'mean_episode_return_'+str(position): torch.mean(torch.stack([_r for _r in mean_episode_return_buf[position]])).item(),
            'loss_'+str(position): loss.item(),
        }

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)  # type: ignore
        optimizer.step()

        for actor_model in actor_models:
            if actor_model.get_agent(position).use_net:
                actor_model.get_agent(position).load_state_dict(agent.state_dict())
        return stats


class DMCTrainer:
    def __init__(self,
                 env,
                 load_model=False,
                 save_interval=30,
                 num_actor_devices=1,
                 num_actors = 5,
                 training_device=0,
                 savedir='experiments/dmc_result',
                 modeldir='experiments/dmc_result/model.tar',
                 total_frames=10000000000,
                 num_eval_games=10000,
                 exp_epsilon=0.01,
                 batch_size=32,
                 unroll_length=100,
                 num_buffers=50,
                 num_threads=4,
                 max_grad_norm=40,
                 learning_rate=0.0001,
                 alpha=0.99,
                 momentum=0,
                 epsilon=0.00001):
        '''
        Deep Monte-Carlo

        Args:
            env: RLCard environment
            load_model (boolean): Whether loading an existing model
            save_interval (int): Time interval (in minutes) at which to save the model
            num_actor_devices (int): The number devices used for simulation
            num_actors (int): Number of actors for each simulation device
            training_device (int): The index of the GPU used for training models
            savedir (string): Root dir where experiment data will be saved
            modeldir (string): Root dir where experiment data will be reloaded
            total_frames (int): Total environment frames to train for
            exp_epsilon (float): The prbability for exploration
            batch_size (int): Learner batch size
            unroll_length (int): The unroll length (time dimension)
            num_buffers (int): Number of shared-memory buffers
            num_threads (int): Number learner threads
            max_grad_norm (int): Max norm of gradients
            learning_rate (float): Learning rate
            alpha (float): RMSProp smoothing constant
            momentum (float): RMSProp momentum
            epsilon (float): RMSProp epsilon
        '''
        self.env = env # 已创建好的 Env

        self.plogger = FileWriter(
            rootdir=savedir,
        ) # 将 logger 存入 savedir 下

        

        self.T = unroll_length
        self.B = batch_size

        self.load_model = load_model # 是否加载已有模型
        self.savedir = savedir # 存储实验数据的根目录
        self.checkpointpath = modeldir # 模型文件所在处
        self.save_interval = save_interval # 间隔多少 minute 存储一下模型
        self.num_actor_devices = num_actor_devices # 使用模拟器的设备数
        self.num_actors = num_actors # 每个模拟器上的 actor 数
        self.training_device = training_device # GPU 上训练模型的索引号
        self.total_frames = total_frames # 全部环境训练帧数
        self.num_eval_games = num_eval_games # 每次断点评估游戏 reward 的局数
        self.exp_epsilon = exp_epsilon # 𝛆 探索的概率
        self.num_buffers = num_buffers # 学习者的批大小
        self.num_threads = num_threads # 学习者的线程数
        self.max_grad_norm = max_grad_norm # 最大正则梯度
        self.learning_rate = learning_rate # 学习率
        self.alpha = alpha # RMSProp 平滑连续率
        self.momentum = momentum # RMSProp 动力值
        self.epsilon = epsilon # RMSProp 𝛆

        self.action_shape = self.env.action_shape
        if self.action_shape[0] == None:  # One-hot encoding
            self.action_shape = [[self.env.num_actions] for _ in range(self.env.num_players)]

        self.mean_episode_return_buf = [deque(maxlen=100) for _ in range(self.env.num_players)]

    def start(self):        
        # Initialize actor models
        # models = []
        # for device in range(self.num_actor_devices):
        #     model = DMCModel(self.env.state_shape,
        #                      self.action_shape,
        #                      exp_epsilon=self.exp_epsilon,
        #                      device=device) # 创建 num_players 个 DMC Agent 合并为一个 DMC Model
        #     model.share_memory() # 分别对 num_players 个 DMC Agent 的网络部分进行共享内存
        #     model.eval() # 告诉网络，这个阶段是用来测试的，于是模型的参数在该阶段不进行更新
        #     models.append(model) # 对每一个设备都初始化一个 DMC Model
            
        models = []
        for device in range(self.num_actor_devices):
            # 前期向模型学习
            # model = RuleModel(self.env.num_players)
            # models.append(model) # 对每一个设备都初始化一个 DMC Model
            
            # 中期和模型学习
            # model = MultiModel(self.env.state_shape,
            #                    self.action_shape,
            #                    exp_epsilon=self.exp_epsilon,
            #                    device=device)
            # model.share_memory()
            # model.eval()
            # models.append(model) # 对每一个设备都初始化一个 DMC Model

            # 后期自我博弈
            model = DMCModel(self.env.state_shape,
                             self.action_shape,
                             exp_epsilon=self.exp_epsilon,
                             device=device) # 创建 num_players 个 DMC Agent 合并为一个 DMC Model
            model.share_memory() # 分别对 num_players 个 DMC Agent 的网络部分进行共享内存
            model.eval() # 告诉网络，这个阶段是用来测试的，于是模型的参数在该阶段不进行更新
            models.append(model) # 对每一个设备都初始化一个 DMC Model

        # Initialize buffers
        buffers = create_buffers(self.T,
                                 self.num_buffers,
                                 self.env.state_shape,
                                 self.action_shape)

        # Initialize queues
        actor_processes = []
        ctx = mp.get_context('spawn')
        free_queue = []
        full_queue = []
        for device in range(self.num_actor_devices):
            _free_queue = [ctx.SimpleQueue() for _ in range(self.env.num_players)]
            _full_queue = [ctx.SimpleQueue() for _ in range(self.env.num_players)]
            free_queue.append(_free_queue)
            full_queue.append(_full_queue)

        # Learner model for training
        learner_model = DMCModel(self.env.state_shape,
                                 self.action_shape,
                                 device=self.training_device)

        # Create optimizers
        optimizers = create_optimizers(self.env.num_players,
                                       self.learning_rate,
                                       self.momentum,
                                       self.epsilon,
                                       self.alpha,
                                       learner_model)

        # Stat Keys
        stat_keys = []
        for p in range(self.env.num_players):
            stat_keys.append('mean_episode_return_'+str(p))
            stat_keys.append('loss_'+str(p))
        frames, stats = 0, {k: 0 for k in stat_keys}

        # Load models if any
        # if os.path.exists(self.checkpointpath):
        if self.load_model and os.path.exists(self.checkpointpath):
            checkpoint_states = torch.load(
                    self.checkpointpath, map_location="cuda:"+str(self.training_device)
            )
            for p in range(self.env.num_players):
                learner_model.get_agent(p).load_state_dict(checkpoint_states["model_state_dict"][p])
                optimizers[p].load_state_dict(checkpoint_states["optimizer_state_dict"][p])
                for device in range(self.num_actor_devices):
                    if models[device].get_agent(p).use_net:
                        models[device].get_agent(p).load_state_dict(learner_model.get_agent(p).state_dict())
            stats = checkpoint_states["stats"]
            frames = checkpoint_states["frames"]
            log.info(f"Resuming preempted job, current stats:\n{stats}")


        # Starting actor processes
        for device in range(self.num_actor_devices):
            num_actors = self.num_actors
            for i in range(self.num_actors):
                actor = ctx.Process(
                    target=act,
                    args=(i, device, self.T, free_queue[device], full_queue[device], models[device], buffers[device], self.env))
                actor.start()
                actor_processes.append(actor)

        def batch_and_learn(i, device, position, local_lock, position_lock, lock=threading.Lock()):
            """Thread target for the learning process."""
            nonlocal frames, stats
            while frames < self.total_frames:
                batch = get_batch(free_queue[device][position], full_queue[device][position], buffers[device][position], self.B, local_lock)
                _stats = learn(position, models, learner_model.get_agent(position), batch,
                    optimizers[position], self.training_device, self.max_grad_norm, self.mean_episode_return_buf, position_lock)

                with lock:
                    for k in _stats:
                        stats[k] = _stats[k]
                    to_log = dict(frames=frames)
                    to_log.update({k: stats[k] for k in stat_keys})
                    self.plogger.log(to_log)
                    frames += self.T * self.B

        for device in range(self.num_actor_devices):
            for m in range(self.num_buffers):
                for p in range(self.env.num_players):
                    free_queue[device][p].put(m)

        threads = []
        locks = [[threading.Lock() for _ in range(self.env.num_players)] for _ in range(self.num_actor_devices)]
        position_locks = [threading.Lock() for _ in range(self.env.num_players)]

        for device in range(self.num_actor_devices):
            for i in range(self.num_threads):
                for position in range(self.env.num_players):
                    thread = threading.Thread(
                        target=batch_and_learn, name='batch-and-learn-%d' % i, args=(i,device,position,locks[device][position],position_locks[position]))
                    thread.start()
                    threads.append(thread)

        def checkpoint(frames):
            _agents = learner_model.get_agents()
            model_tar_dir = os.path.expandvars(os.path.expanduser('%s/%s' % (self.savedir, 'model_'+str(frames)+'.tar')))
            torch.save({
                'model_state_dict': [_agent.state_dict() for _agent in _agents],
                'optimizer_state_dict': [optimizer.state_dict() for optimizer in optimizers],
                "stats": stats,
                'frames': frames,
            }, model_tar_dir)
            log.info('Saving checkpoint to %s', model_tar_dir)

            # Save the weights for evaluation purpose
            for position in range(self.env.num_players):
                model_weights_dir = os.path.expandvars(os.path.expanduser(
                    '%s/%s' % (self.savedir, str(position)+'_'+str(frames)+'.pth')))
                torch.save(learner_model.get_agent(position), model_weights_dir)

        timer = timeit.default_timer
        try:
            last_checkpoint_time = timer() - self.save_interval * 60
            while frames < self.total_frames:
                start_frames = frames
                start_time = timer()
                time.sleep(5)

                if timer() - last_checkpoint_time > self.save_interval * 60:
                    checkpoint(frames)
                    last_checkpoint_time = timer()

                end_time = timer()
                fps = (frames - start_frames) / (end_time - start_time)
                log.info('After %i frames: @ %.1f fps Stats:\n%s',
                             frames,
                             fps,
                             pprint.pformat(stats))
        except KeyboardInterrupt:
            return
        else:
            for thread in threads:
                thread.join()
            log.info('Learning finished after %d frames.', frames)

        checkpoint(frames)
        self.plogger.close()
