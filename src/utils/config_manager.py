"""
配置管理器 - 统一管理系统配置
提供配置验证、环境变量支持和配置热重载功能
"""

import os
import yaml
import json
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """LLM配置类"""
    provider: str = "zhipu"  # zhipu, vllm, openai
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout: int = 30
    max_retries: int = 3
    semaphore: int = 200  # 新增：控制LLM请求的最大并发数
    
    def __post_init__(self):
        """配置验证"""
        if not self.api_key:
            raise ValueError("API密钥不能为空")
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("温度参数必须在0-2之间")
        if self.max_tokens <= 0:
            raise ValueError("最大token数必须大于0")
        if self.semaphore <= 0:
            raise ValueError("并发数必须大于0")

@dataclass
class EnvConfig:
    """环境配置类"""
    # 数据库配置
    db_enabled: bool = True
    db_type: str = "sqlite"
    db_path: str = "data/agentsociety.db"
    
    # AgentSociety数据目录
    home_dir: str = ".agentsociety-benchmark/agentsociety_data"
    
    # 日志配置
    logging_level: str = "INFO"
    logging_file: str = "logs/agentsociety.log"
    logging_max_size: str = "10MB"
    logging_backup_count: int = 5
    
    # 缓存配置
    cache_enabled: bool = True
    cache_ttl: int = 300
    cache_max_size: int = 1000
    
    # 性能监控
    monitoring_enabled: bool = True
    metrics_interval: int = 60
    health_check_interval: int = 30

@dataclass
class AgentConfig:
    """智能体配置类"""
    enhanced_mode: bool = True
    use_llm: bool = True
    risk_threshold: float = 0.5
    evacuation_threshold: float = 0.6
    mobility_threshold: float = 0.4
    social_influence_weight: float = 0.3
    
    # 新增：决策缓存配置
    decision_cache_enabled: bool = True
    decision_cache_ttl: int = 300
    
    # 新增：并行处理配置
    parallel_processing_enabled: bool = True
    max_workers: int = 4
    
    def __post_init__(self):
        """配置验证"""
        for threshold in [self.risk_threshold, self.evacuation_threshold, self.mobility_threshold]:
            if threshold < 0 or threshold > 1:
                raise ValueError("阈值必须在0-1之间")
        if self.max_workers <= 0:
            raise ValueError("最大工作线程数必须大于0")

@dataclass
class SystemConfig:
    """系统配置类"""
    log_level: str = "INFO"
    log_file: str = "logs/system.log"
    data_dir: str = "data"
    results_dir: str = "results"
    cache_enabled: bool = True
    cache_ttl: int = 300
    
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    env: EnvConfig = field(default_factory=EnvConfig)  # 新增环境配置

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: str = "config"):
        """
        初始化配置管理器
        
        Args:
            config_dir: 配置文件目录
        """
        self.config_dir = Path(config_dir)
        self.config_cache = {}
        self.last_modified = {}
        self._ensure_config_dir()
    
    def _ensure_config_dir(self):
        """确保配置目录存在"""
        self.config_dir.mkdir(exist_ok=True)
    
    def load_official_config(self, config_file: str = "config.yml") -> SystemConfig:
        """
        加载官方格式的配置文件
        
        Args:
            config_file: 配置文件名
            
        Returns:
            SystemConfig: 系统配置对象
        """
        config_path = self.config_dir / config_file
        
        if not config_path.exists():
            logger.warning(f"官方配置文件不存在: {config_path}")
            return self.load_config()  # 回退到标准配置
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # 环境变量替换
            config_data = self._replace_env_vars(config_data)
            
            # 解析官方格式配置
            return self._parse_official_config(config_data)
            
        except Exception as e:
            logger.error(f"官方配置加载失败: {e}")
            return self.load_config()  # 回退到标准配置
    
    def _parse_official_config(self, config_data: Dict[str, Any]) -> SystemConfig:
        """解析官方格式的配置数据"""
        try:
            # 解析LLM配置 - 官方格式是列表
            llm_configs = config_data.get('llm', [])
            llm_config = None
            
            # 优先使用智谱配置
            for llm_item in llm_configs:
                if llm_item.get('provider') == 'zhipu':
                    llm_config = LLMConfig(
                        provider="zhipu",
                        api_key=llm_item.get('api_key', ''),
                        base_url=llm_item.get('base_url', 'https://open.bigmodel.cn/api/paas/v4/'),
                        model=llm_item.get('model', 'glm-4-plus'),
                        temperature=llm_item.get('temperature', 0.7),
                        max_tokens=llm_item.get('max_tokens', 1024),
                        timeout=llm_item.get('timeout', 30),
                        max_retries=llm_item.get('max_retries', 3),
                        semaphore=llm_item.get('semaphore', 200)
                    )
                    break
            
            # 如果没有智谱配置，使用vLLM配置
            if not llm_config:
                for llm_item in llm_configs:
                    if llm_item.get('provider') == 'vllm':
                        llm_config = LLMConfig(
                            provider="vllm",
                            api_key=llm_item.get('api_key', 'default-key'),
                            base_url=llm_item.get('base_url', 'http://localhost:8000/v1'),
                            model=llm_item.get('model', 'Qwen/Qwen2.5-7B-Instruct'),
                            temperature=llm_item.get('temperature', 0.7),
                            max_tokens=llm_item.get('max_tokens', 1024),
                            timeout=llm_item.get('timeout', 30),
                            max_retries=llm_item.get('max_retries', 3),
                            semaphore=llm_item.get('semaphore', 100)
                        )
                        break
            
            # 如果都没有，使用默认配置
            if not llm_config:
                llm_config = LLMConfig()
            
            # 解析环境配置
            env_data = config_data.get('env', {})
            db_data = env_data.get('db', {})
            logging_data = env_data.get('logging', {})
            cache_data = env_data.get('cache', {})
            monitoring_data = env_data.get('monitoring', {})
            
            env_config = EnvConfig(
                db_enabled=db_data.get('enabled', True),
                db_type=db_data.get('type', 'sqlite'),
                db_path=db_data.get('path', 'data/agentsociety.db'),
                home_dir=env_data.get('home_dir', '.agentsociety-benchmark/agentsociety_data'),
                logging_level=logging_data.get('level', 'INFO'),
                logging_file=logging_data.get('file', 'logs/agentsociety.log'),
                logging_max_size=logging_data.get('max_size', '10MB'),
                logging_backup_count=logging_data.get('backup_count', 5),
                cache_enabled=cache_data.get('enabled', True),
                cache_ttl=cache_data.get('ttl', 300),
                cache_max_size=cache_data.get('max_size', 1000),
                monitoring_enabled=monitoring_data.get('enabled', True),
                metrics_interval=monitoring_data.get('metrics_interval', 60),
                health_check_interval=monitoring_data.get('health_check_interval', 30)
            )
            
            # 解析智能体配置
            agent_data = config_data.get('agent', {})
            flood_data = agent_data.get('flood_mobility', {})
            thresholds_data = flood_data.get('thresholds', {})
            decision_cache_data = flood_data.get('decision_cache', {})
            parallel_data = flood_data.get('parallel_processing', {})
            
            agent_config = AgentConfig(
                enhanced_mode=flood_data.get('enhanced_mode', True),
                use_llm=flood_data.get('use_llm', True),
                risk_threshold=thresholds_data.get('risk_assessment', 0.5),
                evacuation_threshold=thresholds_data.get('evacuation_decision', 0.6),
                mobility_threshold=thresholds_data.get('mobility_planning', 0.4),
                social_influence_weight=flood_data.get('guanxi_influence_weight', 0.4),
                decision_cache_enabled=decision_cache_data.get('enabled', True),
                decision_cache_ttl=decision_cache_data.get('ttl', 300),
                parallel_processing_enabled=parallel_data.get('enabled', True),
                max_workers=parallel_data.get('max_workers', 4)
            )
            
            # 解析系统配置
            system_data = config_data.get('system', {})
            
            return SystemConfig(
                log_level=env_config.logging_level,
                log_file=env_config.logging_file,
                data_dir=system_data.get('data_dir', 'data'),
                results_dir=system_data.get('results_dir', 'results'),
                cache_enabled=env_config.cache_enabled,
                cache_ttl=env_config.cache_ttl,
                llm=llm_config,
                agent=agent_config,
                env=env_config
            )
            
        except Exception as e:
            logger.error(f"解析官方配置失败: {e}")
            raise

    def load_config(self, config_name: str = "system") -> SystemConfig:
        """
        加载配置文件
        
        Args:
            config_name: 配置文件名（不含扩展名）
            
        Returns:
            SystemConfig: 系统配置对象
        """
        config_file = self.config_dir / f"{config_name}.yaml"
        
        # 检查文件是否存在
        if not config_file.exists():
            logger.warning(f"配置文件不存在: {config_file}，使用默认配置")
            return self._create_default_config(config_file)
        
        # 检查是否需要重新加载
        if self._should_reload(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                
                # 环境变量替换
                config_data = self._replace_env_vars(config_data)
                
                # 创建配置对象
                config = self._create_config_object(config_data)
                
                # 缓存配置
                self.config_cache[config_name] = config
                self.last_modified[config_name] = config_file.stat().st_mtime
                
                logger.info(f"配置加载成功: {config_file}")
                return config
                
            except Exception as e:
                logger.error(f"配置加载失败: {e}")
                return self._create_default_config(config_file)
        
        return self.config_cache.get(config_name, self._create_default_config(config_file))
    
    def _should_reload(self, config_file: Path) -> bool:
        """检查是否需要重新加载配置"""
        config_name = config_file.stem
        if config_name not in self.config_cache:
            return True
        
        current_mtime = config_file.stat().st_mtime
        cached_mtime = self.last_modified.get(config_name, 0)
        
        return current_mtime > cached_mtime
    
    def _replace_env_vars(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """替换配置中的环境变量"""
        def replace_recursive(obj):
            if isinstance(obj, dict):
                return {k: replace_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_recursive(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
                env_var = obj[2:-1]
                default_value = ""
                if ":" in env_var:
                    env_var, default_value = env_var.split(":", 1)
                return os.getenv(env_var, default_value)
            else:
                return obj
        
        return replace_recursive(config_data)
    
    def _create_config_object(self, config_data: Dict[str, Any]) -> SystemConfig:
        """创建配置对象"""
        try:
            # LLM配置
            llm_data = config_data.get('llm', {})
            
            # 智谱配置优先
            if 'zhipu' in llm_data:
                zhipu_config = llm_data['zhipu']
                llm_config = LLMConfig(
                    provider="zhipu",
                    api_key=zhipu_config.get('api_key', ''),
                    base_url=zhipu_config.get('base_url', 'https://open.bigmodel.cn/api/paas/v4/'),
                    model=zhipu_config.get('model', 'glm-4-plus'),
                    temperature=llm_data.get('api', {}).get('temperature', 0.7),
                    max_tokens=llm_data.get('api', {}).get('max_tokens', 1024),
                    timeout=llm_data.get('api', {}).get('timeout', 30),
                    max_retries=llm_data.get('api', {}).get('max_retries', 3)
                )
            else:
                # vLLM配置
                vllm_config = llm_data.get('vllm_server', {})
                llm_config = LLMConfig(
                    provider="vllm",
                    api_key=vllm_config.get('api_key', 'default-key'),
                    base_url=vllm_config.get('base_url', 'http://localhost:8000/v1'),
                    model=llm_data.get('model', {}).get('name', 'Qwen/Qwen2.5-7B-Instruct'),
                    temperature=llm_data.get('api', {}).get('temperature', 0.7),
                    max_tokens=llm_data.get('api', {}).get('max_tokens', 1024),
                    timeout=llm_data.get('api', {}).get('timeout', 30),
                    max_retries=llm_data.get('api', {}).get('max_retries', 3)
                )
            
            # 智能体配置
            agent_data = config_data.get('agent', {})
            agent_config = AgentConfig(
                enhanced_mode=agent_data.get('enhanced_mode', True),
                use_llm=agent_data.get('use_llm', True),
                risk_threshold=agent_data.get('risk_threshold', 0.5),
                evacuation_threshold=agent_data.get('evacuation_threshold', 0.6),
                mobility_threshold=agent_data.get('mobility_threshold', 0.4),
                social_influence_weight=agent_data.get('social_influence_weight', 0.3)
            )
            
            # 系统配置
            system_data = config_data.get('system', {})
            return SystemConfig(
                log_level=system_data.get('log_level', 'INFO'),
                log_file=system_data.get('log_file', 'logs/system.log'),
                data_dir=system_data.get('data_dir', 'data'),
                results_dir=system_data.get('results_dir', 'results'),
                cache_enabled=system_data.get('cache_enabled', True),
                cache_ttl=system_data.get('cache_ttl', 300),
                llm=llm_config,
                agent=agent_config
            )
            
        except Exception as e:
            logger.error(f"创建配置对象失败: {e}")
            raise
    
    def _create_default_config(self, config_file: Path) -> SystemConfig:
        """创建默认配置"""
        default_config = SystemConfig()
        
        # 保存默认配置到文件
        try:
            config_dict = {
                'system': {
                    'log_level': default_config.log_level,
                    'log_file': default_config.log_file,
                    'data_dir': default_config.data_dir,
                    'results_dir': default_config.results_dir,
                    'cache_enabled': default_config.cache_enabled,
                    'cache_ttl': default_config.cache_ttl
                },
                'llm': {
                    'zhipu': {
                        'api_key': '${ZHIPU_API_KEY:your-api-key-here}',
                        'base_url': 'https://open.bigmodel.cn/api/paas/v4/',
                        'model': 'glm-4-plus'
                    },
                    'vllm_server': {
                        'host': '0.0.0.0',
                        'port': 8000,
                        'api_key': '${VLLM_API_KEY:default-key}',
                        'base_url': 'http://localhost:8000/v1'
                    },
                    'model': {
                        'name': 'Qwen/Qwen2.5-7B-Instruct'
                    },
                    'api': {
                        'timeout': 30,
                        'max_retries': 3,
                        'temperature': 0.7,
                        'max_tokens': 1024
                    }
                },
                'agent': {
                    'enhanced_mode': True,
                    'use_llm': True,
                    'risk_threshold': 0.5,
                    'evacuation_threshold': 0.6,
                    'mobility_threshold': 0.4,
                    'social_influence_weight': 0.3
                }
            }
            
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"创建默认配置文件: {config_file}")
            
        except Exception as e:
            logger.error(f"创建默认配置文件失败: {e}")
        
        return default_config
    
    def save_config(self, config: SystemConfig, config_name: str = "system"):
        """
        保存配置到文件
        
        Args:
            config: 配置对象
            config_name: 配置文件名
        """
        config_file = self.config_dir / f"{config_name}.yaml"
        
        try:
            config_dict = {
                'system': {
                    'log_level': config.log_level,
                    'log_file': config.log_file,
                    'data_dir': config.data_dir,
                    'results_dir': config.results_dir,
                    'cache_enabled': config.cache_enabled,
                    'cache_ttl': config.cache_ttl
                },
                'llm': {
                    config.llm.provider: {
                        'api_key': config.llm.api_key,
                        'base_url': config.llm.base_url,
                        'model': config.llm.model
                    },
                    'api': {
                        'timeout': config.llm.timeout,
                        'max_retries': config.llm.max_retries,
                        'temperature': config.llm.temperature,
                        'max_tokens': config.llm.max_tokens
                    }
                },
                'agent': {
                    'enhanced_mode': config.agent.enhanced_mode,
                    'use_llm': config.agent.use_llm,
                    'risk_threshold': config.agent.risk_threshold,
                    'evacuation_threshold': config.agent.evacuation_threshold,
                    'mobility_threshold': config.agent.mobility_threshold,
                    'social_influence_weight': config.agent.social_influence_weight
                }
            }
            
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"配置保存成功: {config_file}")
            
        except Exception as e:
            logger.error(f"配置保存失败: {e}")
            raise
    
    def validate_config(self, config: SystemConfig) -> bool:
        """
        验证配置有效性
        
        Args:
            config: 配置对象
            
        Returns:
            bool: 配置是否有效
        """
        try:
            # 验证LLM配置
            if not config.llm.api_key:
                logger.error("LLM API密钥不能为空")
                return False
            
            # 验证目录路径
            for dir_path in [config.data_dir, config.results_dir]:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
            
            # 验证日志文件路径
            log_dir = Path(config.log_file).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info("配置验证通过")
            return True
            
        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            return False

# 全局配置管理器实例
config_manager = ConfigManager()