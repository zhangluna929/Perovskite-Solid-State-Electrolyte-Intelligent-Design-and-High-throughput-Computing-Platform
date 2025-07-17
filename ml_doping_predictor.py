"""
基于机器学习的钙钛矿材料掺杂位点预测器
主要功能：
1. 结构特征提取
2. 元素属性特征化
3. 掺杂位点适合性预测
4. 模型训练与评估
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from pymatgen.core import Structure, Element, Site
from pymatgen.analysis.local_env import VoronoiNN, CrystalNN
from pymatgen.analysis.structure_analyzer import OxideType
from matminer.featurizers.site import CrystalNNFingerprint, VoronoiFingerprint
from matminer.featurizers.composition import ElementProperty

import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

class MLDopingSitePredictor:
    """机器学习掺杂位点预测器类"""
    
    def __init__(self, 
                 model_type: str = "random_forest",
                 model_path: Optional[str] = None,
                 data_path: Optional[str] = None):
        """
        初始化预测器
        
        Args:
            model_type: 模型类型 ("random_forest", "xgboost", "neural_net")
            model_path: 预训练模型路径
            data_path: 训练数据路径
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.site_featurizer = CrystalNNFingerprint.from_preset("ops")
        self.elem_featurizer = ElementProperty.from_preset("magpie")
        
        # 加载预训练模型
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            
        # 特征缓存
        self._site_features_cache = {}
        self._elem_features_cache = {}
        
    def extract_site_features(self, structure: Structure, site_idx: int) -> np.ndarray:
        """
        提取位点特征
        
        Args:
            structure: 晶体结构
            site_idx: 位点索引
            
        Returns:
            位点特征向量
        """
        cache_key = (structure, site_idx)
        if cache_key in self._site_features_cache:
            return self._site_features_cache[cache_key]
            
        features = []
        site = structure[site_idx]
        
        # 1. 配位环境特征
        try:
            voronoi = VoronoiNN()
            crystal = CrystalNN()
            
            # Voronoi配位数和多面体特征
            vor_info = voronoi.get_nn_info(structure, site_idx)
            vor_cn = len(vor_info)
            
            # 晶体场特征
            cn_info = crystal.get_nn_info(structure, site_idx)
            cn_fingerprint = self.site_featurizer.featurize(structure, site_idx)
            
            features.extend([
                vor_cn,  # Voronoi配位数
                np.mean([n["weight"] for n in vor_info]),  # 平均Voronoi权重
                len(cn_info),  # 晶体场配位数
                *cn_fingerprint  # 晶体场指纹
            ])
            
        except Exception as e:
            print(f"Warning: 提取位点特征时出错: {e}")
            features.extend([0] * (2 + len(self.site_featurizer.feature_labels())))
            
        # 2. 局部结构特征
        try:
            # 计算局部结构畸变
            local_structure = structure.get_sites_in_sphere(site.coords, 5.0)  # 5Å半径
            distances = [s.distance(site) for s in local_structure]
            angles = []
            for i, s1 in enumerate(local_structure[:-1]):
                for s2 in local_structure[i+1:]:
                    angles.append(site.get_angle(s1, s2))
            
            features.extend([
                np.mean(distances),  # 平均键长
                np.std(distances),   # 键长分布
                np.mean(angles),     # 平均键角
                np.std(angles)       # 键角分布
            ])
            
        except Exception as e:
            print(f"Warning: 计算局部结构特征时出错: {e}")
            features.extend([0] * 4)
            
        # 3. 电子结构特征
        try:
            site_element = Element(site.species_string)
            features.extend([
                site_element.X,           # 电负性
                site_element.atomic_radius,  # 原子半径
                site_element.ionic_radius,   # 离子半径
                site_element.atomic_mass,    # 原子质量
            ])
            
        except Exception as e:
            print(f"Warning: 提取电子结构特征时出错: {e}")
            features.extend([0] * 4)
            
        features = np.array(features, dtype=np.float32)
        self._site_features_cache[cache_key] = features
        return features
        
    def extract_doping_features(self, doping_element: str) -> np.ndarray:
        """
        提取掺杂元素特征
        
        Args:
            doping_element: 掺杂元素符号
            
        Returns:
            元素特征向量
        """
        if doping_element in self._elem_features_cache:
            return self._elem_features_cache[doping_element]
            
        try:
            element = Element(doping_element)
            features = self.elem_featurizer.featurize([doping_element])[0]
            self._elem_features_cache[doping_element] = features
            return features
            
        except Exception as e:
            print(f"Warning: 提取元素特征时出错: {e}")
            return np.zeros(len(self.elem_featurizer.feature_labels()))
            
    def combine_features(self, 
                        site_features: np.ndarray,
                        doping_features: np.ndarray) -> np.ndarray:
        """组合位点和掺杂元素特征"""
        return np.concatenate([site_features, doping_features])
        
    def train(self, 
              training_data: List[Dict],
              validation_split: float = 0.2) -> Dict:
        """
        训练模型
        
        Args:
            training_data: 训练数据列表
            validation_split: 验证集比例
            
        Returns:
            训练结果统计
        """
        # 准备数据
        X = []
        y = []
        
        for data in training_data:
            structure = data["structure"]
            doping_element = data["doping_element"]
            site_idx = data["site_idx"]
            is_suitable = data["is_suitable"]  # 1表示适合，0表示不适合
            
            # 提取特征
            site_features = self.extract_site_features(structure, site_idx)
            doping_features = self.extract_doping_features(doping_element)
            combined_features = self.combine_features(site_features, doping_features)
            
            X.append(combined_features)
            y.append(is_suitable)
            
        X = np.array(X)
        y = np.array(y)
        
        # 数据标准化
        X = self.scaler.fit_transform(X)
        
        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        # 选择并训练模型
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif self.model_type == "xgboost":
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == "neural_net":
            input_dim = X.shape[1]
            self.model = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            
        # 训练模型
        if isinstance(self.model, nn.Module):
            # PyTorch训练
            self.model.train()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.BCELoss()
            
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
            
            for epoch in range(100):
                optimizer.zero_grad()
                outputs = self.model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
                
        else:
            # scikit-learn训练
            self.model.fit(X_train, y_train)
            
        # 评估模型
        train_score = self.evaluate(X_train, y_train)
        val_score = self.evaluate(X_val, y_val)
        
        return {
            "train_accuracy": train_score["accuracy"],
            "train_precision": train_score["precision"],
            "train_recall": train_score["recall"],
            "val_accuracy": val_score["accuracy"],
            "val_precision": val_score["precision"],
            "val_recall": val_score["recall"]
        }
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """评估模型性能"""
        if isinstance(self.model, nn.Module):
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                predictions = (self.model(X_tensor).numpy() > 0.5).astype(int)
        else:
            predictions = self.model.predict(X)
            
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
        return {
            "accuracy": accuracy_score(y, predictions),
            "precision": precision_score(y, predictions),
            "recall": recall_score(y, predictions)
        }
        
    def predict(self, 
                structure: Structure,
                site_idx: int,
                doping_element: str) -> float:
        """
        预测掺杂位点的适合性得分
        
        Args:
            structure: 晶体结构
            site_idx: 位点索引
            doping_element: 掺杂元素
            
        Returns:
            适合性得分 (0-1)
        """
        if not self.model:
            raise ValueError("模型未训练")
            
        # 提取特征
        site_features = self.extract_site_features(structure, site_idx)
        doping_features = self.extract_doping_features(doping_element)
        combined_features = self.combine_features(site_features, doping_features)
        
        # 标准化
        X = self.scaler.transform(combined_features.reshape(1, -1))
        
        # 预测
        if isinstance(self.model, nn.Module):
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                score = self.model(X_tensor).item()
        else:
            score = self.model.predict_proba(X)[0][1]  # 获取正类概率
            
        return score
        
    def save_model(self, path: str):
        """保存模型"""
        if isinstance(self.model, nn.Module):
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "scaler": self.scaler
            }, path)
        else:
            import joblib
            joblib.dump({
                "model": self.model,
                "scaler": self.scaler
            }, path)
            
    def load_model(self, path: str):
        """加载模型"""
        if self.model_type == "neural_net":
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.scaler = checkpoint["scaler"]
        else:
            import joblib
            saved = joblib.load(path)
            self.model = saved["model"]
            self.scaler = saved["scaler"] 