# src/train/reward.py

import math
from typing import List, Dict, Any

def compute_euclidean_distance(pos1, pos2):
    return math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])

def compute_reward(agent_traj) -> float:
    """
    Computes normalized scalar reward in range [-1.0, 1.0].
    
    Key Changes for Normalization:
    1. Base/Status scores scaled down by factor of 100 (e.g., -100 -> -1.0).
    2. Spatial rewards calculated as ratio of screen diagonal (0.0 to 1.0).
    3. Final output clipped to strictly [-1.0, 1.0].
    """
    
    status = agent_traj.failed
    steps = agent_traj.step_count
    
    # CONSTANTS
    MAX_SCREEN_DIAG = 1414.0  # Approx diagonal of 1000x1000
    NEAR_MISS_THRESHOLD = 80.0
    
    # ============================================================
    # 0. Context Extraction
    # ============================================================
    total_gt_steps = getattr(agent_traj, 'total_gt_steps', 1)
    total_gt_steps = max(1, total_gt_steps)
    gt_steps_passed = agent_traj.gt_steps_passed
    
    final_dist = float('inf')
    if hasattr(agent_traj, 'target_bbox') and agent_traj.target_bbox:
        x1, y1, x2, y2 = agent_traj.target_bbox
        tx, ty = (x1 + x2) // 2, (y1 + y2) // 2
        
        if hasattr(agent_traj, 'cursor_path') and agent_traj.cursor_path:
            end_pos = agent_traj.cursor_path[-1]
            final_dist = compute_euclidean_distance(end_pos, (tx, ty))

    # ============================================================
    # 1. Progress Reward (Base Score) -> Range [0.0, 1.0]
    # ============================================================
    # Old: 0 to 100 -> New: 0.0 to 1.0
    if status == 0:
        base_score = 1.0
    else:
        base_score = (gt_steps_passed / total_gt_steps) * 1.0

    # ============================================================
    # 2. Status & Dynamic Penalty -> Range [-1.0, +0.1]
    # ============================================================
    status_score = 0.0
    
    if status == 0:
        # Success Bonus: +0.1 (Total success = 1.1 before clip, encourages speed)
        status_score = +0.1
        
    elif status == 1: 
        # Invalid Token: -1.0 (Instant kill)
        status_score = -1.0
        
    elif status == 2: 
        # Wrong Action Type: -0.5
        status_score = -0.8
        
    elif status == 4: 
        # Timeout: -0.15
        status_score = -0.15
        
    elif status == 3: 
        # Miss Logic
        if final_dist < NEAR_MISS_THRESHOLD:
            # Near Miss: -0.05 (Better than timeout)
            status_score = -0.05
        else:
            # Far Miss: -0.6 (Worse than timeout)
            status_score = -0.6

    # ============================================================
    # 3. Efficiency Penalty -> Range Small Negative
    # ============================================================
    # Old: steps * 0.5 -> New: steps * 0.005
    # e.g., 20 steps = -0.1 penalty
    step_penalty = steps * 0.005
    
    # ============================================================
    # 4. Spatial Shaping (Dense Reward) -> Range [0.0, ~0.2]
    # ============================================================
    spatial_reward = 0.0
    
    if status != 1 and hasattr(agent_traj, 'cursor_path') and agent_traj.cursor_path:
        start_pos = agent_traj.cursor_path[0]
        if hasattr(agent_traj, 'target_bbox') and agent_traj.target_bbox:
             # Normalize distance by screen size (0.0 to 1.0)
             initial_dist = compute_euclidean_distance(start_pos, (tx, ty))
             
             # Improvement ratio
             improvement_raw = initial_dist - final_dist
             improvement_ratio = improvement_raw / MAX_SCREEN_DIAG
             
             # Weight: 0.1. If I move across full screen, I get +0.1
             spatial_reward += improvement_ratio * 0.1
        
        # Proximity Bonus (0.0 to 0.2)
        # Old: 20 points -> New: 0.2 points
        if final_dist < MAX_SCREEN_DIAG:
            proximity_ratio = (1.0 - (final_dist / MAX_SCREEN_DIAG))
            spatial_reward += proximity_ratio * 0.2

    # ============================================================
    # 5. Total Sum & Clipping
    # ============================================================
    total_reward = base_score + status_score - step_penalty + spatial_reward
    
    # Hard Clip to ensure strictly [-1.0, 1.0]
    # This protects against floating point drifts or weird spatial spikes
    total_reward = max(-1.0, min(1.0, total_reward))
    
    return total_reward

def batch_compute_rewards(trajectories: List[Any], **kwargs) -> List[float]:
    return [compute_reward(t) for t in trajectories]