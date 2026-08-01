from __future__ import annotations

from dataclasses import MISSING

from isaaclab.envs.mdp import UniformVelocityCommandCfg, UniformPoseCommandCfg
from isaaclab.utils import configclass


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING

@configclass
class UniformLevelPoseCommandCfg(UniformPoseCommandCfg): 
    limit_ranges: UniformPoseCommandCfg.Ranges = MISSING