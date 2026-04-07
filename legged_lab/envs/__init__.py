# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg

from legged_lab.envs.tienkung.run_cfg import TienKungRunAgentCfg, TienKungRunFlatEnvCfg
from legged_lab.envs.tienkung.run_with_sensor_cfg import (
    TienKungRunWithSensorAgentCfg,
    TienKungRunWithSensorFlatEnvCfg,
)
from legged_lab.envs.tienkung.tienkung_env import TienKungEnv
from legged_lab.envs.tienkung.walk_cfg import (
    TienKungWalkAgentCfg,
    TienKungWalkFlatEnvCfg,
)
from legged_lab.envs.tienkung.walk_with_sensor_cfg import (
    TienKungWalkWithSensorAgentCfg,
    TienKungWalkWithSensorFlatEnvCfg,
)

# e1_21dof
from legged_lab.envs.e1_21dof.e1_21dof_env import E1_21DOF_Env
from legged_lab.envs.e1_21dof.run_cfg import E1_21DOF_RunAgentCfg, E1_21DOF_RunFlatEnvCfg
from legged_lab.envs.e1_21dof.walk_cfg import (
    E1_21DOF_WalkAgentCfg,
    E1_21DOF_WalkFlatEnvCfg,
)

# e1_12dof
from legged_lab.envs.e1_12dof.e1_12dof_env import E1_12DOF_Env
from legged_lab.envs.e1_12dof.run_cfg import E1_12DOF_RunAgentCfg, E1_12DOF_RunFlatEnvCfg
from legged_lab.envs.e1_12dof.walk_cfg import (
    E1_12DOF_WalkAgentCfg,
    E1_12DOF_WalkFlatEnvCfg,
)

# tienkung_12dof
from legged_lab.envs.tienkung_12dof.tienkung_12dof_env import TienKung12DOFEnv
from legged_lab.envs.tienkung_12dof.walk_cfg import (
    TienKung12DOFWalkAgentCfg,
    TienKung12DOFWalkFlatEnvCfg,
)

# g1_12dof
from legged_lab.envs.g1_12dof.g1_12dof_env import G1_12DOFEnv
from legged_lab.envs.g1_12dof.walk_cfg import (
    G1_12DOFWalkAgentCfg,
    G1_12DOFWalkFlatEnvCfg,
)

from legged_lab.utils.task_registry import task_registry

task_registry.register("walk", TienKungEnv, TienKungWalkFlatEnvCfg(), TienKungWalkAgentCfg())
task_registry.register("run", TienKungEnv, TienKungRunFlatEnvCfg(), TienKungRunAgentCfg())
task_registry.register(
    "walk_with_sensor", TienKungEnv, TienKungWalkWithSensorFlatEnvCfg(), TienKungWalkWithSensorAgentCfg()
)
task_registry.register(
    "run_with_sensor", TienKungEnv, TienKungRunWithSensorFlatEnvCfg(), TienKungRunWithSensorAgentCfg()
)


task_registry.register("e1_21dof_walk", E1_21DOF_Env, E1_21DOF_WalkFlatEnvCfg(), E1_21DOF_WalkAgentCfg())
task_registry.register("e1_21dof_run", E1_21DOF_Env, E1_21DOF_RunFlatEnvCfg(), E1_21DOF_RunAgentCfg())

task_registry.register("e1_12dof_walk", E1_12DOF_Env, E1_12DOF_WalkFlatEnvCfg(), E1_12DOF_WalkAgentCfg())
task_registry.register("tienkung_12dof_walk", TienKung12DOFEnv, TienKung12DOFWalkFlatEnvCfg(), TienKung12DOFWalkAgentCfg())
task_registry.register("g1_12dof_walk", G1_12DOFEnv, G1_12DOFWalkFlatEnvCfg(), G1_12DOFWalkAgentCfg())


