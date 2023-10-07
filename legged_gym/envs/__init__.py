# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot
from .go1.go1_stand import Go1Stand
from .go1.go1_stand_config import Go1StandCfg, Go1StandCfgPPO
from .go1.go1_recover import Go1Recover
from .go1.go1_recover_config import Go1RecoverCfg, Go1RecoverCfgPPO
from .go1.go1_fall_dr import Go1FallDr
from .go1.go1_fall_dr_config import Go1FallDrCfg, Go1FallDrCfgPPO
from .go1.selector import Go1Selector
from .go1.selector_config import SelectorCfg, SelectorCfgPPO
from .go1.ball import Go1Ball
from .go1.ball_config import BallCfg, BallCfgPPO
from .go1.curr import Go1Curr
from .go1.curr_config import CurrCfg, CurrCfgPPO


import os

from legged_gym.utils.task_registry import task_registry

task_registry.register( "go1_stand", Go1Stand, Go1StandCfg(), Go1StandCfgPPO() )
task_registry.register( "go1_recover", Go1Recover, Go1RecoverCfg(), Go1RecoverCfgPPO() )
task_registry.register( "dr", Go1FallDr, Go1FallDrCfg(), Go1FallDrCfgPPO() )
task_registry.register( "selector", Go1Selector, SelectorCfg(), SelectorCfgPPO() )
task_registry.register( "ball", Go1Ball, BallCfg(), BallCfgPPO() )
task_registry.register( "curr", Go1Curr, CurrCfg(), CurrCfgPPO() )

