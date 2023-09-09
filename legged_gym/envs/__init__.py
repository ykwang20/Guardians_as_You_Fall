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
from .go1.go1_fall import Go1Fall
from .go1.go1_fall_config import Go1FallCfg, Go1FallCfgPPO
from .go1.go1_fall_long_short_config import Go1Fall_LSCfg, Go1Fall_LSCfgPPO
from .go1.go1_fall_back import Go1FallBack
from .go1.go1_fall_back_config import Go1FallBackCfg, Go1FallBackCfgPPO
from .go1.go1_to_back import Go1ToBack
from.go1.go1_for_to_back_config import Go1ForBackCfg, Go1ForBackCfgPPO
from .go1.selector import Go1Selector
from .go1.selector_config import SelectorCfg, SelectorCfgPPO
from .go1.detector import Go1Detector
from .go1.detector_config import DetectorCfg, DetectorCfgPPO
from .go1.estimator import Go1Estimator
from .go1.estimator_config import EstimatorCfg, EstimatorCfgPPO
from .go1.ball import Go1Ball
from .go1.ball_config import BallCfg, BallCfgPPO
from .go1.pit import Go1Pit
from .go1.pit_config import PitCfg, PitCfgPPO
from .go1.curr import Go1Curr
from .go1.curr_config import CurrCfg, CurrCfgPPO
from .go1.height import Go1Height
from .go1.height_config import HeightCfg, HeightCfgPPO
from .go1.draw import Go1Draw
from .go1.draw_config import Go1DrawCfg, Go1DrawCfgPPO  
from .go1.stats import Stats
from .go1.stats_config import StatsCfg, StatsCfgPPO
from .go1.curr_test import CurrTest
from .go1.curr_test_config import CurrTestCfg, CurrTestCfgPPO


import os

from legged_gym.utils.task_registry import task_registry

task_registry.register( "go1_stand", Go1Stand, Go1StandCfg(), Go1StandCfgPPO() )
task_registry.register( "go1_fall", Go1Fall, Go1FallCfg(), Go1FallCfgPPO() )
task_registry.register( "ls", Go1Fall, Go1Fall_LSCfg(), Go1Fall_LSCfgPPO() )
task_registry.register( "back", Go1FallBack, Go1FallBackCfg(), Go1FallBackCfgPPO() )
task_registry.register( "for_back", Go1ToBack, Go1ForBackCfg(), Go1ForBackCfgPPO() )
task_registry.register( "selector", Go1Selector, SelectorCfg(), SelectorCfgPPO() )
task_registry.register( "detector", Go1Detector, DetectorCfg(), DetectorCfgPPO() )
task_registry.register( "estimator", Go1Estimator, EstimatorCfg(), EstimatorCfgPPO() )
task_registry.register( "ball", Go1Ball, BallCfg(), BallCfgPPO() )
task_registry.register( "pit", Go1Pit, PitCfg(), PitCfgPPO() )
task_registry.register( "curr", Go1Curr, CurrCfg(), CurrCfgPPO() )
task_registry.register( "height", Go1Height, HeightCfg(), HeightCfgPPO() )
task_registry.register( "draw", Go1Draw, Go1DrawCfg(), Go1DrawCfgPPO())
task_registry.register( "stats", Stats, StatsCfg(), StatsCfgPPO() )
task_registry.register( "curr_test", CurrTest, CurrTestCfg(), CurrTestCfgPPO() )

