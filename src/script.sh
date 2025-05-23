#!/bin/bash

# Cart Pole
python main.py --env cart_pole --run_name baseline_none1 --episodes 3000 --baseline none
python main.py --env cart_pole --run_name baseline_none_2 --episodes 3000 --baseline none
python main.py --env cart_pole --run_name baseline_none_3 --episodes 3000 --baseline none
python main.py --env cart_pole --run_name baseline_std_1 --episodes 3000 --baseline std
python main.py --env cart_pole --run_name baseline_std_2 --episodes 3000 --baseline std
python main.py --env cart_pole --run_name baseline_std_3 --episodes 3000 --baseline std
python main.py --env cart_pole --run_name baseline_statevalue_1 --episodes 3000 --baseline stateValue
python main.py --env cart_pole --run_name baseline_statevalue_2 --episodes 3000 --baseline stateValue
python main.py --env cart_pole --run_name baseline_statevalue_3 --episodes 3000 --baseline stateValue
python main.py --env cart_pole --sweep

# Lunar Lander
python main.py --env lunar_lander --run_name baseline_none --episodes 5000 --baseline none
python main.py --env lunar_lander --run_name baseline_std --episodes 5000 --baseline std
python main.py --env lunar_lander --run_name baseline_stateValue --episodes 5000 --baseline stateValue
python main.py --env lunar_lander --run_name baseline_stateValue_20K --episodes 20000 --baseline stateValue
python main.py --env lunar_lander --run_name stateValue_128 --episodes 5000 --baseline stateValue
python main.py --env lunar_lander --run_name stateValue_128 --episodes 5000 --baseline stateValue
python main.py --env lunar_lander --run_name stateValue_128 --episodes 5000 --baseline stateValue
python main.py --env lunar_lander --run_name stateValue_64 --episodes 5000 --baseline stateValue
python main.py --env lunar_lander --run_name stateValue_64 --episodes 5000 --baseline stateValue
python main.py --env lunar_lander --run_name stateValue_64 --episodes 5000 --baseline stateValue
python main.py --env lunar_lander --run_name stateValue_32 --episodes 5000 --baseline stateValue
python main.py --env lunar_lander --run_name stateValue_32 --episodes 5000 --baseline stateValue
python main.py --env lunar_lander --run_name stateValue_32 --episodes 5000 --baseline stateValue
python main.py --env lunar_lander --run_name stateValue_32_scheduler --episodes 10000 --baseline stateValue
python main.py --env lunar_lander --run_name stateValue_32_scheduler --episodes 10000 --baseline stateValue
python main.py --env lunar_lander --run_name stateValue_32_scheduler --episodes 10000 --baseline stateValue
python main.py --env lunar_lander --run_name stateValue_64_scheduler --episodes 10000 --baseline stateValue
python main.py --env lunar_lander --run_name stateValue_64_scheduler --episodes 10000 --baseline stateValue
python main.py --env lunar_lander --run_name stateValue_64_scheduler --episodes 10000 --baseline stateValue
python main.py --env lunar_lander --run_name stateValue_128_scheduler --episodes 10000 --baseline stateValue
python main.py --env lunar_lander --run_name stateValue_128_scheduler --episodes 10000 --baseline stateValue
python main.py --env lunar_lander --run_name stateValue_128_scheduler --episodes 10000 --baseline stateValue
python main.py --env lunar_lander --run_name scheduler_128_norm2 --episodes 10000 --baseline stateValue
python main.py --env lunar_lander --run_name scheduler_128_norm5 --episodes 10000 --baseline stateValue
python main.py --env lunar_lander --run_name scheduler_128_norm10 --episodes 10000 --baseline stateValue
python main.py --env lunar_lander --run_name deep --episodes 10000 --baseline stateValue
python main.py --env lunar_lander --run_name deep --episodes 10000 --baseline stateValue
python main.py --env lunar_lander --run_name deep --episodes 10000 --baseline stateValue