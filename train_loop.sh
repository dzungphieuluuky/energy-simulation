#!/bin/bash

# --- Configuration ---
NUM_EPISODES=30
MCR_PATH="/opt/mcr/R2025a"
SIMULATION_EXECUTABLE="./runSimulationWithAnimation"
SCENARIOS=("high_speed")
NUM_SCENARIOS=${#SCENARIOS[@]}
TRIGGER_TEXT="Simulation completed!"

# --- Main Training Loop ---
echo "Starting Sledgehammer automated training for $NUM_EPISODES episodes..."

for i in $(seq 1 $NUM_EPISODES); do
    scenario_index=$(( (i - 1) % NUM_SCENARIOS ))
    current_scenario=${SCENARIOS[$scenario_index]}
    temp_log="episode_${i}.log"

    echo "-----------------------------------------------------"
    echo "STARTING EPISODE $i / $NUM_EPISODES  |  SCENARIO: $current_scenario"
    echo "-----------------------------------------------------"

    # Set up MATLAB Runtime environment
    export LD_LIBRARY_PATH=.:${MCR_PATH}/runtime/glnxa64:${MCR_PATH}/bin/glnxa64:${MCR_PATH}/sys/os/glnxa64:${MCR_PATH}/sys/opengl/lib/glnxa64

    # 1. Execute the simulation binary IN THE BACKGROUND
    "$SIMULATION_EXECUTABLE" "$current_scenario" > "$temp_log" 2>&1 &
    sim_pid=$!
    echo "Simulation executable started with PID: $sim_pid"

    # 2. Monitor the log file for the trigger text
    echo "Waiting for '$TRIGGER_TEXT' in log file..."
    while true; do
        if grep -q "$TRIGGER_TEXT" "$temp_log"; then
            echo "Trigger text found. Main simulation is complete."
            break
        fi
        if ! kill -0 $sim_pid 2>/dev/null; then
            echo "ERROR: Simulation process PID $sim_pid died unexpectedly. Check log:"
            cat "$temp_log"
            exit 1
        fi
        sleep 5
    done

    # 3. THE SLEDGEHAMMER: Forcefully kill all known simulation and GUI processes by name.
    # We use pkill with SIGKILL (-9) which is un-ignorable.
    echo "Terminating hung processes by name (using SIGKILL)..."
    pkill -9 -f "runSimulationWithAnimation"
    pkill -9 -f "Xvfb"
    pkill -9 -f "fluxbox"
    pkill -9 -f "MATLABWindow"
    pkill -9 -f "matlabwindowhel" # Kills all helpers
    
    # Brief pause to allow the OS to process the kill signals
    sleep 2

    # Final check to ensure the main process is gone before continuing
    wait $sim_pid 2>/dev/null

    echo "EPISODE $i completed and all processes have been terminated."
    rm "$temp_log"

done

echo "-----------------------------------------------------"
echo "Automated training finished all $NUM_EPISODES episodes."
echo "-----------------------------------------------------"
