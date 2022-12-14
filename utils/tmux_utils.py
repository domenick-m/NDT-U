import subprocess
import time
import libtmux
import os

def start_tmux_sweep(agent_gpus, sweep_id):
    if subprocess.getoutput('tmux has-session -t ndt_sweep'):
        server = libtmux.Server()
        session = server.new_session(session_name="ndt_sweep", kill_session=True, attach=False)
        window = session.attached_window

        n_gpus = int(subprocess.getoutput('nvidia-smi --list-gpus | wc -l'))
        pane = window.attached_pane
        pane.send_keys('ndt;c')
        for i in range(n_gpus - 1):
            pane = window.split_window(vertical=True)
            window.select_layout('even-vertical')
            pane.send_keys('ndt;c')
            window.select_layout('even-vertical')

        if agent_gpus == []:
            agent_gpus = [i for i in range(n_gpus)]

        for gpu in agent_gpus:
            window.select_pane(gpu).send_keys(f'./train.py --add {sweep_id} --silent --gpu {gpu}')

        server.attach_session(target_session="ndt_sweep")

    else:
        print("Session exists! \nCall ./train.py --kill to end it.")

# bash = '''#!/usr/bin/env bash
# set -e

# # non-interactive shell won't source in ~/.bashrc by default
# # use our own bash configuration
# if [ -f ~/.bashrc_tmux_safekill ]; then
#   source ~/.bashrc_tmux_safekill
# fi

# # List of variables interacting with env configurations
# # - 'TMUX_SAFEKILL_FORCE'
# # 1 for set to proceed with forcequit approach, otherwise (default) no forcequit applied
# envcfg_use_forcequit_approach=0

# # Bootstrap env configurations
# function read_env_config {
#   if [[ $TMUX_SAFEKILL_FORCE == 1 ]]; then
#     envcfg_use_forcequit_approach=1
#   fi
# }

# function safe_end_procs {
#   old_ifs="$IFS"
#   IFS=$'\\n'

#   for pane_set in $1; do
#     pane_id=$(echo "$pane_set" | awk -F " " '{print $1}')
#     pane_proc=$(echo "$pane_set" | awk -F " " '{print tolower($2)}')
#     cmd="C-c"
#     if [[ "$pane_proc" == "vi" ]] || [[ "$pane_proc" == "vim" ]] || [[ "$pane_proc" == "nvim" ]]; then
#       cmd=$([[ $envcfg_use_forcequit_approach == 1 ]] && echo 'Escape ":qa!" Enter' || echo 'Escape ":qa" Enter')
#     elif [[ "$pane_proc" == "man" ]] || [[ "$pane_proc" == "less" ]]; then
#       cmd='"q"'
#     elif [[ "$pane_proc" == "bash" ]] || [[ "$pane_proc" == "zsh" ]] || [[ "$pane_proc" == "fish" ]]; then
#       cmd='C-c C-u space "exit" Enter'
#     elif [[ "$pane_proc" == "ssh" ]] || [[ "$pane_proc" == "vagrant" ]]; then
#       cmd='Enter "~."'
#     elif [[ "$pane_proc" == "psql" ]] || [[ "$pane_proc" == "mysql" ]]; then
#       cmd='Enter "\q"'
#     fi
#     echo $cmd | xargs tmux send-keys -t "$pane_id"
#   done
#   IFS="$old_ifs"
# }

# function safe_kill_panes_of_current_session {
#   session_name=$(tmux display-message -p '#S')
#   current_panes=$(tmux list-panes -a -F "#{pane_id} #{pane_current_command} #{session_name}\n" | grep "$session_name")

#   SAVEIFS="$IFS"
#   IFS=$'\\n'
#   array=($current_panes)
#   # Restore IFS
#   IFS=$SAVEIFS
#   for (( i=0; i<${#array[@]}; i++ ))
#   do
#     safe_end_procs "${array[$i]}"
#     sleep 0.8
#   done
# }

# read_env_config
# safe_kill_panes_of_current_session
# # Kill a second time to catch any survivors
# safe_kill_panes_of_current_session
# exit 0'''