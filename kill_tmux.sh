#!/bin/bash
bash /home/dmifsud/.tmux/plugins/tmux-safekill/scripts/safekill.sh
bash /home/dmifsud/.tmux/plugins/tmux-safekill/scripts/safekill.sh
# session="ndt_sweep"

# tmux has-session -t $session 2>/dev/null

# if [ $? != 0 ]; then
#   echo "No Sessions Running!"
# else 
#   tmux send-keys -t ${session}.0 "C-c"
#   tmux send-keys -t ${session}.1 "C-c"
#   tmux send-keys -t ${session}.2 "C-c"
#   tmux send-keys -t ${session}.3 "C-c"
#   tmux send-keys -t ${session}.4 "C-c"
#   tmux send-keys -t ${session}.5 "C-c"
#   tmux kill-session -t ${session}.0 
#   tmux kill-session -t ${session}.1 
#   tmux kill-session -t ${session}.2 
#   tmux kill-session -t ${session}.3 
#   tmux kill-session -t ${session}.4 
#   tmux kill-session -t ${session}.5 
#   # tmux send-keys -t ${session}.0 "C-c;C-c;C-c;C-c; exit" ENTER
#   # tmux send-keys -t ${session}.1 "C-c;C-c;C-c;C-c; exit" ENTER
#   # tmux send-keys -t ${session}.2 "C-c;C-c;C-c;C-c;C-c; exit" ENTER
#   # tmux send-keys -t ${session}.3 "C-c;C-c;C-c;C-c;C-c; exit" ENTER
#   # tmux send-keys -t ${session}.4 "C-c;C-c;C-c;C-c;C-c; exit" ENTER
#   # tmux send-keys -t ${session}.5 "C-c;C-c;C-c;C-c;C-c; exit" ENTER
# fi