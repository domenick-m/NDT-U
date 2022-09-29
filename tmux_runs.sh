#!/bin/bash
session="ndt_sweep"

tmux has-session -t $session 2>/dev/null

if [ $? != 0 ]; then
  # tmux new-session -d -s $session -x- -y-
  n_gpus="$(($(nvidia-smi --list-gpus | wc -l) - 1))"
  for i in $( seq 0 $n_gpus )
  do
    if [ $i == 0 ]; then
      echo "Start sweep"
    else
      echo "Add agent"
    fi
  done

  myArray=(1 2 4 5 6)

  for i in ${!myArray[@]};
  do
    if [ $i == 0 ]; then
      echo "Start sweep on ${myArray[$i]}"
    else
      echo "Add agent on ${myArray[$i]}"
    fi
  done
  # tmux split-window -v
  # tmux split-window -v
  # tmux split-window -v
  # tmux split-window -v
  # tmux split-window -v
  # tmux select-layout even-vertical
  # tmux send-keys -t ${session}.0 "ndt;python train_cv.py --sweep -y" ENTER
  # sleep 7
  # tmux send-keys -t ${session}.1 "ndt;python train_cv.py --add" ENTER
  # sleep 7
  # tmux send-keys -t ${session}.2 "ndt;python train_cv.py --add" ENTER
  # sleep 7
  # tmux send-keys -t ${session}.3 "ndt;python train_cv.py --add" ENTER
  # sleep 7
  # tmux send-keys -t ${session}.4 "ndt;python train_cv.py --add" ENTER
  # sleep 7
  # tmux send-keys -t ${session}.5 "ndt;python train_cv.py --add" ENTER
fi

# Attach to created session
# tmux attach-session -t $session