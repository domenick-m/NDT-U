#!/bin/bash
session="ndt_sweep"

tmux has-session -t $session 2>/dev/null

if [ $? != 0 ]; then
  tmux new-session -d -s $session -x- -y-
  tmux split-window -v
  tmux split-window -v
  tmux split-window -v
  tmux split-window -v
  tmux split-window -v
  tmux split-window -v
  tmux select-layout even-vertical
  tmux send-keys -t ${session}.0 "echo test0" ENTER
  tmux send-keys -t ${session}.1 "echo test1" ENTER
  tmux send-keys -t ${session}.2 "echo test2" ENTER
  tmux send-keys -t ${session}.3 "echo test3" ENTER
  tmux send-keys -t ${session}.4 "echo test4" ENTER
  tmux send-keys -t ${session}.5 "echo test5" ENTER
fi

# Attach to created session
tmux attach-session -t $session