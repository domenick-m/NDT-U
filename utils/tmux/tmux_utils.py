
def start_tmux_sweep(ag_gpus):
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

        if ag_gpus == []:
            window.select_pane(0).send_keys(f'./train.py --sweep -y --silent --gpu -1')
        else:
            if ag_gpus == [-1]:
                ag_gpus = [i for i in range(n_gpus)]

            for gpu in ag_gpus:
                if gpu == 0: 
                    window.select_pane(gpu).send_keys(f'./train.py --sweep -y --silent --gpu {gpu}')
                    time.sleep(3)
                else: 
                    window.select_pane(gpu).send_keys(f'./train.py --add --silent --gpu {gpu}')
        
        server.attach_session(target_session="ndt_sweep")

    else:
        print("Session exists! \nCall ./train.py --kill to end it.")