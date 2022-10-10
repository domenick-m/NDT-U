from utils.data.create_local_t5data import get_trial_data
from utils_f import get_config
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime
from utils.plot.plot_true_vs_pred_mvmnt import plot_true_vs_pred_mvmnt


def test():

    config = get_config({})

    std_list = [60]
    lag_list = [80]
    # std_list = [i for i in range(10, 410, 10)]
    # lag_list = [i for i in range(10, 410, 10)]
    lag_max = max(lag_list)

    print('Generating data... (This may take a while)')

    train_trials, test_trials = get_trial_data(config, std_list, lag_max)

    lag_max = int(lag_max / config.data.bin_size)
    lag_list = [int(i / config.data.bin_size) for i in lag_list]

    top_name = ''
    top_score = float('-inf')

    print('\nTraining Decoders...')

    now = datetime.now()
    with open(f'{now.strftime("%d.%m.%Y_%H.%M")}.csv', 'w') as f:
        f.write('lag, std, r2\n')

        for lag in lag_list:
            lag_delta = -(lag_max - lag) if lag_max != lag else None

            for std in std_list:
                tr_ids, velocity, smth_spks,  = [], [], []
                for session in train_trials.values():
                    for block in session['ol_blocks'].values():
                        for tr_id, trial in block.items():
                            tr_ids.append(tr_id)
                            velocity.append(trial.decVel[lag:lag_delta])
                            smth_spks.append(trial[f'spikes_smth_{std}'][:-lag_max])
                
                smth_spks_arr = np.concatenate(smth_spks, 0)
                velocity_arr = np.concatenate(velocity, 0)

                OLE = make_pipeline(
                    StandardScaler(), 
                    GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 0, 9)})
                )
                OLE.fit(smth_spks_arr, velocity_arr)
                score = OLE.score(smth_spks_arr, velocity_arr)
                pred_vel = [OLE.predict(i) for i in smth_spks]

                result = f'lag:{lag * config.data.bin_size} std:{std} score:{score:.4f}'
                print(result)
                f.write(f'{lag * config.data.bin_size},{std},{score:.6f}\n')

                if score > top_score:
                    top_score = score
                    top_name = f'\n\n TOP -> {result}'

                html = plot_true_vs_pred_mvmnt(tr_ids, pred_vel, velocity)

                with open(f"pred_vs_true_mvmnt.html", "w") as f:
                    f.write(html)


        print(top_name)
        

